"""CUDA Graph decode benchmark for the Qwen3 TLE tutorial engine.

This bench intentionally keeps CUDA Graph support local to the benchmark.  It
measures the upside before changing the serving path or kernel interfaces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from kernels import attention_decode, attention_ws  # noqa: E402
from models import Qwen3TLEEngine  # noqa: E402


class RuntimeAttentionDecodeEngine(Qwen3TLEEngine):
    """Use the non-autotuned decode path where REAL_KV_LEN is a runtime arg."""

    def _attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        q_len: int,
        start_pos: int,
        kv_len: int,
        sm_scale: float,
    ) -> torch.Tensor:
        if q_len == 1:
            return attention_decode(
                q,
                k_cache,
                v_cache,
                q_len=q_len,
                start_pos=start_pos,
                kv_len=kv_len,
                sm_scale=sm_scale,
                block_n=128,
            )
        return attention_ws(q, k_cache, v_cache, q_len=q_len, start_pos=start_pos, kv_len=kv_len, sm_scale=sm_scale)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 TLE decode CUDA Graph benchmark")
    parser.add_argument("--model-path", default="/data/dataset/Qwen3-32B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scenario", action="append", default=[],
                        help="Scenario as prompt_len:decode_steps. Can be repeated.")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="build/mega/qwen3-32b")
    return parser


def _parse_scenarios(values: list[str]) -> list[tuple[int, int]]:
    if not values:
        values = ["1:16", "128:16", "512:16", "1024:16"]
    scenarios = []
    for value in values:
        prompt, decode = value.split(":", 1)
        scenarios.append((int(prompt), int(decode)))
    return scenarios


def _random_input_ids(engine: Qwen3TLEEngine, batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.randint(0, int(engine.config.vocab_size), (batch_size, seq_len), device=engine.device,
                         dtype=torch.long)


def _decode_step(engine: Qwen3TLEEngine, token: torch.Tensor) -> torch.Tensor:
    logits = engine.decode(token)
    return torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)


def _compile_decode_path(
    engine: Qwen3TLEEngine,
    input_ids: torch.Tensor,
    *,
    max_seq_len: int,
    decode_steps: int,
) -> None:
    engine.reset_cache(batch_size=input_ids.shape[0], max_seq_len=max_seq_len)
    engine.prefill(input_ids)
    token = _random_input_ids(engine, input_ids.shape[0], 1)
    for _ in range(decode_steps):
        token = _decode_step(engine, token)
    torch.cuda.synchronize()


def _bench_eager_decode(
    engine: Qwen3TLEEngine,
    input_ids: torch.Tensor,
    *,
    max_seq_len: int,
    decode_steps: int,
    warmup: int,
    iters: int,
) -> float:
    batch_size = input_ids.shape[0]
    for _ in range(warmup):
        engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
        engine.prefill(input_ids)
        token = _random_input_ids(engine, batch_size, 1)
        for _ in range(decode_steps):
            token = _decode_step(engine, token)
    torch.cuda.synchronize()

    elapsed = 0.0
    for _ in range(iters):
        engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
        engine.prefill(input_ids)
        token = _random_input_ids(engine, batch_size, 1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(decode_steps):
            token = _decode_step(engine, token)
        end.record()
        torch.cuda.synchronize()
        elapsed += float(start.elapsed_time(end))
    return elapsed / max(iters * decode_steps, 1)


class DecodeGraphRunner:
    def __init__(
        self,
        engine: Qwen3TLEEngine,
        input_ids: torch.Tensor,
        *,
        max_seq_len: int,
        decode_steps: int,
        mode: str,
    ) -> None:
        if mode not in {"per_step", "full_span"}:
            raise ValueError(f"unknown graph mode {mode!r}")
        self.engine = engine
        self.input_ids = input_ids
        self.max_seq_len = max_seq_len
        self.decode_steps = decode_steps
        self.mode = mode
        self.static_token = torch.empty((input_ids.shape[0], 1), device=engine.device, dtype=torch.long)
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.graph: torch.cuda.CUDAGraph | None = None

    def _decode_and_update_token(self) -> None:
        next_token = _decode_step(self.engine, self.static_token)
        self.static_token.copy_(next_token)

    def capture(self) -> None:
        batch_size = self.input_ids.shape[0]
        _compile_decode_path(self.engine, self.input_ids, max_seq_len=self.max_seq_len, decode_steps=self.decode_steps)

        self.engine.reset_cache(batch_size=batch_size, max_seq_len=self.max_seq_len)
        self.engine.prefill(self.input_ids)
        self.static_token.copy_(_random_input_ids(self.engine, batch_size, 1))
        torch.cuda.synchronize()

        if self.mode == "per_step":
            self.graphs = []
            for _ in range(self.decode_steps):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._decode_and_update_token()
                self.graphs.append(graph)
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(self.decode_steps):
                    self._decode_and_update_token()
            self.graph = graph
        torch.cuda.synchronize()

    def replay(self, initial_token: torch.Tensor) -> None:
        self.static_token.copy_(initial_token)
        if self.mode == "per_step":
            for graph in self.graphs:
                graph.replay()
        else:
            assert self.graph is not None
            self.graph.replay()


def _bench_graph_decode(
    runner: DecodeGraphRunner,
    *,
    warmup: int,
    iters: int,
) -> float:
    engine = runner.engine
    input_ids = runner.input_ids
    batch_size = input_ids.shape[0]
    runner.capture()

    for _ in range(warmup):
        engine.reset_cache(batch_size=batch_size, max_seq_len=runner.max_seq_len)
        engine.prefill(input_ids)
        initial_token = _random_input_ids(engine, batch_size, 1)
        runner.replay(initial_token)
    torch.cuda.synchronize()

    elapsed = 0.0
    for _ in range(iters):
        engine.reset_cache(batch_size=batch_size, max_seq_len=runner.max_seq_len)
        engine.prefill(input_ids)
        initial_token = _random_input_ids(engine, batch_size, 1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner.replay(initial_token)
        end.record()
        torch.cuda.synchronize()
        elapsed += float(start.elapsed_time(end))
    return elapsed / max(iters * runner.decode_steps, 1)


def _make_engine(
    engine_cls: type[Qwen3TLEEngine],
    *,
    model_path: str,
    device: str,
    max_seq_len: int,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Qwen3TLEEngine:
    return engine_cls.from_pretrained(
        model_path,
        device=device,
        dtype="bf16",
        max_seq_len=max_seq_len,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        attention_backend="ws",
    )


def _time_mode(
    name: str,
    fn: Callable[[], float],
    *,
    prompt_len: int,
    decode_steps: int,
    max_seq_len: int,
) -> dict:
    torch.cuda.synchronize()
    ms = fn()
    return {
        "scenario": f"prefill{prompt_len}_decode{decode_steps}",
        "mode": name,
        "prompt_len": prompt_len,
        "decode_steps": decode_steps,
        "max_seq_len": max_seq_len,
        "decode_ms_per_token": ms,
        "decode_tokens_per_s": 1000.0 / ms,
    }


def _write_report(path: Path, summary: dict) -> None:
    by_scenario: dict[str, list[dict]] = {}
    for row in summary["rows"]:
        by_scenario.setdefault(row["scenario"], []).append(row)
    lines = [
        "# Qwen3 TLE Decode CUDA Graph Benchmark",
        "",
        f"- Created: `{summary['created_utc']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Device: `{summary['device']}`",
        f"- Warmup: `{summary['warmup']}`",
        f"- Iters: `{summary['iters']}`",
        "",
        "| scenario | mode | decode ms/token | tok/s | speedup vs eager |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario, rows in by_scenario.items():
        eager = next(row["decode_ms_per_token"] for row in rows if row["mode"] == "eager")
        for row in rows:
            speedup = eager / row["decode_ms_per_token"]
            lines.append(
                f"| {scenario} | `{row['mode']}` | {row['decode_ms_per_token']:.3f} | "
                f"{row['decode_tokens_per_s']:.3f} | {speedup:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size != 1:
        raise ValueError("CUDA graph decode bench currently expects batch_size=1")

    scenarios = _parse_scenarios(args.scenario)
    max_seq_len = args.max_seq_len or max(prompt + decode for prompt, decode in scenarios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    eager_engine = _make_engine(
        Qwen3TLEEngine,
        model_path=args.model_path,
        device=args.device,
        max_seq_len=max_seq_len,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    runtime_engine = RuntimeAttentionDecodeEngine(
        config=eager_engine.config,
        weights=eager_engine.weights,
        tokenizer=eager_engine.tokenizer,
        device=eager_engine.device,
        dtype=eager_engine.dtype,
        max_seq_len=eager_engine.max_seq_len,
        attention_backend=eager_engine.attention_backend,
    )

    rows = []
    for prompt_len, decode_steps in scenarios:
        scenario_max_seq_len = args.max_seq_len or (prompt_len + decode_steps)
        input_ids = _random_input_ids(eager_engine, args.batch_size, prompt_len)
        runtime_input_ids = input_ids

        rows.append(
            _time_mode(
                "eager",
                lambda input_ids=input_ids, scenario_max_seq_len=scenario_max_seq_len, decode_steps=decode_steps:
                _bench_eager_decode(eager_engine, input_ids, max_seq_len=scenario_max_seq_len,
                                    decode_steps=decode_steps, warmup=args.warmup, iters=args.iters),
                prompt_len=prompt_len,
                decode_steps=decode_steps,
                max_seq_len=scenario_max_seq_len,
            ))
        rows.append(
            _time_mode(
                "graph_per_step",
                lambda input_ids=input_ids, scenario_max_seq_len=scenario_max_seq_len, decode_steps=decode_steps:
                _bench_graph_decode(
                    DecodeGraphRunner(eager_engine, input_ids, max_seq_len=scenario_max_seq_len,
                                      decode_steps=decode_steps, mode="per_step"),
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                prompt_len=prompt_len,
                decode_steps=decode_steps,
                max_seq_len=scenario_max_seq_len,
            ))
        rows.append(
            _time_mode(
                "graph_full_span",
                lambda input_ids=input_ids, scenario_max_seq_len=scenario_max_seq_len, decode_steps=decode_steps:
                _bench_graph_decode(
                    DecodeGraphRunner(eager_engine, input_ids, max_seq_len=scenario_max_seq_len,
                                      decode_steps=decode_steps, mode="full_span"),
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                prompt_len=prompt_len,
                decode_steps=decode_steps,
                max_seq_len=scenario_max_seq_len,
            ))
        rows.append(
            _time_mode(
                "runtime_attn_eager",
                lambda runtime_input_ids=runtime_input_ids, scenario_max_seq_len=scenario_max_seq_len,
                decode_steps=decode_steps: _bench_eager_decode(runtime_engine, runtime_input_ids,
                                                              max_seq_len=scenario_max_seq_len,
                                                              decode_steps=decode_steps, warmup=args.warmup,
                                                              iters=args.iters),
                prompt_len=prompt_len,
                decode_steps=decode_steps,
                max_seq_len=scenario_max_seq_len,
            ))
        rows.append(
            _time_mode(
                "runtime_attn_graph_full_span",
                lambda runtime_input_ids=runtime_input_ids, scenario_max_seq_len=scenario_max_seq_len,
                decode_steps=decode_steps: _bench_graph_decode(
                    DecodeGraphRunner(runtime_engine, runtime_input_ids, max_seq_len=scenario_max_seq_len,
                                      decode_steps=decode_steps, mode="full_span"),
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                prompt_len=prompt_len,
                decode_steps=decode_steps,
                max_seq_len=scenario_max_seq_len,
            ))
        for row in rows[-5:]:
            print(json.dumps(row, sort_keys=True))

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": args.model_path,
        "device": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    json_path = output_dir / f"decode-cuda-graph-{run_id}.json"
    md_path = output_dir / f"decode-cuda-graph-{run_id}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(md_path, summary)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
