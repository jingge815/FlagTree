"""Compare TLE Qwen3 GQA attention kernels with TileOps/TileLang kernels.

Example:
  conda run -n flagtree python python/tutorials/tle/mega/bench/bench_attention_tilelang_compare.py \
      --tileops-path /tmp/TileOPs --prefill-len 128 --prefill-len 512 \
      --decode-kv-len 128 --decode-kv-len 4096
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from kernels.attention import attention_decode, attention_ws  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TLE vs TileOps/TileLang Qwen3 GQA attention benchmark")
    parser.add_argument("--tileops-path", default="/tmp/TileOPs",
                        help="Path to a TileOPs checkout. If absent, import tileops from PYTHONPATH.")
    parser.add_argument("--output-dir", default="build/mega/qwen3-32b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--prefill-len", type=int, action="append", default=[],
                        help="Prefill sequence length. Can be repeated.")
    parser.add_argument("--decode-kv-len", type=int, action="append", default=[],
                        help="Decode KV length. Can be repeated.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tle-prefill-block-n", type=int, default=128)
    parser.add_argument("--tle-decode-block-n", type=int, default=128)
    parser.add_argument("--decode-num-split", type=int, default=16)
    parser.add_argument("--tilelang-prefill-config", choices=("default", "aligned"), default="aligned")
    parser.add_argument("--tilelang-decode-config", choices=("default", "aligned"), default="aligned")
    return parser


def _append_tileops_path(path: str) -> Path | None:
    tileops_path = Path(path).expanduser().resolve()
    if tileops_path.exists() and str(tileops_path) not in sys.path:
        sys.path.insert(0, str(tileops_path))
    return tileops_path if tileops_path.exists() else None


def _tileops_commit(tileops_path: Path | None) -> str:
    if tileops_path is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(tileops_path), "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _time_cuda(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(iters, 1))


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.float() - b.float())).item())


def _prefill_config(name: str) -> dict[str, int] | None:
    if name == "default":
        return None
    return {"block_m": 128, "block_n": 128, "num_stages": 2, "threads": 256}


def _config_label(name: str, config: dict[str, int] | None) -> str:
    if config is None:
        return f"{name} (TileOps internal default)"
    return f"{name} {json.dumps(config, sort_keys=True)}"


def _decode_config(name: str, num_split: int) -> dict[str, int] | None:
    if name == "default":
        return None
    return {"block_H": 64, "block_N": 128, "num_split": num_split, "num_stages": 2, "threads": 128}


def _bench_prefill(
    *,
    gqa_fwd_kernel_cls: type,
    batch: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    warmup: int,
    iters: int,
    seed: int,
    tle_block_n: int,
    tilelang_config_name: str,
) -> dict[str, Any]:
    torch.manual_seed(seed + seq_len)
    q_tilelang = torch.randn((batch, seq_len, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn((batch, seq_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn((batch, seq_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    q_tle = q_tilelang.reshape(batch * seq_len, heads, head_dim).contiguous()
    scale = 1.0 / math.sqrt(head_dim)
    tilelang_config = _prefill_config(tilelang_config_name)

    tilelang_kernel = gqa_fwd_kernel_cls(
        batch,
        heads,
        kv_heads,
        seq_len,
        head_dim,
        is_causal=True,
        dtype=torch.bfloat16,
        config=tilelang_config,
    )

    tle_out = attention_ws(
        q_tle,
        k_cache,
        v_cache,
        q_len=seq_len,
        start_pos=0,
        kv_len=seq_len,
        sm_scale=scale,
        block_n=tle_block_n,
    )
    tilelang_out, _ = tilelang_kernel(q_tilelang, k_cache, v_cache)
    torch.cuda.synchronize()

    tle_ms = _time_cuda(
        lambda: attention_ws(
            q_tle,
            k_cache,
            v_cache,
            q_len=seq_len,
            start_pos=0,
            kv_len=seq_len,
            sm_scale=scale,
            block_n=tle_block_n,
        ),
        warmup=warmup,
        iters=iters,
    )
    tilelang_ms = _time_cuda(lambda: tilelang_kernel(q_tilelang, k_cache, v_cache), warmup=warmup, iters=iters)
    return {
        "kind": "prefill",
        "scenario": f"prefill{seq_len}",
        "batch": batch,
        "q_len": seq_len,
        "kv_len": seq_len,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "tle_kernel": "attention_ws",
        "tilelang_kernel": "TileOps GQAFwdKernel",
        "tle_config": f"block_n={tle_block_n}, block_m=128, pipe=2, warps=4",
        "tilelang_config": _config_label(tilelang_config_name, tilelang_config),
        "tle_ms": tle_ms,
        "tilelang_ms": tilelang_ms,
        "speedup_tle_vs_tilelang": tilelang_ms / tle_ms,
        "max_abs_diff": _max_diff(tle_out.reshape_as(q_tilelang), tilelang_out),
        "tokens_per_s_tle": batch * seq_len * 1000.0 / tle_ms,
        "tokens_per_s_tilelang": batch * seq_len * 1000.0 / tilelang_ms,
    }


def _bench_decode(
    *,
    gqa_decode_kernel_cls: type,
    batch: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    kv_len: int,
    warmup: int,
    iters: int,
    seed: int,
    tle_block_n: int,
    num_split: int,
    tilelang_config_name: str,
) -> dict[str, Any]:
    torch.manual_seed(seed + 100_000 + kv_len)
    q = torch.randn((batch, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn((batch, kv_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn((batch, kv_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(head_dim)
    tilelang_config = _decode_config(tilelang_config_name, num_split)
    tilelang_kernel = gqa_decode_kernel_cls(
        batch,
        heads,
        kv_heads,
        kv_len,
        head_dim,
        dtype=torch.bfloat16,
        config=tilelang_config,
    )

    tle_out = attention_decode(
        q,
        k_cache,
        v_cache,
        q_len=1,
        start_pos=kv_len - 1,
        kv_len=kv_len,
        sm_scale=scale,
        block_n=tle_block_n,
        num_split=num_split,
    )
    tilelang_out = tilelang_kernel(q, k_cache, v_cache, kv_len)
    torch.cuda.synchronize()

    tle_ms = _time_cuda(
        lambda: attention_decode(
            q,
            k_cache,
            v_cache,
            q_len=1,
            start_pos=kv_len - 1,
            kv_len=kv_len,
            sm_scale=scale,
            block_n=tle_block_n,
            num_split=num_split,
        ),
        warmup=warmup,
        iters=iters,
    )
    tilelang_ms = _time_cuda(lambda: tilelang_kernel(q, k_cache, v_cache, kv_len), warmup=warmup, iters=iters)
    threshold = num_split * tle_block_n
    path = "split" if kv_len >= threshold else "no_split"
    return {
        "kind": "decode",
        "scenario": f"decode_kv{kv_len}",
        "batch": batch,
        "q_len": 1,
        "kv_len": kv_len,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "tle_kernel": f"attention_decode.{path}",
        "tilelang_kernel": f"TileOps GQADecodeKernel.{path}",
        "tle_config": f"block_h=64, valid_block_h={heads // kv_heads}, block_n={tle_block_n}, "
        f"num_split={num_split}, tl.range_num_stages=2",
        "tilelang_config": _config_label(tilelang_config_name, tilelang_config),
        "tle_ms": tle_ms,
        "tilelang_ms": tilelang_ms,
        "speedup_tle_vs_tilelang": tilelang_ms / tle_ms,
        "max_abs_diff": _max_diff(tle_out, tilelang_out),
        "tokens_per_s_tle": batch * 1000.0 / tle_ms,
        "tokens_per_s_tilelang": batch * 1000.0 / tilelang_ms,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "scenario",
        "batch",
        "q_len",
        "kv_len",
        "heads",
        "kv_heads",
        "head_dim",
        "tle_kernel",
        "tilelang_kernel",
        "tle_config",
        "tilelang_config",
        "tle_ms",
        "tilelang_ms",
        "speedup_tle_vs_tilelang",
        "max_abs_diff",
        "tokens_per_s_tle",
        "tokens_per_s_tilelang",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# TLE vs TileOps/TileLang Qwen3 GQA Attention",
        "",
        f"- TileOps path: `{summary['tileops_path']}`",
        f"- TileOps commit: `{summary['tileops_commit']}`",
        f"- TileLang version: `{summary['tilelang_version']}`",
        f"- CUDA device: `{summary['device']}`",
        f"- Warmup: `{summary['warmup']}`",
        f"- Iters: `{summary['iters']}`",
        f"- Shape: batch={summary['batch']}, heads={summary['heads']}, kv_heads={summary['kv_heads']}, "
        f"head_dim={summary['head_dim']}, dtype=bf16",
        "",
        "TileOps source reference: "
        "https://github.com/tile-ai/TileOPs/tree/main/tileops/kernels/attention",
        "",
    ]
    if summary["tileops_ws_prefill_status"] != "available":
        lines.extend([
            "TileOps WS prefill status: skipped.",
            "",
            f"Reason: {summary['tileops_ws_prefill_status']}",
            "",
        ])

    lines.extend([
        "## Kernel Latency",
        "",
        "`speedup` is `TileLang ms / TLE ms`; values above 1.0 mean TLE is faster.",
        "",
        "| kind | scenario | TLE ms | TileLang ms | speedup | TLE tok/s | TileLang tok/s | max abs diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {kind} | {scenario} | {tle_ms} | {tilelang_ms} | {speedup} | {tle_tps} | {tl_tps} | {diff} |".format(
                kind=row["kind"],
                scenario=row["scenario"],
                tle_ms=_fmt(row["tle_ms"]),
                tilelang_ms=_fmt(row["tilelang_ms"]),
                speedup=_fmt(row["speedup_tle_vs_tilelang"]),
                tle_tps=_fmt(row["tokens_per_s_tle"]),
                tl_tps=_fmt(row["tokens_per_s_tilelang"]),
                diff=_fmt(row["max_abs_diff"]),
            ))
    lines.extend([
        "",
        "## Configs",
        "",
        "| scenario | TLE kernel | TLE config | TileLang kernel | TileLang config |",
        "|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['scenario']} | `{row['tle_kernel']}` | `{row['tle_config']}` | "
            f"`{row['tilelang_kernel']}` | `{row['tilelang_config']}` |")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _unique_path(output_dir: Path, stem: str, suffix: str, run_id: str) -> Path:
    path = output_dir / f"{stem}-{run_id}.{suffix}"
    index = 1
    while path.exists():
        path = output_dir / f"{stem}-{run_id}-{index}.{suffix}"
        index += 1
    return path


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.heads % args.kv_heads != 0:
        raise ValueError(f"heads must be divisible by kv_heads, got {args.heads} and {args.kv_heads}")

    tileops_path = _append_tileops_path(args.tileops_path)
    tilelang = importlib.import_module("tilelang")
    tilelang_language = importlib.import_module("tilelang.language")
    gqa_fwd_module = importlib.import_module("tileops.kernels.attention.gqa_fwd")
    gqa_decode_module = importlib.import_module("tileops.kernels.attention.gqa_decode")
    gqa_fwd_kernel_cls = getattr(gqa_fwd_module, "GQAFwdKernel")
    gqa_decode_kernel_cls = getattr(gqa_decode_module, "GQADecodeKernel")

    prefill_lens = args.prefill_len or [128, 512, 1024]
    decode_kv_lens = args.decode_kv_len or [1, 128, 512, 1024, 4096]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    ws_status = "available" if hasattr(tilelang_language, "tma_copy") else (
        "local tilelang.language has no T.tma_copy, so TileOps gqa_fwd_ws.py cannot compile on this environment")
    rows: list[dict[str, Any]] = []
    for seq_len in prefill_lens:
        rows.append(
            _bench_prefill(
                gqa_fwd_kernel_cls=gqa_fwd_kernel_cls,
                batch=args.batch_size,
                heads=args.heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                seq_len=seq_len,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                tle_block_n=args.tle_prefill_block_n,
                tilelang_config_name=args.tilelang_prefill_config,
            ))
    for kv_len in decode_kv_lens:
        rows.append(
            _bench_decode(
                gqa_decode_kernel_cls=gqa_decode_kernel_cls,
                batch=args.batch_size,
                heads=args.heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                kv_len=kv_len,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                tle_block_n=args.tle_decode_block_n,
                num_split=args.decode_num_split,
                tilelang_config_name=args.tilelang_decode_config,
            ))

    summary = {
        "tileops_path": str(tileops_path) if tileops_path is not None else "imported-from-pythonpath",
        "tileops_commit": _tileops_commit(tileops_path),
        "tilelang_version": getattr(tilelang, "__version__", "unknown"),
        "tileops_ws_prefill_status": ws_status,
        "device": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "iters": args.iters,
        "batch": args.batch_size,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "rows": rows,
    }
    csv_path = _unique_path(output_dir, "tilelang-attention-compare", "csv", run_id)
    json_path = _unique_path(output_dir, "tilelang-attention-compare", "json", run_id)
    report_path = _unique_path(output_dir, "tilelang-attention-compare", "md", run_id)
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
