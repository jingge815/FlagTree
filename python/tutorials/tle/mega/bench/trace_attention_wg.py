"""Record detailed FA3/TLE attention WG timestamps and render an SVG lane chart.

Example:
  conda run -n flagtree python python/tutorials/tle/mega/bench/trace_attention_wg.py \
      --fa3-path /tmp/flash-attention-fa3/hopper --seq-len 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from kernels.attention import (  # noqa: E402
    ATTENTION_TRACE_EVENT_NAMES,
    ATTENTION_TRACE_EVENTS,
    ATTENTION_TRACE_LANES,
    attention_ws_trace,
)


RESOURCE_COLORS = {
    "cuda_core": "#e69f00",
    "tma": "#56b4e9",
    "cp_async": "#cc79a7",
    "tensor_core": "#0072b2",
    "wait": "#d55e00",
    "global_store": "#8a6d3b",
}
RESOURCE_LABELS = {
    "cuda_core": "CUDA core",
    "tma": "TMA",
    "cp_async": "cp.async / global-smem copy",
    "tensor_core": "tensor core",
    "wait": "wait / barrier",
    "global_store": "global store",
}
STAGES = (
    ("Q load", 1, 2, "cp_async"),
    ("K0 load", 3, 4, "tma"),
    ("K/V load pipe", 5, 6, "tma"),
    ("producer K loop acquire last", 46, 47, "wait"),
    ("producer K loop copy last", 47, 48, "tma"),
    ("producer K acquire tile1", 72, 73, "wait"),
    ("producer K acquire tile2", 74, 75, "wait"),
    ("producer K acquire tile3", 76, 77, "wait"),
    ("producer V loop acquire last", 49, 50, "wait"),
    ("producer V loop copy last", 50, 51, "tma"),
    ("V tail load", 7, 8, "tma"),
    ("producer V tail acquire", 52, 53, "wait"),
    ("producer V tail copy", 53, 54, "tma"),
    ("pipe drain", 9, 10, "wait"),
    ("wait Q", 16, 17, "wait"),
    ("wait K0", 18, 19, "wait"),
    ("QK first", 20, 21, "tensor_core"),
    ("release K0", 64, 65, "wait"),
    ("softmax first", 22, 23, "cuda_core"),
    ("TLE first mask+max", 22, 55, "cuda_core"),
    ("TLE first exp", 55, 56, "cuda_core"),
    ("TLE first sum/update", 56, 57, "cuda_core"),
    ("wait K loop", 38, 39, "wait"),
    ("QK loop last", 24, 25, "tensor_core"),
    ("wait V loop", 40, 41, "wait"),
    ("PV loop last", 26, 27, "tensor_core"),
    ("release K loop last", 66, 67, "wait"),
    ("release K loop tile1", 78, 79, "wait"),
    ("release K loop tile2", 80, 81, "wait"),
    ("release K loop tile3", 82, 83, "wait"),
    ("softmax loop last", 28, 29, "cuda_core"),
    ("TLE loop mask+max", 28, 58, "cuda_core"),
    ("TLE loop exp", 58, 59, "cuda_core"),
    ("TLE loop sum/update", 59, 60, "cuda_core"),
    ("release V loop last", 68, 69, "wait"),
    ("wait V tail", 42, 43, "wait"),
    ("PV tail", 30, 31, "tensor_core"),
    ("release V tail", 70, 71, "wait"),
    ("consumer drain", 44, 45, "wait"),
    ("rescale", 32, 33, "cuda_core"),
    ("O smem stage", 34, 35, "cuda_core"),
    ("O global store", 36, 37, "global_store"),
)
LANES = ("producer", "consumer_lo", "consumer_hi", "consumer_extra")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace FA3/TLE attention WG stages and render SVG")
    parser.add_argument("--output-dir", default="build/mega/qwen3-32b")
    parser.add_argument("--fa3-path", default="/tmp/flash-attention-fa3/hopper")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tle-block-n", type=int, default=128)
    parser.add_argument("--tle-pipe-capacity", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_fa3(fa3_path: str) -> Any:
    path = Path(fa3_path).expanduser().resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import flash_attn_3._C  # noqa: F401

    return torch.ops.flash_attn_3


def _run_tle_trace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_len: int,
    scale: float,
    block_n: int,
    pipe_capacity: int,
    warmup: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for _ in range(warmup):
        attention_ws_trace(
            q, k, v, q_len=seq_len, start_pos=0, kv_len=seq_len, sm_scale=scale, block_n=block_n,
            pipe_capacity=pipe_capacity,
        )
    torch.cuda.synchronize()
    out, trace = attention_ws_trace(
        q, k, v, q_len=seq_len, start_pos=0, kv_len=seq_len, sm_scale=scale, block_n=block_n,
        pipe_capacity=pipe_capacity,
    )
    torch.cuda.synchronize()
    return out, trace


def _run_fa3_trace(
    fa3_ops: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    warmup: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def run_once(trace: torch.Tensor) -> torch.Tensor:
        out, _softmax_lse, _out_accum, _lse_accum = fa3_ops.fwd(
            q,
            k,
            v,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            scale,
            True,
            -1,
            -1,
            0,
            0.0,
            True,
            None,
            1,
            True,
            0,
            trace,
        )
        return out

    for _ in range(warmup):
        run_once(torch.zeros((len(LANES), ATTENTION_TRACE_EVENTS), device=q.device, dtype=torch.int64))
    torch.cuda.synchronize()
    trace = torch.zeros((len(LANES), ATTENTION_TRACE_EVENTS), device=q.device, dtype=torch.int64)
    out = run_once(trace)
    torch.cuda.synchronize()
    return out, trace


def _trace_segments(trace: list[list[int]]) -> tuple[int, int, dict[str, list[dict[str, Any]]]]:
    points = [value for lane in trace for value in lane if value > 0]
    if not points:
        return 0, 1, {}
    base = min(points)
    end = max(points)
    lanes: dict[str, list[dict[str, Any]]] = {}
    for lane_idx, row in enumerate(trace):
        lane_name = LANES[lane_idx] if lane_idx < len(LANES) else f"wg{lane_idx}"
        segments = []
        for stage_name, start_event, end_event, resource in STAGES:
            if start_event >= len(row) or end_event >= len(row):
                continue
            start = row[start_event]
            stop = row[end_event]
            if start > 0 and stop >= start:
                segments.append(
                    {
                        "stage": stage_name,
                        "start_tick": int(start),
                        "end_tick": int(stop),
                        "start_ns": int(start - base),
                        "duration_ns": int(stop - start),
                        "resource": resource,
                        "color": RESOURCE_COLORS[resource],
                    }
                )
        if segments:
            lanes[lane_name] = segments
    return int(base), int(max(end, base + 1)), lanes


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_svg(traces: dict[str, dict[str, Any]], path: Path) -> None:
    margin_left = 138
    margin_right = 36
    top = 48
    lane_h = 30
    panel_gap = 34
    width = 1260
    timeline_w = width - margin_left - margin_right
    total_ns = max(max(panel["end_tick"] - panel["base_tick"], 1) for panel in traces.values())
    panels_h = []
    for panel in traces.values():
        panels_h.append(38 + max(len(panel["lanes"]), 1) * lane_h)
    height = top + sum(panels_h) + panel_gap * (len(panels_h) - 1) + 86

    def x(ns: int) -> float:
        return margin_left + ns / total_ns * timeline_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#202124} .muted{fill:#5f6368} .axis{stroke:#d0d7de;stroke-width:1} .bar{rx:3;ry:3} .lane{stroke:#eceff3;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-size="18" font-weight="700">Attention WG Trace: TLE vs FA3</text>',
        '<text x="24" y="48" font-size="12" class="muted">Tile (last packed-M block, kv_head=0, batch=0), timestamps from %globaltimer. Colors encode execution resource; loop stages show the last steady-state iteration.</text>',
    ]
    y0 = top + 30
    for panel_name, panel in traces.items():
        lanes = panel["lanes"]
        lines.append(f'<text x="24" y="{y0}" font-size="15" font-weight="700">{_svg_escape(panel_name)}</text>')
        lines.append(f'<line x1="{margin_left}" y1="{y0 + 11}" x2="{width - margin_right}" y2="{y0 + 11}" class="axis"/>')
        row_y = y0 + 30
        for lane_name in LANES:
            if lane_name not in lanes:
                continue
            center = row_y + 13
            lines.append(f'<text x="24" y="{center + 4}" font-size="12">{_svg_escape(lane_name)}</text>')
            lines.append(f'<line x1="{margin_left}" y1="{center}" x2="{width - margin_right}" y2="{center}" class="lane"/>')
            for seg in lanes[lane_name]:
                sx = x(seg["start_ns"])
                ex = x(seg["start_ns"] + seg["duration_ns"])
                bw = max(ex - sx, 1.0)
                label = f'{seg["stage"]} {seg["duration_ns"] / 1000.0:.1f} us'
                lines.append(
                    f'<rect class="bar" x="{sx:.2f}" y="{row_y}" width="{bw:.2f}" height="18" '
                    f'fill="{seg["color"]}" opacity="0.88"><title>{_svg_escape(label)}</title></rect>'
                )
                if bw > 72:
                    lines.append(
                        f'<text x="{sx + 5:.2f}" y="{row_y + 13}" font-size="10" fill="#ffffff">'
                        f'{_svg_escape(label)}</text>'
                    )
            row_y += lane_h
        y0 = row_y + panel_gap

    legend_y = height - 38
    legend_x = margin_left
    for resource, color in RESOURCE_COLORS.items():
        lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="14" height="10" fill="{color}" opacity="0.88"/>')
        lines.append(
            f'<text x="{legend_x + 20}" y="{legend_y + 10}" font-size="11">'
            f'{_svg_escape(RESOURCE_LABELS[resource])}</text>'
        )
        legend_x += 178
    tick_y = height - 64
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tx = margin_left + timeline_w * frac
        ns = total_ns * frac
        lines.append(f'<line x1="{tx:.2f}" y1="{tick_y - 5}" x2="{tx:.2f}" y2="{tick_y + 5}" class="axis"/>')
        lines.append(f'<text x="{tx - 18:.2f}" y="{tick_y + 20}" font-size="10" class="muted">{ns / 1000.0:.1f} us</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def _write_markdown(data: dict[str, Any], path: Path) -> None:
    rows = []
    for backend, panel in data["traces"].items():
        for lane, segments in panel["lanes"].items():
            for seg in segments:
                rows.append(
                    "| {backend} | {lane} | {stage} | {resource} | {start:.1f} | {duration:.1f} |".format(
                        backend=backend,
                        lane=lane,
                        stage=seg["stage"],
                        resource=RESOURCE_LABELS[seg["resource"]],
                        start=seg["start_ns"] / 1000.0,
                        duration=seg["duration_ns"] / 1000.0,
                    )
                )
    body = [
        "# Attention WG Trace",
        "",
        f"- SVG: `{data['svg']}`",
        f"- JSON: `{data['json']}`",
        f"- Scenario: batch={data['scenario']['batch']}, seq_len={data['scenario']['seq_len']}, heads={data['scenario']['heads']}, kv_heads={data['scenario']['kv_heads']}, head_dim={data['scenario']['head_dim']}",
        f"- Max abs diff TLE vs FA3: {data['max_abs_diff']:.6f}",
        "",
        "| backend | lane | stage | resource | start_us | duration_us |",
        "|---|---|---|---|---:|---:|",
        *rows,
        "",
    ]
    path.write_text("\n".join(body))


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(torch.cuda.current_device())
    fa3_ops = _load_fa3(args.fa3_path)

    q_fa3 = torch.randn(
        (args.batch_size, args.seq_len, args.heads, args.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (args.batch_size, args.seq_len, args.kv_heads, args.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    q_tle = q_fa3.reshape(args.batch_size * args.seq_len, args.heads, args.head_dim).contiguous()
    scale = 1.0 / math.sqrt(args.head_dim)

    tle_out, tle_trace = _run_tle_trace(
        q_tle,
        k,
        v,
        seq_len=args.seq_len,
        scale=scale,
        block_n=args.tle_block_n,
        pipe_capacity=args.tle_pipe_capacity,
        warmup=args.warmup,
    )
    fa3_out, fa3_trace = _run_fa3_trace(fa3_ops, q_fa3, k, v, scale=scale, warmup=args.warmup)
    max_abs_diff = float(torch.max(torch.abs(tle_out.reshape_as(fa3_out).float() - fa3_out.float())).item())

    traces: dict[str, dict[str, Any]] = {}
    for name, trace_tensor in (("TLE", tle_trace), ("FA3", fa3_trace)):
        trace = trace_tensor.cpu().tolist()
        base, end, lanes = _trace_segments(trace)
        traces[name] = {
            "base_tick": base,
            "end_tick": end,
            "raw": trace,
            "lanes": lanes,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"attention-wg-trace-prefill{args.seq_len}-{stamp}"
    svg_path = output_dir / f"{stem}.svg"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    data = {
        "scenario": {
            "batch": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "tle_block_n": args.tle_block_n,
            "tle_pipe_capacity": args.tle_pipe_capacity,
        },
        "event_names": ATTENTION_TRACE_EVENT_NAMES,
        "tle_lanes": ATTENTION_TRACE_LANES,
        "lanes": LANES,
        "stages": [
            {"name": name, "start_event": start, "end_event": end, "resource": resource}
            for name, start, end, resource in STAGES
        ],
        "resource_colors": RESOURCE_COLORS,
        "max_abs_diff": max_abs_diff,
        "traces": traces,
        "svg": str(svg_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    _render_svg(traces, svg_path)
    _write_markdown(data, md_path)
    print(f"wrote {svg_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"max_abs_diff={max_abs_diff:.6f}")


if __name__ == "__main__":
    main()
