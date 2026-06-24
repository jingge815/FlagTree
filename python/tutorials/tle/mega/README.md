# Qwen3-32B TLE Tutorial

This directory contains a first end-to-end Qwen3-32B inference path whose
transformer compute is implemented through TLE/Triton tutorial kernels.

## Layout

- `serving/`: interactive chat CLI.
- `bench/`: prefill and decode benchmark entry points.
- `models/`: Qwen3 config, HF weight extraction, KV cache, and engine logic.
- `kernels/`: TLE-backed embedding, RMS inv-scale, linear/fused gate-up-SiLU,
  RoPE/KV cache, and attention kernels.

## Chat

```bash
conda run -n flagtree python python/tutorials/tle/mega/serving/qwen3_32b_chat.py \
  --model-path Qwen/Qwen3-32B \
  --max-new-tokens 256
```

## Benchmark

```bash
conda run -n flagtree python python/tutorials/tle/mega/bench/bench_qwen3_32b.py \
  --model-path Qwen/Qwen3-32B \
  --batch-size 1 \
  --prompt-len 1024 \
  --decode-steps 16
```

Results are appended to `bench/results/qwen3_32b_tle_prefill_decode.jsonl`.

## Pull-Based Linear + Fused RMSNorm

```bash
conda run -n flagtree python build/mega/bench_same_body_linear_fused_rmsnorm.py
```

The active mega tutorial path is the Qwen3 decode shape:
`linear(x) -> fused_add_rms_norm(linear, residual)`. The implementation in
`kernels/linear_fused_rmsnorm.py` uses a pull-based tile-ready prototype:
producer CTAs only publish the output tile flag they produced, and the owner
consumer CTA polls the input tile flags it depends on before running the fused
RMSNorm tile. The old queue-based `task_graph.scheduler/runtime_state` path and
its local-queue/steal benchmark have been removed.

Correctness is covered by
`python/test/tle/integration/test_tle_mega_scheduler.py`; performance is tracked
by `build/mega/bench_same_body_linear_fused_rmsnorm.py`.

## Current Scope

The first version is intentionally a TLE end-to-end engine, not yet a single
persistent CUDA megakernel. The Python engine dispatches TLE/Triton kernels for
each Qwen3 operation and uses one shared KV cache across prefill and decode.
The next step is to fold selected layer stages into persistent `tle.pipe` +
`tle.gpu.warp_specialize` kernels, starting from attention and MLP tiles.
