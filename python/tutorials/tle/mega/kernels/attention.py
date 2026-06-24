"""Reference TLE attention kernels for prefill and decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from ._common import cdiv, next_power_of_2, require_cuda_contiguous


FA3_PACKGQA_DECODE_BLOCK_H = 128
ATTENTION_TRACE_EVENTS = 96
ATTENTION_TRACE_LANES = ("producer", "consumer_lo", "consumer_hi")
ATTENTION_TRACE_EVENT_NAMES = {
    0: "task_start",
    1: "q_load_start",
    2: "q_load_end",
    3: "k0_load_start",
    4: "k0_load_end",
    5: "kv_pipeline_start",
    6: "kv_pipeline_end",
    7: "v_tail_load_start",
    8: "v_tail_load_end",
    9: "pipe_drain_start",
    10: "pipe_drain_end",
    16: "q_wait_start",
    17: "q_wait_end",
    18: "k0_wait_start",
    19: "k0_wait_end",
    20: "qk_first_start",
    21: "qk_first_end",
    22: "softmax_first_start",
    23: "softmax_first_end",
    24: "qk_loop_start",
    25: "qk_loop_end",
    26: "pv_loop_start",
    27: "pv_loop_end",
    28: "softmax_loop_start",
    29: "softmax_loop_end",
    30: "pv_tail_start",
    31: "pv_tail_end",
    32: "rescale_start",
    33: "rescale_end",
    34: "o_smem_store_start",
    35: "o_smem_store_end",
    36: "o_global_store_start",
    37: "o_global_store_end",
    38: "k_loop_wait_start",
    39: "k_loop_wait_end",
    40: "v_loop_wait_start",
    41: "v_loop_wait_end",
    42: "v_tail_wait_start",
    43: "v_tail_wait_end",
    44: "consumer_pipe_drain_start",
    45: "consumer_pipe_drain_end",
    46: "producer_k_loop_acquire_start",
    47: "producer_k_loop_acquire_end",
    48: "producer_k_loop_copy_end",
    49: "producer_v_loop_acquire_start",
    50: "producer_v_loop_acquire_end",
    51: "producer_v_loop_copy_end",
    52: "producer_v_tail_acquire_start",
    53: "producer_v_tail_acquire_end",
    54: "producer_v_tail_copy_end",
    55: "softmax_first_mask_max_end",
    56: "softmax_first_exp_end",
    57: "softmax_first_sum_end",
    58: "softmax_loop_mask_max_end",
    59: "softmax_loop_exp_end",
    60: "softmax_loop_sum_end",
    63: "task_end",
    64: "k0_release_start",
    65: "k0_release_end",
    66: "k_loop_release_start",
    67: "k_loop_release_end",
    68: "v_loop_release_start",
    69: "v_loop_release_end",
    70: "v_tail_release_start",
    71: "v_tail_release_end",
    72: "producer_k_loop_tile1_acquire_start",
    73: "producer_k_loop_tile1_acquire_end",
    74: "producer_k_loop_tile2_acquire_start",
    75: "producer_k_loop_tile2_acquire_end",
    76: "producer_k_loop_tile3_acquire_start",
    77: "producer_k_loop_tile3_acquire_end",
    78: "k_loop_tile1_release_start",
    79: "k_loop_tile1_release_end",
    80: "k_loop_tile2_release_start",
    81: "k_loop_tile2_release_end",
    82: "k_loop_tile3_release_start",
    83: "k_loop_tile3_release_end",
}


@triton.jit
def _attention_trace_tid():
    return tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r", [], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _attention_trace_mark(
    trace,
    active,
    lane: tl.constexpr,
    event: tl.constexpr,
    TRACE_ENABLED: tl.constexpr,
    TRACE_EVENTS: tl.constexpr,
):
    if TRACE_ENABLED:
        tid = _attention_trace_tid()
        if lane == 0:
            leader = tid % 32 == 0
        else:
            leader = tid % 128 == 0
        ts = tl.extra.cuda.globaltimer()
        tl.store(trace + lane * TRACE_EVENTS + event, ts, mask=active & leader)


@triton.jit
def _attention_trace_mark_dynamic_event(
    trace,
    active,
    lane: tl.constexpr,
    event,
    TRACE_ENABLED: tl.constexpr,
    TRACE_EVENTS: tl.constexpr,
):
    if TRACE_ENABLED:
        tid = _attention_trace_tid()
        if lane == 0:
            leader = tid % 32 == 0
        else:
            leader = tid % 128 == 0
        ts = tl.extra.cuda.globaltimer()
        tl.store(trace + lane * TRACE_EVENTS + event, ts, mask=active & leader & (event >= 0) & (event < TRACE_EVENTS))


@triton.jit
def _attention_ws_packgqa_tma_producer(
    q_writer,
    k_writer,
    v_writer,
    q,
    k_desc,
    v_desc,
    batch,
    kv_head,
    packed_m_block,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_Q_TILES: tl.constexpr,
    O_ALIAS_V: tl.constexpr,
    PIPE_BASE,
    Q_PIPE_SEQ,
    trace,
    TRACE_ENABLED: tl.constexpr,
    TRACE_EVENTS: tl.constexpr,
):
    Q_PER_KV: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    q_rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_D))
    q_cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_M, BLOCK_D))
    mask_d = offs_d < HEAD_DIM
    packed_start = packed_m_block * BLOCK_M
    packed_stop = tl.minimum(packed_start + BLOCK_M, Q_LEN * Q_PER_KV)
    max_q_token = (packed_stop - 1) // Q_PER_KV
    max_key = tl.minimum(KV_LEN, START_POS + max_q_token + 1)
    n_block_max = tl.cdiv(max_key, BLOCK_N)
    trace_m_block = tl.cdiv(Q_LEN * Q_PER_KV, BLOCK_M) - 1
    trace_active = (packed_m_block == trace_m_block) & (kv_head == 0) & (batch == 0)

    _attention_trace_mark(trace, trace_active, 0, 0, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, 0, 1, TRACE_ENABLED, TRACE_EVENTS)
    q_slot = q_writer.acquire(Q_PIPE_SEQ)
    q_smem = q_slot.q
    offs_packed = packed_start + tl.arange(0, BLOCK_M)
    q_token = offs_packed // Q_PER_KV
    q_in_group = offs_packed - q_token * Q_PER_KV
    q_head = kv_head * Q_PER_KV + q_in_group
    token_offsets = batch * Q_LEN + q_token
    q_ptrs = q + (token_offsets[:, None] * NUM_Q_HEADS + q_head[:, None]) * HEAD_DIM + offs_d[None, :]
    if FULL_Q_TILES:
        if HEAD_DIM == BLOCK_D:
            q_vals = tl.load(q_ptrs)
        else:
            q_vals = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)
    else:
        q_mask = offs_packed < Q_LEN * Q_PER_KV
        if HEAD_DIM == BLOCK_D:
            q_vals = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        else:
            q_vals = tl.load(q_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q_vals)
    q_writer.commit(Q_PIPE_SEQ)
    _attention_trace_mark(trace, trace_active, 0, 2, TRACE_ENABLED, TRACE_EVENTS)

    n_block = n_block_max - 1
    n0 = n_block * BLOCK_N
    _attention_trace_mark(trace, trace_active, 0, 3, TRACE_ENABLED, TRACE_EVENTS)
    k_slot = k_writer.acquire(PIPE_BASE)
    tle.gpu.copy(k_desc, k_slot.k, [1, 1, BLOCK_N, BLOCK_D], [batch, kv_head, n0, 0])
    k_writer.commit(PIPE_BASE)
    _attention_trace_mark(trace, trace_active, 0, 4, TRACE_ENABLED, TRACE_EVENTS)

    _attention_trace_mark(trace, trace_active, 0, 5, TRACE_ENABLED, TRACE_EVENTS)
    for tile in tl.range(1, n_block_max):
        n_block = n_block_max - 1 - tile
        n0 = n_block * BLOCK_N
        k_seq = PIPE_BASE + tile
        trace_k_acquire_event = 72 + (tile - 1) * 2
        trace_k_tile_active = trace_active & (tile <= 3)
        _attention_trace_mark(trace, trace_active, 0, 46, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark_dynamic_event(
            trace, trace_k_tile_active, 0, trace_k_acquire_event, TRACE_ENABLED, TRACE_EVENTS
        )
        k_slot = k_writer.acquire(k_seq)
        _attention_trace_mark(trace, trace_active, 0, 47, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark_dynamic_event(
            trace, trace_k_tile_active, 0, trace_k_acquire_event + 1, TRACE_ENABLED, TRACE_EVENTS
        )
        tle.gpu.copy(k_desc, k_slot.k, [1, 1, BLOCK_N, BLOCK_D], [batch, kv_head, n0, 0])
        k_writer.commit(k_seq)
        _attention_trace_mark(trace, trace_active, 0, 48, TRACE_ENABLED, TRACE_EVENTS)

        v_tile = tile - 1
        v_n0 = n0 + BLOCK_N
        v_seq = PIPE_BASE + v_tile
        _attention_trace_mark(trace, trace_active, 0, 49, TRACE_ENABLED, TRACE_EVENTS)
        v_slot = v_writer.acquire(v_seq)
        _attention_trace_mark(trace, trace_active, 0, 50, TRACE_ENABLED, TRACE_EVENTS)
        tle.gpu.copy(v_desc, v_slot.v, [1, 1, BLOCK_N, BLOCK_D], [batch, kv_head, v_n0, 0])
        v_writer.commit(v_seq)
        _attention_trace_mark(trace, trace_active, 0, 51, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, 0, 6, TRACE_ENABLED, TRACE_EVENTS)

    v_tile = n_block_max - 1
    v_seq = PIPE_BASE + v_tile
    _attention_trace_mark(trace, trace_active, 0, 7, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, 0, 52, TRACE_ENABLED, TRACE_EVENTS)
    v_slot = v_writer.acquire(v_seq)
    _attention_trace_mark(trace, trace_active, 0, 53, TRACE_ENABLED, TRACE_EVENTS)
    tle.gpu.copy(v_desc, v_slot.v, [1, 1, BLOCK_N, BLOCK_D], [batch, kv_head, 0, 0])
    v_writer.commit(v_seq)
    _attention_trace_mark(trace, trace_active, 0, 54, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, 0, 8, TRACE_ENABLED, TRACE_EVENTS)
    if O_ALIAS_V:
        _attention_trace_mark(trace, trace_active, 0, 9, TRACE_ENABLED, TRACE_EVENTS)
        v_writer.close(PIPE_BASE + v_tile + 1)
        v_writer.pipe.wait_drained()
        _attention_trace_mark(trace, trace_active, 0, 10, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, 0, 63, TRACE_ENABLED, TRACE_EVENTS)


@triton.jit
def _attention_ws_packgqa_consumer(
    q_reader,
    k_reader,
    v_reader,
    out,
    o_smem,
    batch,
    kv_head,
    packed_m_block,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    ROWS_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    O_ALIAS_V: tl.constexpr,
    FULL_Q_TILES: tl.constexpr,
    KV_LEN_MULTIPLE_OF_BLOCK_N: tl.constexpr,
    PIPE_BASE,
    Q_PIPE_SEQ,
    trace,
    TRACE_ENABLED: tl.constexpr,
    TRACE_EVENTS: tl.constexpr,
):
    Q_PER_KV: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    SOFTMAX_SCALE_LOG2: tl.constexpr = SM_SCALE * 1.4426950408889634
    offs_packed = packed_m_block * BLOCK_M + ROW_OFFSET + tl.arange(0, ROWS_M)
    q_token = offs_packed // Q_PER_KV
    q_in_group = offs_packed - q_token * Q_PER_KV
    q_head = kv_head * Q_PER_KV + q_in_group
    offs_d = tl.arange(0, BLOCK_D)
    q_rows = tl.broadcast_to(tl.arange(0, ROWS_M)[:, None], (ROWS_M, BLOCK_D))
    q_cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (ROWS_M, BLOCK_D))
    kv_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    kv_cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    q_mask = offs_packed < Q_LEN * Q_PER_KV
    mask_d = offs_d < HEAD_DIM
    zero = tl.full((), 0, tl.int32)
    Q_PART: tl.constexpr = ROW_OFFSET // ROWS_M
    q_part = tl.full((), Q_PART, tl.int32)

    token_offsets = batch * Q_LEN + q_token
    trace_m_block = tl.cdiv(Q_LEN * Q_PER_KV, BLOCK_M) - 1
    trace_active = (packed_m_block == trace_m_block) & (kv_head == 0) & (batch == 0)
    TRACE_LANE: tl.constexpr = 1 + Q_PART
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 0, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 16, TRACE_ENABLED, TRACE_EVENTS)
    q_wait = q_reader.wait(Q_PIPE_SEQ)
    q_tile = q_wait.slot.q.subslice([ROW_OFFSET, 0], [ROWS_M, BLOCK_D])
    q_tile_ptrs = tle.gpu.local_ptr(q_tile, (q_rows, q_cols))
    q_vals = tl.load(q_tile_ptrs)
    if O_ALIAS_V:
        q_reader.release(Q_PIPE_SEQ)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 17, TRACE_ENABLED, TRACE_EVENTS)
    packed_start = packed_m_block * BLOCK_M
    packed_stop = tl.minimum(packed_start + BLOCK_M, Q_LEN * Q_PER_KV)
    min_q_token = packed_start // Q_PER_KV
    max_q_token = (packed_stop - 1) // Q_PER_KV
    max_key = tl.minimum(KV_LEN, START_POS + max_q_token + 1)
    n_block_max = tl.cdiv(max_key, BLOCK_N)
    n_block_min_causal_mask = (START_POS + min_q_token) // BLOCK_N
    causal_mask_tiles = tl.maximum(n_block_max - n_block_min_causal_mask, 0)
    tail_mask_tiles = tl.where(n_block_max * BLOCK_N > max_key, 1, 0)
    masked_tiles = tl.minimum(tl.maximum(causal_mask_tiles, tail_mask_tiles), n_block_max)

    acc = tl.zeros([ROWS_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([ROWS_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([ROWS_M], dtype=tl.float32)

    n_block = n_block_max - 1
    n0 = n_block * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    if not KV_LEN_MULTIPLE_OF_BLOCK_N:
        key_mask = offs_n < KV_LEN

    _attention_trace_mark(trace, trace_active, TRACE_LANE, 18, TRACE_ENABLED, TRACE_EVENTS)
    k_wait = k_reader.wait(PIPE_BASE)
    k_smem = k_wait.slot.k.slot(zero).slot(zero)
    k_vals = tl.load(tle.gpu.local_ptr(k_smem, (kv_rows, kv_cols)))
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 19, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 20, TRACE_ENABLED, TRACE_EVENTS)
    scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 21, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 64, TRACE_ENABLED, TRACE_EVENTS)
    k_reader.release(PIPE_BASE)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 65, TRACE_ENABLED, TRACE_EVENTS)

    _attention_trace_mark(trace, trace_active, TRACE_LANE, 22, TRACE_ENABLED, TRACE_EVENTS)
    causal_mask = offs_n[None, :] <= (START_POS + q_token[:, None])
    if KV_LEN_MULTIPLE_OF_BLOCK_N:
        if FULL_Q_TILES:
            scores = tl.where(causal_mask, scores, -float("inf"))
        else:
            scores = tl.where(q_mask[:, None] & causal_mask, scores, -float("inf"))
    else:
        if FULL_Q_TILES:
            scores = tl.where(key_mask[None, :] & causal_mask, scores, -float("inf"))
        else:
            scores = tl.where(q_mask[:, None] & key_mask[None, :] & causal_mask, scores, -float("inf"))
    scores_max = tl.max(scores, axis=1)
    if FULL_Q_TILES:
        m_new = scores_max
    else:
        m_new = tl.where(q_mask, scores_max, 0.0)
    safe_m_new = tl.where(m_new == -float("inf"), 0.0, m_new)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 55, TRACE_ENABLED, TRACE_EVENTS)
    p_prev_f32 = tl.math.exp2(scores * SOFTMAX_SCALE_LOG2 - safe_m_new[:, None] * SOFTMAX_SCALE_LOG2)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 56, TRACE_ENABLED, TRACE_EVENTS)
    l_i = tl.sum(p_prev_f32, axis=1)
    p_prev = p_prev_f32.to(tl.bfloat16)
    m_i = m_new
    acc_scale = tl.full([ROWS_M], 1.0, dtype=tl.float32)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 57, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 23, TRACE_ENABLED, TRACE_EVENTS)

    masked_loop_end = tl.maximum(masked_tiles, 1)
    for tile in tl.range(1, masked_loop_end):
        n_block = n_block_max - 1 - tile
        n0 = n_block * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if not KV_LEN_MULTIPLE_OF_BLOCK_N:
            key_mask = offs_n < KV_LEN

        k_seq = PIPE_BASE + tile
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 38, TRACE_ENABLED, TRACE_EVENTS)
        k_wait = k_reader.wait(k_seq)
        k_smem = k_wait.slot.k.slot(zero).slot(zero)
        k_vals = tl.load(tle.gpu.local_ptr(k_smem, (kv_rows, kv_cols)))
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 39, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 24, TRACE_ENABLED, TRACE_EVENTS)
        scores_curr = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 25, TRACE_ENABLED, TRACE_EVENTS)

        acc = acc * acc_scale[:, None]
        v_tile = tile - 1
        v_seq = PIPE_BASE + v_tile
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 40, TRACE_ENABLED, TRACE_EVENTS)
        v_wait = v_reader.wait(v_seq)
        v_smem = v_wait.slot.v.slot(zero).slot(zero)
        v_vals = tl.load(tle.gpu.local_ptr(v_smem, (kv_rows, kv_cols)))
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 41, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 26, TRACE_ENABLED, TRACE_EVENTS)
        acc = acc + tl.dot(p_prev, v_vals, out_dtype=tl.float32)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 27, TRACE_ENABLED, TRACE_EVENTS)
        scores = scores_curr
        trace_k_release_event = 78 + (tile - 1) * 2
        trace_k_tile_active = trace_active & (tile <= 3)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 66, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark_dynamic_event(
            trace, trace_k_tile_active, TRACE_LANE, trace_k_release_event, TRACE_ENABLED, TRACE_EVENTS
        )
        k_reader.release(k_seq)
        _attention_trace_mark_dynamic_event(
            trace, trace_k_tile_active, TRACE_LANE, trace_k_release_event + 1, TRACE_ENABLED, TRACE_EVENTS
        )
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 67, TRACE_ENABLED, TRACE_EVENTS)

        _attention_trace_mark(trace, trace_active, TRACE_LANE, 28, TRACE_ENABLED, TRACE_EVENTS)
        causal_mask = offs_n[None, :] <= (START_POS + q_token[:, None])
        if KV_LEN_MULTIPLE_OF_BLOCK_N:
            if FULL_Q_TILES:
                scores = tl.where(causal_mask, scores, -float("inf"))
            else:
                scores = tl.where(q_mask[:, None] & causal_mask, scores, -float("inf"))
        else:
            tile_mask = key_mask[None, :] & causal_mask
            if FULL_Q_TILES:
                scores = tl.where(tile_mask, scores, -float("inf"))
            else:
                scores = tl.where(q_mask[:, None] & tile_mask, scores, -float("inf"))
        scores_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, scores_max)
        if not FULL_Q_TILES:
            m_new = tl.where(q_mask, m_new, 0.0)
        safe_m_new = tl.where(m_new == -float("inf"), 0.0, m_new)
        if FULL_Q_TILES:
            alpha = tl.math.exp2((m_i - safe_m_new) * SOFTMAX_SCALE_LOG2)
        else:
            alpha = tl.where(q_mask & (m_i != -float("inf")),
                             tl.math.exp2((m_i - safe_m_new) * SOFTMAX_SCALE_LOG2), 0.0)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 58, TRACE_ENABLED, TRACE_EVENTS)
        p = tl.math.exp2(scores * SOFTMAX_SCALE_LOG2 - safe_m_new[:, None] * SOFTMAX_SCALE_LOG2)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 59, TRACE_ENABLED, TRACE_EVENTS)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        p_prev = p.to(tl.bfloat16)
        acc_scale = alpha
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 60, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 68, TRACE_ENABLED, TRACE_EVENTS)
        v_reader.release(v_seq)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 69, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 29, TRACE_ENABLED, TRACE_EVENTS)

    for tile in tl.range(masked_loop_end, n_block_max):
        n_block = n_block_max - 1 - tile

        k_seq = PIPE_BASE + tile
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 38, TRACE_ENABLED, TRACE_EVENTS)
        k_wait = k_reader.wait(k_seq)
        k_smem = k_wait.slot.k.slot(zero).slot(zero)
        k_vals = tl.load(tle.gpu.local_ptr(k_smem, (kv_rows, kv_cols)))
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 39, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 24, TRACE_ENABLED, TRACE_EVENTS)
        scores_curr = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 25, TRACE_ENABLED, TRACE_EVENTS)

        acc = acc * acc_scale[:, None]
        v_tile = tile - 1
        v_seq = PIPE_BASE + v_tile
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 40, TRACE_ENABLED, TRACE_EVENTS)
        v_wait = v_reader.wait(v_seq)
        v_smem = v_wait.slot.v.slot(zero).slot(zero)
        v_vals = tl.load(tle.gpu.local_ptr(v_smem, (kv_rows, kv_cols)))
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 41, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 26, TRACE_ENABLED, TRACE_EVENTS)
        acc = acc + tl.dot(p_prev, v_vals, out_dtype=tl.float32)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 27, TRACE_ENABLED, TRACE_EVENTS)
        trace_k_release_event = 78 + (tile - 1) * 2
        trace_k_tile_active = trace_active & (tile <= 3)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 66, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark_dynamic_event(
            trace, trace_k_tile_active, TRACE_LANE, trace_k_release_event, TRACE_ENABLED, TRACE_EVENTS
        )
        k_reader.release(k_seq)
        _attention_trace_mark_dynamic_event(
            trace, trace_k_tile_active, TRACE_LANE, trace_k_release_event + 1, TRACE_ENABLED, TRACE_EVENTS
        )
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 67, TRACE_ENABLED, TRACE_EVENTS)

        _attention_trace_mark(trace, trace_active, TRACE_LANE, 28, TRACE_ENABLED, TRACE_EVENTS)
        if FULL_Q_TILES:
            scores = scores_curr
        else:
            scores = tl.where(q_mask[:, None], scores_curr, -float("inf"))
        scores_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, scores_max)
        if not FULL_Q_TILES:
            m_new = tl.where(q_mask, m_new, 0.0)
        safe_m_new = tl.where(m_new == -float("inf"), 0.0, m_new)
        if FULL_Q_TILES:
            alpha = tl.math.exp2((m_i - safe_m_new) * SOFTMAX_SCALE_LOG2)
        else:
            alpha = tl.where(q_mask & (m_i != -float("inf")),
                             tl.math.exp2((m_i - safe_m_new) * SOFTMAX_SCALE_LOG2), 0.0)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 58, TRACE_ENABLED, TRACE_EVENTS)
        p = tl.math.exp2(scores * SOFTMAX_SCALE_LOG2 - safe_m_new[:, None] * SOFTMAX_SCALE_LOG2)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 59, TRACE_ENABLED, TRACE_EVENTS)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        p_prev = p.to(tl.bfloat16)
        acc_scale = alpha
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 60, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 68, TRACE_ENABLED, TRACE_EVENTS)
        v_reader.release(v_seq)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 69, TRACE_ENABLED, TRACE_EVENTS)
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 29, TRACE_ENABLED, TRACE_EVENTS)

    acc = acc * acc_scale[:, None]
    v_tile = n_block_max - 1
    v_seq = PIPE_BASE + v_tile
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 42, TRACE_ENABLED, TRACE_EVENTS)
    v_wait = v_reader.wait(v_seq)
    v_smem = v_wait.slot.v.slot(zero).slot(zero)
    v_vals = tl.load(tle.gpu.local_ptr(v_smem, (kv_rows, kv_cols)))
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 43, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 30, TRACE_ENABLED, TRACE_EVENTS)
    acc = acc + tl.dot(p_prev, v_vals, out_dtype=tl.float32)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 31, TRACE_ENABLED, TRACE_EVENTS)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 70, TRACE_ENABLED, TRACE_EVENTS)
    v_reader.release(v_seq)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 71, TRACE_ENABLED, TRACE_EVENTS)
    if O_ALIAS_V:
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 44, TRACE_ENABLED, TRACE_EVENTS)
        v_reader.pipe.wait_drained()
        _attention_trace_mark(trace, trace_active, TRACE_LANE, 45, TRACE_ENABLED, TRACE_EVENTS)

    _attention_trace_mark(trace, trace_active, TRACE_LANE, 32, TRACE_ENABLED, TRACE_EVENTS)
    out_vals = acc / l_i[:, None]
    o_vals = out_vals.to(out.dtype.element_ty)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 33, TRACE_ENABLED, TRACE_EVENTS)
    if O_ALIAS_V:
        o_tile = o_smem.slot(q_part)
        o_tile_ptrs = tle.gpu.local_ptr(o_tile, (q_rows, q_cols))
    else:
        o_tile = o_smem.subslice([ROW_OFFSET, 0], [ROWS_M, BLOCK_D])
        o_tile_ptrs = tle.gpu.local_ptr(o_tile, (q_rows, q_cols))
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 34, TRACE_ENABLED, TRACE_EVENTS)
    tl.store(o_tile_ptrs, o_vals, mask=q_mask[:, None] & mask_d[None, :])
    o_vals = tl.load(o_tile_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 35, TRACE_ENABLED, TRACE_EVENTS)
    out_ptrs = out + (token_offsets[:, None] * NUM_Q_HEADS + q_head[:, None]) * HEAD_DIM + offs_d[None, :]
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 36, TRACE_ENABLED, TRACE_EVENTS)
    tl.store(out_ptrs, o_vals, mask=q_mask[:, None] & mask_d[None, :])
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 37, TRACE_ENABLED, TRACE_EVENTS)
    if not O_ALIAS_V:
        q_reader.release(Q_PIPE_SEQ)
    _attention_trace_mark(trace, trace_active, TRACE_LANE, 63, TRACE_ENABLED, TRACE_EVENTS)


@triton.jit
def _attention_ws_packgqa_tma_kernel(
    q,
    k_desc,
    v_desc,
    out,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
    FULL_Q_TILES: tl.constexpr,
    KV_LEN_MULTIPLE_OF_BLOCK_N: tl.constexpr,
    trace,
    TRACE_ENABLED: tl.constexpr,
    TRACE_EVENTS: tl.constexpr,
):
    packed_m_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)
    HALF_M: tl.constexpr = BLOCK_M // 2
    q_smem = tle.gpu.alloc([1, BLOCK_M, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    k_smem = tle.gpu.alloc([PIPE_CAPACITY, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    v_smem = tle.gpu.alloc([PIPE_CAPACITY, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    O_ALIAS_V: tl.constexpr = PIPE_CAPACITY * BLOCK_N >= BLOCK_M
    zero = tl.full((), 0, tl.int32)
    if O_ALIAS_V:
        o_smem = tle.gpu.alloc([2, HALF_M, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                               alias=v_smem, alias_offset_bytes=0, nv_mma_shared_layout=True)
    else:
        o_smem = q_smem.slot(zero)
    q_pipe = tle.pipe(capacity=1, scope="cta", name="attention_packgqa_q", readers=("lo", "hi"), one_shot=True,
                      q=q_smem)
    k_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="attention_packgqa_k", readers=("lo", "hi"), k=k_smem)
    v_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="attention_packgqa_v", readers=("lo", "hi"), v=v_smem)
    tle.gpu.warp_specialize(
        [
            (
                _attention_ws_packgqa_consumer,
                (
                    q_pipe.reader(name="lo"),
                    k_pipe.reader(name="lo"),
                    v_pipe.reader(name="lo"),
                    out,
                    o_smem,
                    batch,
                    kv_head,
                    packed_m_block,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    SM_SCALE,
                    BLOCK_M,
                    0,
                    HALF_M,
                    BLOCK_N,
                    BLOCK_D,
                    O_ALIAS_V,
                    FULL_Q_TILES,
                    KV_LEN_MULTIPLE_OF_BLOCK_N,
                    0,
                    0,
                    trace,
                    TRACE_ENABLED,
                    TRACE_EVENTS,
                ),
            ),
            (
                _attention_ws_packgqa_consumer,
                (
                    q_pipe.reader(name="hi"),
                    k_pipe.reader(name="hi"),
                    v_pipe.reader(name="hi"),
                    out,
                    o_smem,
                    batch,
                    kv_head,
                    packed_m_block,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    SM_SCALE,
                    BLOCK_M,
                    HALF_M,
                    HALF_M,
                    BLOCK_N,
                    BLOCK_D,
                    O_ALIAS_V,
                    FULL_Q_TILES,
                    KV_LEN_MULTIPLE_OF_BLOCK_N,
                    0,
                    0,
                    trace,
                    TRACE_ENABLED,
                    TRACE_EVENTS,
                ),
            ),
            (
                _attention_ws_packgqa_tma_producer,
                (
                    q_pipe.writer(),
                    k_pipe.writer(),
                    v_pipe.writer(),
                    q,
                    k_desc,
                    v_desc,
                    batch,
                    kv_head,
                    packed_m_block,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    FULL_Q_TILES,
                    O_ALIAS_V,
                    0,
                    0,
                    trace,
                    TRACE_ENABLED,
                    TRACE_EVENTS,
                ),
            ),
        ],
        [4, 1],
        [240, 48],
    )


@triton.jit
def _attention_ws_packgqa_tma_persistent_producer(
    q_writer,
    k_writer,
    v_writer,
    q,
    k_desc,
    v_desc,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_Q_TILES: tl.constexpr,
):
    Q_PER_KV: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    num_m_blocks = tl.cdiv(Q_LEN * Q_PER_KV, BLOCK_M)
    total_tiles = BATCH * NUM_KV_HEADS * num_m_blocks
    tile_id = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    pipe_base = tl.full((), 0, tl.int32)
    q_pipe_seq = tl.full((), 0, tl.int32)
    while tile_id < total_tiles:
        packed_m_block = tile_id % num_m_blocks
        head_batch_tile = tile_id // num_m_blocks
        kv_head = head_batch_tile % NUM_KV_HEADS
        batch = head_batch_tile // NUM_KV_HEADS
        _attention_ws_packgqa_tma_producer(
            q_writer,
            k_writer,
            v_writer,
            q,
            k_desc,
            v_desc,
            batch,
            kv_head,
            packed_m_block,
            Q_LEN,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            START_POS,
            KV_LEN,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            FULL_Q_TILES,
            False,
            pipe_base,
            q_pipe_seq,
            q,
            False,
            16,
        )
        packed_start = packed_m_block * BLOCK_M
        packed_stop = tl.minimum(packed_start + BLOCK_M, Q_LEN * Q_PER_KV)
        max_q_token = (packed_stop - 1) // Q_PER_KV
        max_key = tl.minimum(KV_LEN, START_POS + max_q_token + 1)
        pipe_base += tl.cdiv(max_key, BLOCK_N)
        q_pipe_seq += 1
        tile_id += tile_stride


@triton.jit
def _attention_ws_packgqa_persistent_consumer(
    q_reader,
    k_reader,
    v_reader,
    out,
    o_smem,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    SM_SCALE: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    ROWS_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_Q_TILES: tl.constexpr,
    KV_LEN_MULTIPLE_OF_BLOCK_N: tl.constexpr,
):
    Q_PER_KV: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    num_m_blocks = tl.cdiv(Q_LEN * Q_PER_KV, BLOCK_M)
    total_tiles = BATCH * NUM_KV_HEADS * num_m_blocks
    tile_id = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    pipe_base = tl.full((), 0, tl.int32)
    q_pipe_seq = tl.full((), 0, tl.int32)
    while tile_id < total_tiles:
        packed_m_block = tile_id % num_m_blocks
        head_batch_tile = tile_id // num_m_blocks
        kv_head = head_batch_tile % NUM_KV_HEADS
        batch = head_batch_tile // NUM_KV_HEADS
        _attention_ws_packgqa_consumer(
            q_reader,
            k_reader,
            v_reader,
            out,
            o_smem,
            batch,
            kv_head,
            packed_m_block,
            Q_LEN,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            START_POS,
            KV_LEN,
            SM_SCALE,
            BLOCK_M,
            ROW_OFFSET,
            ROWS_M,
            BLOCK_N,
            BLOCK_D,
            False,
            FULL_Q_TILES,
            KV_LEN_MULTIPLE_OF_BLOCK_N,
            pipe_base,
            q_pipe_seq,
            out,
            False,
            16,
        )
        packed_start = packed_m_block * BLOCK_M
        packed_stop = tl.minimum(packed_start + BLOCK_M, Q_LEN * Q_PER_KV)
        max_q_token = (packed_stop - 1) // Q_PER_KV
        max_key = tl.minimum(KV_LEN, START_POS + max_q_token + 1)
        pipe_base += tl.cdiv(max_key, BLOCK_N)
        q_pipe_seq += 1
        tile_id += tile_stride


@triton.jit
def _attention_ws_packgqa_tma_persistent_kernel(
    q,
    k_desc,
    v_desc,
    out,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    SM_SCALE: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
    FULL_Q_TILES: tl.constexpr,
    KV_LEN_MULTIPLE_OF_BLOCK_N: tl.constexpr,
):
    HALF_M: tl.constexpr = BLOCK_M // 2
    q_smem = tle.gpu.alloc([1, BLOCK_M, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    k_smem = tle.gpu.alloc([PIPE_CAPACITY, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    v_smem = tle.gpu.alloc([PIPE_CAPACITY, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    zero = tl.full((), 0, tl.int32)
    o_smem = q_smem.slot(zero)
    q_pipe = tle.pipe(capacity=1, scope="cta", name="attention_packgqa_q_persistent", readers=("lo", "hi"),
                      q=q_smem)
    k_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="attention_packgqa_k_persistent",
                      readers=("lo", "hi"), k=k_smem)
    v_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="attention_packgqa_v_persistent",
                      readers=("lo", "hi"), v=v_smem)
    tle.gpu.warp_specialize(
        [
            (
                _attention_ws_packgqa_persistent_consumer,
                (
                    q_pipe.reader(name="lo"),
                    k_pipe.reader(name="lo"),
                    v_pipe.reader(name="lo"),
                    out,
                    o_smem,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    SM_SCALE,
                    BATCH,
                    BLOCK_M,
                    0,
                    HALF_M,
                    BLOCK_N,
                    BLOCK_D,
                    FULL_Q_TILES,
                    KV_LEN_MULTIPLE_OF_BLOCK_N,
                ),
            ),
            (
                _attention_ws_packgqa_persistent_consumer,
                (
                    q_pipe.reader(name="hi"),
                    k_pipe.reader(name="hi"),
                    v_pipe.reader(name="hi"),
                    out,
                    o_smem,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    SM_SCALE,
                    BATCH,
                    BLOCK_M,
                    HALF_M,
                    HALF_M,
                    BLOCK_N,
                    BLOCK_D,
                    FULL_Q_TILES,
                    KV_LEN_MULTIPLE_OF_BLOCK_N,
                ),
            ),
            (
                _attention_ws_packgqa_tma_persistent_producer,
                (
                    q_pipe.writer(),
                    k_pipe.writer(),
                    v_pipe.writer(),
                    q,
                    k_desc,
                    v_desc,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    BATCH,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    FULL_Q_TILES,
                ),
            ),
        ],
        [4, 1],
        [240, 48],
    )


@triton.jit
def _attention_decode_no_split_producer(
    q_writer,
    k_writer,
    v_writer,
    q,
    k_desc,
    v_desc,
    REAL_KV_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_group_num: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = hid // (kv_group_num // VALID_BLOCK_H)
    head_start = hid * VALID_BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    rows_h = tl.broadcast_to(tl.arange(0, BLOCK_H)[:, None], (BLOCK_H, BLOCK_D))
    cols_h = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_H, BLOCK_D))
    valid_rows = tl.arange(0, BLOCK_H) < VALID_BLOCK_H
    q_mask = valid_rows[:, None] & (offs_h[:, None] < NUM_Q_HEADS) & (offs_d[None, :] < HEAD_DIM)

    q_slot = q_writer.acquire(0)
    q_ptrs = q + (bid * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM + offs_d[None, :]
    q_vals = tl.load(q_ptrs, mask=q_mask, other=0.0)
    tl.store(tle.gpu.local_ptr(q_slot.q, (rows_h, cols_h)), q_vals)
    q_writer.commit(0)

    for n0 in tl.range(0, REAL_KV_LEN, BLOCK_N):
        tile = n0 // BLOCK_N
        k_slot = k_writer.acquire(tile)
        tle.gpu.copy(k_desc, k_slot.k, [1, 1, BLOCK_N, BLOCK_D], [bid, kv_head, n0, 0])
        k_writer.commit(tile)
        v_slot = v_writer.acquire(tile)
        tle.gpu.copy(v_desc, v_slot.v, [1, 1, BLOCK_N, BLOCK_D], [bid, kv_head, n0, 0])
        v_writer.commit(tile)
    end_tile = tl.cdiv(REAL_KV_LEN, BLOCK_N)
    v_writer.close(end_tile)
    v_writer.pipe.wait_drained()


@triton.jit
def _attention_decode_no_split_consumer(
    q_reader,
    k_reader,
    v_reader,
    out_desc,
    o_smem,
    REAL_KV_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2E_SCALE: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_group_num: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = hid // (kv_group_num // VALID_BLOCK_H)
    head_start = hid * VALID_BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    rows_h = tl.broadcast_to(tl.arange(0, BLOCK_H)[:, None], (BLOCK_H, BLOCK_D))
    rows_n = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    cols_h = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_H, BLOCK_D))
    cols_n = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    head_mask = offs_h < NUM_Q_HEADS
    store_rows = tl.arange(0, BLOCK_H) < VALID_BLOCK_H
    row_mask = store_rows & head_mask
    dim_mask = offs_d < HEAD_DIM
    zero = tl.full((), 0, tl.int32)

    q_wait = q_reader.wait(0)
    q_smem = q_wait.slot.q
    q_smem_ptrs = tle.gpu.local_ptr(q_smem, (rows_h, cols_h))
    q_vals = tl.load(q_smem_ptrs)

    acc_o = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    scores_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    logsum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for n0 in tl.range(0, REAL_KV_LEN, BLOCK_N, num_stages=PIPELINE_STAGES):
        n_idx = n0 + offs_n
        kv_mask = n_idx < REAL_KV_LEN
        tile = n0 // BLOCK_N
        k_wait = k_reader.wait(tile)
        k_smem = k_wait.slot.k.slot(zero).slot(zero)
        k_smem_ptrs = tle.gpu.local_ptr(k_smem, (rows_n, cols_n))
        k_vals = tl.load(k_smem_ptrs)

        scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
        scores = tl.where(row_mask[:, None] & kv_mask[None, :], scores, -float("inf"))
        k_reader.release(tile)
        scores_max_prev = scores_max
        scores_max = tl.maximum(scores_max_prev, tl.max(scores, axis=1))
        scores_max = tl.where(row_mask, scores_max, 0.0)
        scores_scale = tl.where(row_mask, tl.exp2((scores_max_prev - scores_max) * LOG2E_SCALE), 0.0)
        probs = tl.exp2((scores - scores_max[:, None]) * LOG2E_SCALE)
        probs = tl.where(row_mask[:, None] & kv_mask[None, :], probs, 0.0)
        v_wait = v_reader.wait(tile)
        v_smem = v_wait.slot.v.slot(zero).slot(zero)
        v_smem_ptrs = tle.gpu.local_ptr(v_smem, (rows_n, cols_n))
        v_vals = tl.load(v_smem_ptrs)
        acc_o = acc_o * scores_scale[:, None] + tl.dot(probs.to(tl.bfloat16), v_vals, out_dtype=tl.float32)
        logsum = logsum * scores_scale + tl.sum(probs, axis=1)
        v_reader.release(tile)

    v_reader.pipe.wait_drained()
    out_vals = acc_o / tl.where(row_mask, logsum, 1.0)[:, None]
    o_smem_ptrs = tle.gpu.local_ptr(o_smem, (rows_h, cols_h))
    tl.store(
        o_smem_ptrs,
        out_vals.to(q_vals.dtype),
        mask=store_rows[:, None] & head_mask[:, None] & dim_mask[None, :],
    )
    o_tile = o_smem.subslice([0, 0], [VALID_BLOCK_H, BLOCK_D])
    tle.gpu.copy(o_tile, out_desc, [1, VALID_BLOCK_H, BLOCK_D], [bid, head_start, 0])
    q_reader.release(0)


@triton.jit
def _attention_decode_no_split_kernel(
    q,
    k_desc,
    v_desc,
    out_desc,
    REAL_KV_LEN,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2E_SCALE: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_H >= VALID_BLOCK_H, "decode no-split packed compute tile must cover valid heads")
    q_smem = tle.gpu.alloc([1, BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    k_smem = tle.gpu.alloc([PIPELINE_STAGES, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    v_smem = tle.gpu.alloc([PIPELINE_STAGES, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    if PIPELINE_STAGES * BLOCK_N >= BLOCK_H:
        o_smem = tle.gpu.alloc([BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                               alias=v_smem, alias_offset_bytes=0, nv_mma_shared_layout=True)
    else:
        o_smem = tle.gpu.alloc([BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                               nv_mma_shared_layout=True)
    q_pipe = tle.pipe(capacity=1, scope="cta", name="decode_q", one_shot=True, q=q_smem)
    k_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="decode_k", k=k_smem)
    v_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="decode_v", v=v_smem)
    tle.gpu.warp_specialize(
        [
            (
                _attention_decode_no_split_consumer,
                (
                    q_pipe.reader(),
                    k_pipe.reader(),
                    v_pipe.reader(),
                    out_desc,
                    o_smem,
                    REAL_KV_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    LOG2E_SCALE,
                    VALID_BLOCK_H,
                    BLOCK_H,
                    BLOCK_N,
                    BLOCK_D,
                    PIPELINE_STAGES,
                ),
            ),
            (
                _attention_decode_no_split_producer,
                (
                    q_pipe.writer(),
                    k_pipe.writer(),
                    v_pipe.writer(),
                    q,
                    k_desc,
                    v_desc,
                    REAL_KV_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    VALID_BLOCK_H,
                    BLOCK_H,
                    BLOCK_N,
                    BLOCK_D,
                ),
            ),
        ],
        [4],
        [24],
    )


@triton.jit
def _attention_decode_split_kernel(
    q,
    k_cache,
    v_cache,
    partial_out,
    partial_lse,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2E_SCALE: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    SID_OFFSET: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2) + SID_OFFSET
    kv_group_num: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = hid // (kv_group_num // VALID_BLOCK_H)
    split_start = sid * SPLIT_BASE

    head_start = hid * VALID_BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    rows_h = tl.broadcast_to(tl.arange(0, BLOCK_H)[:, None], (BLOCK_H, BLOCK_D))
    rows_n = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    cols_h = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_H, BLOCK_D))
    cols_n = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    head_mask = offs_h < NUM_Q_HEADS
    store_rows = tl.arange(0, BLOCK_H) < VALID_BLOCK_H
    dim_mask = offs_d < HEAD_DIM

    q_smem = tle.gpu.alloc([BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    k_smem = tle.gpu.alloc([BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    v_smem = tle.gpu.alloc([BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)

    q_ptrs = q + (bid * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM + offs_d[None, :]
    q_mask = head_mask[:, None] & dim_mask[None, :]
    q_vals = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q_smem_ptrs = tle.gpu.local_ptr(q_smem, (rows_h, cols_h))
    tl.store(q_smem_ptrs, q_vals)
    q_vals = tl.load(q_smem_ptrs)

    acc_o = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    scores_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    logsum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for local_n0 in tl.range(0, SPLIT_LEN, BLOCK_N, num_stages=PIPELINE_STAGES):
        n_idx = split_start + local_n0 + offs_n
        local_mask = (local_n0 + offs_n) < SPLIT_LEN
        kv_ptrs = ((bid * MAX_SEQ_LEN + n_idx[:, None]) * NUM_KV_HEADS + kv_head) * HEAD_DIM + offs_d[None, :]
        tile_mask = local_mask[:, None] & dim_mask[None, :]
        k_vals = tl.load(k_cache + kv_ptrs, mask=tile_mask, other=0.0)
        k_smem_ptrs = tle.gpu.local_ptr(k_smem, (rows_n, cols_n))
        tl.store(k_smem_ptrs, k_vals)
        k_vals = tl.load(k_smem_ptrs)

        scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
        scores = tl.where(head_mask[:, None] & local_mask[None, :], scores, -float("inf"))
        scores_max_prev = scores_max
        scores_max = tl.maximum(scores_max_prev, tl.max(scores, axis=1))
        scores_scale = tl.exp2((scores_max_prev - scores_max) * LOG2E_SCALE)
        probs = tl.exp2((scores - scores_max[:, None]) * LOG2E_SCALE)
        probs = tl.where(head_mask[:, None] & local_mask[None, :], probs, 0.0)
        v_idx = split_start + local_n0 + offs_n
        v_local_mask = (local_n0 + offs_n) < SPLIT_LEN
        v_ptrs = ((bid * MAX_SEQ_LEN + v_idx[:, None]) * NUM_KV_HEADS + kv_head) * HEAD_DIM + offs_d[None, :]
        v_tile_mask = v_local_mask[:, None] & dim_mask[None, :]
        v_vals = tl.load(v_cache + v_ptrs, mask=v_tile_mask, other=0.0)
        v_smem_ptrs = tle.gpu.local_ptr(v_smem, (rows_n, cols_n))
        tl.store(v_smem_ptrs, v_vals, mask=v_tile_mask)
        v_vals = tl.load(v_smem_ptrs, mask=v_tile_mask, other=0.0)
        acc_o = acc_o * scores_scale[:, None] + tl.dot(probs.to(tl.bfloat16), v_vals, out_dtype=tl.float32)
        logsum = logsum * scores_scale + tl.sum(probs, axis=1)

    out_vals = acc_o / logsum[:, None]
    partial_ptrs = partial_out + (((bid * NUM_Q_HEADS + offs_h[:, None]) * NUM_SPLIT + sid) * HEAD_DIM + offs_d[
        None, :])
    tl.store(partial_ptrs, out_vals.to(partial_out.dtype.element_ty), mask=store_rows[:, None] & q_mask)
    lse = tl.log2(logsum) + scores_max * LOG2E_SCALE
    tl.store(partial_lse + (bid * NUM_Q_HEADS + offs_h) * NUM_SPLIT + sid, lse, mask=store_rows & head_mask)


@triton.jit
def _attention_decode_combine_kernel(
    partial_out,
    partial_lse,
    out,
    NUM_SPLIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    bid = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    lse_offsets = (bid * tl.num_programs(0) + head) * NUM_SPLIT + tl.arange(0, NUM_SPLIT)
    lse_vals = tl.load(partial_lse + lse_offsets).to(tl.float32)
    lse_max = tl.max(lse_vals, axis=0)
    weights_unnorm = tl.exp2(lse_vals - lse_max)
    lse_logsum = tl.log2(tl.sum(weights_unnorm, axis=0)) + lse_max
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for sid in tl.static_range(0, NUM_SPLIT):
        weight = tl.exp2(tl.load(partial_lse + (bid * tl.num_programs(0) + head) * NUM_SPLIT + sid).to(tl.float32) -
                         lse_logsum)
        vals = tl.load(partial_out + (((bid * tl.num_programs(0) + head) * NUM_SPLIT + sid) * HEAD_DIM + offs_d),
                       mask=mask_d, other=0.0).to(tl.float32)
        acc += vals * weight
    tl.store(out + (bid * tl.num_programs(0) + head) * HEAD_DIM + offs_d, acc.to(out.dtype.element_ty), mask=mask_d)


_attention_decode_no_split_kernel_autotuned = triton.autotune(
    configs=_ATTENTION_DECODE_AUTOTUNE_CONFIGS,
    key=["NUM_Q_HEADS", "NUM_KV_HEADS", "HEAD_DIM", "VALID_BLOCK_H", "BLOCK_H"],
    cache_results=True,
)(_attention_decode_no_split_kernel)


_attention_decode_split_kernel_autotuned = triton.autotune(
    configs=_ATTENTION_DECODE_AUTOTUNE_CONFIGS,
    key=[
        "NUM_Q_HEADS",
        "NUM_KV_HEADS",
        "HEAD_DIM",
        "NUM_SPLIT",
        "SPLIT_BASE",
        "SPLIT_LEN",
        "SID_OFFSET",
        "VALID_BLOCK_H",
        "BLOCK_H",
    ],
    cache_results=True,
)(_attention_decode_split_kernel)


def attention_ws(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    kv_len: int,
    sm_scale: float,
    block_n: int | None = None,
    block_m: int = 128,
    persistent: bool = False,
    persistent_ctas_per_sm: int = 1,
) -> torch.Tensor:
    """Causal block-M GQA attention with TLE pipe and dot-based QK/PV stages."""
    require_cuda_contiguous("q", q)
    require_cuda_contiguous("k_cache", k_cache)
    require_cuda_contiguous("v_cache", v_cache)
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("expected q [tokens, q_heads, dim] and cache [batch, max, kv_heads, dim]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("attention_ws currently expects bfloat16 q/k/v tensors")
    tokens, num_q_heads, head_dim = q.shape
    batch, max_seq_len, num_kv_heads, cache_dim = k_cache.shape
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("KV cache shapes do not match query head_dim")
    if tokens != batch * q_len:
        raise ValueError(f"tokens={tokens} does not equal batch={batch} * q_len={q_len}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"GQA requires q heads divisible by kv heads, got {num_q_heads} and {num_kv_heads}")
    if start_pos + q_len > kv_len:
        raise ValueError(f"kv_len={kv_len} must include current query range start_pos={start_pos} q_len={q_len}")
    out = torch.empty_like(q)
    block_d = next_power_of_2(head_dim)
    if head_dim != block_d:
        raise ValueError(f"attention_ws full-D TMA path requires power-of-two head_dim, got {head_dim}")
    packed_m = q_len * (num_q_heads // num_kv_heads)
    full_q_tiles = packed_m % block_m == 0
    selected_block_n = (64 if kv_len >= 1024 else 128) if block_n is None else block_n
    kv_len_multiple_of_block_n = kv_len % selected_block_n == 0
    k_desc = TensorDescriptor(
        k_cache,
        shape=[batch, num_kv_heads, max_seq_len, head_dim],
        strides=[max_seq_len * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1],
        block_shape=[1, 1, selected_block_n, block_d],
    )
    v_desc = TensorDescriptor(
        v_cache,
        shape=[batch, num_kv_heads, max_seq_len, head_dim],
        strides=[max_seq_len * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1],
        block_shape=[1, 1, selected_block_n, block_d],
    )
    if persistent:
        if persistent_ctas_per_sm < 1:
            raise ValueError(f"persistent_ctas_per_sm must be >= 1, got {persistent_ctas_per_sm}")
        total_tiles = cdiv(packed_m, block_m) * num_kv_heads * batch
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        workers = min(total_tiles, sm_count * persistent_ctas_per_sm)
        _attention_ws_packgqa_tma_persistent_kernel[(workers, )](
            q,
            k_desc,
            v_desc,
            out,
            q_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            kv_len,
            sm_scale,
            batch,
            block_m,
            selected_block_n,
            block_d,
            2,
            full_q_tiles,
            kv_len_multiple_of_block_n,
            num_warps=4,
            num_stages=2,
        )
    else:
        _attention_ws_packgqa_tma_kernel[(cdiv(packed_m, block_m), num_kv_heads, batch)](
            q,
            k_desc,
            v_desc,
            out,
            q_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            kv_len,
            sm_scale,
            block_m,
            selected_block_n,
            block_d,
            2,
            full_q_tiles,
            kv_len_multiple_of_block_n,
            out,
            False,
            ATTENTION_TRACE_EVENTS,
            num_warps=4,
            num_stages=2,
        )
    return out


def attention_ws_trace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    kv_len: int,
    sm_scale: float,
    block_n: int | None = None,
    block_m: int = 128,
    pipe_capacity: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the static prefill kernel and record coarse WG phase timestamps for tile (0, 0, 0)."""
    require_cuda_contiguous("q", q)
    require_cuda_contiguous("k_cache", k_cache)
    require_cuda_contiguous("v_cache", v_cache)
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("expected q [tokens, q_heads, dim] and cache [batch, max, kv_heads, dim]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("attention_ws_trace currently expects bfloat16 q/k/v tensors")
    tokens, num_q_heads, head_dim = q.shape
    batch, max_seq_len, num_kv_heads, cache_dim = k_cache.shape
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("KV cache shapes do not match query head_dim")
    if tokens != batch * q_len:
        raise ValueError(f"tokens={tokens} does not equal batch={batch} * q_len={q_len}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"GQA requires q heads divisible by kv heads, got {num_q_heads} and {num_kv_heads}")
    if start_pos + q_len > kv_len:
        raise ValueError(f"kv_len={kv_len} must include current query range start_pos={start_pos} q_len={q_len}")
    out = torch.empty_like(q)
    trace = torch.zeros((len(ATTENTION_TRACE_LANES), ATTENTION_TRACE_EVENTS), device=q.device, dtype=torch.int64)
    block_d = next_power_of_2(head_dim)
    if head_dim != block_d:
        raise ValueError(f"attention_ws_trace full-D TMA path requires power-of-two head_dim, got {head_dim}")
    packed_m = q_len * (num_q_heads // num_kv_heads)
    full_q_tiles = packed_m % block_m == 0
    selected_block_n = (64 if kv_len >= 1024 else 128) if block_n is None else block_n
    kv_len_multiple_of_block_n = kv_len % selected_block_n == 0
    k_desc = TensorDescriptor(
        k_cache,
        shape=[batch, num_kv_heads, max_seq_len, head_dim],
        strides=[max_seq_len * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1],
        block_shape=[1, 1, selected_block_n, block_d],
    )
    v_desc = TensorDescriptor(
        v_cache,
        shape=[batch, num_kv_heads, max_seq_len, head_dim],
        strides=[max_seq_len * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1],
        block_shape=[1, 1, selected_block_n, block_d],
    )
    _attention_ws_packgqa_tma_kernel[(cdiv(packed_m, block_m), num_kv_heads, batch)](
        q,
        k_desc,
        v_desc,
        out,
        q_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        start_pos,
        kv_len,
        sm_scale,
        block_m,
        selected_block_n,
        block_d,
        pipe_capacity,
        full_q_tiles,
        kv_len_multiple_of_block_n,
        trace,
        True,
        ATTENTION_TRACE_EVENTS,
        num_warps=4,
        num_stages=2,
    )
    return out, trace


def attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    kv_len: int,
    sm_scale: float,
    block_h: int = 64,
    block_n: int | None = None,
    num_split: int = 16,
) -> torch.Tensor:
    """FA3-style PackGQA decode attention for a single query token."""
    require_cuda_contiguous("q", q)
    require_cuda_contiguous("k_cache", k_cache)
    require_cuda_contiguous("v_cache", v_cache)
    if q_len != 1:
        raise ValueError(f"attention_decode expects q_len=1, got {q_len}")
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("expected q [batch, q_heads, dim] and cache [batch, max, kv_heads, dim]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("attention_decode currently expects bfloat16 q/k/v tensors")
    tokens, num_q_heads, head_dim = q.shape
    batch, max_seq_len, num_kv_heads, cache_dim = k_cache.shape
    if tokens != batch:
        raise ValueError(f"decode q tokens={tokens} must equal batch={batch}")
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("KV cache shapes do not match query head_dim")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"GQA requires q heads divisible by kv heads, got {num_q_heads} and {num_kv_heads}")
    if start_pos + q_len > kv_len:
        raise ValueError(f"kv_len={kv_len} must include current query range start_pos={start_pos} q_len={q_len}")
    if kv_len <= 0:
        raise ValueError("attention_decode requires kv_len > 0")

    # FA3's hdim128 causal PackGQA decode is the normal FA3 forward mainloop with
    # seqlen_q=1: packed M = q_len * q_heads_per_kv, padded to BlockM=128, and
    # BlockN=128.  Keep the public decode signature compatible, but route the
    # selected implementation through the same PackGQA TMA/WS kernel used by
    # prefill instead of the older TileOps-style split decode kernels.
    selected_block_n = 128 if block_n is None else block_n
    if selected_block_n != 128:
        raise ValueError(f"FA3-aligned decode expects block_n=128, got {selected_block_n}")
    if block_h not in (64, FA3_PACKGQA_DECODE_BLOCK_H):
        raise ValueError(f"FA3-aligned decode expects block_h compatible with 128, got {block_h}")
    if num_split not in (1, 16):
        raise ValueError(f"FA3-aligned decode currently follows FA3 varlen num_splits=1, got {num_split}")

    return attention_ws(
        q,
        k_cache,
        v_cache,
        q_len=q_len,
        start_pos=start_pos,
        kv_len=kv_len,
        sm_scale=sm_scale,
        block_n=128,
        block_m=FA3_PACKGQA_DECODE_BLOCK_H,
        persistent=False,
    )
