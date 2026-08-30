"""Emit PIM IR alongside a kernel's primary compilation.

The compiler's stage loop is strictly linear -- each stage consumes the previous
stage's module -- so ``pimir`` cannot be inserted as a stage between ``ttir`` and
``ttgir``: PIM branches off TTIR rather than sitting on the path to the GPU
binary. Instead a backend calls :func:`emit_pim_ir` once its TTIR is ready. The
module is cloned first, so the primary pipeline is byte-for-byte unaffected.

Enable by setting ``FLAGTREE_EMIT_PIM=1``. The hardware parameters default to the
same values as the ``convert-triton-to-pim`` pass options and can be overridden
per-run:

    FLAGTREE_PIM_NUM_DPUS       (default 1)
    FLAGTREE_PIM_NUM_TASKLETS   (default 16)
    FLAGTREE_PIM_WRAM_BYTES     (default 65536)
    FLAGTREE_PIM_MRAM_BYTES     (default 8589934592, 8GiB)
    FLAGTREE_PIM_DMA_ALIGN      (default 8)
    FLAGTREE_PIM_TARGET         (default "pim:v1")
"""

import os

DEFAULT_TARGET = "pim:v1"
DEFAULT_NUM_DPUS = 1
DEFAULT_NUM_TASKLETS = 16
DEFAULT_WRAM_BYTES = 65536
DEFAULT_MRAM_BYTES = 8 * 2**30
DEFAULT_DMA_ALIGN = 8


def is_enabled() -> bool:
    return os.environ.get("FLAGTREE_EMIT_PIM", "0") == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def pim_options() -> dict:
    return {
        "target": os.environ.get("FLAGTREE_PIM_TARGET", DEFAULT_TARGET),
        "num_dpus": _env_int("FLAGTREE_PIM_NUM_DPUS", DEFAULT_NUM_DPUS),
        "num_tasklets": _env_int("FLAGTREE_PIM_NUM_TASKLETS", DEFAULT_NUM_TASKLETS),
        "wram_bytes": _env_int("FLAGTREE_PIM_WRAM_BYTES", DEFAULT_WRAM_BYTES),
        "mram_bytes": _env_int("FLAGTREE_PIM_MRAM_BYTES", DEFAULT_MRAM_BYTES),
        "dma_align": _env_int("FLAGTREE_PIM_DMA_ALIGN", DEFAULT_DMA_ALIGN),
    }


def make_pimir(ttir_mod):
    """Lower a TTIR module to PIM IR. Returns the new module, leaving the input alone."""
    from triton._C.libtriton import ir, passes

    opts = pim_options()
    # `context` is a Python-side dynamic attribute that keeps the MLIRContext
    # alive alongside the module; a clone does not inherit it, so carry it over
    # explicitly before doing anything that needs the context.
    context = ttir_mod.context
    mod = ttir_mod.clone()
    mod.context = context
    pm = ir.pass_manager(context)
    pm.enable_debug()
    # mram_bytes/dma_align 必须按关键字传：旧的 5 参数位置调用会把 `False`
    # 落进 mram_bytes 形参位（等价于 mram_bytes=0），add_tile_to_budget 拿它
    # 比较 tile footprint 时会让任何张量都判定超预算。
    passes.pim.add_convert_to_pim(
        pm,
        opts["target"],
        num_dpus=opts["num_dpus"],
        num_tasklets=opts["num_tasklets"],
        wram_bytes=opts["wram_bytes"],
        mram_bytes=opts["mram_bytes"],
        dma_align=opts["dma_align"],
        enable_source_remat=False,
    )
    # pim-tile-to-budget 必须排在 pim-explicit-dma 之前：前者负责把超出 WRAM
    # 预算的 tile 切小，后者按最终 tile 建 WRAM staging buffer 并在超预算时
    # signalPassFailure()。反过来排的话 explicit-dma 先按未切分的大 tile 建
    # buffer 就直接失败，tile 切分根本没机会跑（lit 测试
    # tile_to_budget_m_split.mlir / tile_to_budget_small_wram.mlir 用的都是
    # `-pim-tile-to-budget -pim-explicit-dma` 这个顺序）。
    #
    # 且该 pass 硬性要求至少一个 tt.dot（"requires at least one tt.dot"），
    # 对纯逐元素 kernel 跑它只会报错，故按 IR 里有没有 tt.dot 判断。
    if "tt.dot" in str(ttir_mod):
        passes.pim.add_tile_to_budget(pm)
    passes.pim.add_explicit_dma(pm)
    pm.run(mod)
    return mod


def emit_pim_ir(ttir_mod, metadata, dump_manager=None, file_name=None):
    """Write ``<kernel>.pimir`` next to the other stage dumps.

    Never raises: this runs alongside a normal compile, and a problem lowering to
    a secondary target must not break the build the user actually asked for. A
    failure is reported as a warning and skipped.
    """
    if not is_enabled():
        return None

    try:
        pim_mod = make_pimir(ttir_mod)
    except Exception as exc:  # noqa: BLE001 - see docstring
        import warnings
        warnings.warn(f"FLAGTREE_EMIT_PIM: could not lower to PIM IR: {exc}")
        return None

    # metadata["name"] is not populated until a later stage, so fall back to the
    # kernel's own entry function name.
    name = file_name or metadata.get("name") or ttir_mod.get_entry_func_name() or "kernel"
    try:
        if dump_manager is not None:
            dump_manager.put(str(pim_mod), f"{name}.pimir", binary=False)
        else:
            path = _default_output_path(name, metadata)
            if path is None:
                return pim_mod
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(str(pim_mod))
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"FLAGTREE_EMIT_PIM: could not write PIM IR: {exc}")
        return None

    return pim_mod


def _default_output_path(name, metadata):
    """Where to write ``<kernel>.pimir`` when no dump manager was supplied.

    Under TRITON_DUMP_DIR, in a subdirectory named for the kernel hash. The
    compiler's own stage dumps live in a sibling directory keyed on a *different*
    hash (``src.hash()``, which the backend cannot see from make_ttir), so this
    deliberately does not try to land in the same folder -- guessing that key
    would break silently whenever it changed.
    """
    out_dir = os.environ.get("TRITON_DUMP_DIR", "")
    if not out_dir:
        return None
    khash = metadata.get("hash") or "unknown"
    return os.path.join(out_dir, f"pim-{khash}", f"{name}.pimir")
