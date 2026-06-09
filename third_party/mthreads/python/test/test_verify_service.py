"""
验证服务测试脚本 — 支持多芯片、参数化 benchmark、HTML 性能表格

用法:
    python test_verify_service.py --url http://172.24.4.45:8912 --chip nvidia
    python test_verify_service.py --url http://10.7.64.131:8912 --chip huawei --all
    python test_verify_service.py --url http://10.7.66.1:8912 --chip moore --kernel gelu

芯片:
    nvidia / haiguang / tianshu / muxi / pingtouge  → CUDA 兼容, 代码不变
    huawei → cuda 替换为 npu
    moore  → cuda 替换为 musa

测试算子:
    gelu    GELU 激活函数 (参数化 benchmark: 12 组 dtype×shape)
    matmul  矩阵乘法 FP16 (正确性验证)
    add     向量加法 (正确性验证)
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from html import unescape
from typing import Dict, Optional


# ============================================================
# 芯片配置
# ============================================================
CHIP_CONFIG = {
    "nvidia":     {"device_keyword": "cuda",  "desc": "NVIDIA CUDA"},
    "haiguang":   {"device_keyword": "cuda",  "desc": "海光 DCU (CUDA兼容)"},
    "tianshu":    {"device_keyword": "cuda",  "desc": "天数智芯 (CUDA兼容)"},
    "muxi":       {"device_keyword": "cuda",  "desc": "沐曦 (CUDA兼容)"},
    "pingtouge":  {"device_keyword": "cuda",  "desc": "平头哥 (CUDA兼容)"},
    "huawei":     {"device_keyword": "npu",   "desc": "华为昇腾 Ascend"},
    "moore":      {"device_keyword": "musa",  "desc": "摩尔线程 MUSA"},
}


# ============================================================
# HTTP 工具
# ============================================================
def http_post(url: str, data: dict, timeout: int = 600) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": True, "message": f"HTTP {e.code}: {e.reason}",
                "detail": e.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        return {"error": True, "message": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": True, "message": str(e)}


def http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": True, "message": str(e)}


# ============================================================
# 设备后处理
# ============================================================
def apply_device_postprocess(code: str, chip: str) -> str:
    """将代码中的 cuda 替换为目标芯片的设备关键字"""
    if not code:
        return code
    cfg = CHIP_CONFIG.get(chip, {})
    target = cfg.get("device_keyword", "cuda")
    if target == "cuda":
        return code
    return re.sub(r'\bcuda\b', target, code)


# ============================================================
# GELU — Triton 融合实现 (参数化 benchmark)
# ============================================================
GELU_TRITON_CODE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x3)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1), avoids tl.math.tanh
    # which is unavailable on some non-NVIDIA Triton backends (e.g. Muxi/Corex)
    exp2x = tl.math.exp(2.0 * inner)
    tanh_inner = (exp2x - 1.0) / (exp2x + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(y_ptr + offsets, y, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y
'''

GELU_TORCH_CODE = r'''
import torch
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")
'''

GELU_TEST_FUNC = r'''
import torch
from kernel_module import gelu as triton_gelu
from torch_module import gelu as torch_gelu


def test():
    shapes = [
        (1024,), (4096,), (16384,), (65536,), (262144,),
        (1024, 1024), (4096, 4096),
    ]
    total = len(shapes)
    passed = 0
    for shape in shapes:
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        out_triton = triton_gelu(x)
        out_torch = torch_gelu(x)
        if torch.allclose(out_triton, out_torch, rtol=1e-3, atol=1e-5):
            passed += 1
        else:
            diff = (out_triton - out_torch).abs().max().item()
            raise AssertionError(f"Shape {shape}: max_diff={diff:.6e}")
    return {"total_tests": total, "passed_tests": passed, "failed_tests": 0, "errors": []}
'''

GELU_BENCHMARK_FUNC = r'''
import torch
from bench.sandbox.test.test_parametrize import parametrize
from kernel_module import gelu as triton_gelu
from torch_module import gelu as torch_gelu


@parametrize("dtype, shape", [
    (torch.float16, (1024,)),
    (torch.float16, (256, 256)),
    (torch.float16, (4, 512, 512)),
    (torch.float16, (8, 16, 64, 64)),
    (torch.float32, (1024,)),
    (torch.float32, (256, 256)),
    (torch.float32, (4, 512, 512)),
    (torch.float32, (8, 16, 64, 64)),
    (torch.bfloat16, (1024,)),
    (torch.bfloat16, (256, 256)),
    (torch.bfloat16, (4, 512, 512)),
    (torch.bfloat16, (8, 16, 64, 64)),
])
def benchmark_gelu(dtype, shape):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    n_runs = 100

    # Warmup
    for _ in range(10):
        triton_gelu(x)
        torch_gelu(x)
    torch.cuda.synchronize()

    # Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_runs):
        triton_gelu(x)
    end.record()
    torch.cuda.synchronize()
    triton_time_ms = start.elapsed_time(end) / n_runs

    # PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_runs):
        torch_gelu(x)
    end.record()
    torch.cuda.synchronize()
    torch_time_ms = start.elapsed_time(end) / n_runs

    speedup = torch_time_ms / triton_time_ms if triton_time_ms > 0 else 0.0
    return {
        "speedup": speedup,
        "ref_time": torch_time_ms,
        "res_time": triton_time_ms,
    }
'''

# ============================================================
# 矩阵乘法 — 正确性验证
# ============================================================
MATMUL_TRITON_CODE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    C = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](
        A, B, C, M, N, K1,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return C
'''

MATMUL_TORCH_CODE = r'''
import torch


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B
'''

MATMUL_TEST_FUNC = r'''
import torch
from kernel_module import matmul as triton_matmul
from torch_module import matmul as torch_matmul


def test():
    configs = [(128, 256, 128), (256, 512, 256), (512, 1024, 512)]
    total = len(configs)
    passed = 0
    for M, N, K in configs:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        out_triton = triton_matmul(A, B)
        out_torch = torch_matmul(A, B)
        if torch.allclose(out_triton, out_torch, rtol=1e-2, atol=1e-2):
            passed += 1
        else:
            diff = (out_triton.float() - out_torch.float()).abs().max().item()
            raise AssertionError(f"Shape ({M},{N},{K}): max_diff={diff:.6e}")
    return {"total_tests": total, "passed_tests": passed, "failed_tests": 0, "errors": []}
'''

# ============================================================
# 向量加法 — 正确性验证
# ============================================================
ADD_TRITON_CODE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
'''

ADD_TORCH_CODE = r'''
import torch


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
'''

ADD_TEST_FUNC = r'''
import torch
from kernel_module import add as triton_add
from torch_module import add as torch_add


def test():
    total = 3
    passed = 0
    for size in [4096 * 4096, 8192 * 8192, 16384 * 4096]:
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        y = torch.randn(size, device="cuda", dtype=torch.float32)
        out_triton = triton_add(x, y)
        out_torch = torch_add(x, y)
        if torch.allclose(out_triton, out_torch):
            passed += 1
        else:
            diff = (out_triton - out_torch).abs().max().item()
            raise AssertionError(f"Size {size}: max_diff={diff:.6e}")
    return {"total_tests": total, "passed_tests": passed, "failed_tests": 0, "errors": []}
'''


# ============================================================
# 算子注册表
# ============================================================
KERNELS = {
    "gelu": {
        "kernel_name": "gelu",
        "triton_code": GELU_TRITON_CODE,
        "torch_code": GELU_TORCH_CODE,
        "test_func_code": GELU_TEST_FUNC,
        "benchmark_func_code": GELU_BENCHMARK_FUNC,
        "desc": "GELU 激活函数 (Triton融合实现，参数化benchmark)",
    },
    "matmul": {
        "kernel_name": "matmul",
        "triton_code": MATMUL_TRITON_CODE,
        "torch_code": MATMUL_TORCH_CODE,
        "test_func_code": MATMUL_TEST_FUNC,
        "benchmark_func_code": None,
        "desc": "矩阵乘法 FP16 (正确性验证)",
    },
    "add": {
        "kernel_name": "add",
        "triton_code": ADD_TRITON_CODE,
        "torch_code": ADD_TORCH_CODE,
        "test_func_code": ADD_TEST_FUNC,
        "benchmark_func_code": None,
        "desc": "向量加法 (正确性验证)",
    },
}


# ============================================================
# 测试编排
# ============================================================

def run_accuracy_verify(base_url: str, info: dict, chip: str, gpu_id: int) -> dict:
    """
    Step 1: 正确性验证 — POST /verify (verify_type=accuracy)
    """
    pp = lambda code: apply_device_postprocess(code, chip)

    payload = {
        "kernel_name": info["kernel_name"],
        "triton_code": pp(info["triton_code"]),
        "torch_code": pp(info["torch_code"]),
        "test_func_code": pp(info["test_func_code"]),
        "verify_type": "accuracy",
        "gpu_id": gpu_id,
        "timeout": 300,
    }

    url = f"{base_url.rstrip('/')}/verify"
    print(f"  [1/2] 正确性验证 → POST {url}")
    sys.stdout.flush()

    t0 = time.time()
    result = http_post(url, payload, timeout=600)
    elapsed = time.time() - t0

    passed = result.get("passed", False)
    total = result.get("total_tests", 0)
    passed_tests = result.get("passed_tests", 0)
    failed_tests = result.get("failed_tests", 0)

    icon = "✅" if passed else "❌"
    print(f"       {icon} passed={passed}, {passed_tests}/{total} 通过, {failed_tests} 失败, 耗时 {elapsed:.2f}s")

    if result.get("errors"):
        for e in result["errors"][:5]:
            print(f"       ⚠️  {e[:160]}")

    return result


def run_performance_verify(base_url: str, info: dict, chip: str, gpu_id: int) -> dict:
    """
    Step 2: 性能验证 — POST /verify/triton (含 benchmark_func_code)
    返回 legacy 格式，包含 speedup 列表和 html 表格
    """
    pp = lambda code: apply_device_postprocess(code, chip)
    triton_code = pp(info["triton_code"])
    torch_code = pp(info["torch_code"])
    benchmark_code = pp(info.get("benchmark_func_code", ""))

    payload = {
        "triton_kernel_name": info["kernel_name"],
        "triton_kernel_code": triton_code,
        "torch_kernel_name": info["kernel_name"],
        "torch_kernel_code": torch_code,
        "test_func_code": "",
        "benchmark_func_code": benchmark_code,
        "language": "zh_CN",
        "chip": chip,
    }

    url = f"{base_url.rstrip('/')}/verify/triton"
    print(f"  [2/2] 性能验证 → POST {url}")
    if benchmark_code:
        lines = [l for l in benchmark_code.split('\n') if '@parametrize' in l]
        if lines:
            print(f"       @parametrize: {lines[0].strip()[:120]}...")
    sys.stdout.flush()

    t0 = time.time()
    result = http_post(url, payload, timeout=600)
    elapsed = time.time() - t0

    speedup = result.get("speedup")
    if speedup and isinstance(speedup, list):
        # 显示汇总
        avg_row = next((s for s in speedup if s.get("params") == "avg"), None)
        if avg_row:
            avg_sp = avg_row.get("speedup", 0)
            icon = "🚀" if avg_sp >= 1.2 else ("✅" if avg_sp >= 1.0 else "⚠️")
            print(f"       {icon} 平均加速比: {avg_sp:.3f}x, 耗时 {elapsed:.2f}s")
        else:
            print(f"       总条目: {len(speedup)}, 耗时 {elapsed:.2f}s")
    elif result.get("success") is False:
        print(f"       ❌ 失败: {result.get('traceback', '')[:200]}")
    else:
        print(f"       耗时 {elapsed:.2f}s")

    # 打印 HTML 表格
    info_data = result.get("info", {})
    html = info_data.get("html", "")
    if html:
        print()
        print(_render_html_table(html))

    return result


def _render_html_table(html: str) -> str:
    """从 HTML 中提取 rich 表格文本并格式化到终端"""
    # 尝试用 ANSI 颜色渲染（如果终端支持）
    # 回退方案：提取 <pre> 中的纯文本
    import textwrap

    # 提取 <pre> 内容
    pre_match = re.search(r'<pre[^>]*>(.*?)</pre>', html, re.DOTALL)
    if pre_match:
        text = pre_match.group(1)
        text = unescape(text)
        # 清理 HTML 标签（span 等）
        text = re.sub(r'<[^>]+>', '', text)
        # 清理多余的空白行
        lines = text.split('\n')
        # 过滤掉只有颜色代码的行
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                clean_lines.append(line)
        text = '\n'.join(clean_lines)
        return textwrap.indent(text, '     ')

    # 如果没找到 pre，返回 truncate 的 HTML
    return textwrap.indent(html[:2000], '     ')


def run_accuracy_only_test(base_url: str, info: dict, chip: str, gpu_id: int) -> dict:
    """
    仅正确性验证（适用于无 benchmark_func_code 的算子，如 matmul / add）
    走 /verify?verify_type=both 自动做性能测试
    """
    pp = lambda code: apply_device_postprocess(code, chip)

    payload = {
        "kernel_name": info["kernel_name"],
        "triton_code": pp(info["triton_code"]),
        "torch_code": pp(info["torch_code"]),
        "test_func_code": pp(info["test_func_code"]),
        "verify_type": "both",
        "gpu_id": gpu_id,
        "timeout": 300,
    }

    url = f"{base_url.rstrip('/')}/verify"
    print(f"  → POST {url} (verify_type=both)")
    sys.stdout.flush()

    t0 = time.time()
    result = http_post(url, payload, timeout=600)
    elapsed = time.time() - t0

    passed = result.get("passed", False)
    total = result.get("total_tests", 0)
    passed_tests = result.get("passed_tests", 0)
    speedup = result.get("speedup")

    icon = "✅" if passed else "❌"
    parts = [f"{icon} passed={passed}", f"{passed_tests}/{total} 通过"]
    if speedup is not None:
        sp_icon = "🚀" if speedup >= 1.2 else ("✅" if speedup >= 1.0 else "⚠️")
        parts.append(f"加速比={speedup:.3f}x {sp_icon}")
    parts.append(f"耗时 {elapsed:.2f}s")
    print(f"       {' | '.join(parts)}")

    if result.get("errors"):
        for e in result["errors"][:3]:
            print(f"       ⚠️  {e[:160]}")

    return result


def check_health(base_url: str):
    data = http_get(f"{base_url.rstrip('/')}/health")
    print(f"  Health:   {json.dumps(data, ensure_ascii=False) if 'error' not in data else '❌ 不可达'}")

    gpu = http_get(f"{base_url.rstrip('/')}/gpu/status")
    if "error" not in gpu:
        print(f"  Vendor:   {gpu.get('vendor', '?')}")
        print(f"  Devices:  {gpu.get('device_count', 0)} total, {gpu.get('available_devices', 0)} available")
        for d in gpu.get("devices", []):
            print(f"    GPU {d['device_id']}: {d['name']} ({d['free_memory_mb']}/{d['total_memory_mb']} MB free)")
    else:
        print(f"  GPU:      ⚠️  {gpu.get('message', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="KernelGen 验证服务测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python test_verify_service.py --url http://172.24.4.45:8912 --chip nvidia
  python test_verify_service.py --url http://10.7.64.131:8912 --chip huawei --all
  python test_verify_service.py --url http://10.7.66.1:8912 --chip moore --kernel gelu""",
    )
    parser.add_argument("--url", default="http://localhost:8912", help="验证服务地址")
    parser.add_argument("--chip", default="nvidia", choices=list(CHIP_CONFIG.keys()), help="芯片类型")
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--all", action="store_true", help="测试所有算子")
    parser.add_argument("--kernel", default="gelu", choices=list(KERNELS.keys()), help="指定算子")
    parser.add_argument("--health-only", action="store_true", help="仅检查健康状态")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    chip = args.chip
    cfg = CHIP_CONFIG[chip]

    print("=" * 72)
    print("  KernelGen 验证服务测试")
    print("=" * 72)
    print(f"  服务:  {base_url}")
    print(f"  芯片:  {chip} ({cfg['desc']})")
    print(f"  设备:  {cfg['device_keyword']}" +
          (f" (cuda → {cfg['device_keyword']})" if cfg['device_keyword'] != 'cuda' else " (CUDA兼容, 不替换)"))
    print(f"  GPU:   {args.gpu_id}" + (" (自动)" if args.gpu_id == -1 else ""))
    print()

    check_health(base_url)

    if args.health_only:
        return

    kernels_to_test = list(KERNELS.keys()) if args.all else [args.kernel]

    all_passed = True
    for name in kernels_to_test:
        info = KERNELS[name]

        print(f"\n{'─' * 72}")
        print(f"  [{name}] {info['desc']}")
        print(f"{'─' * 72}")

        if info.get("benchmark_func_code"):
            # GELU: 正确性 + 参数化性能
            acc_result = run_accuracy_verify(base_url, info, chip, args.gpu_id)
            if not acc_result.get("passed"):
                all_passed = False
            run_performance_verify(base_url, info, chip, args.gpu_id)
        else:
            # matmul / add: 正确性
            result = run_accuracy_only_test(base_url, info, chip, args.gpu_id)
            if not result.get("passed"):
                all_passed = False

    print(f"\n{'=' * 72}")
    print(f"  测试完成  |  芯片: {chip}  |  服务: {base_url}")
    print(f"{'=' * 72}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()