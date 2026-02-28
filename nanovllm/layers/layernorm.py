from typing import Tuple
import torch
from torch import nn
from typing import Optional, Tuple, Union
import time
from .triton_rmsnorm import skip_rmsnorm_no_view

 

class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

class TritonRMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return skip_rmsnorm_no_view(x, residual, self.weight,self.eps)

def benchmark_module(module, x, residual, num_warmup=10, num_runs=100, device="cuda"):
    # Warmup
    for _ in range(num_warmup):
        if residual is not None:
            _ = module(x, residual)
        else:
            _ = module(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timing
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            if residual is not None:
                _ = module(x, residual)
            else:
                _ = module(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(num_runs):
            if residual is not None:
                _ = module(x, residual)
            else:
                _ = module(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms / num_runs


if __name__ == "__main__":
    # ========== 配置区（直接修改这里）==========
    hidden_size = 4096
    batch_size = 8
    seq_len = 2048
    dtype = torch.bfloat16      # 可选: torch.float16, torch.bfloat16, torch.float32
    with_residual = False        # 设为 False 则测试无 residual 模式
    num_warmup = 10
    num_runs = 100
    # =========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")

    shape = (batch_size, seq_len, hidden_size)
    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None

    # 初始化模型
    rms_norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    triton_rms_norm = TritonRMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)

    # 使用 clone 确保输入完全一致
    x1 = x.clone()
    x2 = x.clone()
    res1 = residual.clone() if residual is not None else None
    res2 = residual.clone() if residual is not None else None

    # 获取输出
    print("Running forward passes for numerical comparison...")
    with torch.no_grad():
        out_pytorch = rms_norm(x1, res1)
        out_triton = triton_rms_norm(x2, res2)

    # 解包 tuple if needed
    if isinstance(out_pytorch, tuple):
        y_pt, res_pt = out_pytorch
        y_tr, res_tr = out_triton
    else:
        y_pt, y_tr = out_pytorch, out_triton
        res_pt = res_tr = None

    # 数值一致性检查
    print("\n[ Numerical Consistency Check ]")
    if res_pt is not None:
        print("Checking output tensor...")
    consistent_y = torch.allclose(y_pt, y_tr, atol=1e-2, rtol=1e-2)
    print(f"  Output match: {consistent_y}")
    
    if res_pt is not None:
        print("Checking residual tensor...")
        consistent_res = torch.allclose(res_pt, res_tr, atol=1e-2, rtol=1e-2)
        print(f"  Residual match: {consistent_res}")
        overall_consistent = consistent_y and consistent_res
    else:
        overall_consistent = consistent_y

    if not overall_consistent:
        max_diff_y = (y_pt - y_tr).abs().max().item()
        print(f"  Max absolute diff in output: {max_diff_y:.6f}")
        if res_pt is not None:
            max_diff_res = (res_pt - res_tr).abs().max().item()
            print(f"  Max absolute diff in residual: {max_diff_res:.6f}")
        print("⚠️  Numerical mismatch detected!")
    else:
        print("✅ Outputs are numerically consistent.")

    print("\n" + "="*60)

    # Benchmark（保持不变）
    print("Benchmarking PyTorch RMSNorm (with torch.compile)...")
    pytorch_latency = benchmark_module(
        rms_norm, x, residual,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device
    )

    print("Benchmarking Triton RMSNorm...")
    triton_latency = benchmark_module(
        triton_rms_norm, x, residual,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device
    )

    # 打印结果
    print("\n" + "=" * 60)
    print(f"Config: bs={batch_size}, seq={seq_len}, hidden={hidden_size}, dtype={dtype}, with_residual={with_residual}")
    print(f"PyTorch RMSNorm latency:  {pytorch_latency:.3f} ms")
    print(f"Triton RMSNorm latency:   {triton_latency:.3f} ms")
    speedup = pytorch_latency / triton_latency
    print(f"Triton speedup:           {speedup:.2f}x")
    print("=" * 60)