import torch
from torch import nn
import torch.nn.functional as F
from .triton_swiglu import swiglu_forward
import time


class SiluAndMul(nn.Module):
    '''
    SwiGLU的trick：x @ gate_up = combine
    conbine -> chunk(2,-1) -> x,y
    silu(x) * y 
    '''
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class TritonSiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return swiglu_forward(x, y)



def benchmark_module(module, x, num_warmup=10, num_runs=100, device="cuda"):
    # Warmup
    for _ in range(num_warmup):
        _ = module(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timing
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = module(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = module(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms / num_runs


if __name__ == "__main__":
    # ========== 配置区（直接修改这里）==========
    hidden_size = 4096          # 注意：SwiGLU 输入是 2 * intermediate_size
    batch_size = 8
    seq_len = 2048
    dtype = torch.bfloat16      # 可选: torch.float16, torch.bfloat16
    num_warmup = 10
    num_runs = 100
    # =========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")

    # SwiGLU 的输入维度必须是偶数（因为要 chunk 成两半）
    assert hidden_size % 2 == 0, "hidden_size must be even for SwiGLU"

    shape = (batch_size, seq_len, hidden_size)
    x = torch.randn(shape, device=device, dtype=dtype)

    # 初始化模块
    #pytorch_swiglu = SiluAndMul().to(device=device, dtype=dtype)
    pytorch_swiglu = TritonSiluAndMul().to(device=device, dtype=dtype)

    # 触发编译
    print("Warming up PyTorch SwiGLU...")
    _ = pytorch_swiglu(x.clone())
    print("Warmup done.")

    # Benchmark PyTorch version
    print("Benchmarking PyTorch SwiGLU (with torch.compile)...")
    pytorch_latency = benchmark_module(
        pytorch_swiglu, x,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device
    )

    triton_latency = None
    triton_swiglu = TritonSiluAndMul().to(device=device, dtype=dtype)
    print("Warming up Triton SwiGLU...")
    _ = triton_swiglu(x.clone())
    print("Warmup done.")

    print("Benchmarking Triton SwiGLU...")
    triton_latency = benchmark_module(
        triton_swiglu, x,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device
    )

    # 打印结果
    print("\n" + "=" * 60)
    print(f"Config: bs={batch_size}, seq={seq_len}, input_hidden={hidden_size}, dtype={dtype}")
    print(f"PyTorch SwiGLU latency:   {pytorch_latency:.3f} ms")
    if triton_latency is not None:
        print(f"Triton SwiGLU latency:    {triton_latency:.3f} ms")
        speedup = pytorch_latency / triton_latency
        print(f"Triton speedup:           {speedup:.2f}x")
    print("=" * 60)