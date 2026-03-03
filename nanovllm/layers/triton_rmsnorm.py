"""
modified from https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/fused/skip_rms_norm.py
"""

import torch
import triton
import triton.language as tl
#from triton_utils import calculate_settings

@triton.jit
def skip_rms_norm_kernel_no_view(
    Y_ptr,
    X_ptr,
    R_ptr,
    W_ptr,
    B,
    S,
    N,
    x_stride_b,
    x_stride_s,
    x_stride_n,
    r_stride_b,
    r_stride_s,
    r_stride_n,
    y_stride_b,
    y_stride_s,
    y_stride_n,
    w_stride,
    eps,
    has_residual: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # pid表示处理的行号: 行索引 = batch_idx * S + seq_idx
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    X_ptr = X_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
    Y_ptr = Y_ptr + batch_idx * y_stride_b + seq_idx * y_stride_s
    # R_ptr只有在has_residual为True时才使用

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + cols * x_stride_n, mask=mask, other=0.0).to(tl.float32)

    # 当有residual时，加载并加上r，然后回写r
    if has_residual:
        R_ptr = R_ptr + batch_idx * r_stride_b + seq_idx * r_stride_s
        r = tl.load(R_ptr + cols * r_stride_n, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(R_ptr + cols * r_stride_n, x, mask=mask)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + cols * w_stride, mask=mask, other=0.0)
    y = (x * rrms).to(tl.float16) * w

    tl.store(Y_ptr + cols * y_stride_n, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm_no_view(
    X:torch.Tensor,
    residual:torch.Tensor | None,
    weight:torch.Tensor,
    eps:float = 1e-5
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # 假设X: [B, S, N]
    # 若X为[B,S,N]，不对其进行view
    # 支持 2D [S, N] 或 3D [B, S, N]
    #input_shape = X.shape
    if X.dim() == 2:
        # 自动添加 batch 维度: [S, N] -> [1, S, N]
        X = X.unsqueeze(0)
        if residual is not None:
            residual = residual.unsqueeze(0)
        was_2d = True
    elif X.dim() == 3:
        was_2d = False
    else:
        raise ValueError(f"Expected X to be 2D or 3D, got {X.dim()}D")
  
    B, S, N = X.shape
    Y = torch.empty_like(X)

    x_stride_b, x_stride_s, x_stride_n = X.stride()
    y_stride_b, y_stride_s, y_stride_n = Y.stride()
    w_stride = weight.stride(0)

    # 如果 residual 不为 None，则确保与X同shape和stride
    if residual is not None:
        residual = residual.contiguous()  # 确保是连续存储
        r_stride_b, r_stride_s, r_stride_n = residual.stride()
        has_residual = True
    else:
        # 如果 residual 是 None，则在kernel中不处理residual
        # 这里给r_stride_*赋默认值，但不会使用
        r_stride_b, r_stride_s, r_stride_n = 0, 0, 0
        has_residual = False

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (B * S,)

    skip_rms_norm_kernel_no_view[grid](
        Y,
        X,
        residual
        if residual is not None
        else X,  # 若无residual，这里传X只是占位，kernel中不使用R_ptr
        weight,
        B,
        S,
        N,
        x_stride_b,
        x_stride_s,
        x_stride_n,
        r_stride_b,
        r_stride_s,
        r_stride_n,
        y_stride_b,
        y_stride_s,
        y_stride_n,
        w_stride,
        eps,
        has_residual=has_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    if was_2d:
        Y = Y.squeeze(0)
        if residual is not None:
            residual = residual.squeeze(0)
            return Y, residual
        else:
            return Y
    else:
        return Y if residual is None else (Y, residual)