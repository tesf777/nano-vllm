import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 生成与 probs 同形状的 指数分布随机数
        # argmax[log(pi)+Gi]=argmax[log(pi)-log(Ei)]=argmax[log(pi/Ei)]
        # 等价于从 softmax 概率分布中做一次多项式采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
