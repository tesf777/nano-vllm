import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_k: int = 0, top_p: float = 1.0):
        """
        Args:
            logits: (batch_size, vocab_size) - 模型输出的logits
            temperatures: (batch_size,) - 采样温度
            top_k: top-k采样，0表示不限制
            top_p: top-p采样，1.0表示不限制
        Returns:
            sample_tokens: (batch_size,) - 采样得到的token id
        """
        # 应用温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)

        # Top-k采样
        if top_k > 0:
            top_k = min(top_k, probs.size(-1))
            values, indices = torch.topk(probs, top_k, dim=-1)
            probs = torch.full_like(probs, float('-inf'))
            probs.scatter_(dim=-1, index=indices, src=values)

        # Top-p (nucleus) 采样
        if top_p < 1.0:
            # 对概率进行降序排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            # 计算累积概率
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 创建掩码，移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 至少保留一个token（概率最大的）
            sorted_indices_to_remove[..., 0] = False
            # 将要移除的位置设为-inf
            probs.scatter_(dim=-1, index=sorted_indices,
                           src=torch.where(sorted_indices_to_remove,
                                           torch.full_like(sorted_probs, float('-inf')),
                                           sorted_probs))

        # Gumbel-Max Trick: 生成与 probs 同形状的 指数分布随机数
        # argmax[log(pi)+Gi]=argmax[log(pi)-log(Ei)]=argmax[log(pi/Ei)]
        # 等价于从 softmax 概率分布中做一次多项式采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
