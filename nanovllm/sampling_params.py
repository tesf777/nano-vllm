from dataclasses import dataclass


@dataclass
class SamplingParams:
    '''
    针对logits的采样策略，用于控制从模型输出 logits 中如何生成下一个 token
    temperature: logits = logits/temp 调节 logits 分布的平滑程度，>1则更平滑，更随机
    max_tokens:单次生成的最大 token 数量
    ignore_eos:忽视结尾提示符
    top_k: 只保留概率最高的k个token，其余被过滤掉（0表示不限制）
    top_p: nucleus sampling，保留累积概率达到p的最小token集合（0-1之间，0或1表示不限制）
    '''
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert 0 <= self.top_p <= 1.0, "top_p must be between 0 and 1"
