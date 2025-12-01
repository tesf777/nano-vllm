from dataclasses import dataclass


@dataclass
class SamplingParams:
    '''
    针对logits的采样策略，用于控制从模型输出 logits 中如何生成下一个 token
    temperature: logits = logits/temp 调节 logits 分布的平滑程度，>1则更平滑，更随机
    max_tokens:单次生成的最大 token 数量
    ignore_eos:忽视结尾提示符
    '''
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
