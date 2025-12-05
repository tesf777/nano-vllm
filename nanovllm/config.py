import os
from dataclasses import dataclass
from transformers import AutoConfig
import argparse

'''
这里的装饰器写法是简化了繁琐的变量赋值，
并且自动生成魔术方法：__repr__变量打印，__eq__比较重载
__post_init__是钩子函数，在init后进行，通常用于加载后验证
'''
@dataclass
class Config:
    model: str
    # 单个batch中允许的最大token总数(prompt + generated tokens)
    max_num_batched_tokens: int = 16384
    # 单个batch允许的最大并发请求数量 (512 * ?) 每个请求数长度不确定
    max_num_seqs: int = 512
    # 当前设置下模型的最大上下文长度4096
    # Qwen3-0.6B 原生支持 32768，但设为 4096，则实际用 4096
    max_model_len: int = 4096
    # GPU显存利用率上限
    gpu_memory_utilization: float = 0.9
    # 张量并行数量 1=单卡
    tensor_parallel_size: int = 1
    # cuda graph:一个优化方法。预先启动算子，并融合部分来减少kernel launch的时间 false为使用cuda graph
    # true可以更利于debug
    enforce_eager: bool = False
    # 使用加载的 Hugging Face 模型配置对象
    hf_config: AutoConfig | None = None
    # -1表示未设置，后续可能从 tokenizer 自动获取
    eos: int = -1
    # 将 KV Cache 分成固定大小的 1 block = 256 tokens
    kvcache_block_size: int = 256
    # -1表示自动计算block块数目，通常根据之前的利用率，去除权重等其他占用，最后留给kvcache
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read config")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="model path",
        default="./Qwen3-0.6B"
    )
    args = parser.parse_args()
    config = Config(model=args.model)
    print(config)

    '''
    Config(model='./Qwen3-0.6B', max_num_batched_tokens=16384, max_num_seqs=512, max_model_len=4096, gpu_memory_utilization=0.9, tensor_parallel_size=1, enforce_eager=False, hf_config=Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
, eos=-1, kvcache_block_size=256, num_kvcache_blocks=-1)
    '''