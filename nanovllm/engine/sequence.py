from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

'''
@property装饰器实现了：a.isTrue() -> a.isTrue
sequence定义了一个用户请求。
decoder-only是自回归生成，input = prompt_token + generated_token

用户请求: "Hello, how are you?"

Tokenize → [1234, 5678, 9101, 1121]

┌──────────────┐
│   PREFILL    │ ←─ 对 [1234, 5678, 9101, 1121] 做一次完整 forward
│ (Context Enc)│    → 计算并缓存所有 K/V
└──────┬───────┘
       ↓
输出 logits[-1] → 采样 → "I"  (第1个生成token)

┌──────────────┐
│   DECODING   │ ←─ 输入 ["I"]，利用 KV Cache 计算新 K/V
└──────┬───────┘
       ↓
采样 → "'m" ...

（循环直到结束）
'''

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    
    # KV Cache 的内存块大小（单位：token）。所有序列共享此值，通常设为 16/32/256，用于对齐 GPU 内存访问。
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 从0开始分配seq的id
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 完整 token 列表 [prompt + generated] :list
        self.token_ids = copy(token_ids)
        # 最后一个 token：用于 decoding
        self.last_token = token_ids[-1]
        # 当前总token数量，num_tokens在自回归生成过程会变长 :len(list)
        self.num_tokens = len(self.token_ids) 
        # num_prompt_tokens是prefill阶段的token，不会增加了。
        self.num_prompt_tokens = len(token_ids)
        # 已缓存到 GPU 的 token 数（用于增量更新）
        self.num_cached_tokens = 0
        # 指向物理 KV Cache 块(Blockid)的索引列表（如 [3, 7, 12]）
        self.block_table = []
        # 将采样策略“扁平化”到序列对象中，避免每次生成时传递 SamplingParams
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        '''
        已经生成的tokens数量，总长度-prefill长度
        '''
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        '''获取prompt的tokens'''
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        '''获取自回归生成的tokens'''
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        '''
        已缓存的 block 数，总缓存token数量//一个block容纳token大小(256)
        '''
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        '''
        一个Seq总共需要的 block 数，这里的做法是向上取整，保证block只多不少
        '''
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        '''
        最后一个block中token的数量：总token数量减去n-1个block的token数量
        因为考虑到不能整除的情况，最后一个block往往未装满，例如: (124/256)
        '''
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        '''
        返回第 i 个逻辑 block 对应的一组(256个) token IDs。
        用于 prefill 阶段：将 prompt 分块送入模型，逐步构建 KV Cache。
        '''
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        '''给[prompt + generated]列表添加自回归生成的token'''
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        ''' 
        使用pickle包序列化时的魔术方法：
        
        如果尚未生成任何 token，保存完整 token_ids；
        否则只保存 last_token（节省序列化开销），因为kvcache已经存有上下文信息了
        Prefill 阶段：需完整 prompt
        Decoding 阶段：只需最后一个 token
        '''
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        '''
        反序列化时重建 token_ids 或 last_token
        '''
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
