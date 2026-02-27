# Nano-vLLM 源码导读

> 面向具备Python及PyTorch基础但对vLLM架构陌生的初学者的系统性学习指南

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [核心类与函数深度解析](#2-核心类与函数深度解析)
3. [关联性分析](#3-关联性分析)
4. [分阶段源码学习路线图](#4-分阶段源码学习路线图)
5. [FlashAttention与其他Triton算子集成指南](#5-flashattention与其他triton算子集成指南)

---

## 1. 项目架构总览

### 1.1 目录结构

```
nano-vllm/
├── nanovllm/                    # 核心代码目录
│   ├── __init__.py              # 包入口，导出LLM和SamplingParams
│   ├── config.py                # 配置类，定义全局配置参数
│   ├── llm.py                   # 用户接口类，继承自LLMEngine
│   ├── sampling_params.py       # 采样参数类，控制生成策略
│   ├── engine/                  # 引擎核心模块
│   │   ├── llm_engine.py       # LLMEngine类，协调所有组件
│   │   ├── scheduler.py         # Scheduler类，请求调度器
│   │   ├── block_manager.py     # BlockManager类，KV Cache内存管理
│   │   ├── sequence.py          # Sequence类，表示一个推理请求
│   │   └── model_runner.py      # ModelRunner类，模型执行器
│   ├── layers/                  # 神经网络层组件
│   │   ├── attention.py         # Attention层（含PagedAttention实现）
│   │   ├── activation.py        # 激活函数层（SiLU等）
│   │   ├── linear.py           # 线性层（含张量并行实现）
│   │   ├── layernorm.py         # RMSNorm层
│   │   ├── embed_head.py        # Embedding和LM Head层
│   │   ├── rotary_embedding.py  # RoPE旋转位置编码
│   │   └── sampler.py          # 采样层
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   └── qwen3.py            # Qwen3模型实现
│   └── utils/                   # 工具模块
│       ├── context.py           # 全局上下文管理
│       └── loader.py            # 模型权重加载器
├── example.py                   # 使用示例
├── bench.py                     # 性能测试脚本
└── README.md                    # 项目说明
```

### 1.2 核心功能模块

| 模块 | 职责 | 边界说明 |
|------|------|----------|
| **Frontend (LLM/LLMEngine)** | 用户API接口，接收请求并返回结果 | 处理输入tokenization和输出detokenization |
| **Scheduler** | 请求调度，决定哪些请求进入推理 | 管理waiting和running队列，实现prefill/decode分离调度 |
| **BlockManager** | KV Cache内存管理，分页内存分配 | 管理物理block池，实现prefix caching |
| **ModelRunner** | 模型执行，管理多GPU推理 | 协调TP通信，分配KV Cache，执行前向传播 |
| **Sequence** | 表示一个推理请求 | 封装prompt、生成tokens、状态、页表等 |
| **Attention** | 注意力计算 | 调用FlashAttention实现PagedAttention |

### 1.3 核心术语说明

| 术语 | 解释 |
|------|------|
| **Prefill** | 预填充阶段，对prompt进行编码并计算所有token的KV Cache |
| **Decode** | 解码阶段，逐个生成新token，利用已缓存的KV |
| **PagedAttention** | 分页注意力，将KV Cache分块存储以支持动态批处理 |
| **KV Cache** | Key-Value缓存，存储各层的注意力键值对，避免重复计算 |
| **Prefix Caching** | 前缀缓存，多个请求共享相同prompt的KV Cache |
| **Tensor Parallelism (TP)** | 张量并行，将模型参数切分到多个GPU |
| **CUDA Graph** | CUDA图优化，预定义计算图减少kernel launch开销 |

---

## 2. 核心类与函数深度解析

### 2.1 配置模块：[config.py](nanovllm/config.py)

#### Config类

| 成员变量 | 类型 | 说明 |
|----------|------|------|
| `model` | str | 模型路径 |
| `max_num_batched_tokens` | int | 单batch最大token数（默认16384） |
| `max_num_seqs` | int | 单batch最大序列数（默认512） |
| `max_model_len` | int | 模型最大上下文长度 |
| `gpu_memory_utilization` | float | GPU显存利用率（默认0.9） |
| `tensor_parallel_size` | int | TP并行数量（默认1） |
| `enforce_eager` | bool | 是否禁用CUDA Graph |
| `eos` | int | 结束token ID |
| `kvcache_block_size` | int | KV Cache块大小（默认256 tokens） |
| `num_kvcache_blocks` | int | KV Cache块数量（-1表示自动计算） |
| `hf_config` | AutoConfig | HuggingFace配置对象 |

---

### 2.2 采样参数：[sampling_params.py](nanovllm/sampling_params.py)

#### SamplingParams类

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0      # 温度参数，控制随机性
    max_tokens: int = 64           # 最大生成token数
    ignore_eos: bool = False       # 是否忽略结束符
```

**采样公式**：`probs = softmax(logits / temperature)`

---

### 2.3 Sequence模块：[sequence.py](nanovllm/engine/sequence.py)

#### SequenceStatus枚举

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 等待调度
    RUNNING = auto()     # 正在推理
    FINISHED = auto()    # 已完成
```

#### Sequence类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `seq_id` | int | 唯一序列ID |
| `status` | SequenceStatus | 当前状态 |
| `token_ids` | list[int] | 完整token列表（prompt + generated） |
| `last_token` | int | 最后一个token（decoding用） |
| `num_tokens` | int | 当前token总数 |
| `num_prompt_tokens` | int | prompt token数（不变） |
| `num_cached_tokens` | int | 已缓存的token数 |
| `block_table` | list[int] | 逻辑块表，存储物理block ID |
| `temperature` | float | 采样温度 |
| `max_tokens` | int | 最大生成token数 |
| `ignore_eos` | bool | 是否忽略结束符 |

**核心属性方法**：

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `is_finished` | bool | 是否已完成 |
| `num_completion_tokens` | int | 已生成token数 |
| `prompt_token_ids` | list | prompt token IDs |
| `completion_token_ids` | list | 生成的token IDs |
| `num_cached_blocks` | int | 已缓存的block数 |
| `num_blocks` | int | 总共需要的block数 |
| `last_block_num_tokens` | int | 最后一个block的token数 |

**核心方法**：

```python
def block(self, i: int) -> list[int]:
    """
    返回第i个逻辑块对应的token IDs
    用于prefill阶段分块处理
    """

def append_token(self, token_id: int):
    """追加新token到序列"""
```

---

### 2.4 Block Manager模块：[block_manager.py](nanovllm/engine/block_manager.py)

#### Block类

表示一个物理KV Cache块。

| 成员 | 类型 | 说明 |
|------|------|------|
| `block_id` | int | 块ID |
| `ref_count` | int | 引用计数 |
| `hash` | int | 块内容的哈希值 |
| `token_ids` | list[int] | 块包含的token IDs |

#### BlockManager类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `block_size` | int | 每个block的token数 |
| `blocks` | list[Block] | 所有物理块 |
| `hash_to_block_id` | dict[int, int] | 哈希到块ID的映射 |
| `free_block_ids` | deque[int] | 空闲块队列 |
| `used_block_ids` | set[int] | 已用块集合 |

**核心方法**：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `compute_hash` | token_ids, prefix | int | 计算块哈希（含前缀） |
| `can_allocate` | seq | bool | 是否有足够空闲块 |
| `allocate` | seq | None | 为序列分配块（支持prefix caching） |
| `deallocate` | seq | None | 释放序列所有块 |
| `can_append` | seq | bool | 是否可以追加新token |
| `may_append` | seq | None | 执行块扩展或缓存更新 |

**核心算法：Prefix Caching**

```
对于每个block：
  1. 计算hash（考虑前缀hash，确保上下文敏感）
  2. 检查hash_to_block_id中是否存在相同hash
  3. 若存在且内容匹配 → 复用该block，增加引用计数
  4. 否则 → 分配新block，更新hash映射
```

---

### 2.5 Scheduler模块：[scheduler.py](nanovllm/engine/scheduler.py)

#### Scheduler类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `max_num_seqs` | int | 单batch最大序列数 |
| `max_num_batched_tokens` | int | 单batch最大token数 |
| `eos` | int | 结束token ID |
| `block_manager` | BlockManager | 块管理器 |
| `waiting` | deque[Sequence] | 等待队列 |
| `running` | deque[Sequence] | 运行队列 |

**核心方法**：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add` | seq | None | 添加请求到等待队列 |
| `schedule` | None | (list[Sequence], bool) | 调度下一批序列，返回(序列列表, 是否prefill) |
| `preempt` | seq | None | 抢占序列，释放资源 |
| `postprocess` | seqs, token_ids | None | 处理生成结果，更新状态 |

**核心算法：Prefill/Decode分离调度**

```python
def schedule(self):
    # ===== Prefill阶段 =====
    scheduled_seqs = []
    num_batched_tokens = 0
    while self.waiting and len(scheduled_seqs) < max_num_seqs:
        seq = self.waiting[0]
        if not enough_memory or not self.block_manager.can_allocate(seq):
            break
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True  # prefill

    # ===== Decode阶段 =====
    while self.running and len(scheduled_seqs) < max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)  # 无法分配，回到等待队列
                break
        else:
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    return scheduled_seqs, False  # decode
```

---

### 2.6 Model Runner模块：[model_runner.py](nanovllm/engine/model_runner.py)

#### ModelRunner类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `config` | Config | 配置对象 |
| `model` | Qwen3ForCausalLM | 模型实例 |
| `sampler` | Sampler | 采样器 |
| `kv_cache` | torch.Tensor | 预分配的KV Cache |
| `world_size` | int | TP并行规模 |
| `rank` | int | 当前GPU rank |
| `event` | Event | 多进程同步事件 |
| `shm` | SharedMemory | 共享内存（TP通信用） |
| `graphs` | dict[int, CUDAGraph] | CUDA Graph缓存 |
| `graph_vars` | dict | CUDA Graph变量 |

**核心方法**：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `warmup_model` | None | None | 预热模型，触发内存分配 |
| `allocate_kv_cache` | None | None | 分配KV Cache内存 |
| `capture_cudagraph` | None | None | 捕获CUDA Graph |
| `prepare_block_tables` | seqs | Tensor | 准备页表 |
| `prepare_prefill` | seqs | (Tensor, Tensor) | 准备prefill输入 |
| `prepare_decode` | seqs | (Tensor, Tensor) | 准备decode输入 |
| `prepare_sample` | seqs | Tensor | 准备采样参数 |
| `run_model` | input_ids, positions, is_prefill | Tensor | 执行模型前向传播 |
| `run` | seqs, is_prefill | list[int] | 执行完整推理步骤 |

**核心算法：KV Cache分配**

```
1. 获取GPU总显存和当前使用量
2. 计算每个block的字节数：
   block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
3. 计算可用block数：
   num_blocks = floor((total * gpu_memory_utilization - used - peak + current) / block_bytes)
4. 预分配张量：
   kv_cache = torch.zeros(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

**核心算法：多进程通信**

```
主进程（rank=0）：
  write_shm(method_name, args) → 序列化并写入共享内存 → event.set()

工作进程（rank>0）：
  event.wait() → read_shm() → 执行方法
```

---

### 2.7 LLM Engine模块：[llm_engine.py](nanovllm/engine/llm_engine.py)

#### LLMEngine类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `model_runner` | ModelRunner | 模型执行器 |
| `tokenizer` | AutoTokenizer | 分词器 |
| `scheduler` | Scheduler | 调度器 |
| `ps` | list[Process] | 子进程列表（TP） |
| `events` | list[Event] | 同步事件列表 |

**核心方法**：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_request` | prompt, sampling_params | None | 添加推理请求 |
| `step` | None | (list, int) | 执行一步推理，返回(完成的序列, token数) |
| `is_finished` | None | bool | 是否所有请求完成 |
| `generate` | prompts, sampling_params | list[dict] | 完整生成流程 |

**数据流转图**：

```
用户请求 (prompt)
    ↓
add_request() → tokenizer.encode() → Sequence对象
    ↓
step() → schedule() → 调度结果 (seqs, is_prefill)
    ↓
model_runner.run() → prepare_input() → run_model() → 采样
    ↓
postprocess() → 更新Sequence状态，检查结束条件
    ↓
循环直到 is_finished()
    ↓
tokenizer.decode() → 输出结果
```

---

### 2.8 Attention模块：[attention.py](nanovllm/layers/attention.py)

#### Attention类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `num_heads` | int | 注意力头数 |
| `head_dim` | int | 每个头的维度 |
| `scale` | float | 缩放因子（head_dim^-0.5） |
| `num_kv_heads` | int | KV头数（GQA/MQA） |
| `k_cache`, `v_cache` | Tensor | 当前层的KV Cache |

**核心方法**：

```python
def forward(self, q, k, v) -> Tensor:
    context = get_context()

    # 1. 存储新KV到Cache
    if k_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    # 2. 执行注意力计算
    if context.is_prefill:
        # Prefill阶段：使用变长FlashAttention
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=context.block_tables  # PagedAttention支持
        )
    else:
        # Decode阶段：使用KV Cache
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True
        )
    return o
```

#### store_kvcache_kernel（Triton内核）

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr, D: tl.constexpr
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return  # 前缀缓存的情况

    # 从输入KV加载
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))

    # 存储到KV Cache的对应slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

---

### 2.9 采样模块：[sampler.py](nanovllm/layers/sampler.py)

#### Sampler类

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: Tensor, temperatures: Tensor):
        # 1. 应用温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 2. 计算概率分布
        probs = torch.softmax(logits, dim=-1)

        # 3. Gumbel-Top-k采样（等价于多项式采样）
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
```

**采样原理**：Gumbel-Top-k等价于从多项分布采样，无需显式计算累积分布。

---

### 2.10 模型定义：[models/qwen3.py](nanovllm/models/qwen3.py)

#### Qwen3ForCausalLM类

```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding (词嵌入层)
│   ├── Qwen3DecoderLayer × num_hidden_layers
│   │   ├── Qwen3Attention
│   │   │   ├── QKVParallelLinear (QKV投影)
│   │   │   ├── RowParallelLinear (输出投影)
│   │   │   ├── RotaryEmbedding (位置编码)
│   │   │   └── Attention (注意力计算)
│   │   ├── Qwen3MLP
│   │   │   ├── MergedColumnParallelLinear (gate+up投影)
│   │   │   ├── RowParallelLinear (down投影)
│   │   │   └── SiluAndMul (激活函数)
│   │   ├── RMSNorm (输入归一化)
│   │   └── RMSNorm (输出归一化)
│   └── RMSNorm (最终归一化)
└── ParallelLMHead (语言模型头)
```

---

### 2.11 线性层模块：[linear.py](nanovllm/layers/linear.py)

#### 张量并行策略

| 层类型 | 并行维度 | 数据流 |
|--------|----------|--------|
| **ColumnParallelLinear** | 输出维度 | 输入完整 → 输出分片 |
| **RowParallelLinear** | 输入维度 | 输入分片 → all_reduce → 输出完整 |
| **VocabParallelEmbedding** | 词汇维度 | 分片embedding → mask → all_reduce |

#### ColumnParallelLinear（列并行）

```
完整权重: W = [W0, W1, ..., Wn-1]  (按列切分)
GPU 0: y = x @ W0
GPU 1: y = x @ W1
...
```

#### RowParallelLinear（行并行）

```
完整权重: W = [W0; W1; ...; Wn-1]  (按行切分)
GPU 0: y = (x @ W0)
GPU 1: y = (x @ W1)
...
all_reduce(y)  # 汇聚结果
```

---

### 2.12 Context模块：[context.py](nanovllm/utils/context.py)

#### Context类（全局上下文）

```python
@dataclass
class Context:
    is_prefill: bool                     # 是否prefill阶段
    cu_seqlens_q: Tensor                 # Query累积序列长度（FlashAttention用）
    cu_seqlens_k: Tensor                 # Key累积序列长度
    max_seqlen_q: int                    # Query最大序列长度
    max_seqlen_k: int                    # Key最大序列长度
    slot_mapping: Tensor                 # Token到物理位置的映射
    context_lens: Tensor                 # 每个序列的上下文长度
    block_tables: Tensor                 # 页表（block ID矩阵）
```

**全局变量设计原因**：避免在所有forward方法中层层传递参数，减少函数调用开销。

---

## 3. 关联性分析

### 3.1 模块依赖图

```
LLM (用户接口)
  ↓
LLMEngine
  ├── Scheduler
  │   ├── BlockManager
  │   └── Sequence
  │
  └── ModelRunner
      ├── Qwen3ForCausalLM
      │   ├── Qwen3Model
      │   │   ├── VocabParallelEmbedding
      │   │   ├── Qwen3DecoderLayer
      │   │   │   ├── Qwen3Attention
      │   │   │   │   ├── Attention (调用FlashAttention)
      │   │   │   │   └── RotaryEmbedding
      │   │   │   └── Qwen3MLP
      │   │   │       └── SiluAndMul
      │   │   └── RMSNorm
      │   └── ParallelLMHead
      │
      └── Sampler
```

### 3.2 关键交互关系

#### 3.2.1 Scheduler ↔ BlockManager

```
Scheduler.schedule():
  → BlockManager.can_allocate(seq)  # 检查资源
  → BlockManager.allocate(seq)      # 分配块

Scheduler.preempt():
  → BlockManager.deallocate(seq)     # 释放块

Scheduler.postprocess():
  → BlockManager.deallocate(seq)     # 完成后释放
```

#### 3.2.2 ModelRunner ↔ Scheduler

```
LLMEngine.step():
  → Scheduler.schedule()  # 获取待推理序列
  → ModelRunner.run()    # 执行推理
  → Scheduler.postprocess()  # 更新状态
```

#### 3.2.3 ModelRunner ↔ BlockManager

```
ModelRunner.prepare_prefill/decode():
  → 读取seq.block_table     # 逻辑页表
  → 生成block_tables Tensor  # 物理页表

ModelRunner.allocate_kv_cache():
  → 设置config.num_kvcache_blocks
  → BlockManager使用此值初始化
```

#### 3.2.4 Attention ↔ Context

```
Attention.forward():
  → context = get_context()  # 获取全局上下文
  → 使用context.is_prefill判断阶段
  → 使用context.slot_mapping定位KV位置
  → 使用context.block_tables实现PagedAttention
```

### 3.3 数据流转路径

```
用户请求
  ↓
LLM.generate()
  ↓
LLMEngine.add_request()
  → tokenizer.encode(prompt) → token_ids
  → Sequence(token_ids, sampling_params)
  → Scheduler.add(seq) → waiting队列
  ↓
循环：
  LLMEngine.step()
    ↓
    Scheduler.schedule()
      → 遍历waiting队列
      → BlockManager.can_allocate() / allocate()
      → 返回 (seqs, is_prefill)
      ↓
    ModelRunner.run(seqs, is_prefill)
      → prepare_block_tables() → 构建页表
      → prepare_prefill/decode() → 构建输入
      → set_context(...) → 设置全局上下文
      → run_model(input_ids, positions, is_prefill)
        → model.forward(input_ids, positions)
          → 逐层前向传播
          → Attention.forward(q, k, v)
            → store_kvcache(k, v, ...)  # 存储KV
            → flash_attn_varlen_func() / flash_attn_with_kvcache()
          → MLP.forward(x)
        → compute_logits(hidden_states)
          → lm_head(hidden_states)
      → Sampler(logits, temperatures) → token_ids
      → reset_context()
      ↓
    Scheduler.postprocess(seqs, token_ids)
      → seq.append_token(token_id)
      → 检查结束条件
      → BlockManager.deallocate(seq) [若完成]
      ↓
    返回完成的序列
  ↓
循环直到 is_finished()
  ↓
tokenizer.decode(completion_token_ids) → 输出文本
```

---

## 4. 分阶段源码学习路线图

### 阶段一：数据结构基础

**学习目标**：理解推理请求的基本数据结构和内存管理

| 文件 | 顺序 | 重点关注 | 预计时间 |
|------|------|----------|----------|
| [config.py](nanovllm/config.py) | 1 | Config类的所有配置参数 | 30分钟 |
| [sampling_params.py](nanovllm/sampling_params.py) | 2 | SamplingParams类，温度参数作用 | 15分钟 |
| [sequence.py](nanovllm/engine/sequence.py) | 3 | Sequence类的所有属性和方法 | 45分钟 |
| [block_manager.py](nanovllm/engine/block_manager.py) | 4 | Block和BlockManager，Prefix Caching原理 | 1小时 |

**阶段总结**：
- 理解Sequence如何表示一个推理请求
- 掌握Block Manager的分页内存管理机制
- 理解Prefix Caching如何实现前缀共享

---

### 阶段二：调度核心

**学习目标**：理解请求调度策略和资源管理

| 文件 | 顺序 | 重点关注 | 预计时间 |
|------|------|----------|----------|
| [scheduler.py](nanovllm/engine/scheduler.py) | 5 | Prefill/Decode分离调度算法 | 1小时 |
| [llm_engine.py](nanovllm/engine/llm_engine.py) | 6 | LLMEngine核心流程，step()方法 | 45分钟 |

**阶段总结**：
- 理解为何需要分离Prefill和Decode
- 掌握抢占（preemption）机制的实现
- 理解Batching策略如何影响吞吐量

---

### 阶段三：推理引擎

**学习目标**：理解模型执行和KV Cache管理

| 文件 | 顺序 | 重点关注 | 预计时间 |
|------|------|----------|----------|
| [context.py](nanovllm/utils/context.py) | 7 | 全局Context设计 | 30分钟 |
| [model_runner.py](nanovllm/engine/model_runner.py) | 8 | KV Cache分配，CUDA Graph，多进程通信 | 2小时 |

**阶段总结**：
- 理解KV Cache如何预分配和管理
- 掌握CUDA Graph优化原理
- 理解多GPU通信机制（共享内存 + NCCL）

---

### 阶段四：神经网络层

**学习目标**：理解模型各层的实现细节

| 文件 | 顺序 | 重点关注 | 预计时间 |
|------|------|----------|----------|
| [attention.py](nanovllm/layers/attention.py) | 9 | PagedAttention，Triton内核 | 1.5小时 |
| [sampler.py](nanovllm/layers/sampler.py) | 10 | Gumbel-Top-k采样 | 30分钟 |
| [linear.py](nanovllm/layers/linear.py) | 11 | 张量并行策略 | 1小时 |
| [activation.py](nanovllm/layers/activation.py) | 12 | SiLU激活函数 | 15分钟 |
| [layernorm.py](nanovllm/layers/layernorm.py) | 13 | RMSNorm实现 | 30分钟 |
| [rotary_embedding.py](nanovllm/layers/rotary_embedding.py) | 14 | RoPE旋转位置编码 | 45分钟 |
| [embed_head.py](nanovllm/layers/embed_head.py) | 15 | Embedding和LM Head的并行 | 30分钟 |

**阶段总结**：
- 理解PagedAttention如何支持动态批处理
- 掌握FlashAttention的调用方式
- 理解张量并行的具体切分策略

---

### 阶段五：模型整合

**学习目标**：理解完整模型的前向传播流程

| 文件 | 顺序 | 重点关注 | 预计时间 |
|------|------|----------|----------|
| [models/qwen3.py](nanovllm/models/qwen3.py) | 16 | 模型架构，各层组合 | 1.5小时 |
| [loader.py](nanovllm/utils/loader.py) | 17 | 权重加载机制，packed_modules | 45分钟 |
| [example.py](example.py) | 18 | 完整使用流程 | 30分钟 |

**阶段总结**：
- 理解Transformer Decoder层的完整前向传播
- 掌握权重如何从HuggingFace格式加载
- 理解packed_modules如何支持融合层

---

### 阶段六：实践与调试

**学习目标**：通过实际运行加深理解

| 任务 | 说明 | 预计时间 |
|------|------|----------|
| 运行example.py | 观察完整推理流程 | 30分钟 |
| 修改参数观察变化 | 调整batch_size、temperature等 | 1小时 |
| 添加print调试 | 在关键位置添加输出 | 1小时 |
| 性能分析 | 使用torch.profiler分析瓶颈 | 2小时 |

---

## 5. FlashAttention与其他Triton算子集成指南

### 5.1 当前FlashAttention集成位置

**文件**：[nanovllm/layers/attention.py](nanovllm/layers/attention.py)

当前项目使用的是 **FlashAttention** 库：

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class Attention(nn.Module):
    def forward(self, q, k, v):
        # ...
        if context.is_prefill:
            o = flash_attn_varlen_func(...)
        else:
            o = flash_attn_with_kvcache(...)
```

### 5.2 引入自定义FlashAttention的修改位置

如果要引入自己的FlashAttention实现（例如xformers或其他优化版本），修改位置：

#### 方案A：替换FlashAttention库

**修改文件**：`nanovllm/layers/attention.py`

```python
# 原始导入
# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# 替换为自定义实现或xformers
from your_flashattention import custom_flash_attn_varlen, custom_flash_attn_with_cache

class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        # ... KV Cache存储逻辑不变 ...

        if context.is_prefill:
            # 替换为自定义prefill注意力
            o = custom_flash_attn_varlen(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            # 替换为自定义decode注意力
            o = custom_flash_attn_with_cache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        return o
```

#### 方案B：新增自定义Triton算子

**新增文件**：`nanovllm/layers/triton_ops.py`

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_prefill_kernel(
    q_ptr, k_ptr, v_ptr,
    output_ptr,
    ...  # 其他参数
):
    # 自定义prefill kernel实现
    pass

def flash_attn_prefill(q, k, v, ...):
    # Python wrapper
    pass

@triton.jit
def flash_attn_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    output_ptr,
    ...  # 其他参数
):
    # 自定义decode kernel实现
    pass

def flash_attn_decode(q, k_cache, v_cache, ...):
    # Python wrapper
    pass
```

**修改文件**：`nanovllm/layers/attention.py`

```python
from nanovllm.layers.triton_ops import flash_attn_prefill, flash_attn_decode

class Attention(nn.Module):
    def forward(self, q, k, v):
        # ...
        if context.is_prefill:
            o = flash_attn_prefill(q, k, v, ...)
        else:
            o = flash_attn_decode(q, k_cache, v_cache, ...)
        return o
```

### 5.3 KV Cache存储Kernel修改

**当前位置**：`nanovllm/layers/attention.py` 的 `store_kvcache_kernel`

如果要修改KV Cache存储逻辑（例如添加压缩、量化等）：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    # 可以添加新参数，如量化参数
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    # 修改：支持量化
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    # key = quantize(key)  # 添加量化逻辑

    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
    # value = quantize(value)  # 添加量化逻辑

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

### 5.4 其他Triton算子集成位置

| 算子类型 | 新增位置 | 修改位置 |
|----------|----------|----------|
| FlashAttention | `nanovllm/layers/triton_ops.py` | `nanovllm/layers/attention.py` |
| KV Cache压缩 | `nanovllm/layers/triton_ops.py` | `nanovllm/layers/attention.py` |
| 线性层融合 | `nanovllm/layers/triton_ops.py` | `nanovllm/layers/linear.py` |
| 激活函数优化 | `nanovllm/layers/triton_ops.py` | `nanovllm/layers/activation.py` |
| 归一化优化 | `nanovllm/layers/triton_ops.py` | `nanovllm/layers/layernorm.py` |

### 5.5 集成xformers的完整示例

**新增文件**：`nanovllm/layers/xformers_wrapper.py`

```python
import torch
from xformers import ops as xops

class XFormersAttention:
    """xformers注意力包装器"""

    @staticmethod
    def prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table=None):
        # xformers的变长注意力实现
        # 注意：xformers可能需要不同的输入格式，需要进行适配
        pass

    @staticmethod
    def decode(q, k_cache, v_cache, context_lens, block_table):
        # xformers的KV Cache实现
        # 注意：xformers的PagedAttention支持需要验证
        pass
```

**修改文件**：`nanovllm/layers/attention.py`

```python
# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.xformers_wrapper import XFormersAttention

class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        # ... KV Cache存储 ...

        if context.is_prefill:
            o = XFormersAttention.prefill(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                block_table=context.block_tables
            )
        else:
            o = XFormersAttention.decode(
                q, k_cache, v_cache,
                context_lens=context.context_lens,
                block_table=context.block_tables
            )
        return o
```

### 5.6 依赖管理

修改 `pyproject.toml` 添加依赖：

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "triton>=2.0.0",
    "flash-attn>=2.0.0",  # 或替换为 xformers
    # "xformers>=0.0.20",
]
```

---

## 附录

### A. 关键概念速查

| 概念 | 说明 | 相关文件 |
|------|------|----------|
| Prefill | 预填充阶段，计算prompt所有token的KV | [model_runner.py](nanovllm/engine/model_runner.py) |
| Decode | 解码阶段，逐token生成，复用KV Cache | [model_runner.py](nanovllm/engine/model_runner.py) |
| PagedAttention | 分页注意力，支持动态批处理 | [attention.py](nanovllm/layers/attention.py) |
| Prefix Caching | 前缀缓存，共享相同prompt的KV | [block_manager.py](nanovllm/engine/block_manager.py) |
| Tensor Parallelism | 张量并行，模型参数切分到多GPU | [linear.py](nanovllm/layers/linear.py) |
| CUDA Graph | 图优化，减少kernel launch开销 | [model_runner.py](nanovllm/engine/model_runner.py) |
| Gumbel Sampling | Gumbel-Top-k采样，高效多项式采样 | [sampler.py](nanovllm/layers/sampler.py) |
| RoPE | 旋转位置编码 | [rotary_embedding.py](nanovllm/layers/rotary_embedding.py) |
| GQA | 分组查询注意力，减少KV Cache | [qwen3.py](nanovllm/models/qwen3.py) |

### B. 性能优化技巧

1. **增大batch_size** → 提高GPU利用率
2. **使用CUDA Graph** → 减少kernel launch开销
3. **启用Prefix Caching** → 复用相同prompt的KV Cache
4. **调整kvcache_block_size** → 平衡内存碎片和利用率
5. **使用Tensor Parallelism** → 支持更大模型

### C. 调试技巧

1. 设置 `enforce_eager=True` 禁用CUDA Graph便于调试
2. 在 `Attention.forward` 中打印 `context` 各字段
3. 使用 `torch.profiler.profile()` 分析性能瓶颈
4. 在 `BlockManager` 中打印分配/释放操作追踪内存

---

*文档生成日期：2026-02-24*
*项目版本：nano-vllm master*
