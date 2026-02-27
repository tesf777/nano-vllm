# Nano-vLLM 深度技术面试题

> 面向高性能计算方向的实习生招聘，覆盖模型推理与显存优化、底层算子实现、模型结构与适配、调度策略与并发四个维度

---

## 一、模型推理与显存优化

### 问题 1.1：PagedAttention 机制中的显存碎片问题

**问题**：
在大模型推理中，传统 KV Cache 采用连续内存分配方式会导致严重的显存碎片问题。请详细解释：
1. 传统连续分配的显存碎片问题是如何产生的？
2. PagedAttention 的分页机制是如何解决这个问题的？
3. 在 block_manager.py 的 `can_append` 方法中，为何判断条件是 `len(seq) % block_size == 1`？

**考察意图**：
- 验证面试者对显存碎片问题的理解深度
- 考察分页内存管理机制的原理掌握
- 检验对边界条件处理逻辑的分析能力

**详细答案**：

**1. 传统连续分配的显存碎片问题**

传统 KV Cache 为每个序列分配一段连续的显存空间。问题场景：

```
初始状态：序列A需要200 tokens，序列B需要300 tokens
显存布局：[AAAAAA..AA] [BBBBBB...BB]

序列A完成释放后：
显存布局：[空闲200] [BBBBBB...BB]

新序列C需要300 tokens，无法放入200的空闲块
→ 外部碎片：无法利用的小块显存
→ 必须等待序列B释放后才能分配，降低吞吐量
```

显存浪费公式：
```
传统方法浪费率 = (各序列预留未使用显存总和) / 总显存
              ≈ (序列数 × 平均预留量) / 总显存
```

**2. PagedAttention 分页机制解决原理**

PagedAttention 将 KV Cache 划分为固定大小的 Block（默认256 tokens）：

```
显存池：[Block0][Block1][Block2][Block3][Block4][Block5]...

序列A的页表（1000 tokens）：[0][5][2][7] → 非连续分配
序列B的页表（500 tokens）：[1][3]      → 可任意插入
序列C的页表（800 tokens）：[4][6][8]   → 动态扩展
```

优势：
- **消除外部碎片**：Block 作为最小分配单元，无碎片
- **支持动态扩展**：序列增长时只需追加新 Block
- **Prefix Caching**：相同前缀的 Block 可共享

**3. `can_append` 方法边界条件解析**

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

判断逻辑：
- `len(seq) % block_size == 1` → True 表示序列长度恰好跨越 Block 边界
- 例如：`block_size = 256`，序列长度从 255 → 256 → 257
  - 长度 255：位于 Block 内部，不需要新 Block
  - 长度 256：刚好填满 Block，生成下一个 token 前无需分配
  - 长度 257：`257 % 256 == 1`，已跨越边界，下一个 token 需要新 Block

设计意图：
- 只在真正需要新 Block 时检查资源
- 避免不必要的资源检查，降低调度开销

---

### 问题 1.2：Prefix Caching 的哈希计算与冲突处理

**问题**：
Prefix Caching 通过哈希复用相同前缀的 KV Cache，但存在哈希冲突风险。请分析：
1. 为何在 compute_hash 中需要引入 prefix 参数？
2. 如果忽略前缀哈希，会产生什么问题？请举例说明
3. BlockManager.allocate 中的 cache_miss 变量处理逻辑是什么？

**考察意图**：
- 考察对上下文敏感性的理解
- 验证哈希冲突处理机制的设计能力
- 检验对缓存一致性保证的掌握

**详细答案**：

**1. 引入 prefix 参数的原因**

在语言模型中，相同的 token 序列在不同上下文中产生完全不同的语义和注意力结果：

```
上下文 A："The capital of France is [Paris]"
上下文 B："The capital of Germany is [Paris]"（错误，但技术上可能）
```

相同的 Block 内容 `[Paris]` 在不同上下文中不应共享。

哈希计算包含前缀：
```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 前缀哈希参与计算
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

递归哈希链：
```
Block 0: H0 = hash(["The"])
Block 1: H1 = hash(["capital"], prefix=H0)
Block 2: H2 = hash(["of"], prefix=H1)
Block 3: H3 = hash(["France"], prefix=H2)
```

确保相同内容的 Block 在不同路径下有不同的哈希值。

**2. 忽略前缀哈希的问题**

错误示例：

```
请求1：Prompt = "Write a story about a [cat]"
请求2：Prompt = "Write a poem about a [cat]"
```

如果不考虑前缀哈希，Block `[cat]` 会被错误共享：

```
请求1的注意力：
  Q: "story"  K: "cat"  → 语义：故事中的猫

请求2的注意力：
  Q: "poem"   K: "cat"  → 语义：诗歌中的猫
```

共享后错误：两个请求都会使用相同的 KV Cache，导致注意力计算错误。

**3. cache_miss 变量处理逻辑**

```python
def allocate(self, seq: Sequence):
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)

        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # 当前及后续 Block 都需要新分配
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 命中缓存，复用 Block
            seq.num_cached_tokens += self.block_size
            block.ref_count += 1
```

设计意图：
- **保证块链一致性**：一旦某 Block 未命中，后续 Block 即使哈希相同也不能复用
- **防止部分复用错误**：前缀不同，后续相同内容的 Block 本质上不同
- **简化逻辑**：使用单一 flag 控制后续分配，提高效率

---

### 问题 1.3：KV Cache 内存浪费的量化分析与优化

**问题**：
在 nano-vllm 中，KV Cache 占据推理显存的绝大部分。请分析：
1. 给定模型配置，计算 KV Cache 的显存占用公式是什么？
2. 假设一个 7B 模型（32层、32头、128维），block_size=256，1000 个 blocks，计算 KV Cache 占用（fp16）
3. 提出至少三种减少 KV Cache 显存浪费的方案

**考察意图**：
- 验证显存预算的计算能力
- 考察对模型配置与显存关系的理解
- 检验显存优化方案的设计能力

**详细答案**：

**1. KV Cache 显存占用公式**

```python
# 单个 Block 的字节数
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size

# 其中：
# 2: K 和 V 两个张量
# num_layers: Transformer 层数
# block_size: 每个 Block 的 token 数
# num_kv_heads: KV 头数（GQA/MQA 可能小于 num_heads）
# head_dim: 每个头的维度
# dtype_size: 数据类型字节数（fp16=2, fp32=4）
```

完整 KV Cache 张量形状：
```python
kv_cache = torch.zeros(
    2,                          # K/V
    num_layers,                 # 层数
    num_blocks,                 # Block 数量
    block_size,                 # 每个 Block 的 token 数
    num_kv_heads,               # KV 头数
    head_dim                    # 头维度
)
```

**2. 计算示例**

```python
# 配置
num_layers = 32
num_kv_heads = 32  # 假设无 GQA
head_dim = 128
block_size = 256
num_blocks = 1000
dtype_size = 2  # fp16

# 单 Block 字节数
block_bytes = 2 * 32 * 256 * 32 * 128 * 2
            = 1,048,576 bytes ≈ 1 MB

# 总 KV Cache 字节数
total_bytes = block_bytes * 1000
            = 1,048,576,000 bytes ≈ 1 GB

# 显存布局
kv_cache shape: (2, 32, 1000, 256, 32, 128)
             = 524,288,000 elements
             = 524M elements × 2 bytes = 1,048,576,000 bytes ≈ 1 GB
```

**3. 减少 KV Cache 显存浪费的方案**

**方案 1：KV Cache 量化**
- 将 fp16 KV Cache 量化为 int8/int4
- 节省显存：50% (int8) 或 75% (int4)
- 挑战：量化后的注意力计算精度损失

```python
# 量化示例
kv_cache_int8 = (kv_cache_fp16 / max_abs).clamp(-128, 127).char()
```

**方案 2：动态调整 block_size**
- 短序列使用较小 block_size，减少碎片
- 长序列使用较大 block_size，减少元数据开销
- 需要支持多种 block_size 的混合分配

**方案 3：激进式 Prefix Caching**
- 启用跨请求的前缀共享
- 设计高效的全局哈希索引
- 挑战：哈希查找开销与收益的平衡

**方案 4：KV Cache 压缩**
- 对稀疏注意力模式使用压缩存储
- 基于注意力权重的稀疏化策略
- 需要支持解压缩的注意力计算

---

## 二、底层算子实现

### 问题 2.1：FlashAttention 的 IO 复杂度优化原理

**问题**：
FlashAttention 通过 Tiling 技术将 O(N²) 的注意力计算优化为 IO 优化的实现。请详细解释：
1. 标准 Attention 的 IO 复杂度是多少？瓶颈在哪里？
2. FlashAttention 如何通过分块计算减少 HBM 访问次数？
3. 在 attention.py 中，为何 prefill 和 decode 阶段调用不同的 FlashAttention 函数？

**考察意图**：
- 验证对 FlashAttention 核心优化原理的理解
- 考察对计算与 IO 优化的区分能力
- 检验对不同阶段注意力计算差异的掌握

**详细答案**：

**1. 标准 Attention 的 IO 复杂度**

标准 Attention 计算公式：
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

IO 复杂度分析：
- **计算复杂度**：O(N²d)，其中 N 是序列长度，d 是头维度
- **HBM 访问次数**：
  - Q, K, V 各加载一次：3 × N × d
  - S = QK^T 每次计算都需从 HBM 加载：N² 次
  - P = softmax(S) 同样需 HBM 访问：N² 次
  - O = P V 访问：N² 次
  - **总 HBM 访问量**：O(N²)，远高于显存带宽

瓶颈：注意力矩阵 S (N×N) 太大，无法放入 SRAM，导致反复 HBM 访问。

**2. FlashAttention 分块优化原理**

FlashAttention 使用分块计算，将大矩阵分解为适合 SRAM 的子块：

```python
# SRAM 容量受限（约 192KB）
# 将 Q, K, V 分为块：Q_bc, K_br, V_br

O = torch.zeros_like(Q)  # 输出累加器

for b in range(B):  # 外层循环：块索引
    # 从 HBM 加载当前块到 SRAM
    Q_bc = Q[b * bc : (b+1) * bc]  # bc × d

    for r in range(R):  # 内层循环：块索引
        # 从 HBM 加载 K 和 V 块
        K_br = K[r * br : (r+1) * br]  # br × d
        V_br = V[r * br : (r+1) * br]  # br × d

        # 在 SRAM 中计算局部注意力
        S_bc = Q_bc @ K_br.T  # bc × br
        P_bc = softmax(S_bc, dim=-1)  # bc × br
        O_bc = P_bc @ V_br  # bc × d

        # 累加到输出
        O[b * bc : (b+1) * bc] += O_bc
```

IO 复杂度优化：
- Q 只加载 B 次（原本 N² 次）
- K, V 各加载 R 次（原本 N² 次）
- 总 HBM 访问量：O(N²d / M)，其中 M 是 SRAM 大小
- 实际加速比：约 2-4x（取决于硬件配置）

**3. prefill 和 decode 阶段的差异**

```python
# attention.py
if context.is_prefill:
    o = flash_attn_varlen_func(
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
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True
    )
```

差异分析：

| 维度 | Prefill 阶段 | Decode 阶段 |
|------|-------------|-------------|
| **序列长度** | 长（整个 prompt） | 短（1 个新 token） |
| **输入 K/V** | 当次生成的 K, V | 历史累积的 K/V Cache |
| **计算模式** | 变长序列批处理 | 单 token + KV Cache |
| **优化目标** | 吞吐量最大化 | 延迟最小化 |
| **函数调用** | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |

设计原因：
- Prefill 需要处理变长序列（不同 prompt 长度不同）
- Decode 的 K/V 已经缓存在预分配的显存中，直接读取
- 不同的内存访问模式需要不同的优化策略

---

### 问题 2.2：Triton Kernel 并行化策略与 store_kvcache 优化

**问题**：
在 nano-vllm 中，store_kvcache_kernel 使用 Triton 实现 KV Cache 存储。请分析：
1. 该 Kernel 的并行化策略是什么？每个 thread 处理什么数据？
2. `if slot == -1: return` 这行代码的作用是什么？在什么场景下触发？
3. 如何优化该 Kernel 以提高写入性能？

**考察意图**：
- 验证对 Triton 并行编程的理解
- 考察对 Prefix Caching 场景的掌握
- 检验 Kernel 性能优化的设计能力

**详细答案**：

**1. Kernel 并行化策略分析**

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr, D: tl.constexpr
):
    idx = tl.program_id(0)  # 每个 program_id 对应一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    # 每个线程处理一个 token 的所有维度
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))

    # 存储到 KV Cache 的对应 slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

并行化策略：
- **并行维度**：Token 维度，每个 program_id 处理一个 token
- **数据划分**：每个 thread 处理单个 token 的全部 head_dim 维度
- **内存访问**：
  - 读取：从 key_ptr/value_ptr 读取当前 token 的 K/V
  - 写入：根据 slot_mapping 写入到 KV Cache 的对应位置

Grid 配置：
```python
grid = lambda META: (num_tokens,)
# 总共启动 num_tokens 个线程块
```

**2. `if slot == -1: return` 的作用**

触发场景：**Prefix Caching 命中**

```python
# BlockManager.allocate 中
if not cache_miss:
    # 缓存命中，不需要重新计算该 Block 的 KV
    seq.num_cached_tokens += self.block_size
    # slot_mapping 中对应的 slot 设为 -1
```

作用：
- **跳过重复计算**：Prefix Cache 已有的 Block 不需要重新存储 KV
- **避免无效写入**：保护已缓存的 KV 不被覆盖
- **性能优化**：减少不必要的内存写入操作

数据流：
```
Prompt: ["Hello", "world", "how", "are", "you"]

请求 1: "Hello world are you"
  Block 0: ["Hello", "world", "how", "are"] → 计算并存储 KV
  Block 1: ["you"] → 计算并存储 KV
  slot_mapping: [0, 1, 2, 3, 4]

请求 2: "Hello world how are you there"
  Block 0: 命中缓存，slot_mapping = [-1, -1, -1, -1, 10, 11, 12]
  只需计算新 token "there" 的 KV
```

**3. Kernel 优化方案**

**优化 1：向量化加载/存储**
```python
# 使用更大粒度的向量化
BLOCK_D = 64  # 根据硬件调整
vec_key = tl.load(key_ptr + idx * key_stride + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < D)
```

**优化 2：合并 K/V 写入**
```python
# 假设 K/V 在内存中交错存储
kv_ptr = key_ptr + idx * (key_stride + value_stride)
kv = tl.load(kv_ptr + tl.arange(0, 2 * D))
tl.store(kv_cache_ptr + cache_offsets, kv[:D])    # K
tl.store(v_cache_ptr + cache_offsets, kv[D:])     # V
```

**优化 3：利用 Shared Memory**
```python
# 多线程协同写入同一 Block
@triton.jit
def store_kvcache_kernel_block(..., BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    token_idx = tl.program_id(1)  # Block 内的 token 偏移
    # ... 使用 shared memory 缓冲，减少 HBM 访问
```

---

### 问题 2.3：采样算子的 Gumbel-Top-k 原理与性能调优

**问题**：
在 sampler.py 中使用了 Gumbel-Top-k 采样实现多项式采样。请分析：
1. 为何 Gumbel-Top-k 等价于多项式采样？数学原理是什么？
2. `@torch.compile` 装饰器的作用是什么？在采样场景下带来什么性能提升？
3. 如何优化采样算子的性能以支持更大 batch size？

**考察意图**：
- 验证对概率采样算法的理解
- 考察对 torch.compile 优化机制的掌握
- 检验采样性能优化的设计能力

**详细答案**：

**1. Gumbel-Top-k 数学原理**

多项式采样：从多项分布 `p = [p₁, p₂, ..., pₙ]` 中采样一个索引。

标准方法（累积分布）：
```python
# 计算累积分布
cdf = torch.cumsum(probs, dim=-1)
# 生成均匀随机数
u = torch.rand((batch_size,))
# 二分查找
sampled_index = torch.searchsorted(cdf, u)
```

问题：需要计算完整 CDF 和二分查找，复杂度高。

Gumbel-Top-k 原理：

根据 Gumbel-Max 定理：
```
设 G₁, G₂, ..., Gₙ 是独立同分布的 Gumbel 随机变量
则 argmaxᵢ(Gᵢ + log(pᵢ)) 是从多项分布 p 采样的结果

Gumbel 随机变量生成：G = -log(-log(u))，其中 u ~ Uniform(0,1)
```

代码实现：
```python
# Gumbel-Top-k 采样
gumbel = -torch.log(-torch.log(torch.empty_like(probs).uniform_(0, 1) + 1e-10) + 1e-10)
# 等价于：gumbel = -torch.log(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10))

# 加对数概率后取 argmax
sampled_index = (torch.log(probs) + gumbel).argmax(dim=-1)
```

简化实现（nano-vllm 中的版本）：
```python
# Gumbel-Top-k 的等价形式
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

数学推导：
```
Xᵢ / Eᵢ ~ Gamma(1, pᵢ)  （Eᵢ ~ Exp(1)）
argmaxᵢ(Xᵢ / Eᵢ) = argmaxᵢ(log(Xᵢ) - log(Eᵢ))
                   = argmaxᵢ(Gᵢ + log(pᵢ))  （Gᵢ = -log(Eᵢ) ~ Gumbel(0,1)）
```

优势：
- 无需计算 CDF
- 一次 argmax 操作完成采样
- 并行度高，适合 GPU 加速

**2. `@torch.compile` 装饰器作用**

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: Tensor, temperatures: Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

torch.compile 优化机制：
- **图融合**：将多个算子融合为单个 kernel，减少内存读写
- **内存规划**：优化中间张量的内存分配和释放
- **自动并行化**：识别可并行的操作并调度

采样场景下的性能提升：
- 减少 softmax、div、exponential 等操作的 kernel launch 开销
- 优化中间张量的内存访问模式
- 实测加速比：约 1.5-2x（取决于 batch size）

**3. 采样算子性能优化方案**

**优化 1：Top-k 预筛选**
```python
# 对于大词表，先筛选 top-k 后再采样
topk_probs, topk_indices = torch.topk(probs, k=100)
sampled_topk = (torch.log(topk_probs) + gumbel[:k]).argmax(dim=-1)
sample_tokens = topk_indices.gather(1, sampled_topk.unsqueeze(1))
```

**优化 2：批量化 Gumbel 生成**
```python
# 预生成一批 Gumbel 随机数，减少频繁调用
gumbel_pool = -torch.log(-torch.log(torch.rand((batch_size, vocab_size, pool_size))))
# 采样时从池中取用
```

**优化 3：温度参数融合**
```python
# 将温度参数融合到 logits 中，避免除法
scaled_logits = logits * (1.0 / temperatures.unsqueeze(dim=1))
probs = torch.softmax(scaled_logits, dim=-1)
```

**优化 4：使用 FasterTransformer 的采样算子**
```python
# 使用高度优化的 CUDA kernel
from fastertransformer import multinomial_sampling
sample_tokens = multinomial_sampling(logits, temperatures)
```

---

## 三、模型结构与适配

### 问题 3.1：Transformer 推理中的张量布局与内存连续性

**问题**：
在 Transformer 推理框架中，张量的内存布局对性能有重要影响。请分析：
1. nano-vllm 中 Attention 的输入张量 Q, K, V 的布局是什么？为何这样设计？
2. FlashAttention 对张量布局有什么要求？如何进行布局转换？
3. 在 decode 阶段，KV Cache 的张量布局如何支持高效的注意力计算？

**考察意图**：
- 验证对张量布局设计的理解
- 考察对内存连续性优化策略的掌握
- 检验对推理阶段张量操作差异的分析能力

**详细答案**：

**1. Attention 输入张量布局分析**

在 nano-vllm 的 Qwen3 模型中，张量布局：

```python
# Q, K, V 的原始输出（来自 QKVParallelLinear）
# Shape: (num_tokens, num_heads, head_dim)
# Layout: token-major，每行是一个 token 的所有头的表示

# 例如：batch_size=2, seq_len=[3, 4], num_heads=8, head_dim=128
# Q_concat shape: (7, 8, 128)
#   - 7 = 3 + 4 (所有序列的 token 总数)
#   - 8 = num_heads
#   - 128 = head_dim
```

设计原因：
- **连续批处理支持**：不同序列的 token 可以连续拼接
- **Prefill 高效计算**：FlashAttention 可直接处理变长序列
- **内存访问局部性**：同一 token 的所有头数据连续存储

**2. FlashAttention 布局要求与转换**

FlashAttention 要求的布局：
```python
# flash_attn_varlen_func 要求
q shape: (total_q_tokens, num_heads, head_dim)
k shape: (total_k_tokens, num_kv_heads, head_dim)
v shape: (total_k_tokens, num_kv_heads, head_dim)

# 对于 GQA/MQA，num_kv_heads <= num_heads
# 需要支持 KV 头的重复广播
```

布局转换（在 Linear 层中完成）：
```python
# QKVParallelLinear 输出
# Shape: (num_tokens, num_q_heads + 2 * num_kv_heads, head_dim)

# 拆分为 Q, K, V
q = qkv[:, :num_q_heads, :]
k = qkv[:, num_q_heads:num_q_heads + num_kv_heads, :]
v = qkv[:, num_q_heads + num_kv_heads:, :]

# 如果是 GQA（num_kv_heads < num_q_heads），需要扩展 K, V
# 方法 1：重复（简单但浪费）
k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

# 方法 2：在注意力计算时广播（FlashAttention 内部处理）
```

**3. KV Cache 布局设计**

KV Cache 的张量形状（来自 ModelRunner.allocate_kv_cache）：
```python
# KV Cache 预分配
kv_cache = torch.zeros(
    2,                          # K/V
    num_layers,                 # 层数
    num_blocks,                 # Block 数量
    block_size,                 # 每个 Block 的 token 数
    num_kv_heads,               # KV 头数
    head_dim                    # 头维度
)
# Shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

Decode 阶段的访问模式：
```python
# flash_attn_with_kvcache 读取 KV Cache
# k_cache / v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
# block_table: (num_seqs, max_num_blocks)

# 通过 block_table 定位每个序列的 KV
# 例如：seq1 的页表 [5, 12, 3] 表示
#   - tokens 0-255: block 5
#   - tokens 256-511: block 12
#   - tokens 512-767: block 3
```

布局优势：
- **Block 级局部性**：同一 Block 的数据连续存储
- **跨层共享**：所有层共享相同的 block_table 逻辑
- **GQA 支持**：KV 头独立存储，节省显存

---

### 问题 3.2：非连续内存处理与 PagedAttention 的实现

**问题**：
PagedAttention 的核心挑战在于处理非连续内存。请深入分析：
1. FlashAttention 如何通过 block_table 参数支持非连续 KV Cache？
2. 在 attention.py 中，context.block_tables 是如何构建的？包含什么信息？
3. 如果要实现支持 MoE 模型的 PagedAttention，需要哪些修改？

**考察意图**：
- 验证对 PagedAttention 实现机制的深入理解
- 考察对页表数据结构设计的掌握
- 检验对新模型架构适配能力

**详细答案**：

**1. FlashAttention 的 block_table 支持**

FlashAttention 通过 block_table 实现非连续 KV Cache 访问：

```python
# block_table 形状：[num_seqs, max_num_blocks]
# 含义：每个序列的逻辑 block 到物理 block 的映射

# 例如：
# block_table = [
#     [5, 12, 3, 8],   # 序列 0 的页表
#     [7, 1, 4],       # 序列 1 的页表
# ]
# 表示序列 0 的第 0 个逻辑块对应物理块 5，第 1 个对应物理块 12，依此类推
```

FlashAttention 内部处理流程：
```python
# 伪代码
for seq_idx in range(num_seqs):
    seq_len = context_lens[seq_idx]
    num_blocks = (seq_len + block_size - 1) // block_size

    for token_idx in range(seq_len):
        block_idx = token_idx // block_size
        offset = token_idx % block_size

        # 通过 block_table 定位物理块
        physical_block = block_tables[seq_idx, block_idx]

        # 计算 KV Cache 的物理位置
        k_pos = (physical_block * block_size + offset) * num_kv_heads * head_dim
        v_pos = k_pos

        # 加载该 token 的 K/V
        k_token = k_cache[k_pos:k_pos + num_kv_heads * head_dim]
        v_token = v_cache[v_pos:v_pos + num_kv_heads * head_dim]
```

**2. context.block_tables 构建过程**

在 ModelRunner.prepare_block_tables 中构建：

```python
def prepare_block_tables(self, seqs):
    # 构建页表张量
    # 形状：(num_seqs, max_num_blocks)

    block_tables = []
    for seq in seqs:
        # 序列的页表
        table = seq.block_table  # 逻辑 block 到物理 block 的映射

        # 填充到 max_num_blocks（用 -1 填充未使用的部分）
        padded_table = table + [-1] * (max_num_blocks - len(table))
        block_tables.append(padded_table)

    # 转换为张量
    return torch.tensor(block_tables, dtype=torch.long, device='cuda')
```

示例：
```python
# 假设有两个序列
seq1 = Sequence(token_ids=[0, 1, 2, ..., 257])  # 258 个 tokens
seq2 = Sequence(token_ids=[10, 11, ..., 200])  # 191 个 tokens

# block_size = 256
# seq1 需要 2 个 blocks: block_table = [5, 12]
# seq2 需要 1 个 block: block_table = [7]

# block_tables（max_num_blocks=2）:
# [
#     [5, 12],   # seq1
#     [7, -1]    # seq2（只有 1 个 block，用 -1 填充）
# ]
```

**3. MoE 模型的 PagedAttention 适配挑战**

MoE（Mixture of Experts）引入的新问题：
```python
# MoE 结构
class MoELayer(nn.Module):
    def forward(self, x):
        # 路由：为每个 token 选择专家
        expert_indices = router(x)  # (num_tokens, top_k)

        # 每个 token 的 K/V 分发到不同专家
        # 问题：如何为不同专家的 K/V 建立统一的 KV Cache？
```

适配方案：

**方案 1：专家级独立 KV Cache**
```python
# 为每个专家维护独立的 KV Cache
kv_cache_per_expert = [
    torch.zeros(...),  # expert 0 的 KV Cache
    torch.zeros(...),  # expert 1 的 KV Cache
    ...
]

# 每个 token 的 block_table 需要同时记录专家信息
block_table_with_expert = [
    [(expert_id, block_id), ...],  # 序列 0 的页表
    ...
]

# 问题：内存开销大，专家间难以共享
```

**方案 2：共享 KV Cache + 专家路由记录**
```python
# 共享 KV Cache，但记录每个 token 的路由信息
kv_cache = torch.zeros(...)  # 共享缓存
expert_routing = torch.zeros((num_tokens, num_layers), dtype=torch.long)

# 注意力计算时，根据路由信息选择对应的专家权重
# 问题：注意力计算复杂，需要额外的专家权重加载
```

**方案 3：Prefix 共享 + Expert 分离**
```python
# 共享的 Prefix Cache（所有专家相同）
shared_prefix_cache = torch.zeros(...)

# 专家特定的 Cache
expert_suffix_cache = [torch.zeros(...) for _ in range(num_experts)]

# 混合注意力计算
# 问题：实现复杂，需要修改 FlashAttention 内核
```

**关键修改点**：
1. `block_table` 扩展：需要记录 `(expert_id, block_id)` 对
2. KV Cache 形状调整：增加专家维度
3. Attention 修改：根据专家路由选择对应的 KV Cache
4. FlashAttention 内核：支持专家级别的注意力计算

---

### 问题 3.3：GQA（Grouped Query Attention）的推理优化

**问题**：
Qwen3 模型使用了 GQA 来减少 KV Cache 显存占用。请分析：
1. GQA 与标准多头注意力（MHA）的区别是什么？显存节省比例如何计算？
2. 在 nano-vllm 的 linear.py 中，如何实现 GQA 的张量并行？
3. GQA 如何影响 FlashAttention 的性能？如何进一步优化？

**考察意图**：
- 验证对 GQA 原理和优势的理解
- 考察对 GQA 张量并行实现的掌握
- 检验对 GQA 性能优化的设计能力

**详细答案**：

**1. GQA 与 MHA 的区别及显存节省**

**标准多头注意力（MHA）**：
```python
# Q, K, V 的头数相同
num_heads = 32
num_kv_heads = 32  # 与 num_heads 相同

# 每个 Q 头有独立的 K 和 V 头
# KV Cache 形状：[num_layers, num_blocks, block_size, 32, head_dim]
```

**分组查询注意力（GQA）**：
```python
num_heads = 32        # Query 头数
num_kv_heads = 8      # KV 头数（32/8 = 4 个 Q 头共享 1 个 KV 头）

# 每 4 个 Q 头共享 1 个 K/V 头
# KV Cache 形状：[num_layers, num_blocks, block_size, 8, head_dim]
```

显存节省比例：
```
KV Cache 显存 ∝ num_kv_heads

GQA 节省比例 = 1 - (num_kv_heads / num_heads)
              = 1 - (8 / 32) = 75% 节省

完整模型 KV Cache（32 层，1000 blocks，256 block_size，128 head_dim，fp16）：
MHA: 2 × 32 × 1000 × 256 × 32 × 128 × 2 = 1,048,576,000 bytes ≈ 1 GB
GQA: 2 × 32 × 1000 × 256 × 8  × 128 × 2 = 262,144,000 bytes ≈ 250 MB
节省：750 MB
```

注意力计算（GQA）：
```python
# Q 形状：[num_tokens, num_heads, head_dim]
# K, V 形状：[num_tokens, num_kv_heads, head_dim]

# 需要扩展 K, V 以匹配 Q 的头数
group_size = num_heads // num_kv_heads  # 4

# 方法 1：重复（显式扩展）
k_expanded = k.repeat_interleave(group_size, dim=1)  # [num_tokens, num_heads, head_dim]
v_expanded = v.repeat_interleave(group_size, dim=1)

# 方法 2：广播（FlashAttention 内部处理）
# 无需显式扩展，计算时自动广播
```

**2. GQA 的张量并行实现**

在 QKVParallelLinear 中的实现：

```python
class QKVParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_kv_heads, world_size):
        self.num_heads_per_rank = num_heads // world_size
        self.num_kv_heads_per_rank = num_kv_heads // world_size

        # 输出特征数：Q 头 + K 头 + V 头
        total_out_features = self.num_heads_per_rank + 2 * self.num_kv_heads_per_rank
        self.weight = nn.Parameter(torch.empty(total_out_features, in_features))

    def forward(self, x):
        # 计算分片后的 Q, K, V
        output = F.linear(x, self.weight)

        # 拆分
        q = output[:, :self.num_heads_per_rank, :]
        k = output[:, self.num_heads_per_rank:self.num_heads_per_rank + self.num_kv_heads_per_rank, :]
        v = output[:, self.num_heads_per_rank + self.num_kv_heads_per_rank:, :]

        return q, k, v
```

并行策略：
```
MHA（num_heads=32, num_kv_heads=32, world_size=4）:
  GPU 0: Q[0:7], K[0:7], V[0:7]
  GPU 1: Q[8:15], K[8:15], V[8:15]
  GPU 2: Q[16:23], K[16:23], V[16:23]
  GPU 3: Q[24:31], K[24:31], V[24:31]

GQA（num_heads=32, num_kv_heads=8, world_size=4）:
  GPU 0: Q[0:7], K[0:1], V[0:1]    (4 个 Q 头共享 1 个 KV 头)
  GPU 1: Q[8:15], K[2:3], V[2:3]   (4 个 Q 头共享 1 个 KV 头)
  GPU 2: Q[16:23], K[4:5], V[4:5]  (4 个 Q 头共享 1 个 KV 头)
  GPU 3: Q[24:31], K[6:7], V[6:7]  (4 个 Q 头共享 1 个 KV 头)
```

**3. GQA 对 FlashAttention 性能的影响及优化**

**性能影响**：
- **优势**：KV Cache 显存减少，可容纳更多序列
- **劣势**：KV 头重复可能导致显存访问模式不够连续

**优化方案 1：KV 头布局优化**
```python
# 调整 KV Cache 的内存布局，使同一 KV 头的多个 Q 头访问更连续
# 原始布局：[block_size, num_kv_heads, head_dim]
# 优化布局：[block_size, head_dim, num_kv_heads]（head 维度优先）
```

**优化方案 2：向量化读取**
```python
# 一次读取多个 KV 头的数据
vec_k = tl.load(k_cache_ptr + block_offset + tl.arange(0, num_kv_heads * head_dim))
```

**优化方案 3：减少 KV 头重复**
```python
# 如果 num_kv_heads 整除 num_heads，使用向量化操作
# 避免逐元素重复，使用更高效的张量操作
k_expanded = k.view(num_tokens, num_kv_heads, group_size, head_dim)
                .expand(-1, -1, group_size, -1)
                .reshape(num_tokens, num_heads, head_dim)
```

---

## 四、调度策略与并发

### 问题 4.1：Continuous Batching 的实现逻辑与挑战

**问题**：
Continuous Batching（也称为迭代级批处理）是大模型推理框架的核心调度策略。请分析：
1. Continuous Batching 与 Static Batching 的本质区别是什么？
2. 在 scheduler.py 中，Prefill 和 Decode 阶段是如何分离调度的？
3. 为何要在 Decode 阶段进行抢占（preemption）？抢占的具体逻辑是什么？

**考察意图**：
- 验证对 Continuous Batching 原理的深入理解
- 考察对调度算法设计的掌握
- 检验对抢占机制实现的分析能力

**详细答案**：

**1. Continuous Batching vs Static Batching**

**Static Batching（静态批处理）**：
```
# 固定 batch size，所有请求必须同时完成
batch = [req1, req2, req3, req4]  # 假设 batch_size=4

# 问题：不同请求生成长度差异大
req1: 需生成 5 tokens
req2: 需生成 50 tokens
req3: 需生成 10 tokens
req4: 需生成 100 tokens

# 必须等待最长请求完成（100 tokens），短请求的资源被浪费
# GPU 利用率 = (5+50+10+100) / (4*100) = 41.25%
```

**Continuous Batching（连续批处理）**：
```
# 动态调整 batch，完成的请求立即移除，新请求立即加入
t=0: batch = [req1, req2, req3, req4]
t=5: req1 完成，batch = [req2, req3, req4, req5]
t=10: req3 完成，batch = [req2, req4, req5, req6]
t=50: req2 完成，batch = [req4, req5, req6, req7]
...

# 优势：
# 1. 长请求不会阻塞短请求
# 2. 始终保持高 GPU 利用率
# 3. 吞吐量显著提升
```

本质区别：
| 维度 | Static Batching | Continuous Batching |
|------|----------------|---------------------|
| **Batch 成员** | 固定，直到全部完成 | 动态，随时进出 |
| **调度粒度** | 请求级 | 迭代级（每次生成后重新调度） |
| **资源利用** | 受限于最长请求 | 接近理论最优 |
| **实现复杂度** | 简单 | 复杂（需要动态调度和抢占） |

**2. Prefill 和 Decode 分离调度**

在 scheduler.py 中的实现：

```python
def schedule(self):
    scheduled_seqs = []

    # ===== Prefill 阶段 =====
    num_batched_tokens = 0
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查是否有足够资源
        if num_batched_tokens + (len(seq) - seq.num_cached_tokens) > self.max_num_batched_tokens:
            break
        if not self.block_manager.can_allocate(seq):
            break

        # 分配 Block
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True  # 返回 prefill=True

    # ===== Decode 阶段 =====
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查是否可以追加新 token
        while not self.block_manager.can_append(seq):
            if self.running:
                # 抢占其他运行中的序列
                self.preempt(self.running.pop())
            else:
                # 无法分配，回到等待队列
                self.preempt(seq)
                break
        else:
            # 追加新 token
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

    return scheduled_seqs, False  # 返回 prefill=False
```

分离原因：
1. **计算模式不同**：
   - Prefill：计算量大（整个 prompt），适合批处理多个请求
   - Decode：计算量小（1 token），适合高并发处理

2. **资源分配不同**：
   - Prefill：需要分配完整的 KV Cache Block
   - Decode：只需检查是否需要新的 Block

3. **延迟要求不同**：
   - Prefill：优先级较低（首字延迟）
   - Decode：优先级较高（生成速度）

**3. Decode 阶段的抢占逻辑**

抢占条件：
```python
while not self.block_manager.can_append(seq):
    if self.running:
        # 抢占其他运行中的序列
        self.preempt(self.running.pop())
    else:
        # 无法分配，回到等待队列
        self.preempt(seq)
        break
```

抢占函数：
```python
def preempt(self, seq):
    """
    抢占序列：
    1. 将序列状态设为 WAITING
    2. 释放已分配的 KV Cache Block
    3. 将序列放回等待队列
    """
    # 1. 更新状态
    seq.status = SequenceStatus.WAITING

    # 2. 释放 Block
    self.block_manager.deallocate(seq)

    # 3. 放回等待队列
    self.waiting.appendleft(seq)
```

抢占场景示例：
```
运行队列：[seq1(需要新block), seq2(长序列), seq3(短序列)]

尝试调度 seq1：
  - seq1 需要新 Block，但 free 池为空
  - 抢占 seq3（最后一个，通常是短序列，成本低）
  - seq3 的 Block 被释放，free 池有资源
  - seq1 获得新 Block，继续运行

结果：
  运行队列：[seq1, seq2]
  等待队列：[seq3]
```

抢占策略：
- **优先抢占短序列**：释放 Block 快，成本低
- **优先抢占低优先级请求**：保证重要请求不被中断
- **避免频繁抢占**：减少状态切换开销

---

### 问题 4.2：迭代级调度与请求级调度的区别

**问题**：
调度粒度是推理框架设计的关键决策。请分析：
1. 迭代级调度和请求级调度的本质区别是什么？
2. 迭代级调度在实现上需要哪些额外支持？如何保证状态一致性？
3. 迭代级调度会带来哪些额外开销？如何缓解？

**考察意图**：
- 验证对不同调度粒度的理解
- 考察对迭代级调度实现挑战的掌握
- 检验对调度开销优化的设计能力

**详细答案**：

**1. 迭代级调度 vs 请求级调度**

**请求级调度**：
```
# 每个请求从开始到结束都在同一个 batch 中
for request in batch:
    # Prefill 阶段
    for token in request.prompt:
        compute_kv(token)

    # Decode 阶段
    while not request.finished:
        new_token = generate_one_token(request)
        request.append(new_token)

# 问题：
# 1. 长请求阻塞短请求
# 2. GPU 资源利用不均衡
# 3. 延迟差异大
```

**迭代级调度**：
```
# 每次迭代（生成一个 token）后重新调度
while not all_finished:
    # 选择当前要处理的请求
    scheduled = scheduler.schedule()

    # Prefill 或 Decode 一轮
    for seq in scheduled:
        if is_prefill:
            process_prefill(seq)
        else:
            new_token = generate_one_token(seq)
            seq.append(new_token)

    # 更新状态，重新调度
    scheduler.postprocess(scheduled, new_tokens)
```

本质区别：

| 维度 | 请求级调度 | 迭代级调度 |
|------|-----------|-----------|
| **调度频率** | 每个请求开始时 | 每次迭代后 |
| **Batch 成员** | 固定不变 | 动态变化 |
| **并发度** | 受限于 batch size | 受限于 max_num_seqs |
| **实现复杂度** | 简单 | 复杂（需要状态管理） |
| **吞吐量** | 受长请求限制 | 更接近理论最优 |

**2. 迭代级调度的额外支持**

**（1）状态管理**
```python
# Sequence 类需要维护完整状态
class Sequence:
    def __init__(self, prompt_tokens, sampling_params):
        self.token_ids = prompt_tokens.copy()  # 完整 token 序列
        self.status = SequenceStatus.WAITING   # 当前状态
        self.num_cached_tokens = 0              # 已缓存的 token 数
        self.block_table = []                   # KV Cache 页表
        self.last_token = prompt_tokens[-1]     # 最后一个 token

    def is_finished(self):
        """检查是否完成"""
        return self.status == SequenceStatus.FINISHED

    def num_completion_tokens(self):
        """已生成的 token 数"""
        return len(self.token_ids) - self.num_prompt_tokens
```

**（2）KV Cache 持久化**
```python
# KV Cache 跨迭代持久保存，不会在每次迭代后释放
# BlockManager 负责管理
class BlockManager:
    def allocate(self, seq):
        # 首次调度时分配 Block
        for i in range(seq.num_blocks):
            block_id = self.free_block_ids[0]
            seq.block_table.append(block_id)

    def may_append(self, seq):
        # 每次生成新 token 后更新 Block
        if len(seq) % self.block_size == 1:
            # 需要新 Block
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
```

**（3）状态一致性保证**
```python
# 调度器需要保证状态转换的正确性
def postprocess(self, seqs, token_ids):
    """处理生成结果，更新状态"""
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)

        # 检查结束条件
        if token_id == self.eos and not seq.ignore_eos:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
        elif seq.num_completion_tokens >= seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
        else:
            # 继续运行
            self.running.append(seq)
```

**（4）抢占与恢复**
```python
def preempt(self, seq):
    """抢占序列，保存状态"""
    seq.status = SequenceStatus.WAITING
    # 保留 seq.block_table 和 seq.token_ids
    # 不释放已缓存的 KV Cache（可以后续复用）
    # 释放未使用的 Block

def resume(self, seq):
    """恢复序列，继续运行"""
    seq.status = SequenceStatus.RUNNING
    # 检查 seq.block_table 是否需要更新
    # 继续从 seq.last_token 开始生成
```

**3. 额外开销及缓解**

**开销 1：调度开销**
```
每次迭代都需要：
- 检查所有序列的状态
- 选择要处理的序列
- 更新队列

缓解方案：
- 使用高效的数据结构（deque, set）
- 批量处理状态更新
- 调度器计算与模型计算并行（异步调度）
```

**开销 2：状态切换开销**
```
抢占和恢复需要：
- 更新序列状态
- 释放/分配 Block
- 修改队列

缓解方案：
- 减少抢占频率（预测序列长度）
- 优化 Block 分配策略（预分配）
- 使用更快的队列操作
```

**开销 3：内存管理开销**
```
跨迭代的 KV Cache 管理：
- 需要维护 Block 引用计数
- 处理 Prefix Caching 的冲突

缓解方案：
- 使用高效的哈希表（xxhash）
- 减少哈希计算频率
- 预计算常用前缀的哈希
```

**开销 4：数据拷贝开销**
```
每次迭代需要：
- 准备输入张量（token_ids, positions）
- 准备上下文（cu_seqlens, block_tables）

缓解方案：
- 复用张量内存（避免频繁分配）
- 使用张量拼接而非拷贝
- 缓存常用张量（如 positions）
```

---

### 问题 4.3：抢占式调度的资源回收机制

**问题**：
在资源受限的情况下，抢占式调度是保证高优先级请求及时完成的关键。请分析：
1. BlockManager 的引用计数机制是如何工作的？如何避免悬垂引用？
2. 抢占一个序列时，哪些资源需要立即回收？哪些资源可以保留？
3. 如果实现"带优先级的抢占"，应该如何修改 scheduler.py？

**考察意图**：
- 验证对资源回收机制的理解
- 考察对抢占策略设计的掌握
- 检验对优先级调度实现的分析能力

**详细答案**：

**1. BlockManager 的引用计数机制**

引用计数维护：

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 引用计数
        self.hash = -1
        self.token_ids = []

    def reset(self):
        self.ref_count = 1  # 重置为 1（被分配）
        self.hash = -1
        self.token_ids = []
```

引用计数更新流程：

```python
def allocate(self, seq):
    """分配 Block 时增加引用计数"""
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)

        if not cache_miss:
            # 命中 Prefix Cache，增加引用计数
            block = self.blocks[block_id]
            block.ref_count += 1
        else:
            # 分配新 Block，引用计数初始化为 1
            block = self._allocate_block(block_id)
            block.ref_count = 1
```

引用计数减少：

```python
def deallocate(self, seq):
    """释放序列时减少引用计数"""
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1

        if block.ref_count == 0:
            # 无引用，回收 Block
            self._deallocate_block(block_id)

    seq.block_table.clear()
```

避免悬垂引用的机制：

**机制 1：倒序释放**
```python
# 倒序遍历，确保后缀 Block 先释放
for block_id in reversed(seq.block_table):
    block = self.blocks[block_id]
    block.ref_count -= 1
    # ...
```

原因：
- Prefix Caching 共享前缀 Block
- 倒序释放确保前缀 Block 的引用计数正确更新
- 避免过早释放共享 Block

**机制 2：引用计数检查**
```python
def _allocate_block(self, block_id: int) -> Block:
    """分配 Block 前检查引用计数"""
    block = self.blocks[block_id]
    assert block.ref_count == 0  # 确保无引用
    block.reset()
    # ...
```

**机制 3：引用计数一致性**
```python
# 每个序列完成时，必须调用 deallocate
# 每个序列抢占时，调用 deallocate（或部分释放）
# 确保引用计数的增减平衡
```

**2. 抢占时的资源回收策略**

立即回收的资源：
```python
def preempt(self, seq):
    """
    立即回收：
    1. 运行状态 → 等待状态
    2. 释放已分配但未使用的 Block
    3. 更新调度队列
    """

    # 1. 更新状态
    seq.status = SequenceStatus.WAITING

    # 2. 释放 Block
    # 注意：根据抢占策略，可以全部释放或部分释放
    # 方案 A：全部释放（激进抢占）
    self.block_manager.deallocate(seq)

    # 方案 B：部分释放（保留 Prefix Cache）
    # 只释放部分 Block，保留前缀
    # 实现更复杂

    # 3. 更新队列
    self.waiting.appendleft(seq)
```

保留的资源（如果使用部分抢占）：
```python
def partial_preempt(self, seq, keep_prefix_blocks):
    """
    保留前缀 Block，只释放后缀 Block

    保留的 Block：
    - 前 keep_prefix_blocks 个 Block
    - 这些 Block 可以被后续请求复用（Prefix Caching）

    释放的 Block：
    - 后续的 Block
    - 这些 Block 的 ref_count 减 1
    """

    # 计算要保留的 Block
    keep_ids = seq.block_table[:keep_prefix_blocks]
    release_ids = seq.block_table[keep_prefix_blocks:]

    # 释放后缀 Block
    for block_id in release_ids:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)

    # 更新 block_table
    seq.block_table = keep_ids
```

**3. 带优先级的抢占实现**

修改 Sequence 类，添加优先级：

```python
class Sequence:
    def __init__(self, prompt_tokens, sampling_params, priority=0):
        self.priority = priority  # 优先级，值越大优先级越高
        # ...
```

修改 Scheduler 的抢占逻辑：

```python
def preempt(self, seq=None):
    """
    带优先级的抢占：
    - 如果指定 seq，抢占该序列
    - 如果不指定 seq，选择低优先级的序列抢占
    """

    if seq is not None:
        # 抢占指定序列
        self._preempt_single(seq)
    else:
        # 选择低优先级的序列抢占
        # 策略 1：选择优先级最低的
        # 策略 2：选择优先级最低且最短的
        # 策略 3：选择优先级最低且最新的

        # 策略 1 实现
        min_priority_seq = min(self.running, key=lambda s: s.priority)
        self._preempt_single(min_priority_seq)

        # 策略 2 实现（考虑序列长度）
        min_priority_seq = min(
            self.running,
            key=lambda s: (s.priority, len(s))
        )
        self._preempt_single(min_priority_seq)
```

修改调度逻辑，考虑优先级：

```python
def schedule(self):
    """
    修改为优先级调度：
    1. 等待队列按优先级排序
    2. 运行队列保持高优先级序列
    """

    # 转换为优先级队列
    waiting_sorted = sorted(self.waiting, key=lambda s: -s.priority)

    # Prefill 阶段：优先调度高优先级请求
    scheduled_seqs = []
    num_batched_tokens = 0

    for seq in waiting_sorted:
        if len(scheduled_seqs) >= self.max_num_seqs:
            break
        if num_batched_tokens + (len(seq) - seq.num_cached_tokens) > self.max_num_batched_tokens:
            break
        if not self.block_manager.can_allocate(seq):
            # 尝试抢占低优先级的序列
            self.preempt_low_priority_for(seq)
            continue

        self.block_manager.allocate(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True

    # Decode 阶段：保持高优先级序列运行
    # 运行队列按优先级排序
    running_sorted = sorted(self.running, key=lambda s: -s.priority)

    for seq in running_sorted:
        if len(scheduled_seqs) >= self.max_num_seqs:
            break

        while not self.block_manager.can_append(seq):
            # 尝试抢占低优先级的序列
            low_priority_seq = self.find_low_priority_running(seq.priority)
            if low_priority_seq:
                self._preempt_single(low_priority_seq)
            else:
                break

        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)

    return scheduled_seqs, False
```

辅助函数：

```python
def preempt_low_priority_for(self, target_seq):
    """
    为目标序列抢占资源：
    - 找到低优先级的序列
    - 抢占直到有足够资源
    """
    target_priority = target_seq.priority

    while not self.block_manager.can_allocate(target_seq):
        # 找到低优先级的序列
        low_priority_seq = self.find_low_priority_running(target_priority)

        if low_priority_seq:
            self._preempt_single(low_priority_seq)
        else:
            break

def find_low_priority_running(self, threshold_priority):
    """
    找到运行中优先级低于阈值的序列
    """
    for seq in self.running:
        if seq.priority < threshold_priority:
            return seq
    return None
```

---

## 总结

本套面试题覆盖了 nano-vllm 项目的核心技术维度：

1. **模型推理与显存优化**：PagedAttention、KV Cache 管理、Prefix Caching
2. **底层算子实现**：FlashAttention、Triton Kernel、采样优化
3. **模型结构与适配**：张量布局、非连续内存、GQA/MoE 适配
4. **调度策略与并发**：Continuous Batching、抢占机制、优先级调度

每道题都包含：
- 具体的技术设问
- 明确的考察意图
- 详细的技术原理和实现细节

适合用于检验候选人：
- 对大模型推理框架的深入理解
- 高性能计算和并行编程能力
- 系统设计和优化思维
- 代码实现和调试能力

---

*文档生成日期：2026-02-25*
*面试题版本：nano-vllm v1.0*
