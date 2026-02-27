from dataclasses import dataclass
import torch


@dataclass
class Context:
    '''
    batch decode时，不同 prompt 长度不同，生成的 token 数也不同。为了高效地将多个变长序列打包（packed）到同一个 batch 中进行并行计算（比如用 FlashAttention），我们需要额外的元数据来告诉 kernel：
    1.每个序列从哪里开始、到哪里结束；
    2.KV cache 存在哪些物理页（block）里；
    3.当前是 prefill 阶段还是 decode 阶段。

    管理推理过程中的上下文状态（Context）
    isprefill: 标志位代表是否处于逐token生成阶段，在用户第一次输入prompt时才进行prefill;
    
    cumulative sequence lengths:意思是累积的序列长度。假如batch有三个seq，长度为4,2,3，则cu_seqlens = [0, 4, 6, 9]代表了seq的起始位置。
    用于在FlashAttention中处理packed sequences;
    
    max_seqlens: 当前 batch 中 query 和 key 的 最大序列长度;
    mapping: PagedAttention 中，KV cache 被分页存储（类似虚拟内存）,

    slot_mapping[i] = physical_page_id * page_size + offset
    表示第 i 个逻辑 token 的 KV 应该存/取到哪个物理 slot。

    context_lens: len([prompt + generated]) 
    用于 decode 阶段确定每个序列的历史长度，从而正确 slice KV cache。
    
    block_tables: block_tables[0] = [10, 25, 7] 表示序列 0 的 KV cache 分布在物理页 10、25、7 上，根据这个表 + slot_mapping 定位具体 slot
    '''
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

# _CONTEXT实例是全局变量，用于在前向传播的时候传递batch相关的信息
# 防止变量层层传递带来的开销。
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
