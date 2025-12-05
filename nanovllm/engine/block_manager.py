from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    '''
    用于组织 Key-Value Cache的数据结构
    大小由Config类中的kvcache_block_size来设定
    - 避免 GPU 内存碎片
    - 支持 非连续内存分配
    - 实现 Prefix Caching（前缀共享）\n
    一个 1000 token 的 prompt 被分成 4 个 block（256×3 + 232），每个 block 对应 GPU 上一段连续的 KV Cache 内存区域。\n
    block_table = [10, 25, 7, 42] 表示：第 0-255 token 的 KV 在物理页 10，256~511 在页 25，依此类推。\n

    '''
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    '''
    num_blocks:表示总共有多少个Block块;\n
    block_size:表示1个Block包含多少tokens (默认256);\n
    blocks:用list来放置Block块;\n
    hash_to_block_id: {hash:block_id} 也就是：hash_to_block_id[h0] = 0;\n
    free_block_ids 和 used_block_ids: 前者初始化时为满，后者初始化时为空。
    [0,1,2...99] 和 ()，在使用了Block0后，变为: [1,2...99,0] 和 (0)
    '''
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # 装饰器利于继承，用法类似静态函数
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        '''
        在维护kv-cache时，需要注意在语言模型中，同一个 token 序列出现在不同上下文中，其语义和注意力结果可能完全不同！\n
        因此，我们在计算当前 block 的哈希时，把前一个 block 的哈希值作为前缀（prefix）一起参与hash。\n
        这样，不同上下文，同时长得相同的Block，就不会被认为是相同的。
        '''
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        '''
        优先从free池中获取闲置的Block，并在used池中标记这个Block_id已经使用
        '''
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        '''
        释放used池中的对应id的Block，并标记这个Block为闲置
        '''
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        '''目前free池的空闲Block可以完整容纳当前Seq，返回True'''
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        '''
        倒置页表，并遍历每一页(Block)，减少引用计数，若归零则回收页
        遍历后，清空页表，清零缓存tokens数量
        '''
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        '''
        如果当前的序列长度恰好为block_size*n余1，刚好需要申请新Block时
        free池中恰好有资源，则返回True
        '''
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        '''
        实际执行 block 扩展或缓存更新的函数，在每次生成新 token 后调用
        - 情况1:刚好需要新Block时，例如tokens:256->257时，
        取free池首的Block，分配并添加页表
        - 情况2:刚好填满一个Block时，例如tokens:255->256时
        - 情况3:其他情况，只是做防御性处理，当前block未满，禁止hash
        '''
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
