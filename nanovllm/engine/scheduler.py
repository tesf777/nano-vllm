from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    '''
    max_num_seqs:单batch最大序列数量\n
    max_num_batched_tokens:单batch最大token总数\n
    block_manager:管理kv缓存块的页表数和每页包含的Block数\n
    waiting：等待的序列\n
    running:已完成prefill，正在decoding的序列\n
    '''
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        '''
        prefill,decode分离的调度实现
        负责操作blockmanager对block进行分配
        prefill优先，分配成功直接返回seq，不进入decode - prefill延迟敏感
        返回 已调度的列表 和 is_prefilled标志位
        '''
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 没有足够空间decode了，抢夺已经prefill的seq
                    self.preempt(self.running.pop())
                else:
                    # 连prefill的序列也没有，则抢夺自己。
                    self.preempt(seq)
                    break
            else:
              # 如果有足够空间，不需要抢夺资源，则直接分配
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs # 确定本轮至少调度了一个seq
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        # 抢夺补偿，添加到等待队列的首位，下一次会优先处理
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        '''
        自回归生成的拼接：
        model_run一个循环之后，将生成的n个token_id，拼接到对应n个seq的末尾
        顺带考虑end of seq符号
        '''
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 存在中止符号，并且该token是中止token
                # 或是已经达到decode上限
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
