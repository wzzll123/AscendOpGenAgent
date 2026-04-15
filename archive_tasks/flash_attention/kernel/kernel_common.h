#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template<typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *ptr = reinterpret_cast<int32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i, ++ptr) {
        *ptr = *(tiling32 + i);
    }
}

// 1D block scheduler: distributes totalBlocks evenly across cores
class BlockScheduler1D {
public:
    __aicore__ inline void Init(int totalBlocks, int numCores, int coreIdx)
    {
        int blocksPerCore = totalBlocks / numCores;
        int remainder = totalBlocks % numCores;
        startBlock_ = coreIdx * blocksPerCore + (coreIdx < remainder ? coreIdx : remainder);
        endBlock_ = startBlock_ + blocksPerCore + (coreIdx < remainder ? 1 : 0);
        current_ = startBlock_;
    }

    __aicore__ inline bool HasNext() { return current_ < endBlock_; }

    __aicore__ inline int Next()
    {
        return current_++;
    }

private:
    int startBlock_;
    int endBlock_;
    int current_;
};

#endif // KERNEL_COMMON_H
