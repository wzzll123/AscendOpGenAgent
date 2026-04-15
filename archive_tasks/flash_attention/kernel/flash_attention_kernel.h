#ifndef FLASH_ATTENTION_KERNEL_H
#define FLASH_ATTENTION_KERNEL_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "flash_attention_tiling.h"
#include "kernel_common.h"
#include "flash_attention_cube.h"
#include "flash_attention_vec.h"

using namespace AscendC;

class FlashAttentionKernel {
    static constexpr uint32_t C0 = 32 / sizeof(half);  // 16

public:
    __aicore__ inline FlashAttentionKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR output, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe)
    {
        pipe_ = pipe;
        CopyTiling(&tiling_, tilingGM);

        uint32_t dim = tiling_.dim;
        uint32_t seqLen = tiling_.seqLen;
        uint32_t dimAlign = AlignUp(dim, C0);
        uint32_t totalElements = tiling_.batch * tiling_.heads * seqLen * dim;
        uint32_t kvLoops = seqLen / BLOCK_N;
        int totalBlocks = (seqLen / BLOCK_M) * tiling_.heads * tiling_.batch;

        // Core index: AIC and AIV share the same physical core
        int coreIdx;
        if ASCEND_IS_AIC {
            coreIdx = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            coreIdx = GetBlockIdx() / GetSubBlockNum();
        }

        int numCores = GetBlockNum();
        sched_.Init(totalBlocks, numCores, coreIdx);

        // Set up GM tensors
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(q), totalElements);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(k), totalElements);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(v), totalElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output), totalElements);

        // Workspace layout per core:
        // Keep one slot per kv loop for correctness while debugging the
        // current ring-buffer lowering mismatch.
        // ws_s:     kvLoops * BLOCK_M * BLOCK_N * sizeof(float)
        // ws_p:     kvLoops * BLOCK_M * BLOCK_N * sizeof(half)
        // ws_o:     kvLoops * BLOCK_M * dimAlign * sizeof(float)
        // ws_meta:  kvLoops * BLOCK_M * 3 * sizeof(float)
        //           [max (kvLoops*BLOCK_M), sum (kvLoops*BLOCK_M), exp (kvLoops*BLOCK_M)]
        // ws_acc_o: kvLoops * BLOCK_M * dimAlign * sizeof(float)
        uint64_t wsSSize = (uint64_t)kvLoops * BLOCK_M * BLOCK_N;
        uint64_t wsPSize = (uint64_t)kvLoops * BLOCK_M * BLOCK_N;
        uint64_t wsOSize = (uint64_t)kvLoops * BLOCK_M * dimAlign;
        uint64_t wsMetaSize = (uint64_t)kvLoops * BLOCK_M * 3;
        uint64_t wsAccOSize = (uint64_t)kvLoops * BLOCK_M * dimAlign;

        uint64_t perCoreBytes = wsSSize * sizeof(float) +
                                wsPSize * sizeof(half) +
                                wsOSize * sizeof(float) +
                                wsMetaSize * sizeof(float) +
                                wsAccOSize * sizeof(float);

        GM_ADDR wsBase = workspace + coreIdx * perCoreBytes;

        GM_ADDR wsSPtr = wsBase;
        GM_ADDR wsPPtr = wsSPtr + wsSSize * sizeof(float);
        GM_ADDR wsOPtr = wsPPtr + wsPSize * sizeof(half);
        GM_ADDR wsMetaPtr = wsOPtr + wsOSize * sizeof(float);
        GM_ADDR wsAccOPtr = wsMetaPtr + wsMetaSize * sizeof(float);

        wsSGm_.SetGlobalBuffer((__gm__ float *)(wsSPtr), wsSSize);
        wsPGm_.SetGlobalBuffer((__gm__ half *)(wsPPtr), wsPSize);
        wsOGm_.SetGlobalBuffer((__gm__ float *)(wsOPtr), wsOSize);
        wsMetaGm_.SetGlobalBuffer((__gm__ float *)(wsMetaPtr), wsMetaSize);
        wsAccOGm_.SetGlobalBuffer((__gm__ float *)(wsAccOPtr), wsAccOSize);

        if ASCEND_IS_AIC {
            cubeKernel_.Init(tiling_, qGm_, kGm_, vGm_, wsSGm_, wsPGm_, wsOGm_);
            cubeKernel_.InitBuffers(*pipe_);
        }

        if ASCEND_IS_AIV {
            vecKernel_.Init(tiling_, wsSGm_, wsPGm_, wsOGm_, wsMetaGm_, wsAccOGm_, outGm_);
            vecKernel_.InitBuffers(pipe_);
            // InitFreeSlots: Pre-set PRELAUNCH free flags for bidirectional handshake
            // This allows MM1/MM2 to write the first PRELAUNCH slots without waiting
            for (int i = 0; i < PRELAUNCH; i++) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_VEC1_MM1);  // S slot free
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_MM2_VEC1);  // P slot free (MM2 reads P)
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_VEC2_MM2);  // O slot free
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t seqLen = tiling_.seqLen;
        uint32_t seqBlocks = seqLen / BLOCK_M;
        uint32_t kvLoops = seqLen / BLOCK_N;

        while (sched_.HasNext()) {
            int blockIdx = sched_.Next();

            // Decode block index -> (bz, by, bx)
            int bx = blockIdx % seqBlocks;
            int tmp = blockIdx / seqBlocks;
            int by = tmp % tiling_.heads;
            int bz = tmp / tiling_.heads;

            if ASCEND_IS_AIC {
                // Load Q tile to L1 (once per block)
                cubeKernel_.LoadQ(bz, by, bx);

                // Ring pipeline loop
                for (uint32_t t = 0; t < kvLoops + PRELAUNCH; t++) {
                    if (t < kvLoops) {
                        int slotProd = t;
                        cubeKernel_.ComputeMM1(bz, by, t, slotProd);
                    }
                    if (t >= PRELAUNCH) {
                        int nowK = t - PRELAUNCH;
                        int slotCons = nowK;
                        cubeKernel_.ComputeMM2(bz, by, nowK, slotCons);
                    }
                }
            }

            if ASCEND_IS_AIV {
                vecKernel_.InitState();
                vecKernel_.curBz_ = bz;
                vecKernel_.curBy_ = by;
                vecKernel_.curBx_ = bx;

                // Ring pipeline loop with per-loop GM softmax/accumulation state
                int vec1Loop = 0;
                int vec2Loop = 0;

                for (uint32_t t = 0; t < kvLoops + PRELAUNCH; t++) {
                    if (t < kvLoops) {
                        int slotProd = t;
                        bool isFirstVec1 = (vec1Loop == 0);
                        vecKernel_.ComputeVec1(t, slotProd, vec1Loop, isFirstVec1);
                        vec1Loop++;
                    }
                    if (t >= PRELAUNCH) {
                        int nowK = t - PRELAUNCH;
                        int slotCons = nowK;
                        bool isFirstVec2 = (vec2Loop == 0);
                        bool isLastVec2 = (nowK == (int)kvLoops - 1);
                        vecKernel_.ComputeVec2(nowK, slotCons, vec2Loop,
                                               isFirstVec2, isLastVec2);
                        vec2Loop++;
                    }
                }
            }
        }
    }

private:
    TPipe *pipe_;
    FlashAttentionTiling tiling_;
    BlockScheduler1D sched_;

    GlobalTensor<half> qGm_, kGm_, vGm_, outGm_;
    GlobalTensor<float> wsSGm_, wsOGm_, wsMetaGm_, wsAccOGm_;
    GlobalTensor<half> wsPGm_;

    FlashAttentionCube cubeKernel_;
    FlashAttentionVec vecKernel_;
};

#endif // FLASH_ATTENTION_KERNEL_H
