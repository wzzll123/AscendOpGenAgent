#ifndef FLASH_ATTENTION_CUBE_H
#define FLASH_ATTENTION_CUBE_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "flash_attention_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class FlashAttentionCube {
    static constexpr uint32_t C0 = 32 / sizeof(half);  // 16

public:
    __aicore__ inline FlashAttentionCube() {}

    __aicore__ inline void Init(const FlashAttentionTiling &tiling,
                                GlobalTensor<half> &qGm, GlobalTensor<half> &kGm, GlobalTensor<half> &vGm,
                                GlobalTensor<float> &wsSGm, GlobalTensor<half> &wsPGm,
                                GlobalTensor<float> &wsOGm)
    {
        tiling_ = tiling;
        qGm_ = qGm;
        kGm_ = kGm;
        vGm_ = vGm;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        dimAlign_ = AlignUp(tiling_.dim, C0);
    }

    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint32_t dim = tiling_.dim;
        // L1 buffers (TBuf for persistent Q, reloadable KV and P)
        pipe.InitBuffer(qBufL1_, BLOCK_M * dim * sizeof(half));
        pipe.InitBuffer(kvBufL1_, BLOCK_N * dim * sizeof(half));
        pipe.InitBuffer(pBufL1_, BLOCK_M * BLOCK_N * sizeof(half));
        // L0 buffers (TQue with depth=2 for double-buffering to avoid L0B conflict)
        pipe.InitBuffer(queL0A_, 2, BLOCK_M * BASE_K * sizeof(half));
        pipe.InitBuffer(queL0B_, 2, BASE_K * BLOCK_N * sizeof(half));
        // L0C is shared by MM1 (64x64) and MM2 (64x128); size it for the larger MM2 tile.
        pipe.InitBuffer(queL0C_, 1, BLOCK_M * BASE_K * sizeof(float));
    }

    // Load Q tile to L1 (once per block)
    __aicore__ inline void LoadQ(int bz, int by, int bx)
    {
        uint32_t dim = tiling_.dim;
        uint32_t seqLen = tiling_.seqLen;
        uint64_t qOffset = ((uint64_t)bz * tiling_.heads * seqLen + (uint64_t)by * seqLen + (uint64_t)bx * BLOCK_M) * dim;
        LocalTensor<half> qL1 = qBufL1_.Get<half>();
        LoadNdGmToNzL1(qL1, qGm_[qOffset], BLOCK_M, dim, dim);
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(ev);
        WaitFlag<HardEvent::MTE2_MTE1>(ev);
    }

    // BMM1: S = Q @ K^T, result to workspace_s
    __aicore__ inline void ComputeMM1(int bz, int by, int t, int slot)
    {
        // Wait for Vec1 to release S slot (bidirectional handshake)
        CrossCoreWaitFlag<0x2>(SYNC_VEC1_MM1);

        uint32_t dim = tiling_.dim;
        uint32_t seqLen = tiling_.seqLen;

        // Load K[bz, by, t*blockN:(t+1)*blockN, :] to L1
        uint64_t kOffset = ((uint64_t)bz * tiling_.heads * seqLen + (uint64_t)by * seqLen + (uint64_t)t * BLOCK_N) * dim;
        LocalTensor<half> kvL1 = kvBufL1_.Get<half>();
        LoadNdGmToNzL1(kvL1, kGm_[kOffset], BLOCK_N, dim, dim);

        // Wait for K L1 load
        event_t evMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(evMte2);
        WaitFlag<HardEvent::MTE2_MTE1>(evMte2);

        // Alloc L0C
        LocalTensor<float> cL0 = queL0C_.AllocTensor<float>();
        LocalTensor<half> qL1 = qBufL1_.Get<half>();

        // K-dimension loop: Q[blockM, dim] * K^T[dim, blockN] = S[blockM, blockN]
        uint32_t kTiles = dim / BASE_K;
        uint32_t mActSizeAlign = AlignUp(BLOCK_M, C0);
        uint32_t nActSizeAlign = AlignUp(BLOCK_N, C0);
        for (uint32_t ki = 0; ki < kTiles; ki++) {
            // Load Q slice to L0A (match reference block cube path)
            LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
            {
                uint32_t kActSizeAlign = BASE_K;
                uint32_t mLoop = mActSizeAlign / C0;
                uint64_t qL1Offset = (uint64_t)ki * BLOCK_M * BASE_K;
                for (uint32_t i = 0; i < mLoop; i++) {
                    LoadData2DParams loadData2DParams;
                    loadData2DParams.startIndex = i;
                    loadData2DParams.repeatTimes = kActSizeAlign / C0;
                    loadData2DParams.srcStride = mActSizeAlign / C0;
                    loadData2DParams.dstGap = 0;
                    loadData2DParams.ifTranspose = false;
                    LoadData(aL0[C0 * i * kActSizeAlign], qL1[qL1Offset], loadData2DParams);
                }
            }
            queL0A_.EnQue(aL0);

            // Load K slice to L0B (match reference block cube path)
            LocalTensor<half> bL0 = queL0B_.AllocTensor<half>();
            {
                uint32_t kActSizeAlign = BASE_K;
                uint64_t kL1Offset = (uint64_t)ki * BLOCK_N * BASE_K;
                LoadData2DParams loadData2DParams;
                loadData2DParams.startIndex = 0;
                loadData2DParams.repeatTimes = (kActSizeAlign / C0) * (nActSizeAlign / C0);
                loadData2DParams.srcStride = 1;
                loadData2DParams.dstGap = 0;
                loadData2DParams.ifTranspose = false;
                LoadData(bL0, kvL1[kL1Offset], loadData2DParams);
            }
            queL0B_.EnQue(bL0);

            // Wait for L0A/L0B ready, then Mmad
            aL0 = queL0A_.DeQue<half>();
            bL0 = queL0B_.DeQue<half>();

            MmadParams mmadParams;
            mmadParams.m = BLOCK_M;
            mmadParams.n = BLOCK_N;
            mmadParams.k = BASE_K;
            mmadParams.cmatrixInitVal = (ki == 0);
            mmadParams.cmatrixSource = false;
            Mmad(cL0, aL0, bL0, mmadParams);

            queL0A_.FreeTensor(aL0);
            queL0B_.FreeTensor(bL0);
        }

        // Handle remaining K dimension
        uint32_t kRemain = dim % BASE_K;
        if (kRemain > 0) {
            uint32_t kAligned = AlignUp(kRemain, C0);

            LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
            {
                uint32_t mLoop = mActSizeAlign / C0;
                uint64_t qL1Offset = (uint64_t)kTiles * BLOCK_M * BASE_K;
                for (uint32_t i = 0; i < mLoop; i++) {
                    LoadData2DParams loadData2DParams;
                    loadData2DParams.startIndex = i;
                    loadData2DParams.repeatTimes = kAligned / C0;
                    loadData2DParams.srcStride = mActSizeAlign / C0;
                    loadData2DParams.dstGap = 0;
                    loadData2DParams.ifTranspose = false;
                    LoadData(aL0[C0 * i * kAligned], qL1[qL1Offset], loadData2DParams);
                }
            }
            queL0A_.EnQue(aL0);

            LocalTensor<half> bL0 = queL0B_.AllocTensor<half>();
            {
                uint64_t kL1Offset = (uint64_t)kTiles * BLOCK_N * BASE_K;
                LoadData2DParams loadData2DParams;
                loadData2DParams.startIndex = 0;
                loadData2DParams.repeatTimes = (kAligned / C0) * (nActSizeAlign / C0);
                loadData2DParams.srcStride = 1;
                loadData2DParams.dstGap = 0;
                loadData2DParams.ifTranspose = false;
                LoadData(bL0, kvL1[kL1Offset], loadData2DParams);
            }
            queL0B_.EnQue(bL0);

            aL0 = queL0A_.DeQue<half>();
            bL0 = queL0B_.DeQue<half>();

            MmadParams mmadParams;
            mmadParams.m = BLOCK_M;
            mmadParams.n = BLOCK_N;
            mmadParams.k = kAligned;
            mmadParams.cmatrixInitVal = (kTiles == 0);
            mmadParams.cmatrixSource = false;
            Mmad(cL0, aL0, bL0, mmadParams);

            queL0A_.FreeTensor(aL0);
            queL0B_.FreeTensor(bL0);
        }

        // Fixpipe L0C -> workspace_s[slot]
        queL0C_.EnQue(cL0);
        cL0 = queL0C_.DeQue<float>();

        uint64_t wsOffset = (uint64_t)slot * BLOCK_M * BLOCK_N;
        FixpipeParamsV220 fixParams;
        fixParams.mSize = BLOCK_M;
        fixParams.nSize = BLOCK_N;
        fixParams.srcStride = mActSizeAlign;
        fixParams.dstStride = BLOCK_N;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        Fixpipe(wsSGm_[wsOffset], cL0, fixParams);

        queL0C_.FreeTensor(cL0);

        // Signal Vec1: S data ready (READY_S)
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM1_VEC1);
    }

    // BMM2: O_tmp = P @ V, result to workspace_o
    __aicore__ inline void ComputeMM2(int bz, int by, int t, int slot)
    {
        uint32_t dim = tiling_.dim;
        uint32_t seqLen = tiling_.seqLen;

        // Wait for Vec1 to write P (READY_P)
        CrossCoreWaitFlag<0x2>(SYNC_VEC1_MM2);

        // Load P[blockM, blockN] from workspace_p to L1
        uint64_t pOffset = (uint64_t)slot * BLOCK_M * BLOCK_N;
        LocalTensor<half> pL1 = pBufL1_.Get<half>();
        LoadNdGmToNzL1(pL1, wsPGm_[pOffset], BLOCK_M, BLOCK_N, BLOCK_N);

        // Load V[bz, by, t*blockN:(t+1)*blockN, :] to L1
        uint64_t vOffset = ((uint64_t)bz * tiling_.heads * seqLen + (uint64_t)by * seqLen + (uint64_t)t * BLOCK_N) * dim;
        LocalTensor<half> kvL1 = kvBufL1_.Get<half>();
        LoadNdGmToNzL1(kvL1, vGm_[vOffset], BLOCK_N, dim, dim);

        // Wait L1 loads
        event_t evMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(evMte2);
        WaitFlag<HardEvent::MTE2_MTE1>(evMte2);

        // Wait for Vec2 to release O slot (FREE_O) before writing
        CrossCoreWaitFlag<0x2>(SYNC_VEC2_MM2);

        // N-dimension loop: P[blockM, blockN] * V[blockN, dim] = O[blockM, dim]
        // Keep a focused dump chain so we can validate kvL1 -> bL0 -> cL0 against CPU slices.
        uint32_t nTiles = dim / BASE_K;
        uint32_t mActSizeAlign = AlignUp(BLOCK_M, C0);
        for (uint32_t ni = 0; ni < nTiles; ni++) {
            // Load P to L0A via the verified NZ->ZZ helper.
            LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
            LoadNzL1ToZzL0A(aL0, pL1, BLOCK_M, BLOCK_N, BLOCK_M);
            queL0A_.EnQue(aL0);

            // Load current V[:, ni*BASE_K:(ni+1)*BASE_K] tile to L1, then convert NZ->ZN.
            uint64_t vSliceOffset = vOffset + (uint64_t)ni * BASE_K;
            LoadNdGmToNzL1(kvL1, vGm_[vSliceOffset], BLOCK_N, BASE_K, dim);
            event_t evVSlice = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
            SetFlag<HardEvent::MTE2_MTE1>(evVSlice);
            WaitFlag<HardEvent::MTE2_MTE1>(evVSlice);

            LocalTensor<half> bL0 = queL0B_.AllocTensor<half>();
            LoadNzL1ToZnL0B(bL0, kvL1, BLOCK_N, BASE_K, BLOCK_N);
            queL0B_.EnQue(bL0);

            aL0 = queL0A_.DeQue<half>();
            bL0 = queL0B_.DeQue<half>();

            LocalTensor<float> cL0 = queL0C_.AllocTensor<float>();

            MmadParams mmadParams;
            mmadParams.m = BLOCK_M;
            mmadParams.n = BASE_K;
            mmadParams.k = BLOCK_N;
            mmadParams.cmatrixInitVal = true;  // Each N-slice is independent
            mmadParams.cmatrixSource = false;
            Mmad(cL0, aL0, bL0, mmadParams);

            queL0A_.FreeTensor(aL0);
            queL0B_.FreeTensor(bL0);

            // Fixpipe to workspace_o with stride
            queL0C_.EnQue(cL0);
            cL0 = queL0C_.DeQue<float>();

            uint64_t wsBase = (uint64_t)slot * BLOCK_M * dimAlign_ + (uint64_t)ni * BLOCK_M * BASE_K;
            FixpipeParamsV220 fixParams;
            fixParams.mSize = BLOCK_M;
            fixParams.nSize = BASE_K;
            fixParams.srcStride = mActSizeAlign;
            fixParams.dstStride = dimAlign_;
            fixParams.ndNum = 1;
            fixParams.srcNdStride = 0;
            fixParams.dstNdStride = 0;
            Fixpipe(wsOGm_[wsBase], cL0, fixParams);

            queL0C_.FreeTensor(cL0);
        }

        // Handle remaining N dimension
        uint32_t nRemain = dim % BASE_K;
        if (nRemain > 0) {
            uint32_t nAligned = AlignUp(nRemain, C0);

            LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
            LoadNzL1ToZzL0A(aL0, pL1, BLOCK_M, BLOCK_N, BLOCK_M);
            queL0A_.EnQue(aL0);

            uint64_t vSliceOffset = vOffset + (uint64_t)nTiles * BASE_K;
            LoadNdGmToNzL1(kvL1, vGm_[vSliceOffset], BLOCK_N, nRemain, dim);
            event_t evVSlice = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
            SetFlag<HardEvent::MTE2_MTE1>(evVSlice);
            WaitFlag<HardEvent::MTE2_MTE1>(evVSlice);

            LocalTensor<half> bL0 = queL0B_.AllocTensor<half>();
            uint32_t kActSizeAlign = AlignUp(BLOCK_N, C0);
            uint32_t kLoop = kActSizeAlign / C0;
            for (uint32_t i = 0; i < kLoop; i++) {
                LoadData2DParams loadData2DParams;
                loadData2DParams.startIndex = i;
                loadData2DParams.repeatTimes = nAligned / C0;
                loadData2DParams.srcStride = kActSizeAlign / C0;
                loadData2DParams.dstGap = 0;
                loadData2DParams.ifTranspose = true;
                uint64_t kL1Offset = (uint64_t)i * C0;
                LoadData(bL0[C0 * i * nAligned], kvL1[kL1Offset], loadData2DParams);
            }
            queL0B_.EnQue(bL0);

            aL0 = queL0A_.DeQue<half>();
            bL0 = queL0B_.DeQue<half>();

            LocalTensor<float> cL0 = queL0C_.AllocTensor<float>();

            MmadParams mmadParams;
            mmadParams.m = BLOCK_M;
            mmadParams.n = nAligned;
            mmadParams.k = BLOCK_N;
            mmadParams.cmatrixInitVal = true;
            Mmad(cL0, aL0, bL0, mmadParams);

            queL0A_.FreeTensor(aL0);
            queL0B_.FreeTensor(bL0);

            queL0C_.EnQue(cL0);
            cL0 = queL0C_.DeQue<float>();

            uint64_t wsBase = (uint64_t)slot * BLOCK_M * dimAlign_ + (uint64_t)nTiles * BLOCK_M * BASE_K;
            FixpipeNzL0cToNdGm(wsOGm_[wsBase], cL0, nAligned, BLOCK_M);

            queL0C_.FreeTensor(cL0);
        }

        // Signal Vec2: O data ready (READY_O)
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM2_VEC2);

        // Signal Vec1: P slot free (FREE_P) - Vec1 can write next P
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM2_VEC1);
    }

private:
    FlashAttentionTiling tiling_;
    uint32_t dimAlign_;

    GlobalTensor<half> qGm_, kGm_, vGm_;
    GlobalTensor<float> wsSGm_;
    GlobalTensor<half> wsPGm_;
    GlobalTensor<float> wsOGm_;

    // L1: TBuf (persistent, no pipeline depth needed)
    TBuf<TPosition::A1> qBufL1_;
    TBuf<TPosition::A1> kvBufL1_;
    TBuf<TPosition::A1> pBufL1_;

    // L0A/L0B: TQue with depth=2 for double-buffering (avoids L0B read/write conflict)
    TQue<TPosition::A2, 2> queL0A_;
    TQue<TPosition::B2, 2> queL0B_;

    // L0C: TQue depth=1
    TQue<TPosition::CO1, 1> queL0C_;
};

#endif // FLASH_ATTENTION_CUBE_H
