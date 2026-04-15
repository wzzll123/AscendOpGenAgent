#ifndef FLASH_ATTENTION_VEC_H
#define FLASH_ATTENTION_VEC_H

#include "kernel_operator.h"
#include "flash_attention_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

// SoftmaxFlashV2 configuration: WITHOUT_BRC mode, max/sum/exp shape is [M, 1]
constexpr SoftmaxConfig FA_SOFTMAX_CFG = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

// Constants for softmax state buffers
constexpr uint32_t SOFTMAX_TMP_BUF_SIZE = 2048;  // 2KB per slot
constexpr float SOFTMAX_NEG_INF = -1073741824.0f; // -2^30

class FlashAttentionVec {
    static constexpr uint32_t C0 = 32 / sizeof(half);  // 16
    static constexpr uint32_t HALF_M = BLOCK_M / 2;    // 32, each AIV processes half
    // brcbNum: number of float32 in one 32-byte block
    static constexpr uint16_t BRCB_NUM = 32 / sizeof(float);  // 8
    // Max rows per Vec2 M-chunk to fit in UB
    static constexpr uint32_t VEC2_M_CHUNK = 8;

public:
    __aicore__ inline FlashAttentionVec() {}

    __aicore__ inline void Init(const FlashAttentionTiling &tiling,
                                GlobalTensor<float> &wsSGm, GlobalTensor<half> &wsPGm,
                                GlobalTensor<float> &wsOGm, GlobalTensor<float> &wsMetaGm,
                                GlobalTensor<float> &wsAccOGm,
                                GlobalTensor<half> &outGm)
    {
        tiling_ = tiling;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        wsMetaGm_ = wsMetaGm;
        wsAccOGm_ = wsAccOGm;
        outGm_ = outGm;
        vid_ = GetSubBlockIdx();
        dimAlign_ = AlignUp(tiling_.dim, C0);
        kvLoops_ = tiling_.seqLen / BLOCK_N;
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        uint32_t dim = dimAlign_;

        // Shared input queue: used by Vec1 for S scores, Vec2 for O chunks
        // Size = max(HALF_M * BLOCK_N * sizeof(float), VEC2_M_CHUNK * dim * sizeof(float))
        uint32_t inputBufSize = HALF_M * BLOCK_N * sizeof(float);
        uint32_t vec2ChunkSize = VEC2_M_CHUNK * dim * sizeof(float);
        if (vec2ChunkSize > inputBufSize) {
            inputBufSize = vec2ChunkSize;
        }
        pipe->InitBuffer(inputQue1_, 2, inputBufSize);

        // Shared output queue: used by Vec1 for cast P, Vec2 for cast O / intermediate float store
        uint32_t outputBufSize = HALF_M * BLOCK_N * sizeof(half);
        uint32_t vec2OutHalf = VEC2_M_CHUNK * dim * sizeof(half);
        uint32_t vec2OutFloat = VEC2_M_CHUNK * dim * sizeof(float);  // intermediate store needs float
        if (vec2OutHalf > outputBufSize) outputBufSize = vec2OutHalf;
        if (vec2OutFloat > outputBufSize) outputBufSize = vec2OutFloat;
        pipe->InitBuffer(outputQue1_, 1, outputBufSize);

        // Temp buffer for SoftmaxFlashV2 scratch space
        pipe->InitBuffer(tmpBuf_, 16 * 1024);  // 16KB

        // Softmax state for one iteration only; long-lived recurrence state lives in GM
        pipe->InitBuffer(softmaxMaxBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxSumBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxExpBuf_, SOFTMAX_TMP_BUF_SIZE);

        // Default initial values for first iteration
        pipe->InitBuffer(softmaxMaxDefaultBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxSumDefaultBuf_, SOFTMAX_TMP_BUF_SIZE);

        // Brcb expands each row scalar to one full 32B datablock (8 fp32 values).
        // For HALF_M rows, the worst-case output footprint is AlignUp(HALF_M, 8) datablocks.
        uint32_t brcbRowsAlign = ((HALF_M + BRCB_NUM - 1) / BRCB_NUM) * BRCB_NUM;
        uint32_t brcbSize = brcbRowsAlign * BRCB_NUM * sizeof(float);
        pipe->InitBuffer(brcbBuf_, brcbSize);

        // Get persistent handles
        softmaxMaxUb_ = softmaxMaxBuf_.Get<float>();
        softmaxSumUb_ = softmaxSumBuf_.Get<float>();
        softmaxExpUb_ = softmaxExpBuf_.Get<float>();

        softmaxMaxDefaultUb_ = softmaxMaxDefaultBuf_.Get<float>();
        softmaxSumDefaultUb_ = softmaxSumDefaultBuf_.Get<float>();

        // Initialize defaults
        Duplicate(softmaxMaxDefaultUb_, SOFTMAX_NEG_INF,
                  SOFTMAX_TMP_BUF_SIZE / sizeof(float));
        Duplicate(softmaxSumDefaultUb_, 0.0f,
                  SOFTMAX_TMP_BUF_SIZE / sizeof(float));
    }

    // Reset state at start of each block (nothing in UB to reset now)
    __aicore__ inline void InitState()
    {
        // No persistent UB accumulator; accumulated O and online softmax state live in workspace GM
    }

    // ================================================================
    // Vec1: Online softmax using SoftmaxFlashV2 intrinsic
    // ================================================================
    __aicore__ inline void ComputeVec1(int t, int slot, int loop, bool isFirst)
    {
        // Wait for AIC BMM1 to write S scores (READY_S)
        CrossCoreWaitFlag<0x2>(SYNC_MM1_VEC1);

        uint32_t tileSize = HALF_M * BLOCK_N;
        uint32_t rowBase = vid_ * HALF_M;
        uint32_t stateBase = loop * BLOCK_M + rowBase;

        // Load S scores from workspace_s[slot, vid*halfM..., :]
        LocalTensor<float> sUb = inputQue1_.AllocTensor<float>();
        uint64_t sOffset = (uint64_t)slot * BLOCK_M * BLOCK_N +
                           (uint64_t)rowBase * BLOCK_N;
        DataCopy(sUb, wsSGm_[sOffset], tileSize);
        inputQue1_.EnQue(sUb);
        sUb = inputQue1_.DeQue<float>();
        event_t evSMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evSMte2V);
        WaitFlag<HardEvent::MTE2_V>(evSMte2V);

        // Apply scale: S *= smScale (= 1/sqrt(d))
        Muls(sUb, sUb, tiling_.smScale, tileSize);
        PipeBarrier<PIPE_V>();

        // SoftmaxFlashV2: in-place softmax with online m_i/l_i tracking
        LocalTensor<uint8_t> softmaxTmpUb = tmpBuf_.Get<uint8_t>();

        SoftMaxShapeInfo srcShape;
        srcShape.srcM = HALF_M;
        srcShape.srcK = BLOCK_N;
        srcShape.oriSrcM = HALF_M;
        srcShape.oriSrcK = BLOCK_N;

        SoftMaxTiling smTiling = SoftMaxFlashV2TilingFunc(
            srcShape, sizeof(float), sizeof(float),
            softmaxTmpUb.GetSize(), true, false);

        // Determine input m_i / l_i from previous iteration
        LocalTensor<float> inMaxTensor;
        LocalTensor<float> inSumTensor;

        if (isFirst) {
            inMaxTensor = softmaxMaxDefaultUb_;
            inSumTensor = softmaxSumDefaultUb_;
        } else {
            uint32_t prevStateBase = (loop - 1) * BLOCK_M + rowBase;
            event_t evStateMte3Mte2 = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(evStateMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evStateMte3Mte2);
            LocalTensor<float> inStateUb = inputQue1_.AllocTensor<float>();
            DataCopy(inStateUb, wsMetaGm_[prevStateBase], HALF_M);
            inputQue1_.EnQue(inStateUb);
            inStateUb = inputQue1_.DeQue<float>();
            LocalTensor<float> inSumUb = outputQue1_.AllocTensor<float>();
            DataCopy(inSumUb, wsMetaGm_[kvLoops_ * BLOCK_M + prevStateBase], HALF_M);
            outputQue1_.EnQue(inSumUb);
            inSumUb = outputQue1_.DeQue<float>();
            inMaxTensor = inStateUb;
            inSumTensor = inSumUb;
        }

        // SoftmaxFlashV2 WITHOUT_BRC:
        //   dst (in-place) = e^{x - m_new} (unnormalized P)
        //   outSum = l_new = exp_corr * l_old + rowsum(e^{x - m_new})
        //   outMax = m_new = max(m_old, rowmax(x))
        //   outExp = exp_corr = e^{m_old - m_new}
        SoftmaxFlashV2<float, true, true, false, false, FA_SOFTMAX_CFG>(
            sUb,                        // dst: softmax output (in-place)
            softmaxSumUb_,              // outSum: new l_i
            softmaxMaxUb_,              // outMax: new m_i
            sUb,                        // src: scaled scores
            softmaxExpUb_,              // outExp: correction factor
            inSumTensor,                // inSum: previous l_i
            inMaxTensor,                // inMax: previous m_i
            softmaxTmpUb,               // tmp scratch
            smTiling, srcShape);

        PipeBarrier<PIPE_V>();

        // Persist current online softmax state to GM for later Vec2 / next loop use
        event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evVMte3);
        WaitFlag<HardEvent::V_MTE3>(evVMte3);
        DataCopy(wsMetaGm_[stateBase], softmaxMaxUb_, HALF_M);
        DataCopy(wsMetaGm_[kvLoops_ * BLOCK_M + stateBase], softmaxSumUb_, HALF_M);
        DataCopy(wsMetaGm_[2 * kvLoops_ * BLOCK_M + stateBase], softmaxExpUb_, HALF_M);
        if (!isFirst) {
            inputQue1_.FreeTensor(inMaxTensor);
            outputQue1_.FreeTensor(inSumTensor);
        }
        PipeBarrier<PIPE_MTE3>();

        // Cast float32 -> half, write P to workspace_p
        LocalTensor<half> pHalf = outputQue1_.AllocTensor<half>();
        Cast(pHalf, sUb, RoundMode::CAST_ROUND, tileSize);
        outputQue1_.EnQue(pHalf);
        pHalf = outputQue1_.DeQue<half>();

        uint64_t pOffset = (uint64_t)slot * BLOCK_M * BLOCK_N +
                           (uint64_t)rowBase * BLOCK_N;
        DataCopy(wsPGm_[pOffset], pHalf, tileSize);
        outputQue1_.FreeTensor(pHalf);
        inputQue1_.FreeTensor(sUb);
        PipeBarrier<PIPE_MTE3>();

        // Signal AIC: S slot free (FREE_S) - MM1 can write next S
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_VEC1_MM1);

        // Signal AIC: P data ready (READY_P) - MM2 can read P
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_VEC1_MM2);
    }

    // ================================================================
    // Vec2: Rescale accumulated O and add new contribution
    // Uses workspace GM for accumulated O (not persistent UB)
    // Processes in M-chunks of VEC2_M_CHUNK rows
    // ================================================================
    __aicore__ inline void ComputeVec2(int t, int slot, int loop, bool isFirst, bool isLast)
    {
        // Wait for AIC BMM2 to write O_tmp (READY_O)
        CrossCoreWaitFlag<0x2>(SYNC_MM2_VEC2);

        uint32_t dim = dimAlign_;

        // Process HALF_M rows in chunks of VEC2_M_CHUNK
        uint32_t mChunk = VEC2_M_CHUNK;
        uint32_t numChunks = HALF_M / mChunk;
        uint32_t tailChunk = HALF_M % mChunk;

        for (uint32_t ci = 0; ci < numChunks + (tailChunk > 0 ? 1 : 0); ci++) {
            uint32_t startRow = ci * mChunk;
            uint32_t dealRows = (ci < numChunks) ? mChunk : tailChunk;
            uint32_t chunkSize = dealRows * dim;
            uint32_t rowOffset = vid_ * HALF_M + startRow;
            uint64_t stateRowBase = (uint64_t)loop * BLOCK_M + rowOffset;

            // Load new O_tmp (= P @ V) from workspace_o for this chunk.
            // MM2 writes ws_o in row-major [BLOCK_M, dimAlign] layout, so each chunk is contiguous.
            LocalTensor<float> oNewUb = inputQue1_.AllocTensor<float>();
            uint64_t oOffset = (uint64_t)slot * BLOCK_M * dim + (uint64_t)rowOffset * dim;
            DataCopy(oNewUb, wsOGm_[oOffset], chunkSize);
            inputQue1_.EnQue(oNewUb);
            oNewUb = inputQue1_.DeQue<float>();
            event_t evONewMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evONewMte2V);
            WaitFlag<HardEvent::MTE2_V>(evONewMte2V);

            // Vec1 stores wsMeta via MTE3; Vec2 reads it back via MTE2 in later
            // pipeline iterations, so the GM read-after-write edge needs an
            // explicit MTE3 -> MTE2 sync.
            event_t evMetaMte3Mte2 = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(evMetaMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evMetaMte3Mte2);
            DataCopy(softmaxExpUb_, wsMetaGm_[2 * kvLoops_ * BLOCK_M + stateRowBase], dealRows);
            DataCopy(softmaxSumUb_, wsMetaGm_[kvLoops_ * BLOCK_M + stateRowBase], dealRows);
            event_t evMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(evMte2S);
            WaitFlag<HardEvent::MTE2_S>(evMte2S);

            if (!isFirst) {
                // Sync: wait for previous iteration's MTE3 store to complete before MTE2 load
                event_t evMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(evMte3Mte2);
                WaitFlag<HardEvent::MTE3_MTE2>(evMte3Mte2);

                // Load previous accumulated O from ws_acc_o
                LocalTensor<float> oPrevUb = inputQue1_.AllocTensor<float>();
                uint64_t accOffset = ((uint64_t)(loop - 1) * BLOCK_M + rowOffset) * dim;
                DataCopy(oPrevUb, wsAccOGm_[accOffset], chunkSize);
                inputQue1_.EnQue(oPrevUb);
                oPrevUb = inputQue1_.DeQue<float>();

                // RowMuls: oPrev[row, :] *= exp_correction[row]
                RowMulsImpl(oPrevUb, oPrevUb, softmaxExpUb_, dealRows, dim);
                PipeBarrier<PIPE_V>();

                // Accumulate: oNew += oPrev_rescaled
                Add(oNewUb, oNewUb, oPrevUb, chunkSize);
                PipeBarrier<PIPE_V>();

                inputQue1_.FreeTensor(oPrevUb);
            }

            if (isLast) {
                // Final: divide by l_final (sumexp)
                RowDivsImpl(oNewUb, oNewUb, softmaxSumUb_, dealRows, dim);
                PipeBarrier<PIPE_V>();

                // Cast and write to output
                FinalizeOutputChunk(oNewUb, startRow, dealRows);
            } else {
                // Store intermediate accumulated O to ws_acc_o
                PipeBarrier<PIPE_V>();
                LocalTensor<float> oOutUb = outputQue1_.AllocTensor<float>();
                DataCopy(oOutUb, oNewUb, chunkSize);
                outputQue1_.EnQue(oOutUb);
                oOutUb = outputQue1_.DeQue<float>();

                uint64_t accOutOffset = ((uint64_t)loop * BLOCK_M + rowOffset) * dim;
                DataCopy(wsAccOGm_[accOutOffset], oOutUb, chunkSize);
                outputQue1_.FreeTensor(oOutUb);
                PipeBarrier<PIPE_MTE3>();
            }

            inputQue1_.FreeTensor(oNewUb);
        }

        // Signal AIC: O slot free (FREE_O) - MM2 can write next O
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_VEC2_MM2);
    }

private:
    // RowMuls: dst[row, :] = src[row, :] * scale[row]
    __aicore__ inline void RowMulsImpl(LocalTensor<float> &dst, LocalTensor<float> &src,
                                       LocalTensor<float> &scale,
                                       uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; row++) {
            float alpha = scale.GetValue(row);
            Muls(dst[row * cols], src[row * cols], alpha, cols);
        }
    }

    // RowDivs: dst[row, :] = src[row, :] / scale[row]
    __aicore__ inline void RowDivsImpl(LocalTensor<float> &dst, LocalTensor<float> &src,
                                       LocalTensor<float> &scale,
                                       uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; row++) {
            float inv = 1.0f / scale.GetValue(row);
            Muls(dst[row * cols], src[row * cols], inv, cols);
        }
    }

    // Write final output chunk: cast float32 -> half and copy to GM
    __aicore__ inline void FinalizeOutputChunk(LocalTensor<float> &oUb,
                                                uint32_t startRow, uint32_t dealRows)
    {
        uint32_t actualDim = tiling_.dim;
        uint32_t dim = dimAlign_;
        uint32_t seqLen = tiling_.seqLen;

        LocalTensor<half> outHalf = outputQue1_.AllocTensor<half>();

        if (dim == actualDim) {
            // Contiguous: cast entire chunk at once
            Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);
            PipeBarrier<PIPE_V>();

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.heads * seqLen +
                                (uint64_t)curBy_ * seqLen +
                                (uint64_t)curBx_ * BLOCK_M +
                                (uint64_t)vid_ * HALF_M +
                                (uint64_t)startRow) * actualDim;
            DataCopy(outGm_[outBase], outHalf, dealRows * actualDim);
        } else {
            // Non-contiguous: cast row by row to skip alignment padding
            for (uint32_t i = 0; i < dealRows; i++) {
                Cast(outHalf[i * actualDim], oUb[i * dim],
                     RoundMode::CAST_ROUND, actualDim);
            }
            PipeBarrier<PIPE_V>();

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.heads * seqLen +
                                (uint64_t)curBy_ * seqLen +
                                (uint64_t)curBx_ * BLOCK_M +
                                (uint64_t)vid_ * HALF_M +
                                (uint64_t)startRow) * actualDim;
            for (uint32_t i = 0; i < dealRows; i++) {
                DataCopy(outGm_[outBase + i * actualDim],
                         outHalf[i * actualDim], actualDim);
            }
        }

        PipeBarrier<PIPE_MTE3>();
        outputQue1_.FreeTensor(outHalf);
    }

public:
    // Store current block coordinates for Finalize
    int curBz_, curBy_, curBx_;

private:
    FlashAttentionTiling tiling_;
    uint32_t vid_;
    uint32_t dimAlign_;
    uint32_t kvLoops_;

    // GM workspace pointers
    GlobalTensor<float> wsSGm_;
    GlobalTensor<half> wsPGm_;
    GlobalTensor<float> wsOGm_;
    GlobalTensor<float> wsMetaGm_;
    GlobalTensor<float> wsAccOGm_;   // accumulated O workspace
    GlobalTensor<half> outGm_;

    // Shared queues (used by both Vec1 and Vec2)
    TQue<TPosition::VECIN, 2> inputQue1_;
    TQue<TPosition::VECOUT, 1> outputQue1_;

    // Temp buffer for SoftmaxFlashV2
    TBuf<TPosition::VECCALC> tmpBuf_;

    // Softmax state for current iteration
    TBuf<TPosition::VECCALC> softmaxMaxBuf_;
    TBuf<TPosition::VECCALC> softmaxSumBuf_;
    TBuf<TPosition::VECCALC> softmaxExpBuf_;
    LocalTensor<float> softmaxMaxUb_;
    LocalTensor<float> softmaxSumUb_;
    LocalTensor<float> softmaxExpUb_;

    // Default initial values
    TBuf<TPosition::VECCALC> softmaxMaxDefaultBuf_;
    TBuf<TPosition::VECCALC> softmaxSumDefaultBuf_;
    LocalTensor<float> softmaxMaxDefaultUb_;
    LocalTensor<float> softmaxSumDefaultUb_;

    // Brcb expansion buffer (AlignUp(HALF_M, 8) datablocks, 8 fp32 per datablock)
    TBuf<TPosition::VECCALC> brcbBuf_;
};

#endif // FLASH_ATTENTION_VEC_H
