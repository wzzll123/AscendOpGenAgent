#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "flash_attention_tiling.h"

extern "C" void flash_attention_do(uint32_t blockDim, void *stream,
                                   uint8_t *q, uint8_t *k, uint8_t *v,
                                   uint8_t *output, uint8_t *workspace,
                                   uint8_t *tiling);

static inline uint32_t AlignUpHost(uint32_t num, uint32_t align)
{
    return (num + align - 1) / align * align;
}

namespace my_flash_attention {

at::Tensor run_flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    // Input validation
    TORCH_CHECK(q.dim() == 4, "q must be 4D [batch, heads, seqLen, dim]");
    TORCH_CHECK(k.dim() == 4, "k must be 4D [batch, heads, seqLen, dim]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [batch, heads, seqLen, dim]");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "v must be float16");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q.sizes()[2] % BLOCK_M == 0, "seqLen must be divisible by BLOCK_M");
    TORCH_CHECK(q.sizes()[2] % BLOCK_N == 0, "seqLen must be divisible by BLOCK_N");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    uint32_t batch = q.sizes()[0];
    uint32_t heads = q.sizes()[1];
    uint32_t seqLen = q.sizes()[2];
    uint32_t dim = q.sizes()[3];
    uint32_t dimAlign = AlignUpHost(dim, 16);  // C0=16 for half

    int totalBlocks = (seqLen / BLOCK_M) * heads * batch;
    uint32_t usedCoreNum = totalBlocks < MAX_CORES ? totalBlocks : MAX_CORES;

    // Allocate output
    at::Tensor output = at::zeros_like(q);

    // Fill tiling struct
    at::Tensor t = at::empty({(int64_t)sizeof(FlashAttentionTiling)},
                             at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<FlashAttentionTiling *>(t.data_ptr());
    tiling_ptr->batch = batch;
    tiling_ptr->heads = heads;
    tiling_ptr->seqLen = seqLen;
    tiling_ptr->dim = dim;
    tiling_ptr->blockM = BLOCK_M;
    tiling_ptr->blockN = BLOCK_N;
    tiling_ptr->smScale = 1.0f / sqrtf((float)dim);
    auto tiling_npu = t.to(at::kPrivateUse1);

    // Compute workspace size per core
    uint32_t kvLoops = seqLen / BLOCK_N;
    uint64_t wsSSize = (uint64_t)kvLoops * BLOCK_M * BLOCK_N * sizeof(float);
    uint64_t wsPSize = (uint64_t)kvLoops * BLOCK_M * BLOCK_N * sizeof(uint16_t);  // half
    uint64_t wsOSize = (uint64_t)kvLoops * BLOCK_M * dimAlign * sizeof(float);
    uint64_t wsMetaSize = (uint64_t)kvLoops * BLOCK_M * 3 * sizeof(float);
    uint64_t wsAccOSize = (uint64_t)kvLoops * BLOCK_M * dimAlign * sizeof(float);
    uint64_t perCoreBytes = wsSSize + wsPSize + wsOSize + wsMetaSize + wsAccOSize;
    uint64_t totalWsBytes = perCoreBytes * usedCoreNum;

    at::Tensor w = at::zeros({(int64_t)totalWsBytes},
                             at::device(at::kPrivateUse1).dtype(at::kByte));

    // Ensure workspace is properly initialized before kernel launch
    c10_npu::getCurrentNPUStream().synchronize();

    // Launch kernel
    flash_attention_do(usedCoreNum, acl_stream,
                       reinterpret_cast<uint8_t*>(q.data_ptr<at::Half>()),
                       reinterpret_cast<uint8_t*>(k.data_ptr<at::Half>()),
                       reinterpret_cast<uint8_t*>(v.data_ptr<at::Half>()),
                       reinterpret_cast<uint8_t*>(output.data_ptr<at::Half>()),
                       reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>()),
                       reinterpret_cast<uint8_t*>(tiling_npu.data_ptr<uint8_t>()));

    return output;
}

} // namespace my_flash_attention

PYBIND11_MODULE(flash_attention_ascendc, m)
{
    m.doc() = "Flash Attention AscendC kernel";
    m.def("run_flash_attention", &my_flash_attention::run_flash_attention,
          "Flash Attention forward (Q, K, V) -> Output");
}
