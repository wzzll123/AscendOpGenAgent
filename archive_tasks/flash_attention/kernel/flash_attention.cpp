#include "flash_attention_kernel.h"

extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    FlashAttentionKernel kernel;
    kernel.Init(q, k, v, output, workspace, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void flash_attention_do(uint32_t blockDim, void *stream,
                                   uint8_t *q, uint8_t *k, uint8_t *v,
                                   uint8_t *output, uint8_t *workspace,
                                   uint8_t *tiling)
{
    flash_attention_custom<<<blockDim, nullptr, stream>>>(q, k, v, output, workspace, tiling);
}
#endif
