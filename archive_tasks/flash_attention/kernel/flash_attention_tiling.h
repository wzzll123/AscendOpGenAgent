#ifndef FLASH_ATTENTION_TILING_H
#define FLASH_ATTENTION_TILING_H

#include <cstdint>

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t PRELAUNCH = 2;
constexpr uint32_t RING_SLOTS = PRELAUNCH + 1;  // 3
constexpr uint32_t MAX_CORES = 20;              // Ascend910B3 AIC cores used by this kernel

// Cross-core sync signals (constexpr, not in tiling)
// Three data streams need bidirectional handshake (READY + FREE):
// S: MM1→Vec1, P: Vec1→MM2, O: MM2→Vec2
constexpr uint32_t SYNC_MM1_VEC1 = 0;  // Cube → Vec1: S ready (READY_S)
constexpr uint32_t SYNC_VEC1_MM1 = 1;  // Vec1 → Cube: S slot free (FREE_S)
constexpr uint32_t SYNC_VEC1_MM2 = 2;  // Vec1 → Cube: P ready (READY_P)
constexpr uint32_t SYNC_MM2_VEC1 = 3;  // Cube → Vec1: P slot free (FREE_P)
constexpr uint32_t SYNC_MM2_VEC2 = 4;  // Cube → Vec2: O ready (READY_O)
constexpr uint32_t SYNC_VEC2_MM2 = 5;  // Vec2 → Cube: O slot free (FREE_O)

#pragma pack(push, 8)
struct FlashAttentionTiling {
    int32_t batch;
    int32_t heads;
    int32_t seqLen;
    int32_t dim;
    int32_t blockM;    // 64
    int32_t blockN;    // 64
    float   smScale;   // 1/sqrt(dim)
};
#pragma pack(pop)

#endif // FLASH_ATTENTION_TILING_H
