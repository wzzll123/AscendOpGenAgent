## TileLang 设计转换到 AscendC Kernel 关键原则

本文档讨论如何将已经完成的 TileLang 设计，系统地转换为等价的 AscendC 实现。本文中的 DSL 特指 TileLang。后文中的 tiling、绑定层、主 kernel 类、子模块拆分与同步关系，都应以 TileLang 尤其是 tile-level 设计为直接来源。

1. **先看 Mapping，再查 API 文档**
   转换时，应先阅读 `@references/TileLang-AscendC-API-Mapping.md`，先确认 TileLang API 到 AscendC API 的映射关系，再去阅读 `@references/AscendC_knowledge/` 目录下对应的具体 API 文档。
   知识库入口：`api_reference/INDEX.md`

2. **禁止在 C++ 中直接调用 torch / ATen 计算接口**
   `pybind11.cpp` 中禁止使用 `torch::*`、`torch::nn::functional::*`、`ATen` 库或任何 `at::*` 计算接口来实现或替代核心计算，包括但不限于 `at::einsum`、`at::matmul`、`at::softmax`、`at::bmm` 等。绑定层只负责参数检查、输出与 workspace 分配、tiling 填充和 kernel launch，不允许把核心计算留在 C++/ATen 侧。
---

## TileLang 到 AscendC 转换总览

一个从 TileLang 设计转换得到的 AscendC 实现，通常包含 4 部分：

1. **Host 侧准备**：`xxx_tiling.h` + `pybind11.cpp`
   这两部分共同构成 AscendC 的 host 侧准备逻辑，通常需要结合 TileLang kernel 的 host 信息一起整理。`xxx_tiling.h` 负责定义 shape、block size、tile size、workspace 深度等参数；`pybind11.cpp` 负责 Python 接口、输入校验、输出与 workspace 分配、tiling 构造和 kernel launch。

2. **公共工具**：如 `kernel_common.h`、`workspace_queue.h`、`matmul_tile.h`, `vector_tile.h`
   提供调度、数据搬运、workspace 管理等通用能力。

3. **Kernel 入口**：`xxx.cpp`
   定义 `__global__ __aicore__` kernel 和 `extern "C"` launch 函数。

4. **主 Kernel 类与计算子模块**：一个或多个 `*.h`
   主 `Kernel` 类负责 `Init()` / `Process()` 主流程，管理 GM tensor、调度和流水。若 TileLang 中存在多个 `T.prim_func`，将对应的主 `Kernel` 类拆到多个独立头文件中，例如 `xxx_merge_n_kernel.h`、`xxx_single_row_kernel.h`。若算子属于 C/V 融合算子，或者 TileLang 设备侧存在多个有明确职责分工的 `Scope`，则可在这一部分下继续按计算阶段拆分子模块，例如 `matmul.h`、`leakyrelu.h`；通常每个 `Scope` 对应一个子模块，职责应与原 TileLang 设计中的计算阶段一一对应。对于纯 Vector 算子，或者虽然有 host / queue / buffer 管理但设备侧只有单个 Vector 计算阶段 / 单个 Vector `Scope` 的简单算子，主 `Kernel` 类本身通常就承载全部计算逻辑，不再额外拆分子模块。

### `T.prim_func` 到 AscendC Kernel 的映射规则

TileLang 中有多少个 `T.prim_func`，AscendC 侧就至少要有多少个独立的 kernel 实现单元。不要把多个 `T.prim_func` 折叠进同一个 `Kernel` 类里再靠运行时分支区分。

具体要求：

- 每个 `T.prim_func` 都应对应一个独立的主 `Kernel` 类。
- 每个 `T.prim_func` 都应对应至少一个独立的 `__global__ __aicore__` kernel 入口和一个匹配的 `extern "C"` launch 函数。
- 如果同一个 `T.prim_func` 需要按 dtype 分成多个实现，例如 fp16 / fp32 / int8，则可以在该 `prim_func` 之下派生出更多 `extern` 入口；但主 `Kernel` 类的个数仍应首先与 `T.prim_func` 的个数对齐。
- Host 侧 `pybind11.cpp` 负责根据 shape、dtype 或其他 trace-time 条件，选择调用哪个 `extern` 入口；这种选择逻辑不应反向合并掉 `prim_func` 级别的结构差异。

例如，若 TileLang 提供 `merge_n` 和 `single_row` 两个 `T.prim_func`，则 AscendC 至少应有两个主 `Kernel` 类、两个 `__global__ __aicore__` 入口和两个 `extern "C"` launch 函数；若两者还各自支持多种 dtype，则 `extern` 数量可以更多，但主 `Kernel` 类仍至少是两个。

---

## 第一章：Host 侧准备（摘要）

完整实现细节与代码示例见 `@references/dsl2Ascendc_host.md`。

**要点**：

- **Tiling 参数一致性**：所有 kernel 组件（Cube/Vector/Host）必须使用同一组 baseM/baseN/baseK 常量，参数不匹配会导致错误的内存访问。
- **Tiling Struct**：在 Host 侧预计算 `nTiles`、`nTilesPerH` 等派生量写入 tiling struct，避免 kernel 里重复除法。
- **绑定层职责**：`pybind11.cpp` 负责参数检查、输出分配、workspace 分配、tiling 构造、kernel launch。绑定函数只接收 DSL 显式输入张量，不接收输出和 workspace。
- **模块名**：推荐 `_<op_name>_ext`，不要与任务目录同名。
- **Workspace**：只要 DSL 声明了 workspace 参数或 `workspace_idx`，就必须分配 workspace。

---

## 第二章：公共工具（摘要）

### 1. tile 层公共工具：`matmul_tile.h` / `vector_tile.h`

这一类文件主要承载 tile 级的数据搬运、分块计算封装和局部流水组织，是把 DSL 里的 tile-level 设计落到 AscendC 时最常见的公共工具。

- 纯 Vector 算子：通常需要构建 `vector_tile.h` 一类工具，可以参考 `rms_norm` 的 kernel 写法，例如 `archive_tasks/rms_norm/kernel/` 下对 `vector_tile.h` 的使用
- 纯 Cube 算子：通常需要构建 `matmul_tile.h` 一类工具，可以参考 `matmul_leakyrelu` 的 kernel 写法，例如 `archive_tasks/matmul_leakyrelu/kernel/matmul_tile.h`
- C/V 融合算子：通常两类工具都要结合具体分工一起看

### 2. `workspace_queue.h` 与跨核同步

这一部分主要对应 AIC / AIV 之间通过 workspace 传递中间结果、并通过 cross-core flag 建立 producer / consumer 协同的场景，常见于 C/V 融合算子。

- 纯 Vector 算子和纯 Cube 算子：默认**不要阅读** `@references/dsl2Ascendc_cross_core_sync.md`
- C/V 融合算子，或 DSL 中出现跨核协同 / workspace 生产者-消费者关系 / cross-core flag：**必须阅读** `@references/dsl2Ascendc_cross_core_sync.md`

---

## 第三章：Kernel 入口（摘要）
**KERNEL_TYPE 与 DSL vec_num 对应关系**：

| DSL `vec_num` | KERNEL_TYPE | 每个 block 组成 |
|:---|:---|:---|
| 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV |
| 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV |

**代码结构要求**：
- `Process()` 封装工作负载循环，调用 `CopyInX` / `ComputeX` / `CopyOutX`
- 每个阶段函数定义为 `__aicore__ inline`

---

## 第四章：主 Kernel 类与计算子模块（摘要）

完整实现细节与代码示例见：

- 纯 Vector 算子：`@references/dsl2Ascendc_compute_vector.md`
- 纯 Cube 算子：`@references/dsl2Ascendc_compute_cube.md`
- C/V 融合算子：先看 `@references/dsl2Ascendc_compute_cv.md`，再结合 `@references/dsl2Ascendc_compute_cube.md` 和 `@references/dsl2Ascendc_compute_vector.md`

### UB tmp / workBuf 规则

TileLang `T.reduce_*`、`T.tile.broadcast` 在 DSL 层不显式传 `tmp`，但 AscendC 底层 API 往往仍需要 UB `tmp` / `workBuf`。转译时不要照抄 TileLang 签名省略临时 buffer。

- 默认用 `AscendC::TBuf<AscendC::TPosition::VECCALC>` 申请，并在使用时取成 `AscendC::LocalTensor<uint8_t>` 或对应 dtype。
- 初次转译优先用保守方案，不要让 `reduce`、`broadcast`、cast buffer 和输入输出 queue 过早复用同一块 UB。
- 对 shape-based `Broadcast` / `Reduce`，优先按 `2 * logical_tile_elems * sizeof(uint8_t)` 申请。
- 对 count-based `ReduceSum(dst, src, workBuf, count)`，优先按 `count * sizeof(acc_dtype)` 申请；`float` accumulate 时通常是 `count * sizeof(float)`。
- 同一阶段若同时有 broadcast 和 reduce，默认分别申请独立 tmp。

参考 `archive_tasks/rms_norm/kernel`：

- `rms_norm_merge_n_kernel.h` 中 `gammaBroadcastTmpBuf_`、`scaleBroadcastTmpBuf_`、`reduceBuf_` 都按 `2 * tile_elems * sizeof(uint8_t)` 申请。
- `rms_norm_splitd_kernel.h` 中 `reduceBuf_` 按 `count * sizeof(float)` 申请，`tempBuf_` 单独保存 `x * x` 中间结果。

**常见陷阱速查表**：

| 问题 | 症状 | 解决方法 |
|:---|:---|:---|
| local UB buffer 该用 `TQue` 却写成 `TBuf` | 结果系统性错误或生命周期错乱 | 在 `T.serial` 中的输入/输出 buffer 用 `TQue` |
| TQue depth=0 用了返回值形式 API | 编译报错 | VECIN/VECOUT 必须用引用形式 |
| TBuf DataCopy 后缺少 PipeBarrier | 结果随机错误 | 插入 `PipeBarrier<PIPE_MTE2>()` |
| Fixpipe 未同步 | 流同步超时 | CrossCoreSetFlag 使用 `PIPE_FIX` |
| 内层循环中重复 DeQue | 结果错误 | 每个外层迭代只 DeQue 一次 |
| Fixpipe dstStride = baseN | tile 间数据覆盖 | dstStride 应设为完整行宽 N |
| WholeReduceMax 参数错误 | 编译报错 | 改用 `ReduceMax(dst, src, workBuf, count)` |
| 使用不存在的 Divs | 编译报错 | 改用 `Muls(dst, src, 1.0f/scaleVal, count)` |
