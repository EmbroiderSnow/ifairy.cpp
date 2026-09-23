# iFairy 推理代码盘点与独立仓库整理

## 来源与边界

- 源仓库：`/Users/1806-admin/Projects/llama.cpp`。
- 源提交：`fe79b5dc5a449922caa64a37451948f30c124318`。
- 独立仓库：`/Users/1806-admin/Projects/llama.cpp-ifairy`，分支 `codex/ifairy-only`。
- 按原始 `general.architecture=ifairy` 整理。保留 CPU（通用、ARM/NEON 和现有 x86 条件编译路径）；未移植 GPU 支持。
- 源仓库的三个未跟踪 Row4/Nemotron 文件未复制、未修改。
- 本次是源码拆分和可运行性验证，没有进行性能优化或宣称速度提升。

## 原仓库还保留了什么

| 层次 | 原有实现位置 | 本次处理 |
| --- | --- | --- |
| 架构、元数据与张量名 | `src/llama-arch.*`、`src/llama-model-loader.cpp`、`src/llama-model.cpp` | 保留 iFairy；加载器只允许 `ifairy` |
| 完整 transformer 图 | `llm_build_ifairy` | 保留 Q/K/V、复数 RoPE、attention、post-attention norm、ReLU² FFN、残差、最终输出；其他架构构造器删除 |
| 复数图辅助 | `src/llama-graph.cpp` 的 `ifairy_build_norm/ffn/attn` | 保留 |
| 存储、量化、反量化、索引编码 | `ggml/src/ggml-common.h`、`ggml-quants.*`、`ggml.c` | 保留 `IFAIRY=40`、`IFAIRY_Q16=41` 及 legacy 64-value 兼容类型；不重编号现有 GGUF 类型 |
| CPU matmul 与量化策略 | `ggml/src/ggml-cpu/ggml-cpu.c`、`legacy-ifairy/legacy-ifairy-cpu.cpp` | 保留 vecdot 和 tensor-scale 激活策略 |
| ARM 内核 | `ggml/src/ggml-cpu/legacy-ifairy/arm/` | 原有 iFairy NEON/dotprod 文件保留 |
| LUT、打包与兼容 W2 | `ggml/src/ggml-cpu/legacy-ifairy/lut/`、`wide-linear*`、`lut-qgemm*` | 保留历史 iFairy 算子及单测；普通 matmul 的 LUT 分派缺口见下文 |
| 复数基础算子 | `ggml/src/ggml-cpu/ops.cpp`、`binary-ops.cpp`、`unary-ops.cpp` | 保留 split/merge、add/mul、norm、ReLU²、RoPE |
| GGUF 转换 | `gguf-py/convert_ifairy.py`、gguf 包中的 iFairy 常量和映射 | 保留原始入口；删除 Fairy2i 转换器及共享的 Fairy2i Python 包 |
| 验证 | `test-legacy-ifairy-direct.cpp`、`test-legacy-ifairy.cpp`、`test-ifairy-ref.py`、backend-op 用例 | 保留；新增完整模型加载与推理隔离回归 |

原始 iFairy 与 Fairy2i 共用过部分基础设施。iFairy 专属目录并不意味着所有 iFairy 逻辑都已独立：量化、CPU 分派和复数算子仍散布于公共 ggml 文件，已逐一移除其中的 Fairy2i 专用部分。

## 模型契约

- 元数据前缀 `ifairy`：context length、embedding length、block count、feed-forward length、attention head count、KV head count、`attention.layer_norm_epsilon`、vocab size。RoPE 参数继续使用原有默认值与可选 GGUF 覆盖。
- `token_embd` 是以 F32 容器承载的 BF16 实部/虚部对；普通量化线性权重为 `GGML_TYPE_IFAIRY`，每块 256 个复数；norm 权重为实部/虚部展开后的 F32；output 通常为 F16。
- 设复数 embedding 宽度为 E、FFN 宽度为 F：token embedding `[E,V]`，output norm `[2E]`，output `[2E,V]`，Q/O `[E,E]`，K/V `[E,E*n_head_kv/n_head]`，FFN gate/up `[E,F]`，down `[F,E]`。形状按 ggml 的 ne 顺序书写。
- 默认每头实数展开宽度是 `2*E/n_head`，不能在拆分共享分支时丢掉倍数 2。
- 量化 matmul 保持 `w * conj(x)`，不能替换为普通复数 `w*x`。
- 合成 fixture 覆盖 prefill、多次单 token decode、KV cache 复用、同路径重复一致性；使用非零确定性权重，不代表语言质量。

## 删除了什么

1. `LLM_ARCH_FAIRY2I`、对应的 GGUF schema / attention-layout / numeric-profile / tensor 加载分支，以及 Fairy2i 图构造器。
2. Fairy2i tile64_v2 / ACT_Q16_64 / bundle-codes 类型实现、专用 W1/W2、exact BF16 norm/RoPE/SiLU/mul/attention 算子、CPU 模块和构建开关。
3. Fairy2i Python 转换、校验、测试包，以及混用 Fairy2i Python 模块的旧 `legacy_ifairy` 量化导出包；原始 `convert_ifairy.py` 不依赖这个包。
4. Metal、OpenCL、CUDA 等后端实现目录；当前原始 iFairy 本就没有完整 GPU 路由。
5. 其他模型的计算图构造器、示例、server/多模态等无关应用与历史性能报告。通用 ggml 类型、tokenizer、采样及共享 API 元数据保留，它们不构成其他模型的可加载推理实现。

生产目录 `src/ include/ common/ ggml/ gguf-py/` 没有 `fairy2i` 引用；这个字符串仅出现在说明、源码审计脚本和负向测试中。编译出的动态库导出符号也进行了同样检查。

## 还剩下的工作 / 已知限制

### 普通 iFairy 的 LUT 执行入口需要接回

这是源提交已有的状态，不是此次拆分新增的行为：

- `ggml_legacy_ifairy_cpu_supports_op()` 只接受 `GGML_OP_IFAIRY_WIDE_LINEAR_W2`。
- `ggml_legacy_ifairy_cpu_prepare_graph()` 对普通 MUL_MAT 做资格判断与权重预打包。
- `ggml_compute_forward_mul_mat()` 的实际执行仍使用 vecdot，没有调用 iFairy LUT QGEMM。
- 因而 `GGML_IFAIRY_LUT=1`、`ifairy_lut: can_mul_mat=true` 和 LUT 内核单测通过都不能用作整模型 LUT 加速证据。

本版保留此兼容行为，默认使用已验证的直接路径。若下一步接入 LUT，需要独立完成工作区大小、线程分片、激活预处理与执行分派的设计，再用真实模型做 correctness 和顺序运行的 benchmark 验证。

### 真实权重、质量与平台验证

- 本地所检查的 `models/` 和 `/Users/1806-admin/Checkpoints` 中未找到原始 iFairy 推理权重。未运行真实 700M checkpoint 的文本生成、困惑度或吞吐基线。
- 没有测试 x86/AVX512、Linux 或 Android 实机；保留了现有条件编译和通用回退代码。
- 转换脚本仍从当前工作目录读取 tokenizer，且保留原有分片/名称映射行为；需要真实 checkpoint 才能确认全流程转换兼容性。
- 历史 `test-legacy-ifairy` 中基础 RoPE 和复数矩阵测试有简化检查；不能只看其 PASS。另用 backend-op 参考比较及新增完整模型 fixture 补充验证。
- clang-tidy 按原配置运行，没有错误退出，但有现有共享代码和风格告警；没有做无关全库整改。

## 验证结果

环境：Apple M5 Max / macOS arm64，Apple clang 21.0.0，CMake Release (`-O3 -DNDEBUG`)；CPU native flags 包含 `-mcpu=native+dotprod+i8mm+nosve+sme`。构建 `-j 8`；合成模型回归使用 2 个推理线程。没有运行吞吐基准。

| 验证 | 结果 | 原始记录 |
| --- | --- | --- |
| Release + LUT 完整构建 | 通过，含 CLI/bench/perplexity/微基准和测试 | `validation/build-lut.log` |
| Release 直接路径完整构建 | 通过 | `validation/build-direct.log` |
| LUT 构建 CTest | 5/5 通过 | `validation/ctest-lut.log` |
| 直接路径 CTest | 3/3 通过 | `validation/ctest-direct.log` |
| 合成模型加载 / prefill / decode / 重复性 | 通过，批量与串行最终 logits 最大差值 0 | `validation/model.log` |
| 拒绝非 iFairy 架构 | fairy2i、llama、qwen3、unknown 均拒绝 | `validation/model.log` |
| 生产代码与导出符号隔离 | 通过 | `validation/isolation.log` |
| Python gguf 包导入与 iFairy 类型映射 | 通过；未运行真实转换 | `validation/python-gguf.log` |
| git clang-format | 无需修改 | `validation/clang-format.log` |
| clang-tidy（13 个修改/新增 C/C++ 源文件） | 所有退出码 0；保留告警 | `validation/clang-tidy-summary.json` 及对应日志 |

合成模型的测试数据在运行中临时生成并清理。源仓库未修改。交付仓库不包含真实权重、旧 GPU/Fairy2i 源码或其他模型的计算图构造器。
