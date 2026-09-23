# 本机 x86 iFairy 700M 推理与 LUT 验证报告

测试日期：2026-09-23，Asia/Shanghai。接入前基线：`a6256d61ffd3ebe4cba06fdb3b46e0b7eb78feaa`；第 0 节补测对应的 LUT 修复及测试源码归档于提交 `a243488`。

## 0. LUT 接入后补测（当前结论）

同日后续已补齐普通二维 IFAIRY `MUL_MAT` 的 LUT 执行分派、工作区和激活预处理。当前实现已实际执行 LUT；下方第 1 节起保留的是接入前记录，其中“只预打包”“开关前后 logits 相同”不是当前结论。完整改动与两平台复测见 [LUT 接入报告](LUT_MUL_MAT_INTEGRATION_REPORT_2026-09-23.md)。

本机直接构建 CTest **3/3**、LUT 构建 **6/6** 通过；新回归验证 144 次计算图执行与资源缺失回退，内存/未定义行为检查复测通过。真实模型的 prefill/decode 共 336 次矩阵乘法、3,006 个抽样分量通过独立双精度复数参考检查。

7 个真实模型配置累计检查 **14,784,000** 个有限 logits。LUT auto 的独立重复、lut16 对照、批量/串行 prefill 完整 trace 均逐字节相同；832-token prefill + 256-token 生成完成。LUT 专用量化与默认直接路径不同，二者输出不逐位相同；相同行级量化的直接对照也存在整模型差异。未验证语料质量或 PyTorch 等价性。

| 线程 / 模式 | pp128 (token/s) | pp512 (token/s) | tg128 (token/s) |
| --- | ---: | ---: | ---: |
| 4 / 默认直接 | 49.46 ± 1.62 | 47.24 ± 0.14 | 35.57 ± 0.27 |
| 4 / LUT auto | 143.97 ± 1.75 | 148.99 ± 1.27 | 71.17 ± 1.91 |
| 8 / 默认直接 | 70.64 ± 0.97 | 67.74 ± 2.41 | 46.61 ± 0.99 |
| 8 / LUT auto | 216.24 ± 4.10 | 198.88 ± 15.91 | 76.15 ± 0.44 |
| 8 / 行级量化直接对照 | 89.77 ± 9.19 | 72.33 ± 3.39 | 43.55 ± 0.58 |

8 线程的默认直接与 LUT auto 对照，pp128/pp512 约 **3.06×/2.94×**，tg128 约 **1.63×**；这是不同量化配置的整体吞吐比较。auto 峰值 RSS **1,681.29 MiB**，默认直接 **1,170.93 MiB**。详细条件和限制见新报告。

## 1. 接入前结论（历史记录）

仓库仍有完整的 x86 CPU 推理实现，包括 iFairy AVX2 vecdot、LUT QGEMM 和通用回退。本机已成功构建并运行真实 700M GGUF。默认直接构建通过 3/3 项回归，LUT 构建通过 5/5 项回归；模型加载、文本生成、KV cache 复用以及固定配置的重复运行均完成。

**LUT 编译开关和运行时开关已开启，但普通 iFairy 模型的矩阵乘法仍执行 vecdot。** 当前源码只为普通 `MUL_MAT` 做 LUT 资格判断和权重预打包，未接通 LUT QGEMM 执行分派。下文整模型吞吐不能作为 LUT 加速结果；独立 LUT 内核另有测试证据。

**默认 LLAMAFILE 路径存在真实模型的批量/串行预填充数值差异。** 对照构建关闭 `GGML_LLAMAFILE`、保持 LUT 开启后，本次两组提示词的预填充和后续 32 步生成均恢复一致。建议本机需要稳定复现时使用文末 `build-lut-reference` 配置。本报告确认的是所列用例的运行和数值检查结果，不作所有输入、全部模型质量或 PyTorch 等价性的保证。

本次未修改推理 C/C++ 源码、模型准入规则、packed BF16 复数表示或 `w * conj(x)` 语义，未新增模型架构或 GPU 后端，未复制或提交真实模型权重。

## 2. 源码检查：x86 实现与实际 LUT 路径

| 项目 | 源码位置及检查结果 |
| --- | --- |
| x86 后端纳入构建 | [CPU CMakeLists](../ggml/src/ggml-cpu/CMakeLists.txt)，约 285 行，加入 `arch/x86/quants.c` 和 `repack.cpp`；本机使用 `-march=native` |
| iFairy AVX2 直接点积 | [x86/quants.c](../ggml/src/ggml-cpu/arch/x86/quants.c)，`ggml_vec_dot_ifairy_q16_K`，约 1473 行；`__AVX2__` 下使用 256 位 SIMD，否则调用 generic 回退 |
| 复数约定和存储 | 同函数约 1697–1702 行：实部为 `ac + bd`，虚部为 `bc - ad`，输出打包成两个 BF16 |
| x86 LUT 内核 | [ggml-ifairy-lut-qgemm.cpp](../ggml/src/ggml-cpu/legacy-ifairy/lut/ggml-ifairy-lut-qgemm.cpp)，约 679、1005 行起保留 AVX2 shuffle/累加内核，也保留 ARM 与标量路径 |
| 可选 legacy AVX512 | [wide-linear.cpp](../ggml/src/ggml-cpu/legacy-ifairy/wide-linear.cpp)，约 127、213 行有 W2 AVX512 条件分支；本次 `GGML_LEGACY_IFAIRY_CPU_AVX512=OFF`，未验证此独立开关 |
| 普通模型的 LUT 缺口 | [legacy-ifairy-cpu.cpp](../ggml/src/ggml-cpu/legacy-ifairy/legacy-ifairy-cpu.cpp) 的 `supports_op()` 仅接受 `IFAIRY_WIDE_LINEAR_W2`；`prepare_graph()` 为普通 `MUL_MAT` 预打包；[ggml-cpu.c](../ggml/src/ggml-cpu/ggml-cpu.c) 的 `ggml_compute_forward_mul_mat()` 仍调用 vecdot |
| 模型隔离 | `scripts/check-ifairy-only.py` 通过；回归测试仍拒绝 `fairy2i`、`llama`、`qwen3` 和 `unknown` |

LUT 调试运行出现 `ifairy_lut: can_mul_mat=true`，它只证明资格检查成功。F16 输出层等普通浮点张量出现 `type mismatch` 表示不适用 iFairy LUT，并非模型加载失败。见 [调试日志](validation/local-x86-20260923/cli-lut-debug.stderr.log) 和 [实际编译命令](validation/local-x86-20260923/compile-evidence.json)。

## 3. 本机、模型与构建

| 项目 | 实测配置 |
| --- | --- |
| CPU | AMD Ryzen 7 H 255 w/ Radeon 780M Graphics，8 核 / 16 线程 |
| 指令集 | 支持 AVX2、FMA、AVX512；运行日志报告 AVX2、AVX512、AVX512_VBMI/VNNI/BF16 可用 |
| 系统 | Arch Linux x86_64，Linux `7.2.3-arch1-3` |
| 内存 | 约 30 GiB RAM，无 swap |
| 工具链 | GCC `16.2.1 20260810`，CMake `3.22.1-g37088a8`，Python `3.14.7` |
| 编译 | Release，`-O3 -DNDEBUG -march=native`，OpenMP 开启，ccache 关闭，`-j 8` |
| 后端 | CPU，`-ngl 0`；本机 Radeon GPU 未参与推理 |
| 模型 | `/home/zybi/projects/Fairy-plus-minus-i-700M/ifairy.gguf`，已有 GGUF，未重新转换 |
| GGUF | `general.architecture=ifairy`，267 个张量：168 IFAIRY、98 F32、1 F16 |
| 模型形状 | 24 层，复数 embedding 1536，FFN 4096，16 heads，词表 32000，训练 context 2048 |
| 模型大小 | 文件 576,123,520 字节；加载器 tensor buffer 548.73 MiB；模型标称 700M，加载器按 GGUF 张量元素统计 827.35 M |

模型 SHA-256：

```text
2211f40ec7dd56ed424f4bcaccc65869b8deffbbcea65e84806b619e380da6b3
```

| 构建目录 | LUT 编译 | LLAMAFILE | 验证 |
| --- | --- | --- | --- |
| `build-rel` | OFF | ON | 完整构建；CTest 3/3；真实模型直接路径 |
| `build-lut` | ON | ON | 完整构建；CTest 5/5；LUT 开关及实现选项对照 |
| `build-lut-reference` | ON | OFF | 完整构建；CTest 5/5；关闭 LLAMAFILE 的单变量对照与本机复现配置 |

首次直接/LUT 构建分别用时 77.607 / 77.923 秒，不含配置；配置分别为 1.608 / 1.024 秒。编译存在已有的 packed BF16 指针别名告警和测试代码枚举转换告警，但没有编译错误。LUT 配置时强制关闭其他后端的 CMake 提示符合 CPU-only 约束。

## 4. 正确性、文本生成和边界

### 回归与真实 logits

回归覆盖直接 vecdot、复数基础算子、LUT 打包/内核、合成模型加载、prefill/decode、重复性和架构隔离。两个标准构建的合成模型批量/串行 logits 最大差值均为 `0.00000000`。完整输出见 [direct CTest](validation/local-x86-20260923/ctest-direct-detailed.log) 和 [LUT CTest](validation/local-x86-20260923/ctest-lut-detailed.log)。

真实模型额外使用 [check_logits.py](validation/local-x86-20260923/check_logits.py) 直接调用当前 `libllama.so`。每组配置独立进程，8 个推理线程、context 2048、batch/ubatch 512、F16 KV、关闭 Flash Attention。两个提示词分别为 `The capital of France is`（5 tokens）和 `Once upon a time, in a small village`（9 tokens）。每个提示词检查 prefill 后以及 32 次 greedy decode 后的完整词表 logits：`2 × 33 × 32000 = 2,112,000` 个浮点值。

| 检查 | 结果 |
| --- | --- |
| direct 构建与 LUT 构建运行时关闭 | 两组提示词的全部 logits 字节流 SHA-256 和生成 token IDs 一致 |
| LUT 关闭与开启 `auto` | 全部一致；所有 logits 有限，无 NaN/Inf |
| LUT `auto` 独立进程重复运行 | 全部一致 |
| LUT `auto`、`lut16`、`lut_c` | 全部一致；普通 matmul 仍执行同一 vecdot 路径 |
| 逐 token prefill 下 LUT 关闭/开启 | 全部一致 |
| 默认 LLAMAFILE：批量与串行 prefill | 法国首都提示词一致；故事提示词有差异，详见下表 |
| 关闭 LLAMAFILE：批量与串行 prefill | 两组提示词完整 logits 哈希及生成 token IDs 一致；prefill 最大绝对差值均为 0 |
| 关闭 LLAMAFILE：LUT 关闭/开启 | 两组提示词全部 logits 与生成 token IDs 一致 |

对故事提示词，默认构建在**同一提示词、尚未发生生成分歧时**的 32000 个 prefill logits 差异如下，不能用后续不同 token 序列的 logits 代替该比较：

| 指标 | 值 |
| --- | ---: |
| 最大绝对差值 | 0.2316303253 |
| 平均绝对差值 | 0.0433072086 |
| RMS 差值 | 0.0547536547 |
| prefill top-1 是否一致 | 是 |
| 首个生成 token 分歧 | 第 15 个 |

关闭 LUT 仍复现；只关闭 LLAMAFILE 后两种 prefill 的 logits 哈希一致，且与默认构建的串行路径一致。这将差异缩小到启用 LLAMAFILE 的计算路径，但本次未逐算子定位误差，也未与原始 PyTorch 进行精度对齐。因此不能将合成模型回归通过扩展为默认配置对所有真实输入均满足批量/串行一致性。原始数据见 [数值检查汇总](validation/local-x86-20260923/correctness-summary.json)。

### 文本与较长输入

CLI 使用 `--seed 42 --temp 0 -no-cnv`，Flash Attention 保持 CLI 默认 `auto`；上述 logits 探针及下文 benchmark 则显式/默认关闭 Flash Attention。法国首都提示词生成以 `Paris, the largest city in France` 开头；direct 与 LUT 开启时的 64-token 输出相同。故事提示词完成 128-token 生成。使用随报告保存的 [长提示词](validation/local-x86-20260923/long-prompt.txt)，默认 LUT 配置和关闭 LLAMAFILE 的 LUT 配置均完成 832-token prefill（跨两个 ubatch）和 256-token 生成，在 context 2048 内正常退出。关闭 LLAMAFILE 的 [完整回归记录](validation/local-x86-20260923/ctest-reference-detailed.log)、[故事生成](validation/local-x86-20260923/cli-reference-story.stdout.log) 和 [长输入运行日志](validation/local-x86-20260923/cli-reference-long.stderr.log) 均已保存。

文本可读，但故事出现重复，法国首都续写有事实错误，刻意重复的长提示词也引出了重复输出。本次只能证明生成链路可运行；没有语言质量基准或原始模型对照，不能把这些现象归因为 LUT，也不能宣称语言质量达标。CLI 首个生成 token 来自 prefill logits，因此生成 256 tokens 时计时器显示 255 次 decode 属正常统计口径。

### 内存

相同两提示词 logits 探针的进程峰值 RSS：LUT 关闭约 1,190.09 MiB，开启 `auto` 约 1,696.93 MiB，增加约 506.84 MiB。这是整进程 `getrusage()` 峰值，并非内核专属内存统计。源码中的额外权重预打包可以解释这类额外开销；当前普通模型尚未获得相应的 LUT 执行收益。context 2048 的 F16 KV buffer 为 576 MiB，计算 buffer 为 95.01 MiB。

## 5. 性能测量

采用 `llama-bench` 的随机 token 工作负载，`-b 512 -ub 512 -ngl 0 -r 3`，保留默认 warmup。`pp128/pp512` 分别测 128/512-token 预填充，`tg128` 测从空 context 开始的 128-token decode；它不同于 832-token 长输入之后的生成。表中为工具输出的平均 tokens/s ± 样本标准差，不含加载/预打包耗时。

所有基准按顺序执行，期间没有编译或其他本次推理测试并发运行。使用系统默认调度，未绑定 CPU、未锁定频率，CPU governor 为 `powersave`，boost 保持系统设置。因此数据是本次本机条件下的测量，不代表硬件最大性能。

| 构建 / 运行时 LUT | 线程 | pp128，tokens/s | pp512，tokens/s | tg128，tokens/s |
| --- | ---: | ---: | ---: | ---: |
| `build-lut` / OFF | 4 | 43.98 ± 1.93 | 40.60 ± 1.17 | 32.52 ± 0.05 |
| `build-lut` / ON | 4 | 45.03 ± 0.04 | 41.54 ± 0.12 | 32.50 ± 0.09 |
| `build-lut` / OFF | 8 | 72.03 ± 0.15 | 68.65 ± 0.11 | 44.25 ± 0.11 |
| `build-lut` / ON | 8 | 72.20 ± 0.12 | 68.85 ± 0.10 | 45.18 ± 0.10 |
| `build-lut-reference` / ON，LLAMAFILE OFF | 8 | 71.57 ± 0.08 | 66.80 ± 0.03 | 45.08 ± 0.17 |

原始样本与环境字段：[LUT OFF](validation/local-x86-20260923/bench-lut-off.stdout.log)、[LUT ON](validation/local-x86-20260923/bench-lut-on.stdout.log)、[LLAMAFILE OFF + LUT ON](validation/local-x86-20260923/bench-reference-on.stdout.log)。每个单元格均来自 3 次测量，未选择性剔除样本。

本次测试的 4/8 线程中，8 线程更快。用于复现的 LLAMAFILE OFF 配置测得 pp128 **71.57 tokens/s**、pp512 **66.80 tokens/s**、tg128 **45.08 tokens/s**；与默认 LUT 配置接近。较长 CLI 输入后的 decode 测得 32.49 tokens/s，包含更深的 KV cache 和 CLI 路径差异，不能与空 context 的 tg128 直接比较。

对照 `build-lut` 中运行时 LUT 关闭/开启的数据，是在相同二进制中测量配置开销；其普通 iFairy matmul 均走 vecdot，不能将波动归因于 LUT 矩阵乘法加速。

独立 `ifairy-microbench` 直接调用 LUT 内核，合成权重和激活，单线程，`M=K=1536, N=1`，10 次 warmup、100 次迭代：QGEMM 为 **53.75 µs/次**，融合 preprocess+QGEMM 为 **55.80 µs/次**。两种模式输入构造不同，checksum 不作相互比较；这是内核可运行性与局部时延记录，不能换算为真实模型 tokens/s。其数值回归依据为上文 LUT CTest，日志见 [QGEMM](validation/local-x86-20260923/lut-kernel-qgemm.stdout.log) 和 [fused](validation/local-x86-20260923/lut-kernel-fused.stdout.log)。

基准复现命令（先运行结束再切换 `GGML_IFAIRY_LUT=1`）：

```sh
GGML_IFAIRY_LUT=0 GGML_IFAIRY_LUT_DEBUG=0 \
  ./build-lut/bin/llama-bench \
  -m "$HOME/projects/Fairy-plus-minus-i-700M/ifairy.gguf" \
  -ngl 0 -t 4,8 -b 512 -ub 512 -p 128,512 -n 128 -r 3 -o json

GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_DEBUG=0 \
  ./build-lut-reference/bin/llama-bench \
  -m "$HOME/projects/Fairy-plus-minus-i-700M/ifairy.gguf" \
  -ngl 0 -t 8 -b 512 -ub 512 -p 128,512 -n 128 -r 3 -o json
```

## 6. 本机复现命令

从仓库根目录运行。完整命令、退出码及耗时见 [42 条命令记录](validation/local-x86-20260923/commands.json)，全部退出码为 0；[原始记录目录](validation/local-x86-20260923/) 保留各命令的 stdout/stderr 和独立元数据。命令成功退出与数值一致性是不同检查，后者见上文汇总。

```sh
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=OFF -DGGML_CCACHE=OFF
cmake --build build-rel -j 8
ctest --test-dir build-rel --output-on-failure

cmake -B build-lut -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=ON -DGGML_CCACHE=OFF
cmake --build build-lut -j 8
ctest --test-dir build-lut --output-on-failure
```

本次用于规避所测 prefill 一致性差异的配置：

```sh
cmake -B build-lut-reference -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=ON -DGGML_LLAMAFILE=OFF \
  -DGGML_CCACHE=OFF
cmake --build build-lut-reference -j 8
ctest --test-dir build-lut-reference --output-on-failure

GGML_IFAIRY_LUT=1 ./build-lut-reference/bin/llama-cli \
  -m "$HOME/projects/Fairy-plus-minus-i-700M/ifairy.gguf" \
  -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 \
  -p "The capital of France is" -n 64 \
  --seed 42 --temp 0 -no-cnv --simple-io

GGML_IFAIRY_LUT=1 python3 \
  docs/validation/local-x86-20260923/check_logits.py \
  build-lut-reference "$HOME/projects/Fairy-plus-minus-i-700M/ifairy.gguf"
# 末尾添加 serial 可复现逐 token prefill；比较输出 JSON 中的 logits_sha256。
```

上述探针的 ctypes 结构布局对应本报告提交的 `include/llama.h`，不是跨版本 Python API。更换源码版本应先核对 ABI。

本次没有修改 C/C++ 文件，故仓库要求针对变更 C/C++ 运行的 `git clang-format --diff` 和 `clang-tidy` 无适用文件；未将它们记录为已执行。基准均串行运行，没有并发 `llama-bench`。权重保留在原目录，报告与日志不包含权重。
