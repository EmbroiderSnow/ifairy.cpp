# 普通 MUL_MAT 接入 LUT：实现与本机、手机复测

日期：2026-09-23。基线提交：`a6256d61ffd3ebe4cba06fdb3b46e0b7eb78feaa`；已验证的 LUT 修复及测试源码归档于提交 `a243488`。实验在提交前运行，二进制中的 `build_commit` 仍为基线，变更源码可用记录中的 SHA-256 核对。原始记录、命令、校验和及复现脚本位于 [validation/lut-integration-20260923](validation/lut-integration-20260923/)。

## 1. 结论与实现范围

普通二维 `GGML_TYPE_IFAIRY × F32/IFAIRY_Q16 → F32` 矩阵乘法已接入 CPU LUT。现在 `GGML_IFAIRY_LUT=1` 同时触发权重打包、激活预处理和 LUT QGEMM 执行。本机与 Samsung SM-F9660 的调试日志各记录 **336 次 `ifairy_lut: executed MUL_MAT`**，包含预热和五 token 提示词的实际计算，证明执行分派已生效。只出现 `can_mul_mat=true` 仍不能作为执行证据。

- 在 CPU 扩展层接入算子资格检查、工作区规划、激活量化、共享 LUT 预处理、同步屏障、16 行权重 tile 分片和 packed BF16 输出。
- 保留 `w * conj(x)`、旧 BF16 复数布局及所有原有直接计算路径。支持单 token 和批量预填充，允许激活行间存在跨步。
- 不连续/视图权重、高维广播、普通 `IFAIRY64`、其他类型继续回退原路径；兼容 W2 算子的原有 LUT 路径保留。缺失权重打包、工作区不足或预量化激活超出安全范围时，计算入口拒绝 LUT，执行层回退；规划时已预留直接量化所需空间。
- 修复了单列 LUT 预处理时多个线程同时写缩放数组的数据竞争。另用 `unique_ptr` 管理打包元数据，保留缓存重置后 tensor extra 的有效性，并在后端卸载时释放元数据；内存检查发现的泄漏已复测消除。
- 未改变模型准入/架构隔离，未加入模型图或 GPU 后端，未将真实权重放入仓库。

主要实现：[legacy-ifairy-cpu.cpp](../ggml/src/ggml-cpu/legacy-ifairy/legacy-ifairy-cpu.cpp)、[ggml-cpu.c](../ggml/src/ggml-cpu/ggml-cpu.c)、[LUT QGEMM](../ggml/src/ggml-cpu/legacy-ifairy/lut/ggml-ifairy-lut-qgemm.cpp)、[打包缓存](../ggml/src/ggml-cpu/legacy-ifairy/lut/ggml-ifairy-lut-transform.cpp)。

## 2. 激活量化与正确性口径

int8 LUT 的每个表项累加两个有符号激活。x86 默认直接路径使用范围 ±127 的量化，直接用于这些表项会饱和。因此接入使用仓库已有的 LUT 专用量化：

| 配置 | F32（packed BF16 complex）激活处理 |
| --- | --- |
| `GGML_IFAIRY_LUT=0` 或未设置 | 原有直接路径不变 |
| LUT 开启，`auto` / `lut16` | 每个输入行共享缩放，42.6 缩放、夹到 ±42 |
| LUT 开启，`lut_c` | 每个 256 元素块独立缩放，42.6 缩放、夹到 ±42 |
| 输入已是 `IFAIRY_Q16` | 保留原缩放与量化值；任一值不在 [-63,63] 时回退直接计算 |

这意味着 LUT 与原默认直接路径并非相同的数值配置，**不能要求两个路径的完整 logits、生成 token 或模型质量逐位相同，也不能把吞吐比全部归因于查表内核本身。** 即便采用相同的行级量化，浮点累加次序及 BF16 舍入仍可能导致多层模型输出分歧。本次用独立复数公式验证算子语义，同时独立记录整模型数值差异，没有以“生成文本可读”替代数值检查。

## 3. 回归与实际执行证据

| 检查 | 结果 |
| --- | --- |
| 本机 Release 直接构建 | CTest **3/3** |
| 本机 Release LUT 构建 | CTest **6/6** |
| 本机 LUT、LLAMAFILE 关闭的对照构建 | CTest **6/6** |
| 手机 ARM64 LUT 构建 | 执行对应的 **6 项**测试命令，全部返回 0（设备端未调用 CTest） |
| 新 `test-ifairy-lut-mul-mat` | 两个平台均通过 144 次计算图执行及缺失打包/工作区回退检查 |
| AddressSanitizer / UndefinedBehaviorSanitizer / LeakSanitizer | 新回归复测通过，无报告错误 |
| 真实模型算子抽查 | 336 次矩阵乘法，3,006 个实部/虚部分量，独立双精度参考 **0 失败** |
| 架构隔离 | `scripts/check-ifairy-only.py` 通过 |
| 格式与静态检查 | 对变更 C/C++ 文件运行 merge-base `git clang-format --diff`；变更源文件运行 `clang-tidy -p build-rel`，LUT 条件编译代码另用 LUT 数据库补查 |

新回归覆盖线程数 1/4/8，M=1/17/65、N=1/3/9、K=256/512/1536，两种激活存储、全部实现选项、非整 tile 尾部、重复使用计算图并更换激活，以及开关关闭/未设置、跨步权重、高维广播和超范围量化输入的回退。测试读取真实执行日志计数，并独立解码权重和计算复数乘积；若删除新执行分派，测试会失败。参考误差门限是每分量 `0.008*abs(reference)+1e-5`，未放宽旧测试。

真实模型抽查在本机运行，通过诊断回调读取五 token 提示词 prefill 和一步 decode 的实际输入、权重、输出；使用同一激活量化后，在 Python 中逐元素计算 `w * conj(x)`，不调用 LUT/vecdot 作为参考。覆盖各层的首/中/末行以及首/末输入列，最大误差/门限比 **0.48114**。采用行级量化的直接对照同样通过 336 次/3,006 分量检查（最大比值 **0.48544**），见 [直接路径逐算子记录](validation/lut-integration-20260923/host-direct-real-matmul-oracle.json)。两个路径都满足算子参考检查，并不意味着经过 24 层和自回归生成后得到相同 logits。诊断会改变计算图分段，计时不计入吞吐结果。详情见 [参考检查脚本](validation/lut-integration-20260923/check_matmul_oracle.py) 和 [逐算子记录](validation/lut-integration-20260923/host-real-matmul-oracle.json)。

静态检查全部退出码为 0，但共享代码仍有现有告警，测试中按量化块计算缩放的有意整除也有提示；未声明零告警。

## 4. 真实模型端到端检查

模型：外部路径 `~/projects/Fairy-plus-minus-i-700M/ifairy.gguf`，576,123,520 字节，SHA-256 `2211f40ec7dd56ed424f4bcaccc65869b8deffbbcea65e84806b619e380da6b3`。本机为 AMD Ryzen 7 H 255 / x86 AVX2；手机为 SM-F9660 / SM8750 / Android 16 / ARM64 NEON。沿用原报告的系统和编译环境；手机为 NDK r30-beta1，`armv8.6-a+dotprod+i8mm+fp16`，OpenMP OFF；本机 GCC 16.2.1，native/OpenMP ON。两端标准 LUT 构建均保留 LLAMAFILE ON。

每个平台分别运行 7 个独立探针进程：默认关闭、auto、auto 重复、lut16、lut_c、auto 串行 prefill、相同行级量化的直接对照。每进程包含两组提示词，各 33×32,000 logits，共 **2,112,000** 个值；每个平台累计 **14,784,000** 个值，全部有限。context=2048、batch/ubatch=512、8 线程、Flash Attention OFF；对照结果均来自相同提示词和贪心生成 32 tokens。

| 同一平台内的比较 | 本机 | 手机 |
| --- | --- | --- |
| auto 与独立进程重复 | 全 trace 逐字节一致 | 全 trace 逐字节一致 |
| auto 与 lut16 | 全 trace 逐字节一致 | 全 trace 逐字节一致 |
| auto 批量与串行 prefill | 全 trace 逐字节一致 | 全 trace 逐字节一致 |
| auto 与原默认直接路径 | 不一致 | 不一致 |
| auto 与相同行级量化的直接对照 | 不一致 | 不一致 |
| auto 与 lut_c | 不一致，量化粒度不同 | 不一致，量化粒度不同 |

相同输入的首个 logits 向量，auto 与原默认直接路径的最大绝对差分别为本机 **1.75863 / 1.34125**、手机 **2.43701 / 1.44824**（两提示词）；与行级量化直接对照分别为本机 **1.17207 / 1.15452**、手机 **1.13587 / 1.08106**。生成分歧后不再把后续 logits 差当作同输入误差。本机行级量化对照使用 LLAMAFILE OFF，手机对照使用同一个 LUT 二进制且运行时关闭 LUT。详细 token、哈希及首 logits 统计见 [本机汇总](validation/lut-integration-20260923/host-correctness-summary.json) 和 [手机汇总](validation/lut-integration-20260923/phone-correctness-summary.json)。

本次 LUT auto 的跨平台首 logits 最大差为 **1.55367 / 1.38328**，完整 trace 不一致；没有宣称 x86 与 ARM 逐位等价，见 [跨平台记录](validation/lut-integration-20260923/cross-platform-lut-summary.json)。

两端都完成故事提示词 128-token 生成，以及 **832-token prefill + 256-token 生成**（跨两个 ubatch），退出码为 0。文本仍有重复；未做语料困惑度、任务准确率或 PyTorch 对齐，因此本次确认运行链路、查表执行和所列数值检查，不宣称语言质量与原路径完全等价。

探针峰值 RSS（整进程）本机默认关闭 **1,170.93 MiB**、auto **1,681.29 MiB**；手机分别 **1,140.76 / 1,648.71 MiB**。额外约 508–510 MiB 主要来自权重打包及工作区，仍是实际内存代价。

## 5. 吞吐

配置：CPU、`-b 512 -ub 512 -p 128,512 -n 128 -r 3 -o json`，F16 KV、Flash Attention OFF，不绑定核心。每个指标 3 次，表中为均值 ± 标准差。所有 `llama-bench` 进程顺序执行，本机结束后才启动手机基准。

### 本机

| 线程 / 模式 | pp128 (token/s) | pp512 (token/s) | tg128 (token/s) |
| --- | ---: | ---: | ---: |
| 4 / 默认直接 | 49.46 ± 1.62 | 47.24 ± 0.14 | 35.57 ± 0.27 |
| 4 / LUT auto | 143.97 ± 1.75 | 148.99 ± 1.27 | 71.17 ± 1.91 |
| 8 / 默认直接 | 70.64 ± 0.97 | 67.74 ± 2.41 | 46.61 ± 0.99 |
| 8 / LUT auto | 216.24 ± 4.10 | 198.88 ± 15.91 | 76.15 ± 0.44 |
| 8 / 行级量化直接对照 | 89.77 ± 9.19 | 72.33 ± 3.39 | 43.55 ± 0.58 |

8 线程下，LUT auto 相比默认直接配置，pp128/pp512 分别约 **3.06× / 2.94×**，tg128 约 **1.63×**。这是已测试配置的整体吞吐比；两者量化不同，不能作为严格同精度的纯内核加速比。另列的行级量化直接对照设置 `GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1`、LLAMAFILE OFF，LUT 相对其 tg128 约 **1.75×**，但二者完整 logits 仍不等价。

首轮 4 线程关闭 LUT 的测试期间曾运行静态检查，已排除该轮并在其他主机任务完成后单独重测；表中使用 `host-bench-4t-off-clean`，原始首轮仍保留供审计。其余主机指标使用各自顺序测试结果，数据及完整命令见 [本机基准汇总](validation/lut-integration-20260923/host-benchmark-summary.json)、`bench-commands.json`、`bench-recheck-commands.json`、`bench-same-quant-commands.json`。

### 手机

重新连接后已取回全部结果：四组在设备端均正常完成，退出码为 0，共 **36 个计时样本**。ADB 断连导致主机连接命令返回 255，没有中止设备上的这四组测试。根据设备时间戳，测试在 09:45:32–09:53:49 UTC（北京时间 17:45:32–17:53:49）顺序完成，三次组间空闲均为 90 秒，无需重跑。最终版本回归程序也已重新部署并在手机通过 **144 次计算图执行**；模型和全部 12 个部署文件的 SHA-256 均已复核。

| 运行顺序 | 线程 / 模式 | pp128 (token/s) | pp512 (token/s) | tg128 (token/s) |
| ---: | --- | ---: | ---: | ---: |
| 1 | 4 / 默认直接 | 75.00 ± 6.74 | 48.12 ± 0.10 | 33.31 ± 0.15 |
| 2 | 4 / LUT auto | 66.26 ± 6.50 | 36.43 ± 1.21 | 28.79 ± 0.12 |
| 3 | 8 / LUT auto | 117.94 ± 11.56 | 67.66 ± 1.52 | 31.27 ± 1.54 |
| 4 | 8 / 默认直接 | 135.16 ± 6.08 | 69.22 ± 0.52 | 32.19 ± 1.86 |

**本轮手机没有测到 LUT 吞吐提升。** 8 线程下 ON/OFF 的 pp128、pp512、tg128 观测比为 **0.873× / 0.977× / 0.971×**；4 线程为 **0.883× / 0.757× / 0.864×**。LUT 确实进入整模型计算，但接入成功不等于在 ARM 上加速。这些是当前实现、量化和设备状态共同作用的整体吞吐；不能直接套用本机 x86 的提升比例。

| 测量 | 电池温度，°C，前 → 后 | Thermal Status，前 → 后 | 缓存 SKIN 温度，°C，前 → 后 |
| --- | --- | --- | --- |
| 4t OFF | 33.4 → 34.1 | 0 → 1 | 37.9 → 40.1 |
| 4t ON | 34.2 → 34.8 | 0 → 1 | 37.9 → 40.0 |
| 8t ON | 34.7 → 35.3 | 0 → 2 | 37.9 → 42.0 |
| 8t OFF | 35.2 → 35.7 | 1 → 2 | 39.9 → 42.0 |

各组开始时 policy0/policy6 的最高频率均恢复至 **3.53/4.47 GHz**，但两组 4 线程结束时降至 **1.79/1.69 GHz**，两组 8 线程结束时降至 **1.36/1.40 GHz**。手机全程 USB 供电，电量从 56% 到 58%，未绑定核心或锁频。90 秒空闲没有完全统一起始温度；温度记录是组前/组后快照，SKIN 来自 thermalservice 缓存，不能表示连续温度曲线。固定顺序、温控和不同量化使本轮结果不足以单独归因 LUT 内核的性能影响，也不能据此断言所有手机场景都会变慢。

原始均值、标准差、全部样本和时间见 [手机基准汇总](validation/lut-integration-20260923/phone-benchmark-summary.json)，温度与频率见 [温控汇总](validation/lut-integration-20260923/phone-thermal-summary.json)。[恢复记录](validation/lut-integration-20260923/phone-interruption.json)、[最终回归日志](validation/lut-integration-20260923/phone-lut-dispatch-final.log) 和 [最终部署校验和](validation/lut-integration-20260923/android-deployment-final.json) 一并保存。


## 6. 使用与复现

```sh
cmake -B build-lut -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_LEGACY_IFAIRY_CPU_LUT=ON
cmake --build build-lut -j 8
ctest --test-dir build-lut --output-on-failure
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=auto \
  build-lut/bin/llama-cli -m ~/projects/Fairy-plus-minus-i-700M/ifairy.gguf \
  -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 \
  -p 'The capital of France is' -n 64 --seed 42 --temp 0 -no-cnv
```

需要验证实际执行时额外设置 `GGML_IFAIRY_LUT_DEBUG=1`，检查 `executed MUL_MAT`；正式计时关闭 debug。手机的 [功能测试脚本](validation/lut-integration-20260923/run-phone-functional.sh) 与 [顺序 benchmark 脚本](validation/lut-integration-20260923/run-phone-bench.sh) 已保存。测试二进制位于 `/data/local/tmp/ifairy-lut-integration-20260923/bin`，复用上轮手机测试目录内的同一模型；未安装 App 或更改系统频率/功耗设置。

大体积 logits 原始二进制保留在本机 `/tmp/ifairy-lut-integration`、`/tmp/ifairy-lut-phone-results`、恢复连接后的 `/tmp/ifairy-lut-phone-results-resumed` 及手机测试目录，不纳入仓库。仓库只保存源码、脚本、日志、摘要和校验和。旧报告中的“只打包、不执行”及原 ON/OFF 数值相同结论是**接入前历史结果**，不能用来描述当前实现。
