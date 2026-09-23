# Android 手机 iFairy 700M 推理与 LUT 实验报告

日期：2026-09-23，Asia/Shanghai。设备：三星 `SM-F9660`，系统报告 SoC 为 `SM8750`。

## 0. LUT 接入后补测（当前结论）

同日后续已将新的 LUT 执行分派交叉编译并部署到 SM-F9660，已验证的修复及测试源码归档于提交 `a243488`。手机实际执行了 **6 项回归**（含新增普通 MUL_MAT 回归），均返回 0；调试日志出现 **336 次 `executed MUL_MAT`**。下方第 1 节起是接入前记录，不代表当前实现。详细实现、证据与数值边界见 [LUT 接入报告](LUT_MUL_MAT_INTEGRATION_REPORT_2026-09-23.md)。

7 个真实模型配置累计检查 **14,784,000** 个有限 logits；LUT auto 的独立重复、lut16 对照及批量/串行 prefill，完整 trace 均逐字节一致。故事 128-token 生成和 832-token prefill + 256-token 生成均完成。LUT 专用量化与默认直接路径不同，完整 logits 和生成 token 不保证相同，不能据此宣称语言质量等价。auto 峰值 RSS **1,648.71 MiB**，默认直接 **1,140.76 MiB**。

重新连接后已取回完整基准：四组设备端退出码均为 0，断连期间仍正常完成，没有重跑或丢弃样本。最终版本回归程序重新部署后，通过 **144 次计算图执行**；模型及 12 个部署文件的 SHA-256 均已复核。以下是接入后的当前结果，下方第 5 节保留接入前历史性能。

| 线程 / 模式 | pp128，token/s | pp512，token/s | tg128，token/s |
| --- | ---: | ---: | ---: |
| 4 / 默认直接 | 75.00 ± 6.74 | 48.12 ± 0.10 | 33.31 ± 0.15 |
| 4 / LUT auto | 66.26 ± 6.50 | 36.43 ± 1.21 | 28.79 ± 0.12 |
| 8 / LUT auto | 117.94 ± 11.56 | 67.66 ± 1.52 | 31.27 ± 1.54 |
| 8 / 默认直接 | 135.16 ± 6.08 | 69.22 ± 0.52 | 32.19 ± 1.86 |

每格为 3 次均值 ± 样本标准差，共 36 个样本；CPU、F16 KV、Flash Attention OFF，batch/ubatch=512。按表顺序执行，组间各空闲 90 秒，USB 供电，不绑定核心。8 线程 LUT 的 pp128/pp512/tg128 分别为直接配置的 **0.873× / 0.977× / 0.971×**，**本轮手机没有测到 LUT 加速**。

电池温度从首组开始的 33.4°C 升至末组结束的 35.7°C；四组 Thermal Status 前后依次为 0→1、0→1、0→2、1→2。各组开始时 policy0/policy6 频率上限为 3.53/4.47 GHz，4 线程组结束时降至 1.79/1.69 GHz，8 线程组结束时降至 1.36/1.40 GHz。空闲后温度仍不完全一致，加上量化配置不同，不能把观测吞吐差异单独归因于查表内核。LUT 的运行及数值检查通过，与是否获得性能收益是两个独立结论。

新测试使用 `/data/local/tmp/ifairy-lut-integration-20260923/bin`；当前原始样本见 [手机基准汇总](validation/lut-integration-20260923/phone-benchmark-summary.json)，温控见 [温度与频率记录](validation/lut-integration-20260923/phone-thermal-summary.json)，最终验证见 [回归日志](validation/lut-integration-20260923/phone-lut-dispatch-final.log) 和 [部署清单](validation/lut-integration-20260923/android-deployment-final.json)。

## 1. 接入前结论（历史记录）

同一份 700M GGUF 已在已连接的 Android 手机上完成直接路径、LUT 开关对照、真实模型数值检查和文本生成测试。手机实际执行了与主机 CTest 对应的直接构建 **3/3**、LUT 构建 **5/5** 测试命令，全部通过；没有将主机交叉编译成功当成手机运行通过。

两组真实提示词、每组 prefill 加 32 步 decode 的完整 logits，在直接构建、LUT 构建运行时关闭/开启、`auto/lut16/lut_c`、重复运行、批量/逐 token prefill 之间均一致。每种配置检查 **2,112,000 个有限 logits**，8 种配置共 **16,896,000 个**，未发现 NaN/Inf。此前 x86 测试的批量/串行差异未在手机的这两组用例中复现，手机维持 `GGML_LLAMAFILE=ON`。

**当前普通 iFairy 整模型仍使用 ARM vecdot，开启 LUT 只增加权重预打包，并未接入普通 `MUL_MAT` 的 LUT 执行分派。** 独立 LUT 内核可以运行，但整模型速度不能标为 LUT 加速性能。手机与 x86 的 logits 和部分生成 token 不同，见第 4 节；本次未验证跨平台或 PyTorch 数值等价，也未进行语言质量验收。

生产推理源码和模型准入逻辑未改动。本次增加 [原生真实模型探针](../tests/test-ifairy-real-model.cpp) 及其 CMake 构建目标，以便在没有 Python 的手机上复现数值检查。探针需要显式传入模型路径，不加入无需真实权重的默认 CTest。

## 2. 环境与模型

| 项目 | 实测配置 |
| --- | --- |
| 设备标识 | Samsung `SM-F9660`，`ro.soc.model=SM8750` |
| 系统 | Android 16 / API 36，`arm64-v8a`，Linux `6.6.98-android15-8` |
| CPU | 8 核；CPU 0–5 最高频率 3,532,800 kHz，CPU 6–7 最高频率 4,473,600 kHz；内核 governor 均为 `walt` |
| 可用指令集 | `/proc/cpuinfo` 包含 NEON/ASIMD、ASIMDDP、FP16、i8mm、BF16；编译目标为 `armv8.6-a+dotprod+i8mm+fp16` |
| 内存 | 系统 `MemTotal=11,351,264 KiB`，约 10.83 GiB；首次记录 `MemAvailable` 约 6.6 GiB |
| 编译工具 | 本机 Linux 上的 Android NDK `30.0.14904198`，r30-beta1，Clang 21；CMake Release，`-j 8` |
| Android 编译 API / STL | API 28，`c++_static`；生成 AArch64 Android 二进制，在 API 36 手机执行 |
| CPU 配置 | `GGML_NATIVE=OFF`，ARM dotprod 保持默认开启，OpenMP OFF，LLAMAFILE ON，ccache OFF；ggml 自带线程池正常使用 4/8 线程 |
| 推理后端 | CPU，`-ngl 0`；没有 GPU 后端或 Android 应用层 |
| 手机测试目录 | `/data/local/tmp/ifairy-validation-20260923`；使用 adb shell，无需 root |
| 电源和设置 | USB 连接并充电；保留系统调度、频率、屏幕和省电设置，没有锁频或修改温控策略 |

源码基线为 `a6256d61ffd3ebe4cba06fdb3b46e0b7eb78feaa`，加本次测试探针。模型来自本机 `/home/zybi/projects/Fairy-plus-minus-i-700M/ifairy.gguf`，文件 **576,123,520 字节**；传到手机后重新计算 SHA-256，与上一份本机报告完全一致：

```text
2211f40ec7dd56ed424f4bcaccc65869b8deffbbcea65e84806b619e380da6b3
```

GGUF 为 `general.architecture=ifairy`，24 层、embedding 1536、FFN 4096、16 heads、词表 32000、训练 context 2048；267 个张量中 168 IFAIRY、98 F32、1 F16。本次直接使用已有 GGUF，未重新转换 checkpoint。权重只复制到所连接手机的测试目录，未写入仓库。

原始环境和二进制指纹：[设备信息](validation/android-sm-f9660-20260923/device-info.log)、[构建环境](validation/android-sm-f9660-20260923/environment.json)、[编译命令](validation/android-sm-f9660-20260923/compile-evidence.json)、[部署文件 SHA-256](validation/android-sm-f9660-20260923/deployment-manifest.json)、[手机模型 SHA-256](validation/android-sm-f9660-20260923/model-device-sha256.stdout.log)。部署前只去掉调试段，保留主机构建中的调试信息。

## 3. 构建、ARM 路径与回归

| 构建 | LUT 编译开关 | 手机执行结果 |
| --- | --- | --- |
| `build-android-direct` | OFF | direct kernel、合成模型、复数基础算子，3/3 通过 |
| `build-android-lut` | ON | 上述 3 项、legacy LUT、LUT 开启的合成模型，5/5 通过 |

交叉编译分别耗时 **46.004 / 49.002 秒**，不含配置和后续测试探针头文件完善后的增量编译。手机没有安装 CTest，使用 [run-functional.sh](validation/android-sm-f9660-20260923/run-functional.sh) 执行与 `tests/CMakeLists.txt` 相同的测试程序、算子筛选和环境变量；日志按测试分别保存，退出码逐条检查。

直接路径注册见 [ARM ifairy-quants.c](../ggml/src/ggml-cpu/legacy-ifairy/arm/ifairy-quants.c)：`ggml_ifairy_vecdot_init()` 优先选择可用的 dotprod，其次 NEON，最后 generic。编译记录包含独立 `ifairy-dotprod.c` 和 ARM quants 源文件；该手机具备 dotprod。LUT 构建额外包含 ARM NEON LUT 内核。未删改直接、NEON 或 scalar 回退。

普通 `MUL_MAT` 仍由 [ggml-cpu.c](../ggml/src/ggml-cpu/ggml-cpu.c) 的 vecdot 路径执行；[legacy-ifairy-cpu.cpp](../ggml/src/ggml-cpu/legacy-ifairy/legacy-ifairy-cpu.cpp) 的 LUT 执行入口只接受 legacy W2。调试日志中的 `can_mul_mat=true` 仅是资格检查，不能作为整模型执行 LUT QGEMM 的证据。这与 [本机 x86 报告](LOCAL_X86_LUT_INFERENCE_REPORT_2026-09-23.md) 的源码检查一致。

回归包括 `w * conj(x)`、packed BF16、prefill/decode、同路径重复性，以及对 `fairy2i/llama/qwen3/unknown` 的预期拒绝。源码隔离脚本也通过。增加探针后，本机 Release/LUT 构建重新完成，主机 CTest 再次 **3/3、5/5** 通过。

构建过程中，CMake 的一次 ARM 宏探测未带交叉编译 `--target`，打印 `unknown target CPU` / `Failed to get ARM features` 提示；配置本身返回 0。实际编译命令明确包含 `--target=aarch64-none-linux-android28` 和 ARM `-march`，生成的程序已在手机执行。保留该配置日志，未将其记为编译失败或静默隐藏。

## 4. 真实模型数值和文本检查

### 手机内的一致性

原生探针与上一轮 Python 探针使用相同参数：context 2048、batch/ubatch 512、8 线程、F16 KV、Flash Attention OFF；提示词为 `The capital of France is`（5 tokens）和 `Once upon a time, in a small village`（9 tokens）。每个提示词记录 prefill 与 32 次 greedy decode 后的 33 × 32000 个 F32 logits，在设备检查有限性，原始字节拉回本机后计算 SHA-256。

先在 x86 上验证新 C++ 探针：两组完整 logits 哈希与上一轮 Python 探针一致，见 [探针交叉校验](validation/android-sm-f9660-20260923/probe-host-crosscheck.json)。然后在手机独立进程间比较：

| 对照 | 完整 logits 哈希 / token IDs |
| --- | --- |
| direct 构建 vs LUT 构建、运行时关闭 | 两组均一致 |
| LUT OFF vs ON / `auto` | 两组均一致 |
| LUT ON 独立进程重复运行 | 两组均一致 |
| `auto` vs `lut16` vs `lut_c` | 两组均一致 |
| batched vs serial prefill | 两组均一致；首个 logits 向量最大绝对差值均为 0 |
| serial prefill 下 LUT OFF vs ON | 两组均一致 |

法国首都提示词的完整 logits SHA-256 为 `40fb7908291bb4c6afeb9f2912c3c120957b7d628f5c2aebb3561978c754cf53`，故事提示词为 `ec5c7c152c0be13ffa21042aac4c5af953729dd6269ec159b1049160a8d1438c`。详细结果及每个进程峰值 RSS 见 [correctness-summary.json](validation/android-sm-f9660-20260923/correctness-summary.json)。

### 跨平台差异

使用相同原生探针比较手机与 x86 直接构建，在相同提示词、尚未发生生成分歧的 prefill logits 上测得：

| 提示词 | 最大绝对差值 | 平均绝对差值 | 首个生成 token 分歧 |
| --- | ---: | ---: | ---: |
| 法国首都 | 1.40843582 | 0.24353399 | 第 14 个 |
| 故事 | 0.61886635 | 0.11068297 | 第 15 个 |

两种平台的完整 logits 哈希不同，不能声称跨平台逐位等价。这里同时存在 ISA、编译器和浮点计算路径差异；本次未逐算子归因，也没有原始 PyTorch 参考结果，不能仅凭同设备 LUT 对照通过就认定跨平台精度完全正确。以上差异作为实验发现保留，不纳入“通过”的一致性项目。

### 文本、长输入与内存

CLI 使用 `--seed 42 --temp 0 -no-cnv`，Flash Attention 保持 CLI 默认 `auto`。法国首都提示词完成 64-token 生成，以 `Paris, the largest city in the country` 开头；手机 direct 与 LUT 开启时的输出逐字相同。故事提示词完成 128-token 生成。长输入采用上一份报告相同的 832-token 人工重复段落，context 2048、batch/ubatch 512，生成上限 256 tokens；原始输入保存在 [long-prompt.txt](validation/local-x86-20260923/long-prompt.txt)。

长输入实际完成 **832-token prefill + 256-token 生成**，退出码 0：prefill 12,518.38 ms（66.46 tokens/s），255 次单 token decode 11,142.73 ms（22.88 tokens/s），CLI 推理计时总计 23,699.33 ms。第一个生成 token 使用 prefill logits，所以生成 256 tokens 对应 255 次 decode；上述总计不含 2,071.70 ms 的模型加载/初始化。完整记录见 [长输入 stdout](validation/android-sm-f9660-20260923/cli-long.stdout.log) 和 [stderr](validation/android-sm-f9660-20260923/cli-long.stderr.log)。

生成文本可读，但含重复和事实性错误；没有语言质量或困惑度基准，本报告只对所列运行、数值检查和性能数据负责。

相同真实模型探针的进程峰值 RSS，LUT 构建运行时 OFF 为 **1140.88 MiB**，ON 为 **1647.69 MiB**，增加 **506.81 MiB**；直接构建为 1140.79 MiB。使用 Android `getrusage()` 的整进程最大 RSS，包含模型映射、KV、计算缓冲区及其他运行内存，不是单独的内核工作区。LUT 权重预打包带来了额外内存开销，当前普通模型没有相应的 LUT 执行收益。

## 5. 吞吐和温度

`llama-bench` 参数与本机测试一致：`-ngl 0 -b 512 -ub 512 -p 128,512 -n 128 -r 3 -o json`，F16 KV、Flash Attention OFF，保留默认 warmup。使用同一个 LUT 构建测运行时 OFF/ON，测试 4/8 线程。顺序为 4t OFF → 4t ON → 8t ON → 8t OFF，每次等待前一进程结束；脚本见 [run-bench.sh](validation/android-sm-f9660-20260923/run-bench.sh)。

| 运行顺序 | 线程 | 运行时 LUT | pp128，tokens/s | pp512，tokens/s | tg128，tokens/s |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 4 | OFF | 77.73 ± 15.16 | 47.64 ± 0.05 | 37.62 ± 1.08 |
| 2 | 4 | ON | 46.10 ± 0.45 | 42.43 ± 0.62 | 32.91 ± 0.90 |
| 3 | 8 | ON | 73.21 ± 0.66 | 68.54 ± 0.66 | 31.45 ± 2.31 |
| 4 | 8 | OFF | 63.02 ± 3.12 | 58.83 ± 2.50 | 30.41 ± 2.44 |

原始每次样本见 [benchmark-summary.json](validation/android-sm-f9660-20260923/benchmark-summary.json)。保留全部测量，没有剔除第一组 pp128 的高波动样本。手机 8t LUT ON 本轮测得 **pp128 73.21、pp512 68.54、tg128 31.45 tokens/s**。上一轮 x86 同类配置为 72.20、68.85、45.18 tokens/s；硬件、编译器、OpenMP 和温控条件不同，这些数据只能作为各自环境下的观测。

性能数字是工具输出的平均 tokens/s ± 样本标准差，每格 3 次测量。`pp128/pp512` 测预填充，`tg128` 从空 context 测 128-token decode；不包含加载、ADB 传输或 LUT 权重预打包时间。没有同时运行其他本次手机推理任务，没有锁频、绑定 CPU 或改变温控。USB 充电及系统后台调度均属于测量环境。

连续负载期间，手机出现明显频率上限收紧和温控状态上升：

| 测量 | 电池温度，°C | 系统 Thermal Status | SKIN 温度，°C |
| --- | --- | --- | --- |
| 4t OFF 前 → 后 | 32.1 → 32.8 | 0 → 0 | 37.9 → 37.9 |
| 4t ON 前 → 后 | 32.8 → 33.8 | 0 → 1 | 37.9 → 40.0 |
| 8t ON 前 → 后 | 33.8 → 34.4 | 1 → 2 | 40.0 → 42.0 |
| 8t OFF 前 → 后 | 34.4 → 35.0 | 2 → 2 | 42.0 → 42.0 |

第一组开始时，CPU policy0/policy6 的 `scaling_max_freq` 为 3.53/4.47 GHz；第一组结束时已降至约 1.79/1.69 GHz；最后两组降至约 **1.36/1.40 GHz**。手机处于屏幕唤醒和 USB 充电状态，基准开始/结束的电量均为 31%。日志见 [thermal-summary.json](validation/android-sm-f9660-20260923/thermal-summary.json) 及每组 `.before.log/.after.log`。

这些结果反映连续负载升温过程，**不是相同温度下的公平开关对照，也不是冷机峰值**。不能依据这四组结果推断 LUT 开关的独立性能效应，或宣称 4/8 线程在所有温度下的最佳选择；如需得出这些结论，应另做相同温度、供电和频率条件的实验。本次未关闭系统温控以追求更高数字。

LUT 配置 ON/OFF 的普通 matmul 都使用相同 ARM vecdot，表中差异不能解释为 LUT 算法加速；独立 LUT 内核的局部耗时也不能换算成整模型 tokens/s。

独立 LUT 微基准使用合成权重/激活、单线程、`M=K=1536, N=1`，10 次 warmup、100 次迭代：QGEMM **253.10 µs/次**，融合 preprocess+QGEMM **262.32 µs/次**。其执行先于上述四组整模型基准。两模式输入构造不同，checksum 不应相互比较；内核数值正确性的证据来自 legacy LUT 回归。见 [QGEMM 日志](validation/android-sm-f9660-20260923/kernel-qgemm.stdout.log) 和 [fused 日志](validation/android-sm-f9660-20260923/kernel-fused.stdout.log)。

## 6. 复现、记录与改动范围

在仓库根目录构建（将 `ON` 改为 `OFF`、使用独立目录即可复现直接构建）：

```sh
cmake -B build-android-lut \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/Android/Sdk/ndk/30.0.14904198/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16 \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=ON -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF
cmake --build build-android-lut -j 8
```

部署模型到测试目录的 `models/ifairy.gguf`，将相应 `bin/` 下的执行文件和四个 `lib*.so` 放到手机测试目录的 `direct/`、`lut/`，复制 `tests/ifairy-test-data/` 和长提示词，然后执行报告保存的脚本。设备端单独生成命令：

```sh
cd /data/local/tmp/ifairy-validation-20260923
LD_LIBRARY_PATH="$PWD/lut" GGML_IFAIRY_LUT=1 ./lut/llama-cli \
  -m models/ifairy.gguf -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 \
  -p 'The capital of France is' -n 64 --seed 42 --temp 0 \
  -no-cnv --no-display-prompt --simple-io

LD_LIBRARY_PATH="$PWD/lut" GGML_IFAIRY_LUT=1 ./lut/test-ifairy-real-model \
  models/ifairy.gguf results/recheck batched 8
```

探针输出两份 `.f32` 原始 logits 和 JSON；将 `batched` 改为 `serial` 可复现另一种 prefill。二进制 logits 留在手机和本机 `/tmp/ifairy-phone-results`，仓库只保留日志、哈希和差异摘要。原始材料索引见 [记录目录](validation/android-sm-f9660-20260923/)。

新增 C++ 已按目标 merge-base 执行 `git clang-format --diff`，没有格式差异；为处理临时索引跨文件系统写入失败，在 `/tmp` 的本地共享克隆中检查完全相同的源码，实际仓库暂存区未改变。`clang-tidy -p build-rel` 只检查新增 C++ 文件，退出 0，保留一个关于系统 `rusage` 头文件归属的 include-cleaner 告警，源码已包含公开的 `<sys/resource.h>`。未修改或调优生产内核，未提交模型权重、手机二进制或其他模型架构。
