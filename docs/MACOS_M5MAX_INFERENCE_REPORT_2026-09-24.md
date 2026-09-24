# Apple M5 Max / macOS 真实模型验证

日期：2026-09-24。推理与转换源码提交：`6fbc73c4723cb0295d5300088f776fcfb09991e0`。本次从固定的 Hugging Face 快照下载模型，使用仓库原有脚本转换，再运行 CLI、原生数值探针和顺序吞吐测试。未修改推理或转换实现。

原始命令、退出码、时间、环境、日志和摘要保存在 [macos-m5max-20260924](validation/macos-m5max-20260924/)。权重和 GGUF 位于 Git 忽略的 `models/`，完整 logits 位于本机 `/tmp/ifairy-macos-validation`，均不提交。

## 1. 环境

| 项目 | 实测值 |
| --- | --- |
| 计算机 | Apple M5 Max，18 个物理/逻辑 CPU 核心，128 GiB RAM |
| 操作系统 | macOS 27.0，build 26A428，arm64 |
| 编译器 | Apple clang 21.0.0 (`clang-2100.3.34.2`) |
| 工具链 | Command Line Tools `27.0.0.0.1788430756`；macOS SDK 27.0 |
| 构建工具 | CMake/CTest 4.4.2，GNU Make 3.81 |
| Python | 3.12.13 |
| 转换依赖 | PyTorch 2.14.0；NumPy 2.5.3；safetensors 0.8.0；Transformers 4.52.4；tokenizers 0.21.4；huggingface_hub 0.36.2；本仓库 gguf 0.17.1 |
| CPU 构建 | Release，`-O3 -DNDEBUG`，native ARM；OpenMP OFF、LLAMAFILE ON、ccache OFF、curl OFF，构建 `-j 8` |
| 推理 | CPU only，`-ngl 0`；GPU 未参与 |
| 供电与调度 | AC 电源，电量记录 80%，系统自动电源模式；未绑定 CPU、未锁频、未修改温控 |

完整依赖见 [pip freeze](validation/macos-m5max-20260924/environment-python-packages.stdout.log)，实际编译命令见 [compile-evidence.json](validation/macos-m5max-20260924/compile-evidence.json)。每组基准前后记录 `pmset -g therm`；均没有已记录的热/性能警告。该命令不是连续温度或频率遥测，不能由此证明完全没有限频。系统桌面应用继续运行。

## 2. 模型、下载和转换

- 来源：[PKU-DS-LAB/Fairy-plus-minus-i-700M](https://huggingface.co/PKU-DS-LAB/Fairy-plus-minus-i-700M)。
- 固定 revision：`c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b`。
- `model.safetensors`：3,112,026,544 字节，SHA-256 `7d098568f2e51cec2ba2540d29830cae90466a015b44f04d4606ed97bdd4a535`。
- 生成的 GGUF：576,123,520 字节，SHA-256 `39f77bb18ba42b4612dd2bd770f7267143e7e20be6038ec8a33028a48851721e`。
- 架构 `ifairy`，24 层，embedding 1536，FFN 4096，16 heads，词表 32000，context 2048；267 个张量：168 IFAIRY、98 F32、1 F16。

GGUF 与先前 x86/Android 报告中的 `2211f40e…da6b3` 文件大小及记录的模型形状相同，但 **SHA-256 不同**。未获得旧文件进行逐张量对照，不能断言两份权重等价；跨设备吞吐表是各自实测结果，不是严格同文件的硬件对照。

| 阶段 | 墙钟时间 | 说明 |
| --- | ---: | --- |
| 下载完整快照 | 118.275 s | 10 个文件，实际网络条件下的首次下载 |
| 转换为 GGUF | 64.947 s | 原始 `gguf-py/convert_ifairy.py`，在模型目录运行 |
| 转换峰值 RSS | 5.45 GiB | `/usr/bin/time -l`：5,850,218,496 字节，整进程 |
| 直接构建 configure / build | 2.782 / 16.795 s | 新建构建目录 |
| LUT 构建 configure / build | 2.456 / 17.028 s | 新建构建目录 |

构建与下载阶段部分重叠；下面的推理吞吐阶段没有本任务的构建、下载或转换并行。转换退出 0 并输出成功消息；日志保留 tokenizer 的 `Unknown separator token '<s>' in TemplateProcessing<pair>` 提示，后续模型加载和生成均成功。模型逐文件指纹见 [checkpoint.json](validation/macos-m5max-20260924/checkpoint.json)，转换日志见 [convert.stderr.log](validation/macos-m5max-20260924/convert.stderr.log)。

## 3. README 生成示例

运行 README 中的直接模式命令：4 个推理/批处理线程，context 2048，batch/ubatch 512，seed 42、temperature 0，提示词 `The capital of France is`，生成上限 32 tokens。

实际 stdout：

```text
The capital of France is Paris, the largest city in the country and the second most populous in France after Lyon. The city is also known as the "City of Light" and
```

这是实际模型生成，含事实错误，不作为正确答案。命令退出 0，墙钟时间 **0.675 s**，包含进程启动和加载；此前已做模型探针，文件缓存是热的。CLI 报告 load 84.16 ms，5-token prompt 30.76 ms，31 次计时 decode 259.18 ms。进程墙钟与 CLI 内部计时范围不同，不能混用。日志见 [stdout](validation/macos-m5max-20260924/cli-demo.stdout.log)、[stderr](validation/macos-m5max-20260924/cli-demo.stderr.log) 和 [命令记录](validation/macos-m5max-20260924/cli-demo.json)。

## 4. 功能和数值验证

| 检查 | 结果 |
| --- | --- |
| 直接构建 | CTest 3/3，通过；2.238 s |
| LUT 构建 | CTest 6/6，通过；2.333 s |
| 普通 matmul 回归 | 144 次计算图执行通过，含独立复数参考、尾部、跨步、广播、范围和资源回退检查 |
| 架构隔离 | `scripts/check-ifairy-only.py` 通过 |
| 真实模型实际执行 | 调试用例记录 336 次 `ifairy_lut: executed MUL_MAT` |
| 有限 logits | 七个独立进程，每个 2,112,000 个值，共 **14,784,000**，全部有限 |
| 独立重复 | auto 与 repeat 的两组完整 logits trace 逐字节相同 |
| 实现选项对照 | auto 与 lut16 完整 trace 相同 |
| prefill 对照 | auto 的 batched / serial 完整 trace 相同 |
| 较长生成 | 故事提示词 128-token 生成完成，墙钟 1.771 s |
| 长提示词 | 832-token prefill + 256-token 生成完成，墙钟 7.104 s |

七种配置为直接关闭、auto、auto 独立重复、lut16、lut_c、auto 串行 prefill、相同行级激活量化的直接对照。均使用同一个 LUT 构建、8 线程、context 2048、batch/ubatch 512、F16 KV、Flash Attention OFF。两提示词分别为 `The capital of France is` 和 `Once upon a time, in a small village`，每个含 prefill 后及 32 步贪心生成的 33 个 logits 向量。由原生探针和独立 NumPy 读取各自检查有限性；SHA-256 用于完整 trace 比较。

各模式的 logits 并不普遍相同。auto 对默认直接路径的两个首 logits 向量最大绝对差为 **3.08130 / 1.43140**；对 lut_c 为 **2.63477 / 1.32568**；对相同行级量化直接对照为 **1.00928 / 1.22363**。生成分歧后不把后续 logits 差解释为同输入误差。这些检查不证明 PyTorch 对齐、困惑度或任务准确率；长文本存在重复，未进行语言质量验收。

原生探针整进程峰值 RSS：默认直接 **1,144.91 MiB**，auto **1,653.81 MiB**。macOS 的 `ru_maxrss` 单位为字节，摘要转换为 MiB。详情见 [数值摘要](validation/macos-m5max-20260924/correctness-summary.json) 和 [回归详细日志](validation/macos-m5max-20260924/ctest-lut-detail.log)。本次没有重跑先前 x86 报告的真实模型逐算子 oracle 或 sanitizers，不将其结果当成本机结果。

## 5. CPU 吞吐

同一个 LUT 构建，运行时分别使用直接和 auto 模式。命令使用 `-ngl 0 -b 512 -ub 512 -p 128,512 -n 128 -r 3 -o json`，F16 KV，Flash Attention OFF，保留默认 warmup。每项三次测量，下表为平均 tokens/s；原始样本与标准差保留在 JSON 中。顺序为 4t direct → 4t auto → 8t auto → 8t direct，所有进程串行运行，组间没有额外冷却等待。

| 线程 | 模式 | pp128 | pp512 | tg128 |
| ---: | --- | ---: | ---: | ---: |
| 4 | 直接 | 171.31 | 158.28 | 111.67 |
| 4 | LUT auto | 190.01 | 174.02 | 94.24 |
| 8 | 直接 | 279.55 | 260.95 | 129.42 |
| 8 | LUT auto | 302.66 | 288.07 | 104.97 |

这些结果仅反映此模型文件、编译设置和桌面运行条件。模式之间的完整输出不相同，不作为等精度的内核加速比。基准不计入下载、转换和模型加载耗时；warmup 保持开启。见 [吞吐摘要及全部样本](validation/macos-m5max-20260924/benchmark-summary.json) 和各 `bench-*.json` 命令记录。

## 6. 复现

先按 README 配置 Python 环境。仓库根目录下运行：

```sh
. .venv/bin/activate
python docs/validation/macos-m5max-20260924/run-validation.py environment
python docs/validation/macos-m5max-20260924/run-validation.py build
python docs/validation/macos-m5max-20260924/run-validation.py download
python docs/validation/macos-m5max-20260924/run-validation.py convert
python docs/validation/macos-m5max-20260924/run-validation.py functional
python docs/validation/macos-m5max-20260924/run-validation.py bench
python docs/validation/macos-m5max-20260924/summarize.py
```

脚本以自身位置确定仓库根目录，保存命令、退出码和墙钟时间；固定 `models/` 中的模型路径和 `/tmp/ifairy-macos-validation` 中的 logits 路径。重跑会更新同名日志和摘要，下载缓存与已有构建目录会改变耗时；测量 fresh build 时应使用新的构建目录。不要同时运行其他 `llama-bench` 进程。
