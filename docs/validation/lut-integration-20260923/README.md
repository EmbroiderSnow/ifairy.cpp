# 2026-09-23 普通 MUL_MAT LUT 接入与复测证据

总结见 [实现与两平台复测报告](../../LUT_MUL_MAT_INTEGRATION_REPORT_2026-09-23.md)。旧测试记录保留在相邻的 `local-x86-20260923`、`android-sm-f9660-20260923` 目录，新旧结果不得混用。

## 复现与原始记录

- `run-host.py`：本机 build/probes/bench 三阶段；`*-commands.json` 保存实际执行参数、环境、时间、返回码。模型始终在仓库外。
- `run-phone-functional.sh`：手机 6 项回归、7 个真实模型探针、3 个 CLI 测试，16 个命令均已完成并拉取，见 `phone-functional-exit-summary.json`。
- `run-phone-bench.sh`：手机 4/8 线程 LUT ON/OFF 顺序基准；组间 90 秒空闲，记录温度与频率限制。重新连接后已取回四组完整结果，设备端退出码均为 0，共 36 个样本；断线未中止设备基准，见已更新的 `phone-interruption.json`。
- `summarize_phone_bench.py`：从原始 JSON、退出码、时间戳及温控快照重建 `phone-benchmark-summary.json` / `phone-thermal-summary.json`，检查组间 90 秒空闲及测量配置。所有样本保留，没有观察到手机 LUT 加速；注意温控和量化差异。
- `phone-lut-dispatch-final.log`：重连后最终回归程序在手机通过 144 次计算图执行。`android-pending-regression.json` 保留先前待部署工件记录并更新为部署/验证完成，附运行命令和返回码。
- `host-benchmark-summary.json`：本机可引用的 5 组结果。`host-bench-4t-off` 为受静态检查干扰的首轮，排除；使用后续独立复测 `host-bench-4t-off-clean`。
- `summarize.py host /tmp/ifairy-lut-integration`、`summarize.py phone /tmp/ifairy-lut-phone-results-resumed`：从完整 logits 重算有限值检查、SHA-256、首 logits 和生成 token 对照。结果在 `*-correctness-summary.json`。
- `check_matmul_oracle.py BUILD MODEL`：按本仓库 Linux x86_64 ABI，对模型实际计算的 IFAIRY matmul 抽取首/中/末行、首/末列，用独立双精度复数和验证。设置 `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=auto` 检查 LUT；设置 `GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1`，使用 `build-lut-reference` 检查同量化直接路径。两次各 336 算子、3,006 分量，均无失败。诊断回调改变图分段，不用于计时。
- `host-cli-debug.stderr.log` / `phone-cli-debug.stderr.log`：各 336 条实际 `executed MUL_MAT` 记录。此计数包括预热，不是仅凭资格检查日志推断。

## 检查与版本

`build-configurations.json` 保存实际 CMake 选项；`source-sha256.json` 保存 C/C++ 变更内容哈希；`android-deployment.json` 保存初次部署的剥离调试符号二进制哈希，`android-deployment-final.json` 保存重新部署最终回归程序后的清单，其余文件不变。`phone-resumed-artifacts.sha256` 是重连后在设备上重新计算的全部二进制及模型指纹，已与清单和本机模型指纹核对。实验运行时改动尚未提交，二进制中的 `build_commit` 仍指向基线 `a6256d6`；已验证的源码随后归档于提交 `a243488`，与 `source-sha256.json` 一致，不可将二进制的基线字段当作新实现的唯一标识。

`git-clang-format.log` 为对 merge-base 的变更文件检查。为避免原工作区只读 Git 索引限制，在 `/tmp` 的共享克隆中复制完全相同文件并标记新增文件后执行 `git clang-format --diff`；`source-sha256.json` 可核对复制内容。主工作区索引未修改。

`clang-tidy-summary.json` 及日志保存变更源文件的检查。直接构建数据库不包含 LUT 专属源文件时补充 include 参数，另外用 `build-lut` 数据库检查实际启用分支；有告警，退出码均为 0。未扫描无关源文件。

`sanitize-test.log` 是首次检测到旧打包元数据泄漏的记录；修复后的通过记录为 `sanitize-retest.log`。`sanitize-result.json` 区分初次失败与最终结果。测试保持 AddressSanitizer、UndefinedBehaviorSanitizer 和泄漏检查开启。

仓库不保存 GGUF、原始 `.f32` logits、模型分片或测试 ELF 文件。大文件位于 `/tmp` 和手机测试目录；本目录仅含脚本、文字日志、汇总及校验和。
