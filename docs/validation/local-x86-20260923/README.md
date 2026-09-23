# 2026-09-23 本机 x86 验证原始记录

主报告：[本机 x86 iFairy 700M 推理与 LUT 验证](../../LOCAL_X86_LUT_INFERENCE_REPORT_2026-09-23.md)。

| 文件 | 内容 |
| --- | --- |
| `environment.json`、`hardware.stdout.log`、`compile-evidence.json` | 系统、模型指纹及真实编译命令 |
| `commands.json` | 按时间排列的 42 条命令，含工作目录、参数、退出码和墙钟耗时 |
| `<tag>.json` | 单条命令元数据；不等同于命令的输出结果 |
| `<tag>.stdout.log`、`<tag>.stderr.log` | 原始输出；空日志表示该流没有输出 |
| `ctest-*-detailed.log` | CTest 完整测试输出，包括预期的架构拒绝信息 |
| `check_logits.py` | 当前提交的 ctypes 真实模型数值探针，无额外 Python 依赖 |
| `correctness-summary.json` | logits 哈希、LUT 开关一致性、批量/串行差异及关闭 LLAMAFILE 的对照结果 |
| `benchmark-summary.json` | 三组顺序运行的 llama-bench 原始 JSON 结果集合 |
| `long-prompt.txt` | 长输入测试用的人工构造重复段落 |

数值探针额外支持 `IFAIRY_PROBE_DUMP_DIR=/tmp/some-directory`，将两组提示词的首个完整 logits 向量写为本机端序 F32 文件 `prompt-0.f32`、`prompt-1.f32`，便于复算报告中的 prefill 最大/平均绝对差值。报告中的转储保留在 `/tmp/ifairy-prefill-{batched,serial}` 和 `/tmp/ifairy-reference-{batched,serial}`，未作为仓库交付文件；完整复现命令已写入 `commands.json`。

这些记录均不含模型权重。首次 `cli-direct-smoke` 与 LUT 编译有时间重叠，其 CLI 计时未用于报告性能表；正式 `bench-*` 测量在构建和正确性测试全部结束后串行执行。
