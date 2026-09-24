# Android SM-F9660 实验原始记录

| 文件 | 内容 |
| --- | --- |
| `device-info.log`、`environment.json` | 手机、工具链和构建配置 |
| `compile-evidence.json` | 真实 ARM 编译命令 |
| `deployment-manifest.json` | 部署到手机的程序/库大小与 SHA-256；二进制本身不在仓库 |
| `model-device-sha256.stdout.log` | 手机上对同一 GGUF 的哈希校验 |
| `run-functional.sh` | 23 项顺序执行的设备端回归、真实模型、文本和内核测试 |
| `run-bench.sh` | 4 组顺序执行的基准，含每组前后温度、频率上限和电源快照 |
| `<tag>.stdout.log`、`<tag>.stderr.log` | 每项原始输出；空文件表示该流没有输出 |
| `<tag>.exit`、`.started`、`.finished` | 设备端命令退出码和秒精度 UTC 起止时间 |
| `<tag>.json` | 本机构建/工具调用的参数、耗时与退出码；与 stdout 内容分开保存 |
| `correctness-summary.json` | 全部 logits SHA-256、生成 token IDs、RSS、设备内一致性及跨平台差异 |
| `probe-host-crosscheck.json` | 新原生探针与此前本机 Python 探针的一致性 |
| `device-exit-summary.json` | 27 项设备端命令全部退出 0；数值比较另见 correctness 汇总 |
| `benchmark-summary.json`、`thermal-summary.json` | 所有性能样本和对应温度/频率快照 |
| `clang-format-final.log`、`clang-tidy-final.*` | 新增 C++ 的格式与静态检查 |

二进制 logits 在手机测试目录及本机 `/tmp/ifairy-phone-results`，未写入仓库。每份 `.f32` 按步连续存放 33 × 32000 个 little-endian F32；前 32000 个值为同一提示词的 prefill logits。完整文件 SHA-256 用于同路径一致性，前一向量用于批量/串行及跨平台误差比较；生成分歧之后的不同序列不用于该误差统计。

记录保留了初始 ADB 沙箱连接失败及成功重试；这类工具准备失败不是设备推理失败。`clang-format.log` 是临时索引路径失败记录，`clang-format-final.log` 为成功的最终检查。手机基准期间，温控和频率上限改变，报告没有将四组测试包装成相同温度下的性能对照。
