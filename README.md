# llama.cpp — iFairy CPU 推理专用版

从原仓库提交 `fe79b5dc5a449922caa64a37451948f30c124318` 整理，唯一可加载的模型架构是 GGUF `general.architecture = "ifairy"`。其他架构在加载器中明确拒绝。保留 llama.cpp 的通用分词、采样、KV cache、线程调度和 iFairy CPU 实现。

Fairy2i 模型加载、W1/W2/tile64/bundle 专用实现和转换器已从源码删除；其他模型的计算图构造器、GPU 后端实现和无关应用也已删除。原始 iFairy 的 GGUF 类型编号及复数计算语义保持不变。现有 legacy `IFAIRY64` / `IFAIRY_WIDE_LINEAR_W2` 作为 iFairy 底层兼容算子保留，它们不提供其他模型的加载或推理入口。

完整盘点、已知缺口与验证结果见 [整理说明](docs/IFAIRY_INVENTORY.md)。

## 构建与测试

默认构建直接 CPU 推理路径，不依赖 Python 或模型权重：

```sh
cmake -B build-direct -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-direct -j 8
ctest --test-dir build-direct --output-on-failure
```

包含历史 LUT 内核和它们的回归测试：

```sh
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=ON
cmake --build build-rel -j 8
ctest --test-dir build-rel --output-on-failure
python3 scripts/check-ifairy-only.py
```

**当前普通 iFairy `MUL_MAT` 实际仍走 vecdot。** `GGML_IFAIRY_LUT=1` 会做预打包，但此提交的普通 matmul 没有接入 LUT 执行分派。LUT 算子单测通过、或日志 `can_mul_mat=true`，均不意味着整模型已使用 LUT 加速。默认直接路径已完成合成模型的 prefill/decode 回归。

## 运行真实模型

```sh
./build-direct/bin/llama-cli -m /absolute/path/ifairy.gguf \
  --gpu-layers 0 -t 4 -b 32 -p "I believe life is" -n 32 \
  --seed 42 --temp 0 -no-cnv
```

仓库不附带真实权重；本次没有完成真实 700M 权重生成质量、困惑度或吞吐验证。

## GGUF 转换

保留 `gguf-py/convert_ifairy.py`。它需要原有 Python 依赖（PyTorch、safetensors、NumPy 和 gguf 包依赖）；这些依赖不影响 C++ 推理构建，可在虚拟环境中从仓库根目录执行 `pip install -r requirements.txt` 安装。转换器目前从工作目录读取 tokenizer，因此应进入 checkpoint 目录再运行：

```sh
cd /absolute/path/checkpoint
PYTHONPATH=/absolute/path/llama.cpp-ifairy/gguf-py \
  python3 /absolute/path/llama.cpp-ifairy/gguf-py/convert_ifairy.py \
  . /absolute/path/ifairy.gguf
```

模型转换未用真实 checkpoint 验证。参考测试 JSON 已随源码附带；如需重新生成，安装 NumPy 后运行 `python3 tests/test-ifairy-ref.py`。

## 保留的工具

- `llama-cli`、`llama-bench`、`llama-perplexity`
- `ifairy-actq-microbench`、`ifairy-vecdot-microbench`
- 开启 LUT 构建时的 `ifairy-microbench`

沿用原项目 [MIT 许可证](LICENSE) 和第三方许可证。
