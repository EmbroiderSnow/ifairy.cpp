# iFairy CPU inference

CPU inference for iFairy models, with checkpoint conversion and text generation through `llama-cli`.

## 1. System requirements

### Software dependencies and tested versions

Building requires a C11/C++17 compiler, **CMake ≥ 3.14**, and a build tool such as Make. The C++ dependencies are bundled in this repository; system runtime libraries come with the operating system/toolchain. Git is needed to clone the source (tested on macOS: 2.54.0, Apple Git-157). Android deployment additionally requires `adb`; its version was not recorded in the device validation records. Python and internet access are required to download and convert the demo model.

| Component | macOS / ARM64 | Linux / x86_64 | Android / ARM64 |
| --- | --- | --- | --- |
| Tested operating system | macOS 27.0, build 26A428 | Arch Linux, kernel `7.2.3-arch1-3` | Android 16 / API 36, kernel `6.6.98-android15-8` |
| Test device | Apple M5 Max, 128 GiB RAM | AMD Ryzen 7 H 255, approximately 30 GiB RAM | Samsung SM-F9660 / SM8750, 10.83 GiB RAM |
| Compiler | Apple clang 21.0.0 (`clang-2100.3.34.2`) | GCC `16.2.1 20260810` | NDK Clang 21.0.0, build 14475230 |
| SDK / toolchain | Command Line Tools `27.0.0.0.1788430756`; macOS SDK 27.0 | Native GCC toolchain | NDK `30.0.14904198` (r30-beta1); API 28 build target, `arm64-v8a`, `c++_static` |
| Build tools | CMake/CTest 4.4.2; GNU Make 3.81 | CMake 3.22.1-g37088a8; Make | Cross-compiled on the Linux host using CMake and the NDK |

Real-model inference was tested on all three systems. The complete download, conversion and generation demo below was tested on macOS with these Python dependencies:

| Dependency | Tested version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch / safetensors | 2.14.0 / 0.8.0 |
| Transformers / tokenizers | 4.52.4 / 0.21.4 |
| huggingface_hub | 0.36.2 |
| NumPy / PyYAML / tqdm | 2.5.3 / 6.0.3 / 4.70.1 |
| Bundled `gguf` | 0.17.1, installed from `gguf-py/` |

The [complete Python dependency list](docs/validation/macos-m5max-20260924/environment-python-packages.stdout.log) includes transitive dependencies and their tested versions.

### Hardware requirements

No non-standard hardware or GPU is required. The demo downloads approximately **3.12 GB** and produces a **0.58 GB** GGUF; allow additional disk space for Python packages and build files. On the test Mac, conversion used **5.45 GiB peak RAM**, and direct inference used approximately **1.12 GiB** at context length 2048. These are process measurements; leave additional memory for the operating system.

## 2. Installation guide

Prebuilt binaries for macOS ARM64, Linux x86_64 and Android ARM64 are available
from [GitHub Releases](https://github.com/EmbroiderSnow/ifairy.cpp/releases).
See the included README or [binary usage instructions](docs/binary-usage.md) for
minimum system requirements, checksum verification and installation. With a
prebuilt archive, use its `bin/llama-cli` in place of `build-direct/bin/llama-cli`
in the demo below; the Python download and conversion steps still apply.

Install the compiler, CMake, Make and Python 3.12 first. On macOS, `xcode-select --install` installs the Command Line Tools. From a terminal:

```sh
git clone https://github.com/EmbroiderSnow/ifairy.cpp.git
cd ifairy.cpp

cmake -S . -B build-direct \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=OFF \
  -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF
cmake --build build-direct -j 8

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt \
  "torch==2.14.0" "safetensors==0.8.0" \
  "transformers==4.52.4" "tokenizers==0.21.4" \
  "huggingface_hub==0.36.2" "numpy==2.5.3" \
  "PyYAML==6.0.3" "tqdm==4.70.1"
```

The executable is `build-direct/bin/llama-cli`. Reduce `-j 8` if fewer cores or less memory are available.

### Typical installation time

Allow **1–5 minutes** to compile on a conventional desktop with prerequisites installed. The fresh direct build on the Apple M5 Max took **2.78 s to configure and 16.80 s to compile** using eight jobs. This excludes tool installation, Python package downloads and model download/conversion.

## 3. Demo

The demo downloads [PKU-DS-LAB/Fairy-plus-minus-i-700M](https://huggingface.co/PKU-DS-LAB/Fairy-plus-minus-i-700M), converts it to GGUF and generates a continuation of the short input prompt `The capital of France is`. Small numerical reference datasets are also included in [tests/ifairy-test-data](tests/ifairy-test-data/).

Run the following steps in the same shell, from the repository root, with `.venv` activated.

### Download the model

Download the complete snapshot, including weights, configuration and tokenizer files:

```sh
repo_dir="$PWD"
model_dir="$repo_dir/models/Fairy-plus-minus-i-700M"
gguf_file="$repo_dir/models/Fairy-plus-minus-i-700M.gguf"

python - <<'PYMODEL'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PKU-DS-LAB/Fairy-plus-minus-i-700M",
    revision="c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b",
    local_dir="models/Fairy-plus-minus-i-700M",
)
PYMODEL
```

### Convert to GGUF

The converter reads tokenizer files from the current directory, so run it inside the downloaded model directory:

```sh
(
  cd "$model_dir" &&
  PYTHONPATH="$repo_dir/gguf-py" \
    python "$repo_dir/gguf-py/convert_ifairy.py" . "$gguf_file"
)
```

Successful conversion creates `models/Fairy-plus-minus-i-700M.gguf` and prints `转换成功！GGUF 文件已保存至:` followed by the output path.

### Run inference

```sh
GGML_IFAIRY_LUT=0 "$repo_dir/build-direct/bin/llama-cli" \
  -m "$gguf_file" -ngl 0 -t 4 -tb 4 \
  -c 2048 -b 512 -ub 512 \
  -p "The capital of France is" -n 32 \
  --seed 42 --temp 0 -no-cnv --simple-io
```

### Expected output

The CLI prints model-loading information, the prompt and generated text, followed by timing diagnostics. A successful run exits with status `0` after at most 32 new tokens. The observed output on the test Mac began:

```text
The capital of France is Paris, the largest city in the country
```

This is an excerpt of generated text; the exact continuation can vary by platform and settings.

### Expected runtime

Measured on the Apple M5 Max on 2026-09-24:

| Stage | Wall time |
| --- | ---: |
| Download the model snapshot | 118.28 s |
| Convert to GGUF | 64.95 s |
| Load and generate up to 32 tokens | 0.68 s |

The CLI measurement used four threads and warm filesystem caches, and includes process startup and model loading. On a conventional desktop, allow **several minutes for preparation and a few seconds for generation** as a planning estimate. Download time depends on the network; conversion and generation times depend on the hardware. Measurements are recorded in the [demo timing record](docs/validation/macos-m5max-20260924/summary.json).

## 4. Instructions for use

### Run on your own text

Save a UTF-8 prompt in a text file and provide a compatible iFairy GGUF checkpoint:

```sh
./build-direct/bin/llama-cli \
  -m /absolute/path/ifairy.gguf -ngl 0 -t 4 -c 2048 \
  -f /absolute/path/prompt.txt -n 128 \
  --seed 42 --temp 0 -no-cnv --simple-io
```

Use `-p "your prompt"` instead of `-f` for inline text. Adjust `-n` for the output length and `-c` for the context length supported by your model. This repository loads only checkpoints with `general.architecture=ifairy`.

For another original iFairy checkpoint, follow the conversion step above with your own input directory and output path. The input directory must contain `config.json`, all `.safetensors` weights, the shard index if applicable, and compatible tokenizer files including `tokenizer.json`.

To use LUT, build into `build-lut` with `-DGGML_LEGACY_IFAIRY_CPU_LUT=ON` instead of `OFF`, then use `build-lut/bin/llama-cli` with `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=auto` and the same inference arguments.

## 5. License and source code

This code is distributed under the **MIT License**; see [LICENSE](LICENSE) and the included [third-party licenses](licenses/).

Source repository: [EmbroiderSnow/ifairy.cpp](https://github.com/EmbroiderSnow/ifairy.cpp).
