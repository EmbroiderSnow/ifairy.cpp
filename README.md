# iFairy CPU inference with llama.cpp

This repository provides a CPU inference implementation for the original iFairy model. It loads GGUF files with `general.architecture = "ifairy"` and implements model loading, prompt processing (prefill), autoregressive decoding, token sampling, and KV caching. Complex matrix multiplication preserves the original `w * conj(x)` convention and packed BF16 representation.

The code was extracted from llama.cpp revision `fe79b5dc5a449922caa64a37451948f30c124318`. The tested extraction revision is `609eacf`. Other model graph builders, Fairy2i-specific inference and conversion implementations, and GPU backend implementations have been removed. Other GGUF architectures are rejected during model loading. Legacy `IFAIRY64` and `IFAIRY_WIDE_LINEAR_W2` primitives remain for compatibility; they do not provide an entry point for loading other model architectures.

The package includes source code, small synthetic reference datasets, and a self-contained inference demo. No pretrained model, Python installation, GPU, or network connection is required to run the C++ demo once the build tools are installed. See the [code inventory and implementation limitations](docs/IFAIRY_INVENTORY.md) for further details.

## 1. System requirements

### Required software and tested versions

The installation and demo below were verified on the following system. Version numbers describe the tested environment, not a claim that every other version is supported.

| Component | Requirement | Tested version or configuration |
| --- | --- | --- |
| Operating system | macOS with the Command Line Tools and SDK for a native build | macOS 27.0, build 26A428, arm64 |
| C/C++ compiler | C11 and C++17 support | Apple clang 21.0.0 (`clang-2100.3.34.2`) |
| Apple Command Line Tools / SDK | Required for the tested macOS build | Command Line Tools `27.0.0.0.1788430756`; macOS SDK 27.0 |
| CMake and CTest | CMake ≥ 3.14, as declared by the project | 4.4.2 |
| Build tool | Make for the commands below | GNU Make 3.81 |
| C/C++ runtime, system threads, Accelerate | Supplied by the tested operating system and toolchain | Bundled macOS versions; no separate installation |
| ggml and other bundled C/C++ dependencies | Use the copies included in this source package | Versions vendored in extraction revision `609eacf` |

The validated build explicitly disables OpenMP, ccache, and libcurl. These are not prerequisites. A system-installed ggml library is not used. Git is only needed if obtaining the source through Git; it is not required when using the source archive.

Only the macOS/arm64 environment above has been validated for this extraction. Existing generic CPU and other architecture-specific fallback code remains, but Linux, Windows, Android, x86, and AVX512 builds have not been validated here.

### Hardware

No non-standard hardware or accelerator is required. The software uses the CPU; no CUDA toolkit, GPU, or GPU memory is needed. ARM NEON/dotprod paths are selected when available. The test computer was an **Apple M5 Max with 128 GiB of RAM**. That memory capacity is not a requirement for the small demo. For a real model, available RAM must accommodate the model, KV cache, and compute buffers; requirements therefore depend on the checkpoint and context length.

### Optional Python dependencies

Python is only needed for reference-data regeneration, the source-isolation audit, or checkpoint conversion. It is not part of the C++ installation or demo.

| Workflow / dependency | Version information | Validation status |
| --- | --- | --- |
| Reference-data generator: Python / NumPy | Python 3.12.13 / NumPy 2.5.3 | Used to generate the bundled reference JSON files |
| Source-isolation audit | Python 3; standard library only | Executed successfully |
| Local `gguf` package | 0.17.1, from `gguf-py/` | Package import and iFairy type mapping checked |
| `gguf` import dependencies | NumPy 2.5.3, PyYAML 6.0.3, tqdm 4.70.1 | Used in the import check; declared lower bounds are in [package metadata](gguf-py/pyproject.toml) |
| Full checkpoint conversion | PyTorch, safetensors, and Transformers, in addition to the dependencies above | Versions are not pinned in this package; no complete checkpoint conversion has been validated |

The converter calls a Hugging Face tokenizer through Transformers. Install that dependency explicitly if using the optional conversion workflow; it is not currently included in the root `requirements.txt`. This repository does not yet supply a validated, version-locked conversion environment.

## 2. Installation guide

### Obtain the source

Extract the supplied source archive and enter its root directory, which contains `CMakeLists.txt`:

```sh
tar -xzf llama.cpp-ifairy-source.tar.gz
cd llama.cpp-ifairy
```

Alternatively, clone this inference repository:

```sh
git clone https://github.com/EmbroiderSnow/ifairy.cpp.git
cd ifairy.cpp
```

If the source is already available as a checkout, enter that directory instead. The related research project linked below is separate from this inference repository.

### Build the CPU runtime and demo

Ensure that the compiler, CMake, and Make listed above are installed. On macOS, `xcode-select --install` installs the Command Line Tools if they are missing; CMake must also be available on `PATH`. Check your tools with `clang --version`, `cmake --version`, and `make --version`.

Run from the source root:

```sh
cmake -S . -B build-direct \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_CCACHE=OFF
cmake --build build-direct -j 8
```

Executables and their shared libraries are placed in `build-direct/bin/`. Run them from that directory layout; no system-wide installation or administrator privileges are needed. Reduce `-j 8` on a computer with fewer available cores or limited memory.

### Installation time

A fresh build in an empty build directory on the test computer took **2.55 seconds to configure and 16.00 seconds to compile**, approximately **19 seconds in total**, using eight parallel build jobs and no compiler cache. This includes the retained tools and tests, and excludes installing the compiler/CMake or downloading the source.

For planning on a conventional desktop with the prerequisites already installed, allow **1–5 minutes**. This is an estimate, not a measurement on additional computers. Tool installation and optional Python dependency downloads can take longer and are not included.

## 3. Demo: inference on a small synthetic dataset

### Included data

The primary demo is [test-ifairy-model.cpp](tests/test-ifairy-model.cpp). It generates a deterministic, one-layer iFairy GGUF fixture locally with embedding and feed-forward widths of 256, four attention heads, a vocabulary size of 32, and context length 32. It uses non-zero synthetic weights and token IDs `[1, 2, 3, 4]`.

The fixture is written to the operating system's temporary directory, loaded by the normal model loader, and deleted after the test. It is intended to demonstrate inference mechanics rather than language quality: it has no text tokenizer and does not generate meaningful sentences.

Small numerical reference datasets are also included:

- [quant_test.json](tests/ifairy-test-data/quant_test.json): quantization/dequantization reference data.
- [rope_test.json](tests/ifairy-test-data/rope_test.json): rotary-position reference data.
- [matmul_test.json](tests/ifairy-test-data/matmul_test.json): complex matrix-multiplication reference data.

These JSON files are consumed by the legacy test in the optional LUT build described below. They can be regenerated using [test-ifairy-ref.py](tests/test-ifairy-ref.py), which uses NumPy and random seed 42.

### Run the demo

After installation, run from the source root:

```sh
GGML_IFAIRY_LUT=0 ./build-direct/bin/test-ifairy-model
```

The demo checks model loading, three-token prefill, single-token decoding, serial versus batched prompt processing, finite logits, repeatability, and rejection of unsupported model architectures. It uses two CPU inference threads.

### Expected output

After diagnostic output, a successful run exits with status `0` and prints:

```text
iFairy model load, prefill, decode, repeatability and isolation passed; max difference 0.00000000
```

The displayed difference was zero on the tested system. The test requires a maximum absolute difference below `0.05` between batched and serial final logits, and exact repeatability when rerunning the same path.

Messages rejecting `fairy2i`, `llama`, `qwen3`, and `unknown` architectures are **expected negative-test output**, not a demo failure. They verify that this repository only loads iFairy models.

To run the complete default test suite:

```sh
ctest --test-dir build-direct --output-on-failure
```

Expected summary:

```text
100% tests passed out of 3
```

### Demo runtime

The standalone demo took **1.29 seconds** in the fresh-build verification. The subsequent three-test suite took **0.64 seconds** including process startup; caches were already warm. On a conventional desktop, allow **a few seconds, approximately 1–10 seconds**, as a planning estimate. These times exclude compilation and do not measure pretrained-model token throughput.

## 4. Instructions for use with your own data

### Run an existing iFairy GGUF model

Supply a compatible pretrained GGUF checkpoint whose architecture is `ifairy`, including its tokenizer metadata. Renaming another model's architecture field does not make its tensors compatible. Model weights are not included in this package.

```sh
./build-direct/bin/llama-cli \
  -m /absolute/path/ifairy.gguf \
  --gpu-layers 0 -t 4 -c 0 -b 32 \
  -p "I believe life is" -n 32 \
  --seed 42 --temp 0 -no-cnv
```

Replace the model path and prompt with your own. `-c 0` uses the model's declared context length; reduce the context if necessary for available RAM. The command prints generated text and runtime diagnostics, stopping after at most 32 generated tokens or an end-of-generation token. Actual text depends on the checkpoint and prompt; no reference sentence is prescribed.

To use a UTF-8 prompt file, replace `-p "I believe life is"` with `-f /absolute/path/prompt.txt`. Keep the same checkpoint, prompt, context, thread count, seed, and sampling settings when comparing repeated runs.

### Convert an original iFairy checkpoint (optional; not validated end to end)

The retained [conversion script](gguf-py/convert_ifairy.py) expects an original ComplexNetLM/iFairy checkpoint with `config.json`, its `.safetensors` weights, any required shard index, and compatible tokenizer files including `tokenizer.json`. It is not a converter for arbitrary Hugging Face models.

From the repository root, create an isolated Python environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt transformers
repo_dir="$PWD"
```

Then enter the checkpoint directory because the converter currently reads tokenizer files from the working directory:

```sh
cd /absolute/path/checkpoint
PYTHONPATH="$repo_dir/gguf-py" \
  python "$repo_dir/gguf-py/convert_ifairy.py" \
  . /absolute/path/ifairy.gguf
```

The output is an iFairy GGUF file that can be supplied to `llama-cli`. Record the installed package versions with `python -m pip freeze` if attempting conversion. No compatible real checkpoint was available for this extraction's validation, so conversion success, pretrained-model generation quality, perplexity, and throughput remain unverified.

## 5. Additional validation and reproducibility

To build the retained LUT primitives and run their regression tests:

```sh
cmake -S . -B build-rel \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_LEGACY_IFAIRY_CPU_LUT=ON \
  -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF
cmake --build build-rel -j 8
ctest --test-dir build-rel --output-on-failure
```

The recorded LUT build passed **5/5 tests**. The source-isolation check can additionally be run with Python:

```sh
python3 scripts/check-ifairy-only.py
```

Expected output:

```text
PASS: no Fairy2i production implementation; only the iFairy graph builder remains.
```

**Known LUT limitation:** ordinary iFairy `MUL_MAT` currently executes through vecdot. Enabling `GGML_IFAIRY_LUT=1` prepares packed weights but does not connect ordinary matmul to LUT execution. The legacy W2 primitive has a LUT execution path. Passing LUT primitive tests or seeing `can_mul_mat=true` is not evidence of end-to-end LUT acceleration.

The retained tools include `llama-cli`, `llama-bench`, `llama-perplexity`, `ifairy-actq-microbench`, and `ifairy-vecdot-microbench`; LUT builds additionally include `ifairy-microbench`. Do not run multiple `llama-bench` processes simultaneously when collecting performance measurements.

[Validation records](docs/validation/) and the [implementation inventory](docs/IFAIRY_INVENTORY.md) document the extraction checks. The demo reproduces these functional checks, not quantitative results from a manuscript. Reproducing manuscript results would additionally require the exact checkpoints, datasets, evaluation commands, and reference metrics, which are not supplied here.

## 6. License and source availability

This extracted llama.cpp inference repository is distributed under the **MIT License**, as specified in [LICENSE](LICENSE). Preserve the copyright notices and the applicable [third-party licenses](licenses/). A license stated for a separate research repository does not replace the license of this code package.

Related iFairy research project: [PKULab1806/Fairy-plus-minus-i](https://github.com/PKULab1806/Fairy-plus-minus-i).

Source repository: [EmbroiderSnow/ifairy.cpp](https://github.com/EmbroiderSnow/ifairy.cpp). The extraction is also available as a source archive. For a review submission, distribute the complete source package, including this README, `tests/ifairy-test-data/`, the synthetic demo source, and license files, or provide an accessible link to that exact package. The implementation description and remaining limitations are available in the [inventory](docs/IFAIRY_INVENTORY.md).
