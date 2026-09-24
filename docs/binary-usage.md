# iFairy CPU binaries

This archive contains `bin/llama-cli`, `bin/llama-bench`, `bin/llama-perplexity`
and a small, weight-free self-test, `bin/test-ifairy-model`. Only iFairy GGUF
models are supported. Model weights and the Python conversion environment are
not included. See `build-info.json` for the exact source commit, build options,
system requirements and validation performed on this archive.

## Choose an archive

| Archive suffix | Requirements |
| --- | --- |
| `macos-arm64` | Apple Silicon; macOS 13 or newer |
| `linux-x86_64-avx2` | x86_64 with AVX2, FMA, F16C and SSE4.2; glibc 2.35+ and libstdc++6 (Ubuntu 22.04 or newer) |
| `android-arm64` | Android 9 / API 28 or newer; ARM64 with ARMv8.2-A and dot product |

The llama/ggml libraries are linked into the executables. Android also includes
the C++ runtime; desktop binaries use the operating system's standard runtime
libraries. macOS binaries are ad-hoc signed by the toolchain, not notarized.

## Desktop usage

Download the matching `.tar.gz` and `SHA256SUMS` from the same release. Verify
the archive against its entry in `SHA256SUMS` using `shasum -a 256` on macOS or
`sha256sum` on Linux, then extract it with `tar -xzf <archive>.tar.gz`.
Change into the extracted directory and run:

```sh
./bin/llama-cli --help
GGML_IFAIRY_LUT=0 ./bin/test-ifairy-model
GGML_IFAIRY_LUT=1 ./bin/test-ifairy-model

./bin/llama-cli -m /absolute/path/Fairy-plus-minus-i-700M.gguf \
  -ngl 0 -t 4 -c 2048 -p "The capital of France is" -n 32 \
  --seed 42 --temp 0 -no-cnv --simple-io
```

For model download and GGUF conversion, follow the demo in the
[repository README](https://github.com/EmbroiderSnow/ifairy.cpp#3-demo).
Substitute this archive's `bin/llama-cli` for `build-direct/bin/llama-cli`.
To use LUT, prefix the same command with
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=auto`; use `GGML_IFAIRY_LUT=0` for direct inference.

## Android usage

Extract the Android archive on your computer, enable USB debugging on the device,
connect it, and run these commands from the extracted directory:

```sh
adb shell mkdir -p /data/local/tmp/ifairy
adb push bin/. /data/local/tmp/ifairy/
adb push /absolute/path/Fairy-plus-minus-i-700M.gguf /data/local/tmp/ifairy/model.gguf
adb shell 'cd /data/local/tmp/ifairy && chmod 755 llama-* test-ifairy-model'
adb shell 'cd /data/local/tmp/ifairy && GGML_IFAIRY_LUT=0 ./test-ifairy-model'
adb shell 'cd /data/local/tmp/ifairy && GGML_IFAIRY_LUT=1 ./test-ifairy-model'
adb shell 'cd /data/local/tmp/ifairy && ./llama-cli -m model.gguf -ngl 0 -t 4 -c 2048 -p "The capital of France is" -n 32 --seed 42 --temp 0 -no-cnv --simple-io'
```

Android archives are cross-compiled and checked for architecture and runtime
dependencies in CI; CI does not execute them on an Android device. The included
self-test can verify model loading, prefill, decode, repeatability and rejection
of other model architectures on your device without downloading weights.

## License

See `LICENSE` (MIT) and `licenses/` for bundled third-party notices.
