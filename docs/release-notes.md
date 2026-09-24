CPU inference binaries for iFairy models, including `llama-cli`, `llama-bench`,
`llama-perplexity` and a weight-free model self-test.

| Download suffix | System requirements |
| --- | --- |
| `macos-arm64.tar.gz` | Apple Silicon; macOS 13+ |
| `linux-x86_64-avx2.tar.gz` | Linux x86_64; AVX2/FMA/F16C/SSE4.2; glibc 2.35+ and libstdc++6 |
| `android-arm64.tar.gz` | Android 9 / API 28+; ARM64 with ARMv8.2-A + dot product |

Download an archive and verify its SHA-256 against `SHA256SUMS`. Each archive
includes executable tools, usage instructions, licenses and `build-info.json`
with the source commit and build configuration. Model weights are downloaded
separately; follow the [download → GGUF conversion → inference demo](https://github.com/EmbroiderSnow/ifairy.cpp#3-demo).

macOS and Linux builds pass the direct and LUT test suites and smoke tests after
archive extraction. Android builds pass cross-compilation and ELF architecture,
linker and dependency checks; they are **not executed on Android in CI**.
Run the included `test-ifairy-model` on your device for a weight-free self-test.
macOS binaries are not notarized. Only iFairy models and CPU inference are supported.
