# iFairy-only repository

Only general.architecture=ifairy is supported. Keep the model admission check and
architecture-isolation regression intact. Do not add other model graphs or GPU backends.
Preserve the legacy packed BF16 complex representation and w * conj(x) semantics.
CPU direct, ARM NEON and LUT fallbacks must remain available.

Build: cmake -B build-rel -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
Test: cmake --build build-rel -j 8 && ctest --test-dir build-rel --output-on-failure
Also validate -DGGML_LEGACY_IFAIRY_CPU_LUT=ON in a separate build.
Run git clang-format --diff against the target merge-base for changed C/C++ files,
and clang-tidy -p build-rel on changed C/C++ source files only.
Do not commit real model weights. Do not run concurrent llama-bench jobs.
