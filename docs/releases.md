# Building and publishing releases

`.github/workflows/release.yml` builds macOS ARM64, Linux x86_64 AVX2 and Android
ARM64 archives on pull requests, pushes to `main`, and manual workflow runs.
Each platform compiles both direct and LUT configurations. Desktop jobs run
CTest in both builds and smoke-test the extracted archive. Android jobs use
NDK `30.0.16248370`, API 28 and `c++_static`, and inspect the cross-compiled ELF
files; they do not claim Android runtime validation.

Only a push of a stable version tag `vMAJOR.MINOR.PATCH` publishes a GitHub
Release. After all three jobs succeed, the publishing job creates a draft,
uploads the three archives and `SHA256SUMS`, then publishes it. Normal builds
have read-only repository permissions; only the publishing job can write a
release. Actions are pinned to commit SHAs. No personal access token is needed.

To publish from an up-to-date, validated `main` checkout:

```sh
git tag -a v0.1.0 -m "iFairy.cpp v0.1.0"
git push origin v0.1.0
```

Use a new version for each release. If a build fails, fix it before tagging the
next version; do not move an existing public tag. A retry can finish an existing
draft, but the workflow refuses to overwrite an already published release.
Update `docs/release-notes.md` before tagging if release-specific notes are needed.

The same build/package script can run locally on the matching host:

```sh
python3 scripts/build-release.py --platform macos-arm64 --version v0.1.0 --jobs 3
```

Linux builds require GCC/G++ 11; Android builds require `ANDROID_NDK_HOME` pointing
to the pinned NDK on Linux x86_64. Build directories must be fresh. Archives are
written to `out/release/`; no model download or Python third-party package is
needed to build them. `GGML_NATIVE=OFF` and explicit CPU targets prevent binaries
from depending on the CI runner's newer instruction sets.
