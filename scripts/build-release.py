#!/usr/bin/env python3
"""Build and validate portable CPU binaries, then create one release archive."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("llama-cli", "llama-bench", "llama-perplexity", "test-ifairy-model")
REQUIREMENTS = {
    "macos-arm64": "macOS 13 or later; Apple Silicon (ARM64)",
    "linux-x86_64-avx2": (
        "Linux x86_64; glibc >= 2.35 and libstdc++6 (Ubuntu 22.04 or newer); "
        "AVX2, FMA, F16C and SSE4.2"
    ),
    "android-arm64": "Android 9 / API 28 or later; ARM64 with ARMv8.2-A + dot product",
}


def run(*args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def output(*args):
    return run(*args, stdout=subprocess.PIPE, text=True).stdout.strip()


def platform_flags(target):
    if target == "macos-arm64":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Build macos-arm64 on an Apple Silicon Mac")
        return ["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
                "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"]
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Build Linux/Android archives on a Linux x86_64 host")
    if target == "linux-x86_64-avx2":
        return ["-DCMAKE_C_COMPILER=gcc-11", "-DCMAKE_CXX_COMPILER=g++-11",
                "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON",
                "-DGGML_F16C=ON", "-DGGML_SSE42=ON", "-DGGML_BMI2=OFF",
                "-DGGML_AVX512=OFF", "-DGGML_AVX_VNNI=OFF"]
    ndk = Path(os.environ["ANDROID_NDK_HOME"])
    if "30.0.16248370" not in (ndk / "source.properties").read_text():
        raise RuntimeError("Android release builds require NDK 30.0.16248370")
    return [f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
            "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-28",
            "-DANDROID_STL=c++_static", "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"]


def check_binary(binary, target):
    description = output("file", binary)
    print(description)
    if target == "macos-arm64":
        if "Mach-O 64-bit executable arm64" not in description:
            raise RuntimeError(f"Wrong binary architecture: {description}")
        dependencies = output("otool", "-L", binary).splitlines()[1:]
        if any(not line.strip().startswith(("/usr/lib/", "/System/Library/"))
               for line in dependencies):
            raise RuntimeError(f"Non-system dependency: {dependencies}")
    elif target == "linux-x86_64-avx2":
        if "ELF 64-bit" not in description or "x86-64" not in description:
            raise RuntimeError(f"Wrong binary architecture: {description}")
        dependencies = output("ldd", binary)
        if any(value in dependencies for value in ("not found", "libllama", "libggml")):
            raise RuntimeError(f"Unbundled dependency: {dependencies}")
    else:
        if "ELF 64-bit" not in description or "ARM aarch64" not in description:
            raise RuntimeError(f"Wrong binary architecture: {description}")
        readelf = (Path(os.environ["ANDROID_NDK_HOME"]) /
                   "toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf")
        headers = output(readelf, "--program-headers", binary)
        if "/system/bin/linker64" not in headers:
            raise RuntimeError("Expected Android linker64")
        dependencies = output(readelf, "--dynamic", binary)
        needed = set(re.findall(r"Shared library: \[(.*?)\]", dependencies))
        if not needed or needed - {"libc.so", "libm.so", "libdl.so", "liblog.so"}:
            raise RuntimeError(f"Unexpected Android dependencies: {needed}")


def smoke_test(directory):
    for name in TOOLS[:3]:
        run(directory / "bin" / name, "--help", cwd=directory,
            stdout=subprocess.DEVNULL)
    for mode in ("0", "1"):
        run(directory / "bin/test-ifairy-model", cwd=directory,
            env={**os.environ, "GGML_IFAIRY_LUT": mode, "GGML_IFAIRY_LUT_IMPL": "auto"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=REQUIREMENTS, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+|snapshot-[0-9a-f]{7,40}", args.version):
        parser.error("version must be vMAJOR.MINOR.PATCH or snapshot-COMMIT")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    os.chdir(ROOT)
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
             "-DGGML_NATIVE=OFF", "-DGGML_BACKEND_DL=OFF", "-DGGML_OPENMP=OFF",
             "-DGGML_CCACHE=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=ON",
             "-DLLAMA_BUILD_TOOLS=ON"] + platform_flags(args.platform)
    native = args.platform != "android-arm64"
    for mode in ("OFF", "ON"):
        build = ROOT / f"build-release-{args.platform}-{mode.lower()}"
        # Do not reuse caches with stale compiler/ISA settings for release binaries.
        if build.exists():
            raise RuntimeError(f"Use a fresh build directory; already exists: {build}")
        run("cmake", "-S", ROOT, "-B", build, *flags,
            f"-DGGML_LEGACY_IFAIRY_CPU_LUT={mode}")
        run("cmake", "--build", build, "--parallel", args.jobs)
        if native:
            run("ctest", "--test-dir", build, "--output-on-failure")

    name = f"ifairy-{args.version}-{args.platform}"
    destination = ROOT / "out/release"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{name}.tar.gz"
    if archive.exists():
        raise RuntimeError(f"Archive already exists: {archive}")
    with tempfile.TemporaryDirectory(prefix="ifairy-package-") as temp:
        package = Path(temp) / name
        (package / "bin").mkdir(parents=True)
        for tool in TOOLS:
            binary = package / "bin" / tool
            shutil.copy2(build / "bin" / tool, binary)
            check_binary(binary, args.platform)
        shutil.copy2(ROOT / "LICENSE", package)
        shutil.copytree(ROOT / "licenses", package / "licenses")
        if not native:
            notices = list(Path(os.environ["ANDROID_NDK_HOME"]).glob("NOTICE*"))
            if not notices:
                raise RuntimeError("Cannot package Android C++ runtime without NDK notices")
            for notice in notices:
                shutil.copy2(notice, package / "licenses" / f"NDK-{notice.name}")
        shutil.copy2(ROOT / "docs/binary-usage.md", package / "README.md")
        validation = ("Direct and LUT CTest suites; relocated CLI help and synthetic "
                      "model tests in both modes" if native else
                      "Cross-compiled direct and LUT builds; ELF architecture, linker "
                      "and dependency checks. Not executed on Android in CI.")
        metadata = {
            "version": args.version,
            "commit": output("git", "rev-parse", "HEAD"),
            "platform": args.platform,
            "requirements": REQUIREMENTS[args.platform],
            "cmake": output("cmake", "--version").splitlines()[0],
            "cmake_flags": flags + ["-DGGML_LEGACY_IFAIRY_CPU_LUT=ON"],
            "build_host": platform.platform(),
            "validation": validation,
            "binaries_sha256": {
                tool: hashlib.sha256((package / "bin" / tool).read_bytes()).hexdigest()
                for tool in TOOLS
            },
        }
        (package / "build-info.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(package, arcname=name)
        # Exercise the actual archive after extraction, away from the source/build trees.
        unpacked = Path(temp) / "unpacked"
        with tarfile.open(archive) as bundle:
            bundle.extractall(unpacked)
        for tool in TOOLS:
            binary = unpacked / name / "bin" / tool
            if not os.access(binary, os.X_OK):
                raise RuntimeError(f"Archive lost executable permissions: {tool}")
            if hashlib.sha256(binary.read_bytes()).hexdigest() != metadata["binaries_sha256"][tool]:
                raise RuntimeError(f"Archive checksum mismatch: {tool}")
        if native:
            smoke_test(unpacked / name)
    print(f"Created {archive}", flush=True)


if __name__ == "__main__":
    main()
