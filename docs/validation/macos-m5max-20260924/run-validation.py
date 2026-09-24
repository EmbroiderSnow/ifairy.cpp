"""Reproduce the recorded macOS runs; weights and raw logits stay outside docs."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
REVISION = "c274e9bb0b9a82fbe0bc20eeedbf4b8a3fcd358b"
MODEL_DIR = ROOT / "models/Fairy-plus-minus-i-700M"
MODEL = ROOT / "models/Fairy-plus-minus-i-700M.gguf"
DIRECT = "build-macos-validation-direct"
LUT = "build-macos-validation-lut"
SCRATCH = Path("/tmp/ifairy-macos-validation")
SCRATCH.mkdir(exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(tag, command, env=None, cwd=ROOT):
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("GGML_IFAIRY_") or key.startswith("GGML_LEGACY_IFAIRY_"):
            del environment[key]
    environment.update(env or {})
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    with (EVIDENCE / f"{tag}.stdout.log").open("w") as out:
        with (EVIDENCE / f"{tag}.stderr.log").open("w") as err:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=out, stderr=err)
    record = {
        "tag": tag,
        "command": command,
        "cwd": str(cwd),
        "environment": env or {},
        "started_utc": started,
        "seconds": round(time.monotonic() - start, 3),
        "returncode": result.returncode,
    }
    (EVIDENCE / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(tag, result.returncode, record["seconds"], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def settings(lut="0", impl="auto", act="0", debug="0"):
    return {
        "GGML_IFAIRY_LUT": lut,
        "GGML_IFAIRY_LUT_IMPL": impl,
        "GGML_IFAIRY_VEC_DOT_ACT_TENSOR": act,
        "GGML_IFAIRY_LUT_DEBUG": debug,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["environment", "build", "download", "convert", "functional", "bench"]
    )
    phase = parser.parse_args().phase
    if phase == "environment":
        commands = {
            "source-commit": ["git", "rev-parse", "HEAD"],
            "os": ["sw_vers"],
            "kernel": ["uname", "-a"],
            "cpu": [
                "sysctl",
                "machdep.cpu.brand_string",
                "hw.memsize",
                "hw.physicalcpu",
                "hw.logicalcpu",
            ],
            "compiler": ["clang", "--version"],
            "cmake": ["cmake", "--version"],
            "make": ["make", "--version"],
            "sdk": ["xcrun", "--show-sdk-version"],
            "clt": ["pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables"],
            "power": ["pmset", "-g", "custom"],
            "power-source": ["pmset", "-g", "batt"],
            "thermal": ["pmset", "-g", "therm"],
            "python": [sys.executable, "--version"],
            "python-packages": [sys.executable, "-m", "pip", "freeze"],
        }
        for tag, command in commands.items():
            run("environment-" + tag, command)
        source = [
            "gguf-py/convert_ifairy.py",
            "ggml/src/ggml-cpu/legacy-ifairy/legacy-ifairy-cpu.cpp",
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "tests/test-ifairy-real-model.cpp",
        ]
        (EVIDENCE / "source-sha256.json").write_text(
            json.dumps({s: sha256(ROOT / s) for s in source}, indent=2) + "\n"
        )
    elif phase == "build":
        for build, lut in [(DIRECT, "OFF"), (LUT, "ON")]:
            run(
                "configure-" + build,
                [
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    build,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    "-DGGML_LEGACY_IFAIRY_CPU_LUT=" + lut,
                    "-DGGML_LLAMAFILE=ON",
                    "-DLLAMA_CURL=OFF",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_CCACHE=OFF",
                ],
            )
            run("build-" + build, ["cmake", "--build", build, "-j", "8"])
            run("ctest-" + build, ["ctest", "--test-dir", build, "--output-on-failure"])
        run("source-isolation", [sys.executable, "scripts/check-ifairy-only.py"])
    elif phase == "download":
        code = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='PKU-DS-LAB/Fairy-plus-minus-i-700M', revision={REVISION!r}, "
            f"local_dir={str(MODEL_DIR)!r})"
        )
        run("download", [sys.executable, "-c", code])
        files = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(MODEL_DIR.iterdir())
            if p.is_file()
        }
        (EVIDENCE / "checkpoint.json").write_text(
            json.dumps(
                {
                    "repo_id": "PKU-DS-LAB/Fairy-plus-minus-i-700M",
                    "revision": REVISION,
                    "files": files,
                },
                indent=2,
            )
            + "\n"
        )
    elif phase == "convert":
        run(
            "convert",
            [
                "/usr/bin/time",
                "-l",
                sys.executable,
                str(ROOT / "gguf-py/convert_ifairy.py"),
                ".",
                str(MODEL),
            ],
            env={"PYTHONPATH": str(ROOT / "gguf-py")},
            cwd=MODEL_DIR,
        )
        (EVIDENCE / "gguf.json").write_text(
            json.dumps(
                {
                    "filename": MODEL.name,
                    "bytes": MODEL.stat().st_size,
                    "sha256": sha256(MODEL),
                    "source_revision": REVISION,
                },
                indent=2,
            )
            + "\n"
        )
    elif phase == "functional":
        for name, lut, impl, mode, act in [
            ("off", "0", "auto", "batched", "0"),
            ("auto", "1", "auto", "batched", "0"),
            ("repeat", "1", "auto", "batched", "0"),
            ("lut16", "1", "lut16", "batched", "0"),
            ("lut-c", "1", "lut_c", "batched", "0"),
            ("serial", "1", "auto", "serial", "0"),
            ("same-quant-direct", "0", "auto", "batched", "1"),
        ]:
            run(
                "probe-" + name,
                [
                    str(ROOT / LUT / "bin/test-ifairy-real-model"),
                    str(MODEL),
                    str(SCRATCH / ("probe-" + name)),
                    mode,
                    "8",
                ],
                settings(lut, impl, act),
            )
        cli = [
            str(ROOT / LUT / "bin/llama-cli"),
            "-m",
            str(MODEL),
            "-ngl",
            "0",
            "-t",
            "8",
            "-tb",
            "8",
            "-c",
            "2048",
            "-b",
            "512",
            "-ub",
            "512",
            "--seed",
            "42",
            "--temp",
            "0",
            "-no-cnv",
            "--simple-io",
        ]
        run(
            "cli-debug",
            cli + ["-p", "The capital of France is", "-n", "1"],
            settings("1", debug="1"),
        )
        for name, extra in [
            ("story", ["-p", "Once upon a time, in a small village", "-n", "128"]),
            ("long", ["-f", "docs/validation/local-x86-20260923/long-prompt.txt", "-n", "256"]),
        ]:
            run("cli-" + name, cli + extra, settings("1"))
        run(
            "cli-demo",
            [
                str(ROOT / DIRECT / "bin/llama-cli"),
                "-m",
                str(MODEL),
                "-ngl",
                "0",
                "-t",
                "4",
                "-tb",
                "4",
                "-c",
                "2048",
                "-b",
                "512",
                "-ub",
                "512",
                "-p",
                "The capital of France is",
                "-n",
                "32",
                "--seed",
                "42",
                "--temp",
                "0",
                "-no-cnv",
                "--simple-io",
            ],
            settings(),
        )
    elif phase == "bench":
        for threads, lut in [(4, "0"), (4, "1"), (8, "1"), (8, "0")]:
            tag = f"bench-{threads}t-" + ("auto" if lut == "1" else "direct")
            run(tag + "-power-before", ["pmset", "-g", "therm"])
            run(
                tag,
                [
                    str(ROOT / LUT / "bin/llama-bench"),
                    "-m",
                    str(MODEL),
                    "-ngl",
                    "0",
                    "-t",
                    str(threads),
                    "-b",
                    "512",
                    "-ub",
                    "512",
                    "-p",
                    "128,512",
                    "-n",
                    "128",
                    "-r",
                    "3",
                    "-o",
                    "json",
                ],
                settings(lut),
            )
            run(tag + "-power-after", ["pmset", "-g", "therm"])


if __name__ == "__main__":
    main()
