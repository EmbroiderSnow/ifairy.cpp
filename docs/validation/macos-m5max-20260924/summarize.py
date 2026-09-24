"""Summarize the saved runs and raw logits produced by run-validation.py."""

import hashlib
import json
from pathlib import Path

import numpy as np


EVIDENCE = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/ifairy-macos-validation")
NAMES = ["off", "auto", "repeat", "lut16", "lut-c", "serial", "same-quant-direct"]


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def save(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    probes = {}
    arrays = {}
    for name in NAMES:
        result = read(f"probe-{name}.stdout.log")
        traces = []
        arrays[name] = []
        for i, prompt in enumerate(result["results"]):
            path = SCRATCH / f"probe-{name}-prompt-{i}.f32"
            raw = path.read_bytes()
            logits = np.frombuffer(raw, dtype=np.float32).reshape(33, result["n_vocab"])
            assert np.isfinite(logits).all(), f"Non-finite logits: {path}"
            assert logits.size == prompt["finite_logits"]
            arrays[name].append(logits)
            traces.append(
                {
                    "prompt": prompt["prompt"],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "values": logits.size,
                    "all_finite": True,
                    "generated_ids": prompt["generated_ids"],
                }
            )
        probes[name] = {
            "peak_rss_mib": result["max_rss_native"] / 1024**2,
            "seconds": read(f"probe-{name}.json")["seconds"],
            "traces": traces,
        }
    comparisons = {}
    for other in ["repeat", "lut16", "serial", "off", "lut-c", "same-quant-direct"]:
        comparisons["auto_vs_" + other] = [
            {
                "trace_byte_identical": probes["auto"]["traces"][i]["sha256"]
                == probes[other]["traces"][i]["sha256"],
                "first_vector_max_abs_diff": float(
                    np.max(np.abs(arrays["auto"][i][0] - arrays[other][i][0]))
                ),
            }
            for i in range(2)
        ]
    save(
        "correctness-summary.json",
        {
            "configurations": probes,
            "comparisons": comparisons,
            "total_finite_logits": sum(t["values"] for p in probes.values() for t in p["traces"]),
            "rss_units": "MiB, macOS ru_maxrss divided by 1024^2",
        },
    )
    benchmarks = []
    for threads, mode in [(4, "direct"), (4, "auto"), (8, "auto"), (8, "direct")]:
        tag = f"bench-{threads}t-{mode}"
        rows = read(tag + ".stdout.log")
        benchmarks.append({"tag": tag, "threads": threads, "mode": mode, "rows": rows})
    save("benchmark-summary.json", benchmarks)
    execution = (
        (EVIDENCE / "cli-debug.stderr.log").read_text().count("ifairy_lut: executed MUL_MAT")
    )
    summary = {
        "source_commit": (EVIDENCE / "environment-source-commit.stdout.log").read_text().strip(),
        "gguf": read("gguf.json"),
        "checkpoint_revision": read("checkpoint.json")["revision"],
        "download_seconds": read("download.json")["seconds"],
        "conversion_seconds": read("convert.json")["seconds"],
        "cli_demo_seconds": read("cli-demo.json")["seconds"],
        "debug_executed_matmuls": execution,
        "total_finite_logits": sum(t["values"] for p in probes.values() for t in p["traces"]),
        "probe_rss_mib": {name: probes[name]["peak_rss_mib"] for name in ["off", "auto"]},
    }
    save("summary.json", summary)
    print(json.dumps(summary, indent=2))
    for group in benchmarks:
        print(
            group["tag"],
            [
                (f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}", r["avg_ts"])
                for r in group["rows"]
            ],
        )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
