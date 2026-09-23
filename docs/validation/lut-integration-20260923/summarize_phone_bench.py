"""Rebuild phone benchmark and thermal summaries from saved device logs."""
from datetime import datetime
import json
from pathlib import Path
import re


base = Path(__file__).resolve().parent
tags = ["phone-bench-4t-off", "phone-bench-4t-on", "phone-bench-8t-on", "phone-bench-8t-off"]
runs = []
thermal = {}
previous_finish = None
for tag in tags:
    started = (base / (tag + ".started")).read_text().strip()
    finished = (base / (tag + ".finished")).read_text().strip()
    start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
    finish_time = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    idle = (start_time - previous_finish).total_seconds() if previous_finish else None
    assert idle is None or idle >= 90
    assert finish_time > start_time
    previous_finish = finish_time
    code = int((base / (tag + ".exit")).read_text())
    assert code == 0
    raw = json.loads((base / (tag + ".stdout.log")).read_text())
    assert len(raw) == 3
    metrics = {}
    for row in raw:
        assert row["n_batch"] == row["n_ubatch"] == 512
        assert row["n_gpu_layers"] == 0 and not row["flash_attn"]
        assert row["type_k"] == row["type_v"] == "f16"
        assert row["backends"] == "CPU" and row["cpu_mask"] == "0x0"
        assert row["n_threads"] == (4 if "4t" in tag else 8)
        assert len(row["samples_ts"]) == len(row["samples_ns"]) == 3
        metric = ("pp" + str(row["n_prompt"])) if row["n_prompt"] else ("tg" + str(row["n_gen"]))
        metrics[metric] = {key: row[key] for key in ("avg_ts", "stddev_ts", "samples_ts", "samples_ns")}
    assert set(metrics) == {"pp128", "pp512", "tg128"}
    runs.append({"tag": tag, "threads": raw[0]["n_threads"], "lut_enabled": tag.endswith("-on"),
                 "started_utc": started, "finished_utc": finished, "idle_before_s": idle,
                 "exit_code": code, "raw_json": tag + ".stdout.log", "metrics": metrics})
    thermal[tag] = {}
    for phase in ("before", "after"):
        text = (base / (tag + "." + phase + ".log")).read_text()
        policies = re.findall(r"/sys/devices/system/cpu/cpufreq/(policy\d+)\s+(\d+)\s+(\d+)", text)
        thermal[tag][phase] = {
            "recorded_utc": text.splitlines()[0],
            "battery_temperature_c": int(re.search(r"^  temperature: (\d+)$", text, re.M)[1]) / 10,
            "battery_level_percent": int(re.search(r"^  level: (\d+)$", text, re.M)[1]),
            "usb_powered": "USB powered: true" in text,
            "thermal_status": int(re.search(r"Thermal Status: (\d+)", text)[1]),
            "cached_skin_temperature_c": float(re.search(r"mValue=([\d.]+), mType=3, mName=SKIN", text)[1]),
            "cpu_policy_khz": {name: {"current": int(cur), "max": int(maximum)} for name, cur, maximum in policies},
        }
        assert len(policies) == 2

ratios = {}
for threads in (4, 8):
    off = next(r for r in runs if r["threads"] == threads and not r["lut_enabled"])
    on = next(r for r in runs if r["threads"] == threads and r["lut_enabled"])
    ratios[str(threads)] = {m: on["metrics"][m]["avg_ts"] / off["metrics"][m]["avg_ts"] for m in off["metrics"]}

summary = {"runs": runs, "total_measured_samples": 36, "lut_on_over_off_observed_throughput": ratios,
           "limitations": ["Different activation quantization and full-model outputs between LUT and default direct.",
                           "Sequential, fixed-order runs; USB charging, no affinity or frequency lock.",
                           "Thermal throttling observed; 90-second idle did not equalize all starting conditions.",
                           "Snapshots are endpoints, not continuous telemetry; SKIN values are thermalservice cached readings."]}
(base / "phone-benchmark-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(base / "phone-thermal-summary.json").write_text(json.dumps(thermal, indent=2) + "\n")
print(json.dumps(ratios, indent=2))
