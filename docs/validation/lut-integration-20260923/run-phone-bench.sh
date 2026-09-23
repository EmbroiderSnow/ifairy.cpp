#!/system/bin/sh
# Run only when no host/phone llama-bench is active. No phone settings changed.
set -u
cd /data/local/tmp/ifairy-lut-integration-20260923 || exit 1
export LD_LIBRARY_PATH="$PWD/bin"
export TMPDIR="$PWD"
export GGML_IFAIRY_LUT_DEBUG=0
export GGML_IFAIRY_LUT_IMPL=auto
snapshot() {
    date -u +%Y-%m-%dT%H:%M:%SZ
    dumpsys battery | head -28
    dumpsys thermalservice | head -22
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
        echo "$p"
        cat "$p/scaling_cur_freq" "$p/scaling_max_freq"
    done
}
run() {
    tag=$1
    export GGML_IFAIRY_LUT=$2
    threads=$3
    snapshot > "results/$tag.before.log" 2>&1
    date -u +%Y-%m-%dT%H:%M:%SZ > "results/$tag.started"
    ./bin/llama-bench -m /data/local/tmp/ifairy-validation-20260923/models/ifairy.gguf -ngl 0 -t "$threads" -b 512 -ub 512 -p 128,512 -n 128 -r 3 -o json > "results/$tag.stdout.log" 2> "results/$tag.stderr.log"
    rc=$?
    echo "$rc" > "results/$tag.exit"
    date -u +%Y-%m-%dT%H:%M:%SZ > "results/$tag.finished"
    snapshot > "results/$tag.after.log" 2>&1
    printf '%s exit=%s\n' "$tag" "$rc"
    [ "$rc" -eq 0 ] || exit "$rc"
}
# Fixed 90s idle intervals reduce carry-over heat; snapshots establish actual conditions.
run phone-bench-4t-off 0 4
sleep 90
run phone-bench-4t-on 1 4
sleep 90
run phone-bench-8t-on 1 8
sleep 90
run phone-bench-8t-off 0 8
