#!/system/bin/sh
# Run from the isolated device test directory after deploying direct/ and lut/.
set -u
cd /data/local/tmp/ifairy-validation-20260923 || exit 1
export TMPDIR="$PWD"
export GGML_IFAIRY_TEST_DATA_DIR="$PWD/tests/ifairy-test-data"
mkdir -p results
failed=0
run() {
    tag=$1
    shift
    date -u +%Y-%m-%dT%H:%M:%SZ > "results/$tag.started"
    "$@" > "results/$tag.stdout.log" 2> "results/$tag.stderr.log"
    rc=$?
    echo "$rc" > "results/$tag.exit"
    date -u +%Y-%m-%dT%H:%M:%SZ > "results/$tag.finished"
    printf '%s exit=%s\n' "$tag" "$rc"
    if [ "$rc" -ne 0 ]; then failed=1; fi
}
ops=IFAIRY_ADD,IFAIRY_MUL,IFAIRY_RMS_NORM,IFAIRY_SPLIT,IFAIRY_MERGE,IFAIRY_RELU2,IFAIRY_ROPE
export LD_LIBRARY_PATH="$PWD/direct"
run direct-kernels env GGML_IFAIRY_LUT=0 ./direct/test-legacy-ifairy-direct
run direct-model env GGML_IFAIRY_LUT=0 ./direct/test-ifairy-model
run direct-ops ./direct/test-backend-ops test -b CPU -o "$ops"
run probe-direct env GGML_IFAIRY_LUT=0 ./direct/test-ifairy-real-model models/ifairy.gguf results/probe-direct batched 8
run cli-direct env GGML_IFAIRY_LUT=0 ./direct/llama-cli -m models/ifairy.gguf -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -p 'The capital of France is' -n 64 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
export LD_LIBRARY_PATH="$PWD/lut"
run lut-direct-kernels env GGML_IFAIRY_LUT=0 ./lut/test-legacy-ifairy-direct
run lut-model-off env GGML_IFAIRY_LUT=0 ./lut/test-ifairy-model
run lut-ops ./lut/test-backend-ops test -b CPU -o "$ops"
run lut-kernels env GGML_IFAIRY_LUT=1 ./lut/test-legacy-ifairy
run lut-model-on env GGML_IFAIRY_LUT=1 ./lut/test-ifairy-model
run probe-lut-off env GGML_IFAIRY_LUT=0 ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-lut-off batched 8
run probe-lut-on env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=auto ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-lut-on batched 8
run probe-lut-repeat env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=auto ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-lut-repeat batched 8
run probe-lut16 env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-lut16 batched 8
run probe-lut-c env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-lut-c batched 8
run probe-lut-serial env GGML_IFAIRY_LUT=1 ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-lut-serial serial 8
run probe-serial-off env GGML_IFAIRY_LUT=0 ./lut/test-ifairy-real-model models/ifairy.gguf results/probe-serial-off serial 8
run cli-lut env GGML_IFAIRY_LUT=1 ./lut/llama-cli -m models/ifairy.gguf -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -p 'The capital of France is' -n 64 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
run cli-story env GGML_IFAIRY_LUT=1 ./lut/llama-cli -m models/ifairy.gguf -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -p 'Once upon a time, in a small village' -n 128 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
run cli-long env GGML_IFAIRY_LUT=1 ./lut/llama-cli -m models/ifairy.gguf -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -f long-prompt.txt -n 256 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
run cli-lut-debug env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_DEBUG=1 ./lut/llama-cli -m models/ifairy.gguf -ngl 0 -t 8 -c 2048 -b 512 -ub 512 -p 'The capital of France is' -n 1 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
run kernel-qgemm ./lut/ifairy-microbench --type ifairy --mode qgemm --m 1536 --k 1536 --iters 100 --warmup 10 --seed 42
run kernel-fused ./lut/ifairy-microbench --type ifairy --mode fused --m 1536 --k 1536 --iters 100 --warmup 10 --seed 42
exit "$failed"
