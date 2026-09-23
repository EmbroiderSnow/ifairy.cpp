#!/system/bin/sh
set -u
cd /data/local/tmp/ifairy-lut-integration-20260923 || exit 1
export LD_LIBRARY_PATH="$PWD/bin"
export TMPDIR="$PWD"
export GGML_IFAIRY_TEST_DATA_DIR=/data/local/tmp/ifairy-validation-20260923/tests/ifairy-test-data
model=/data/local/tmp/ifairy-validation-20260923/models/ifairy.gguf
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
run phone-direct-kernels env GGML_IFAIRY_LUT=0 ./bin/test-legacy-ifairy-direct
run phone-model-off env GGML_IFAIRY_LUT=0 ./bin/test-ifairy-model
run phone-ops ./bin/test-backend-ops test -b CPU -o IFAIRY_ADD,IFAIRY_MUL,IFAIRY_RMS_NORM,IFAIRY_SPLIT,IFAIRY_MERGE,IFAIRY_RELU2,IFAIRY_ROPE
run phone-lut-dispatch ./bin/test-ifairy-lut-mul-mat
run phone-lut-kernels env GGML_IFAIRY_LUT=1 ./bin/test-legacy-ifairy
run phone-model-on env GGML_IFAIRY_LUT=1 ./bin/test-ifairy-model
probe() {
    tag=phone-probe-$1
    enabled=$2
    impl=$3
    mode=$4
    act=$5
    run "$tag" env GGML_IFAIRY_LUT="$enabled" GGML_IFAIRY_LUT_IMPL="$impl" GGML_IFAIRY_VEC_DOT_ACT_TENSOR="$act" GGML_IFAIRY_LUT_DEBUG=0 ./bin/test-ifairy-real-model "$model" "results/$tag" "$mode" 8
}
probe off 0 auto batched 0
probe auto 1 auto batched 0
probe repeat 1 auto batched 0
probe lut16 1 lut16 batched 0
probe lut-c 1 lut_c batched 0
probe serial 1 auto serial 0
probe same-quant-direct 0 auto batched 1
run phone-cli-debug env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_DEBUG=1 ./bin/llama-cli -m "$model" -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -p 'The capital of France is' -n 1 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
run phone-cli-story env GGML_IFAIRY_LUT=1 ./bin/llama-cli -m "$model" -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -p 'Once upon a time, in a small village' -n 128 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
run phone-cli-long env GGML_IFAIRY_LUT=1 ./bin/llama-cli -m "$model" -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -f /data/local/tmp/ifairy-validation-20260923/long-prompt.txt -n 256 --seed 42 --temp 0 -no-cnv --no-display-prompt --simple-io
exit "$failed"
