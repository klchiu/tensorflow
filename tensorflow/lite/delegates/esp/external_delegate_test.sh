set -o errexit
set -o nounset

readonly benchmark_tool=third_party/tensorflow/lite/tools/benchmark/benchmark_model
readonly external_delegate=third_party/tensorflow/lite/delegates/esp/esp_external_delegate.so
readonly model=third_party/tensorflow/lite/delegates/coreml/internal_test/testdata/mobilenet_v2_1.0_224_quantized_weights_fp16.tflite
readonly benchmark_log=/tmp/benchmark.out

die() { echo "$@" >&2; exit 1; }

$benchmark_tool --graph=$model \
    --external_delegate_path=$external_delegate \
    --external_delegate_options='error_during_init:true;error_during_prepare:true' \
    >& $benchmark_log
cat $benchmark_log
grep -q 'EXTERNAL delegate created.' $benchmark_log \
    || die "Didn't find expected log contents"

echo "PASS"
