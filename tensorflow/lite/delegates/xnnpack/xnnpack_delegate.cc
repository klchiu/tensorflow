#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include <stddef.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// #include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"

// [from conv.cc] =============================================================
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/conv.h"

// #if !defined(TFLITE_WITH_RUY)
// #define TFLITE_WITH_MULTITHREADED_EIGEN
// #endif

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
// #if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
// #include "tensorflow/lite/kernels/eigen_support.h"
// #endif
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"
// b/131835803 forces us to include multithreaded_conv.h before optimized_ops.h
// #if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
// #include "tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h"
// #endif
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"
// namespace tflite {
// namespace ops {
// namespace builtin {
// namespace conv {

// This file has 4 implementation of Conv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  // kMultithreadOptimized is a mixture of an Eigen-based kernel when threads
  // are available and kGenericOptimized when we must use only one thread.
  kMultithreadOptimized,
  // The kernel uses use CBLAS interface for matrix multiplication.
  // It's fast when an optimized CBLAS implementation is available (e.g. Apple
  // Accelerate Framework), and it's slow when falling back to naive
  // implementation.
  kCblasOptimized,
};

const int kTensorNotAllocated = -1;

static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024;  // 1GB

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int im2col_id = kTensorNotAllocated;
  int hwcn_weights_id = kTensorNotAllocated;
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;
  int accum_scratch_id = kTensorNotAllocated;
  // Row sums are used to cache filter sums for hybrid zero-point calculations.
  int row_sums_id = kTensorNotAllocated;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // Indexes are the offset to the memory buffer in the array used to keep track
  // of the allocated temporaries.
  int32_t im2col_index;
  int32_t hwcn_weights_index;
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t accum_scratch_index;
  int32_t input_offset_index;
  int32_t row_sums_index;

  bool need_hwcn_weights = false;
  bool have_weights_been_transposed = false;
  bool need_im2col = false;
  // If it's true, it means im2col546 is needed but gets disabled because the
  // temporary im2col tensor requires too much memory (i.e.
  // >= kMaxIm2colBufferSize);
  bool im2col_oversized = false;

  bool supports_multithreaded_kernel = false;
  bool is_hybrid_per_channel = false;
  bool compute_hybrid_row_sums = true;

  // Number of convolution groups.
  int32_t groups = 1;
};

// inline PaddingType RuntimePaddingType(TfLitePadding padding) {
//   switch (padding) {
//     case TfLitePadding::kTfLitePaddingSame:
//       return PaddingType::kSame;
//     case TfLitePadding::kTfLitePaddingValid:
//       return PaddingType::kValid;
//     case TfLitePadding::kTfLitePaddingUnknown:
//     default:
//       return PaddingType::kNone;
//   }
// }

// Naive implementation of transpose for floats. Could be optimized to be more
// cache friendly, but for now it's a one-time cost on first run, and we would
// prefer to remove the need to do this at all eventually.
// void TransposeFloatTensor(const TfLiteTensor* input, TfLiteTensor* output) {
//   const int rows = output->dims->data[1];
//   const int cols = output->dims->data[0];
//   const float* input_data = GetTensorData<float>(input);
//   float* output_data = GetTensorData<float>(output);
//   for (int i = 0; i < rows; ++i) {
//     for (int j = 0; j < cols; ++j) {
//       const float in_value = input_data[i * cols + j];
//       output_data[j * rows + i] = in_value;
//     }
//   }
// }

// } // namespace conv
// } // namespace builtin
// } // namespace ops
// } // namespace tflite

// [end from conv.cc] =============================================================

// #define ESP_RISCV

#ifdef ESP_RISCV
// [humu]: include this header file for using ESP APIs
#define __FIXED
#define BITWIDTH 32
#include "tensorflow/esp_libs/cfg_conv2d.h"
#include "tensorflow/esp_libs/cfg_gemm.h"
#include "tensorflow/esp_libs/cfg_tf_add3.h"
#include "tensorflow/esp_libs/cfg_tf_mult3.h"
#include "tensorflow/esp_libs/cfg_tf_sub3.h"
#include "tensorflow/esp_libs/conv2d_helper.h"
#include "tensorflow/esp_libs/esp_acc_prints.h"
#include "tensorflow/esp_libs/esp_api_include.h"
#include "tensorflow/esp_libs/gemm_helper.h"
// [humu]: end for including ESP APIs
#else
typedef int token_t;
#define NACC 1
#define ACC_TLB_ENTRIES 128
#define ACC_PAGE_SIZE (1 << 20)
#define MAX_SIZE (ACC_PAGE_SIZE * ACC_TLB_ENTRIES)

#endif  // ESP_RISCV

static float ActivationFunctionWithMinMax(float total, int output_activation_min, int output_activation_max) {
  // [humu]: why just return total?
  return total;
}

static void init_array(float* array, int size) {
  srand(time(NULL));
  for (int i = 0; i < size; i++) {
    array[i] = 0.1 * (rand() % 10);
  }
}

static void init_array_0(float* array, int size) {
  srand(time(NULL));
  for (int i = 0; i < size; i++) {
    array[i] = 0.0;
  }
}

#ifdef ESP_RISCV

static void load_buffer_gemm(token_t* acc_buf, uint32_t base_addr_1, float* input_1, uint32_t base_addr_2, float* input_2, unsigned len) {
  // fprintf(stderr, "-- load_buffer_gemm, len = %d\n", len);

  int i;

  // load input_1
  for (i = 0; i < len; i++) {
    acc_buf[len + i] = float2fx(input_1[base_addr_1 + i], FX_IL);
    // fprintf(stderr, "-- load_buffer_gemm, i = %d\n", i);
  }

  // fprintf(stderr, "-- load_buffer_gemm 2\n");

  // load input_2
  for (i = 0; i < len; i++) {
    acc_buf[len + len + i] = float2fx(input_2[base_addr_2 + i], FX_IL);
    // fprintf(stderr, "-- load_buffer_gemm, i = %d\n", i);
  }
}

static void store_buffer_gemm(token_t* acc_buf, uint32_t base_addr_0, float* output_0, unsigned len) {
  // fprintf(stderr, "-- store_buffer_gemm, len = %d\n", len);

  int i;

  for (i = 0; i < len; i++) {
    output_0[base_addr_0 + i] = fx2float(acc_buf[i], FX_IL);
  }
}

static void load_buffer_conv2d(token_t* acc_buf, uint32_t base_addr_1, float* input_1, uint32_t base_addr_2, float* input_2, unsigned len) {
  // fprintf(stderr, "-- load_buffer_conv2d, len = %d\n", len);

  int i;

  // load input_1
  for (i = 0; i < len; i++) {
    acc_buf[len + i] = float2fx(input_1[base_addr_1 + i], FX_IL);
    // fprintf(stderr, "-- load_buffer_conv2d, i = %d\n", i);
  }

  // fprintf(stderr, "-- load_buffer_conv2d 2\n");

  // load input_2
  for (i = 0; i < len; i++) {
    acc_buf[len + len + i] = float2fx(input_2[base_addr_2 + i], FX_IL);
    // fprintf(stderr, "-- load_buffer_conv2d, i = %d\n", i);
  }
}

static void store_buffer_conv2d(token_t* acc_buf, uint32_t base_addr_0, float* output_0, unsigned len) {
  // fprintf(stderr, "-- store_buffer_conv2d, len = %d\n", len);

  int i;

  for (i = 0; i < len; i++) {
    output_0[base_addr_0 + i] = fx2float(acc_buf[i], FX_IL);
  }
}

#endif  // ESP_RISCV

namespace tflite {
namespace {
TfLiteRegistration GetDelegateKernelRegistration(SimpleDelegateInterface* delegate) {
  TfLiteRegistration kernel_registration{};
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = delegate->Name();
  kernel_registration.version = 1;
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<SimpleDelegateKernelInterface*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer, size_t length) -> void* {
    const TfLiteDelegateParams* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    if (params == nullptr) {
      TF_LITE_KERNEL_LOG(context, "NULL TfLiteDelegateParams passed.");
      return nullptr;
    }
    auto* delegate = reinterpret_cast<SimpleDelegateInterface*>(params->delegate->data_);
    std::unique_ptr<SimpleDelegateKernelInterface> delegate_kernel(delegate->CreateDelegateKernelInterface());
    if (delegate_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return delegate_kernel.release();
  };
  kernel_registration.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Delegate kernel was not initialized");
      return kTfLiteError;
    }
    SimpleDelegateKernelInterface* delegate_kernel = reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    return delegate_kernel->Prepare(context, node);
  };
  kernel_registration.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    SimpleDelegateKernelInterface* delegate_kernel = reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    TFLITE_DCHECK(delegate_kernel != nullptr);
    return delegate_kernel->Eval(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* base_delegate) {
  auto* delegate = reinterpret_cast<SimpleDelegateInterface*>(base_delegate->data_);
  auto delegate_options = delegate->DelegateOptions();
  if (delegate_options.max_delegated_partitions <= 0) delegate_options.max_delegated_partitions = std::numeric_limits<int>::max();

  TF_LITE_ENSURE_STATUS(delegate->Initialize(context));
  delegates::IsNodeSupportedFn node_supported_fn = [=](TfLiteContext* context, TfLiteNode* node, TfLiteRegistration* registration,
                                                       std::string* unsupported_details) -> bool {
    return delegate->IsNodeSupportedByDelegate(registration, node, context);
  };
  // TODO(b/149484598): Update to have method that gets all supported nodes.
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  std::vector<int> supported_nodes =
      helper.GetNodesOfFirstNLargestPartitions(delegate_options.max_delegated_partitions, delegate_options.min_nodes_per_partition);

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "%s delegate: %d nodes delegated out of %d nodes with "
                       "%d partitions.\n",
                       delegate->Name(), supported_nodes.size(), helper.num_total_nodes(), helper.num_partitions());
  TfLiteRegistration delegate_kernel_registration = GetDelegateKernelRegistration(delegate);

  return context->ReplaceNodeSubsetsWithDelegateKernels(context, delegate_kernel_registration, BuildTfLiteIntArray(supported_nodes).get(),
                                                        base_delegate);
}
}  // namespace

TfLiteDelegate* TfLiteDelegateFactory::CreateSimpleDelegate(std::unique_ptr<SimpleDelegateInterface> simple_delegate, int64_t flag) {
  if (simple_delegate == nullptr) {
    return nullptr;
  }
  auto delegate = new TfLiteDelegate();
  delegate->Prepare = &DelegatePrepare;
  delegate->flags = flag;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;
  delegate->data_ = simple_delegate.release();
  return delegate;
}

void TfLiteDelegateFactory::DeleteSimpleDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  SimpleDelegateInterface* simple_delegate = reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
  delete simple_delegate;
  delete delegate;
}

namespace xnnpack_test {

// ------------------------------------------------------------------
// XNNPack delegate kernel.
// Init(): called once for one-time initialization
// Prepare(): called for each different instance of this node
// Eval(): called for inference
// These 3 functions work on subgraphs. For example a network is
// conv2d -> conv2d -> conv2d -> add -> conv2d -> conv2d -> conv2d -> add -> reshape -> fully_connected -> softmax
// If conv2d is delegated: Init() x2, Prepare() x2, Eval() x2. Each has 3
// ------------------------------------------------------------------
class XNNPackDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit XNNPackDelegateKernel(const TfLiteXNNPackDelegateOptions& options) : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params) override {
    fprintf(stderr, "[humu]: XNNPACK-ESP Init()\n");

    counter_Eval = 0;

    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);

    fprintf(stderr, "[humu]: XNNPACK-ESP Init(), inputs_.size = %d\n", params->nodes_to_replace->size);
    fprintf(stderr, "[humu]: XNNPACK-ESP Init(), outputs_.size = %d\n", params->nodes_to_replace->size);
    fprintf(stderr, "[humu]: XNNPACK-ESP Init(), builtin_code_.size = %d\n", params->nodes_to_replace->size);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d\n", i);

      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(context, context->GetNodeAndRegistration(context, node_index, &delegated_node, &delegated_node_registration),
                        kTfLiteOk);

      inputs_[i].push_back(delegated_node->inputs->data[0]);    // input
      inputs_[i].push_back(delegated_node->inputs->data[1]);    // filter
      inputs_[i].push_back(delegated_node->inputs->data[2]);    // bias
      outputs_[i].push_back(delegated_node->outputs->data[0]);  // output
      builtin_code_[i] = delegated_node_registration->builtin_code;

      fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d, inputs->data[0] = %d, size = %d\n", i, delegated_node->inputs->data[0], delegated_node->inputs->size);
      fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d, inputs->data[1] = %d, size = %d\n", i, delegated_node->inputs->data[1], delegated_node->inputs->size);
      fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d, inputs->data[2] = %d, size = %d\n", i, delegated_node->inputs->data[2], delegated_node->inputs->size);
      fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d, outputs->data[1] = %d, size = %d\n", i, delegated_node->outputs->data[0], delegated_node->outputs->size);


      fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d, builtin_code = %d\n", i, builtin_code_[i]);
    }

    return kTfLiteOk;

    // return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  // Check if im2col needs to be allocated, as some version of optimized Conv dont
  // use it. If any change is supporting im2col in any of the Conv versions, then
  // it should be updated here as well
  bool IsIm2ColRequired(const TfLiteTensor* input, TfLiteConvParams* params, const TfLiteTensor* filter, OpData* data, bool is_hybrid,
                        KernelType kernel_type) {
    // If HWCN weights are required, Im2Col not required
    if (data->need_hwcn_weights) return false;

    // segregate based on dilated conv & non-dialated conv
    const bool need_dilated_im2col = params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
    const bool need_non_dilated_im2col =
        params->stride_width != 1 || params->stride_height != 1 || filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

    const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

    // Return early as basic requirement is not met
    if (!need_im2col) return false;

    switch (kernel_type) {
      case kReference:
        if (is_hybrid) {
          return true;
        } else {
          return false;
        }
      case kGenericOptimized:
      case kCblasOptimized:
        // `need_im2col` is always satisfied.
        return true;
      case kMultithreadOptimized:
        if (input->type == kTfLiteUInt8 ||  //
            input->type == kTfLiteInt8 ||   //
            input->type == kTfLiteInt16 ||  // quantized.
            !data->supports_multithreaded_kernel) {
          return true;
        } else {
          return false;
        }
      default:
        return false;
    }
  }

  // Allocate temporary tensors (`im2col`, `hwcn_weights` if necessary).
  // Note: `context->AddTensors` might invalidate pointers to existing tensors.
  // Therefore the logic to add tensors are isolated into this function.
  static TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context, TfLiteNode* node, bool is_hybrid, bool is_per_channel,
                                                         KernelType kernel_type, size_t im2col_bytes) {
    auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    TF_LITE_ENSURE(context, node->inputs->size >= 2);
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
    const TfLiteTensor* filter;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

    // If we're using the optimized multithreaded EigenTensor implementation of
    // convolution, it expects the filter weights to be transposed compared to
    // the normal TF Lite buffer format. Typical TF Lite weights are
    // [filter_count, filter_height, filter_width, input_depth], but for the float
    // implementation we need them as [filter_height, filter_width, input_depth,
    // filter_count]. We get to that format by transposing, and create a temporary
    // buffer to store the results.
    // This path is only used for float processing, so only create the buffer if
    // we're running with that data type.
    data->need_hwcn_weights = input->type == kTfLiteFloat32 && data->supports_multithreaded_kernel;

    // We don't always need to allocate im2col. It is only used in some versions
    // of the optimized Conv. This test just mimics something that happens inside
    // optimized_ops.h, in order to avoid a DCHECK(!im2col_data).
    // data->need_im2col =
    //     IsIm2ColRequired(input, params, filter, data, is_hybrid, kernel_type2);
    data->need_im2col = 0;  // [humu]: hardcode it to 0 for now

    // If im2col_oversized is found to be true, we have to fallback to an
    // execution path (like kReference in float/quantized cases) that doesn't
    // require im2col operation. Therefore, we have to skip checking the hybrid
    // case (but not the hybrid-per-channel one) where there's no such a fallback
    // execution path.
    // TODO(b/178743262): Consider making this check conditioned on the available
    // memory of the system, rather than coupling to the mobile platform check.
    if (IsMobilePlatform() && !(is_hybrid && !is_per_channel) && data->need_im2col && im2col_bytes >= kMaxIm2colBufferSizeMobile) {
      data->need_im2col = false;
      data->im2col_oversized = true;
    }
    int temporaries_count = 0;
    if (data->need_im2col) {
      data->im2col_index = temporaries_count;
      if (data->im2col_id == kTensorNotAllocated) {
        context->AddTensors(context, 1, &data->im2col_id);
      }
      ++temporaries_count;
    }
    if (data->need_hwcn_weights) {
      data->hwcn_weights_index = temporaries_count;
      if (data->hwcn_weights_id == kTensorNotAllocated) {
        context->AddTensors(context, 1, &data->hwcn_weights_id);
      }
      ++temporaries_count;
    }

    if (is_hybrid) {
      // Allocate tensor to store the on-the-fly quantized inputs.
      data->input_quantized_index = temporaries_count;
      if (data->input_quantized_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(context, context->AddTensors(context, 1, &data->input_quantized_id));
      }
      ++temporaries_count;

      // Allocate tensor to store the quantization params computed during
      // on-the-fly input quantization.
      data->scaling_factors_index = temporaries_count;
      if (data->scaling_factors_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(context, context->AddTensors(context, 1, &data->scaling_factors_id));
      }
      ++temporaries_count;

      // Allocate tensor to store the accumulators for the matrix multiply.
      data->accum_scratch_index = temporaries_count;
      if (data->accum_scratch_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(context, context->AddTensors(context, 1, &data->accum_scratch_id));
      }
      ++temporaries_count;
      if (is_per_channel) {
        data->input_offset_index = temporaries_count;
        if (data->input_offset_id == kTensorNotAllocated) {
          TF_LITE_ENSURE_OK(context, context->AddTensors(context, 1, &data->input_offset_id));
        }
        ++temporaries_count;

        data->row_sums_index = temporaries_count;
        if (data->row_sums_id == kTensorNotAllocated) {
          TF_LITE_ENSURE_OK(context, context->AddTensors(context, 1, &data->row_sums_id));
        }
        ++temporaries_count;
      }
    }

    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(temporaries_count);

    return kTfLiteOk;
  }

  // [humu]: #define TF_LITE_ENSURE_EQ(context, a, b)
  // Check whether the value `a == b` is true, and if not return kTfLiteError
  // from the current function, while also reporting the location of the error.
  // `a` and `b` may be evaluated more than once, so no side effects or
  // extremely expensive computations should be done.

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    // return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare()\n");

    auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    printf("[humu]: conv.cc, Prepare\n");

    bool has_bias = node->inputs->size == 3;
    // Check number of inputs/outputs
    TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
    TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
    const TfLiteTensor* filter;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

fprintf(stderr, "-------------------->     context->tensors[tensor_index] : %d\n", *(context->tensors[0].dims));

fprintf(stderr, "-------------------->     context->tensors[tensor_index] : %d\n", *(context->tensors[1].dims));

fprintf(stderr, "-------------------->     context->tensors[tensor_index] : %d\n", *(context->tensors[2].dims));

fprintf(stderr, "-------------------->     input->dims->size : %d\n", input->dims->size);

fprintf(stderr, "-------------------->     filter->dims->size : %d\n",filter->dims->size);


    // Check dimensionality of input, filter
    TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare(), debug 1\n");

    TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare(), debug 2\n");

    // Check input channels matching filter
    // Filter input channel can be a factor of channels of input (grouped conv)
    // or equals (normal conv).
    auto input_channel = input->dims->data[3];
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare(), debug 3\n");

    auto filter_input_channel = filter->dims->data[3];
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare(), debug 4, filter_input_channel = %d\n", filter_input_channel);

    TF_LITE_ENSURE_EQ(context, input_channel % filter_input_channel, 0);
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare(), debug 5\n");

    data->groups = input_channel / filter_input_channel;
    fprintf(stderr, "[humu]: XNNPACK-ESP Prepare(), debug 6\n");

    // Check types. (We assume that UINT8 refers to quantized tensors)
    TfLiteType input_type = input->type;
    TF_LITE_ENSURE(context,
                   input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 || input_type == kTfLiteInt8 || input_type == kTfLiteInt16);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

    if (input_type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
    // Filter must have zero zero-points in per-channel quantization.
    if (input_type == kTfLiteInt16 || input_type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);
      const auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
      for (int i = 0; i < affine_quantization->zero_point->size; ++i) {
        TF_LITE_ENSURE_EQ(context, affine_quantization->zero_point->data[i], 0);
      }
    }

    const TfLiteTensor* bias = nullptr;

    // TODO(ahentz): At this point the optimized versions require 'bias'. We can
    // either change that or document that convolution requires it.
    TF_LITE_ENSURE(context, has_bias);

    if (has_bias) {
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &bias));
      if (input_type == kTfLiteUInt8 || input_type == kTfLiteInt8) {
        TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
        TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
      } else if (input_type == kTfLiteInt16) {
        TF_LITE_ENSURE(context, (bias->type == kTfLiteInt32) || (bias->type == kTfLiteInt64));
        TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
      } else {
        TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
      }
      TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
    }

    const bool is_hybrid = (input->type == kTfLiteFloat32 && (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));

    if (is_hybrid && filter->type == kTfLiteInt8 && filter->quantization.type == kTfLiteAffineQuantization && filter->quantization.params &&
        reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)->scale &&
        reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)->scale->size > 1) {
      const auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
      const float scale = affine_quantization->scale->data[0];
      for (int i = 1; i < affine_quantization->scale->size; i++) {
        if (affine_quantization->scale->data[i] != scale) {
          data->is_hybrid_per_channel = true;
          break;
        }
      }
    }

    // The multi-threaded kernel supports neither dilation nor hybrid kernels, and
    // is incompatible with mutable input filters that might change between evals.

    KernelType kernel_type = kMultithreadOptimized;  // [humu]: hardcode the kernel_type to kMultithreadOptimized
    data->supports_multithreaded_kernel = (kernel_type == kMultithreadOptimized) && (context->recommended_num_threads != 1) && !is_hybrid &&
                                          (params->dilation_width_factor == 1) && (params->dilation_height_factor == 1) &&
                                          (filter->allocation_type != kTfLiteArenaRw) && !IsDynamicTensor(filter);

    int channels_in = filter->dims->data[3];
    int channels_out = filter->dims->data[0];
    int width = input->dims->data[2];
    int height = input->dims->data[1];
    int filter_width = filter->dims->data[2];
    int filter_height = filter->dims->data[1];
    int batches = input->dims->data[0];

    // Matching GetWindowedOutputSize in TensorFlow.
    auto padding = params->padding;
    int out_width, out_height;
    data->padding = ComputePaddingHeightWidth(params->stride_height, params->stride_width, params->dilation_height_factor,
                                              params->dilation_width_factor, height, width, filter_height, filter_width, padding,
                                              &out_height, &out_width);

    size_t im2col_type_size;
    TF_LITE_ENSURE_STATUS(GetSizeOfType(context, input->type, &im2col_type_size));
    // Note that we intentionally promote the first multiplicand (i.e. 'batches')
    // to 'size_t' to avoid integer overflow here.
    const size_t im2col_bytes =
        static_cast<size_t>(batches) * out_height * out_width * channels_in * filter_height * filter_width * im2col_type_size;
    TF_LITE_ENSURE_STATUS(
        AllocateTemporaryTensorsIfRequired(context, node, is_hybrid, data->is_hybrid_per_channel, kernel_type, im2col_bytes));

    TF_LITE_ENSURE(context, has_bias);

    // Note that full fixed-point inference requires that all tensors have their
    // parameters set. This is usually done during quantized training or
    // calibration.
    if (input_type != kTfLiteFloat32) {
      TF_LITE_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);
      const auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
      TF_LITE_ENSURE(context, affine_quantization);
      TF_LITE_ENSURE(context, affine_quantization->scale);
      TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 || affine_quantization->scale->size == channels_out));

      data->per_channel_output_multiplier.resize(channels_out);
      data->per_channel_output_shift.resize(channels_out);
      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
          context, input, filter, bias, output, params->activation, &data->output_multiplier, &data->output_shift,
          &data->output_activation_min, &data->output_activation_max, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), channels_out));
    }

    TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
    output_size->data[0] = batches;
    output_size->data[1] = out_height;
    output_size->data[2] = out_width;
    output_size->data[3] = channels_out;
    auto output_status = context->ResizeTensor(context, output, output_size);

    if (output_status != kTfLiteOk) return output_status;

    if (data->need_im2col) {
      node->temporaries->data[data->im2col_index] = data->im2col_id;

      TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);

      auto filter_input_channel = filter->dims->data[3];
      im2col_size->data[0] = output_size->data[0];
      im2col_size->data[1] = output_size->data[1];
      im2col_size->data[2] = output_size->data[2];
      im2col_size->data[3] = filter_input_channel * filter_height * filter_width;

      TfLiteTensor* im2col = &context->tensors[node->temporaries->data[data->im2col_index]];
      im2col->type = input->type;
      if (is_hybrid) {
        im2col->type = filter->type;
      }
      im2col->allocation_type = kTfLiteArenaRw;
      auto im2col_status = context->ResizeTensor(context, im2col, im2col_size);
      if (im2col_status != kTfLiteOk) return im2col_status;
    }

    if (data->need_hwcn_weights) {
      node->temporaries->data[data->hwcn_weights_index] = data->hwcn_weights_id;
      TfLiteIntArray* hwcn_weights_size = TfLiteIntArrayCreate(2);

      // Because we're treating the filter weights as a matrix when we do the
      // transpose, we allocate the buffer with a two-dimensional shape, where one
      // dimension is the number of elements in each filter, and the second is the
      // total number of filters.
      auto filter_input_channel = filter->dims->data[3];
      hwcn_weights_size->data[0] = (filter_height * filter_width * filter_input_channel);
      hwcn_weights_size->data[1] = channels_out;

      TfLiteTensor* hwcn_weights = &context->tensors[node->temporaries->data[data->hwcn_weights_index]];
      hwcn_weights->type = input_type;
      hwcn_weights->name = "Conv_hwcn_weights";
      hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;

      auto hwcn_weights_status = context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
      if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

      // TODO(petewarden): If Resize() is called when the size hasn't actually
      // changed, this will do extra redundant work.
      data->have_weights_been_transposed = false;
    }

    if (is_hybrid) {
      node->temporaries->data[data->input_quantized_index] = data->input_quantized_id;
      TfLiteTensor* input_quantized;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, data->input_quantized_index, &input_quantized));
      input_quantized->type = kTfLiteInt8;
      input_quantized->allocation_type = kTfLiteArenaRw;
      if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
        TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized, input_quantized_size));
      }

      node->temporaries->data[data->scaling_factors_index] = data->scaling_factors_id;
      TfLiteTensor* scaling_factors;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, data->scaling_factors_index, &scaling_factors));
      scaling_factors->type = kTfLiteFloat32;
      scaling_factors->allocation_type = kTfLiteArenaRw;
      // Only one scale factor per batch is typically necessary. See optimized
      // implementation for why we need to allocate for the height of the inputs
      // flattened to 2D.
      TF_LITE_ENSURE(context, channels_in != 0);
      const int height = NumElements(input) / channels_in;
      int scaling_dims[1] = {height};
      if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
        TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
        scaling_factors_size->data[0] = height;
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors, scaling_factors_size));
      }

      node->temporaries->data[data->accum_scratch_index] = data->accum_scratch_id;
      TfLiteTensor* accum_scratch;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, data->accum_scratch_index, &accum_scratch));
      accum_scratch->type = kTfLiteInt32;
      accum_scratch->allocation_type = kTfLiteArenaRw;
      const int scratch_width = batches * out_height * out_width;
      int accum_scratch_dims[2] = {channels_out, scratch_width};
      if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2, accum_scratch_dims)) {
        TfLiteIntArray* accum_scratch_size = TfLiteIntArrayCreate(2);
        accum_scratch_size->data[0] = channels_out;
        accum_scratch_size->data[1] = scratch_width;
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, accum_scratch, accum_scratch_size));
      }

      if (data->is_hybrid_per_channel) {
        const auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
        TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, filter->dims->data[affine_quantization->quantized_dimension]);
        node->temporaries->data[data->input_offset_index] = data->input_offset_id;
        TfLiteTensor* input_offsets;
        TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, data->input_offset_index, &input_offsets));
        input_offsets->type = kTfLiteInt32;
        input_offsets->allocation_type = kTfLiteArenaRw;
        // See above comment for the need to allocate for height of inputs.
        TF_LITE_ENSURE(context, channels_in != 0);
        const int height = NumElements(input) / channels_in;
        const int input_offset_dims[1] = {height};
        if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, input_offset_dims)) {
          TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
          input_offsets_size->data[0] = input_offset_dims[0];
          TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets, input_offsets_size));
        }
        node->temporaries->data[data->row_sums_index] = data->row_sums_id;
        TfLiteTensor* row_sums;
        TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
        row_sums->type = kTfLiteInt32;
        row_sums->name = "Conv_row_sums";
        row_sums->allocation_type = kTfLiteArenaRwPersistent;
        // See above comment for the need to allocate for height of inputs.
        const int row_sums_dims[1] = {channels_out};
        if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
          TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
          row_sums_size->data[0] = row_sums_dims[0];
          TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, row_sums, row_sums_size));
        }
      }
    }

    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // Evaluate the delegated graph. Here we loop over all the delegated nodes.

    fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), inputs_.size() = %ld, counter_Eval = %d\n", inputs_.size(), counter_Eval);
    TfLiteStatus ret = kTfLiteOk;

    if (counter_Eval == 0) {
      counter_Eval += 1;

      auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
      OpData* data = reinterpret_cast<OpData*>(node->user_data);

      fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), debug, data->groups = %d\n", data->groups);

      for (int i = 0; i < builtin_code_.size(); ++i) {
        // Get the node input tensors.
        // Add/Sub operation accepts 2 inputs.
        auto& input_tensor_1 = context->tensors[inputs_[i][0]];
        auto& input_tensor_2 = context->tensors[inputs_[i][1]];
        auto& output_tensor = context->tensors[outputs_[i][0]];

        //  fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), builtin_code = %d\n", builtin_code_[i]);

        // ====================================================================
        // [humu]: uncomment the ones that you want to enable
        // ====================================================================
        if (builtin_code_[i] == kTfLiteBuiltinConv2d) {
          ret = doConv2dAcc(context, node);
          // ret = doConv2dSwEvalImp(context, node);
        }
        // if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) {
        //   ret = doConv2dAcc(context, node);
        // }
        // if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) {
        //   // fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), done 1 doAcc, i = %d\n", i);
        //   // ret = doGemmAcc(context, node);
        //   ret = doFineGrainedFC(context, node);
        // }
        // if (builtin_code_[i] == kTfLiteBuiltinAdd) {
        //   ret = ComputeResult(context, builtin_code_[i], &input_tensor_1, &input_tensor_2, &output_tensor);
        // }
        // if (builtin_code_[i] == kTfLiteBuiltinSub) {
        //   ret = ComputeResult(context, builtin_code_[i], &input_tensor_1, &input_tensor_2, &output_tensor);
        // }
        // if (builtin_code_[i] == kTfLiteBuiltinMul) {
        //   ret = ComputeResult(context, builtin_code_[i], &input_tensor_1, &input_tensor_2, &output_tensor);
        // }

        // fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), done 1 doAcc, i = %d\n", i);
      }
    }
    return ret;

    // return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  void doConv2dSwEvalFloat(TfLiteContext* context, TfLiteNode* node, TfLiteConvParams* params, OpData* data, const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* im2col, TfLiteTensor* hwcn_weights,
                           TfLiteTensor* output) {
    printf("[humu]: doConv2dSwEvalFloat() debug 0\n");

    float output_activation_min, output_activation_max;
    CalculateActivationRange(params->activation, &output_activation_min, &output_activation_max);
    KernelType kernel_type = kMultithreadOptimized;  // [humu]: hardcode the kernel_type to kMultithreadOptimized
    KernelType effective_kernel_type = kernel_type;
    // Fall back to the optimized path if multi-threaded conv is unsupported.
    if ((kernel_type == kMultithreadOptimized) && !data->supports_multithreaded_kernel) {
      effective_kernel_type = kGenericOptimized;
      printf("[humu]: doConv2dSwEvalFloat() debug 1\n");
    }

    // When im2col is needed (which is implied when 'im2col_oversized' is true),
    // the GEMMM-based optimized path requires im2col data be allocated to ensure
    // the correctness. Therefore, when im2col is disabled because of the
    // oversized temporary im2col tensor, fallback to a non-optimized path is
    // needed.
    // See b/178743262 for the detailed motivation.
    if (data->im2col_oversized) {
      effective_kernel_type = kReference;
      printf("[humu]: doConv2dSwEvalFloat() debug 2\n");

#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
      // As detailed by tflite::multithreaded_ops::Conv implementation in
      // multithreaded_conv.h, the Eigen-based execution doesn't need im2col data.
      // Therefore, we could rely on it as a better-optimized fallback than the
      // reference one.
      if (data->supports_multithreaded_kernel) {
        effective_kernel_type = kMultithreadOptimized;
        printf("[humu]: doConv2dSwEvalFloat() debug 3\n");
      }
#endif
    }

    // Grouped convolution is right now only supported on reference kernel.
    if (data->groups != 1) {
      effective_kernel_type = kReference;
      printf("[humu]: doConv2dSwEvalFloat() debug 4, data->groups = %d\n", data->groups);
    }

    ConvParams op_params;
    // op_params.padding_type = RuntimePaddingType(params->padding);
    switch (params->padding) {
      case TfLitePadding::kTfLitePaddingSame:
        op_params.padding_type = PaddingType::kSame;
      case TfLitePadding::kTfLitePaddingValid:
        op_params.padding_type = PaddingType::kValid;
      case TfLitePadding::kTfLitePaddingUnknown:
      default:
        op_params.padding_type = PaddingType::kNone;
    }

    op_params.padding_values.width = data->padding.width;
    op_params.padding_values.height = data->padding.height;
    op_params.stride_width = params->stride_width;
    op_params.stride_height = params->stride_height;
    op_params.dilation_width_factor = params->dilation_width_factor;
    op_params.dilation_height_factor = params->dilation_height_factor;
    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;

    switch (effective_kernel_type) {
      case kReference: {
        printf("[humu]: conv.cc, EvalFloat:kReference \n");

        reference_ops::Conv(op_params, GetTensorShape(input), GetTensorData<float>(input), GetTensorShape(filter),
                            GetTensorData<float>(filter), GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
                            GetTensorData<float>(output), GetTensorShape(im2col), GetTensorData<float>(im2col));
        break;
      }
      case kCblasOptimized:
      case kGenericOptimized: {
        printf("[humu]: conv.cc, EvalFloat:kGenericOptimized \n");

        optimized_ops::Conv(op_params, GetTensorShape(input), GetTensorData<float>(input), GetTensorShape(filter),
                            GetTensorData<float>(filter), GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
                            GetTensorData<float>(output), GetTensorShape(im2col), GetTensorData<float>(im2col),
                            CpuBackendContext::GetFromContext(context));
        break;
      }
      case kMultithreadOptimized: {
        printf("[humu]: conv.cc, EvalFloat:kMultithreadOptimized \n");

#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)

        printf("[humu]: conv.cc, EvalFloat:TFLITE_WITH_MULTITHREADED_EIGEN = 1 \n");

        const float* filter_data;
        if (data->need_hwcn_weights) {
          filter_data = GetTensorData<float>(hwcn_weights);
        } else {
          filter_data = GetTensorData<float>(filter);
        }
        multithreaded_ops::Conv(*eigen_support::GetThreadPoolDevice(context), op_params, GetTensorShape(input), GetTensorData<float>(input),
                                GetTensorShape(filter), filter_data, GetTensorShape(bias), GetTensorData<float>(bias),
                                GetTensorShape(output), GetTensorData<float>(output), GetTensorShape(im2col), GetTensorData<float>(im2col));
        break;
#else   // !defined(TFLITE_WITH_MULTITHREADED_EIGEN)

        printf("[humu]: conv.cc, EvalFloat:TFLITE_WITH_MULTITHREADED_EIGEN = 0 \n");
        // See Register_CONV_2D: we should never be here when TFLITE_WITH_RUY
        // was enabled. We #if out this code in order to get the corresponding
        // binary size benefits.
        TFLITE_DCHECK(false);
#endif  // defined(TFLITE_WITH_MULTITHREADED_EIGEN)
      }
    }
  }

  TfLiteStatus doConv2dSwEvalImp(TfLiteContext* context, TfLiteNode* node) {
    fprintf(stderr, "[humu]: ================== doConv2dSwEvalImp: debug 0\n");

    auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
    const TfLiteTensor* filter;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
    bool has_bias = node->inputs->size == 3;
    const TfLiteTensor* bias = has_bias ? GetInput(context, node, 2) : nullptr;
    TfLiteTensor* im2col = data->need_im2col ? &context->tensors[node->temporaries->data[data->im2col_index]] : nullptr;
    TfLiteTensor* hwcn_weights = data->need_hwcn_weights ? &context->tensors[node->temporaries->data[data->hwcn_weights_index]] : nullptr;

    if (data->need_hwcn_weights && !data->have_weights_been_transposed) {
      // TransposeFloatTensor(filter, hwcn_weights);
      // [humu]: here I hardcode the TransposeFloatTensor function below
      const int rows = hwcn_weights->dims->data[1];
      const int cols = hwcn_weights->dims->data[0];
      const float* input_data = GetTensorData<float>(filter);
      float* output_data = GetTensorData<float>(hwcn_weights);
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          const float in_value = input_data[i * cols + j];
          output_data[j * rows + i] = in_value;
        }
      }

      data->have_weights_been_transposed = true;
    }

    // [humu]: input->type can be kTfLiteFloat32, kTfLiteUInt8, kTfLiteInt8, kTfLiteInt16
    // TFLITE_DCHECK_EQ(input_type, input->type);
    switch (input->type) {  // Already know in/outtypes are same.
      case kTfLiteFloat32:
        // if (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8) {
        //   if (data->is_hybrid_per_channel ||
        //       // TODO(b/162870360): Fallback to PerChannel implementation
        //       // before we have grouped hybrid convolution.
        //       data->groups != 1) {
        //     TF_LITE_ENSURE_OK(context, EvalHybridPerChannel<kernel_type>(
        //                                    context, node, params, data, input,
        //                                    filter, bias, im2col, output));
        //   } else {
        //     TfLiteTensor* accum_scratch =
        //         &context->tensors[node->temporaries
        //                               ->data[data->accum_scratch_index]];
        //     TF_LITE_ENSURE_OK(context,
        //                       EvalHybrid<kernel_type>(context, node, params, data,
        //                                               input, filter, bias, im2col,
        //                                               accum_scratch, output));
        //   }
        // } else {
        doConv2dSwEvalFloat(context, node, params, data, input, filter, bias, im2col, hwcn_weights, output);
        // }
        break;
      // case kTfLiteUInt8:
      //   EvalQuantized<kernel_type>(context, node, params, data, input, filter,
      //                              bias, im2col, output);
      //   break;
      // case kTfLiteInt8:
      //   EvalQuantizedPerChannel<kernel_type>(context, node, params, data, input,
      //                                        filter, bias, output, im2col);
      //   break;
      // case kTfLiteInt16:
      //   EvalQuantizedPerChannel16x8<kernel_type>(
      //       context, node, params, data, input, filter, bias, output, im2col);
      //   break;
      default:
        TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.", TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
    return kTfLiteOk;
  }

  TfLiteStatus doConv2dAcc(TfLiteContext* context, TfLiteNode* node) {
    fprintf(stderr, "[humu]: ================== doConv2dAcc: 0717\n");

    const TfLiteTensor* tensors = context->tensors;
    const TfLiteConvParams* conv_params = static_cast<const TfLiteConvParams*>(node->builtin_data);

    // [humu]: the parameters are wrong (output and input tensors have to switch???)
    const TfLiteTensor* input;
    const TfLiteTensor* filter;
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

    ConvParams params;
    params.padding_type = PaddingType::kSame;
    // params.padding_values.width = data->padding.width;
    // params.padding_values.height = data->padding.height;
    params.stride_width = params.stride_width;
    params.stride_height = params.stride_height;
    // params.dilation_width_factor = params->dilation_width_factor;
    // params.dilation_height_factor = params->dilation_height_factor;
    // params.float_activation_min = output_activation_min;
    // params.float_activation_max = output_activation_max;

    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& filter_shape = GetTensorShape(filter);
    const RuntimeShape& output_shape = GetTensorShape(output);

    // setup in multithreaded_conv.h Conv()
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const PaddingType padding = params.padding_type;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    fprintf(stderr, "[humu]: config -- stride_width: %d\n", stride_width);
    fprintf(stderr, "[humu]: config -- stride_height: %d\n", stride_height);
    fprintf(stderr, "[humu]: config -- pad_width: %d\n", pad_width);
    fprintf(stderr, "[humu]: config -- pad_height: %d\n", pad_height);
    fprintf(stderr, "[humu]: config -- output_activation_min: %f\n", output_activation_min);
    fprintf(stderr, "[humu]: config -- output_activation_max: %f\n", output_activation_max);
    fprintf(stderr, "[humu]: config -- batches: %d\n", batches);
    fprintf(stderr, "[humu]: config -- input_depth: %d\n", input_depth);
    fprintf(stderr, "[humu]: config -- output_depth: %d\n", output_depth);
    fprintf(stderr, "[humu]: config -- input_height: %d\n", input_height);
    fprintf(stderr, "[humu]: config -- input_width: %d\n", input_width);
    fprintf(stderr, "[humu]: config -- filter_height: %d\n", filter_height);
    fprintf(stderr, "[humu]: config -- filter_width: %d\n", filter_width);
    fprintf(stderr, "[humu]: config -- output_height: %d\n", output_height);
    fprintf(stderr, "[humu]: config -- output_width: %d\n", output_width);

    bool do_conv2d_sw = (filter_height == 1 && filter_width == 1 && input_height == 1 && input_width == 1);

    int gops = filter_height * filter_width * input_depth * input_height * input_width * input_depth;
    if (gops < 16384) {
      fprintf(stderr, "[humu]: doConv2dAcc: gops = %d\n", gops);
      do_conv2d_sw = 1;
    }

    if (output_height == 1 && output_width == 1 && input_height == 1 && input_width == 1) {
      fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d, case 2 (output_height = output_width = input_height = input_width = 1)\n",
              do_conv2d_sw);

      do_conv2d_sw = 1;
    }
    if (filter_height != filter_width) {
      fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d, case 3 (filter_height = %d, filter_width = %d)\n", do_conv2d_sw,
              filter_height, filter_width);

      do_conv2d_sw = 1;
    }
    if (filter_height == 1 || filter_width == 1) {
      fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d, case 4 (filter_height = %d, filter_width = %d)\n", do_conv2d_sw);

      do_conv2d_sw = 1;
    }

    ///*
    fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d\n", do_conv2d_sw);

    if (!do_conv2d_sw) {
#ifdef ESP_RISCV
      fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d, case 5\n", do_conv2d_sw);

      acc_buf = (token_t*)esp_alloc(MAX_SIZE);

      // set ACC parameters
      conv2d_cfg_000[0].n_channels = output_depth;
      conv2d_cfg_000[0].feature_map_height = output_height;
      conv2d_cfg_000[0].feature_map_width = output_width;
      conv2d_cfg_000[0].n_filters = output_depth;
      conv2d_cfg_000[0].filter_dim = filter_height;
      if (padding == tflite::PaddingType::kSame) {
        conv2d_cfg_000[0].is_padded = 1;
      } else {
        conv2d_cfg_000[0].is_padded = 0;
      }
      conv2d_cfg_000[0].stride = 1;     // should be the same as stride_cols
      conv2d_cfg_000[0].do_relu = 0;    // this function doesn't do relu (?)
      conv2d_cfg_000[0].pool_type = 0;  // this function doesn't do pooling (?)
      conv2d_cfg_000[0].batch_size = batches;

      // load_buffer_conv2d(acc_buf, )

      // print_conv2d_cfg(&cfg_conv2d[0], &conv2d_cfg_000[0]);
      cfg_conv2d[0].hw_buf = acc_buf;

      //  fprintf(stderr, "[humu]: doConv2dAcc: esp_run()\n");
      esp_run_no_print(cfg_conv2d, NACC);

      // store_buffer_conv2d(acc_buf, )

      esp_free(acc_buf);

#endif  // ESP_RISCV
    }
    // */

    return kTfLiteOk;
  }

  TfLiteStatus doGemmAcc(TfLiteContext* context, TfLiteNode* node) {
    fprintf(stderr, "[humu]: ================== doGemmAcc: \n");

    constexpr int kInputTensor = 0;
    constexpr int kWeightsTensor = 1;
    constexpr int kBiasTensor = 2;
    constexpr int kOutputTensor = 0;
    constexpr int kShuffledInputWorkspaceTensor = 1;

    const TfLiteTensor* tensors = context->tensors;

    auto* params = reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    // OpData* data = reinterpret_cast<OpData*>(node->user_data);

    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
    const TfLiteTensor* filter;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kWeightsTensor, &filter));
    const TfLiteTensor* bias = (node->inputs->size == 3) ? GetOptionalInputTensor(context, node, kBiasTensor) : nullptr;
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));
    // Do nothing if expected output is empty.
    if (NumElements(output) == 0) {
      fprintf(stderr, "[humu]: ================== doGemmAcc: NumElements(output) == 0\n");
      return kTfLiteOk;
    }

    // if (filter->dims->data[1] == 0) {
    //   fprintf(stderr, "[humu]: ================== doGemmAcc: filter->dims->data[1] == 0\n");
    //   memset(output->data.data, 0, output->bytes);
    //   return kTfLiteOk;
    // }

    RuntimeShape input_shape = GetTensorShape(input);
    RuntimeShape weights_shape = GetTensorShape(filter);
    RuntimeShape bias_shape = GetTensorShape(bias);
    RuntimeShape output_shape = GetTensorShape(output);

    // ruy::profiler::ScopeLabel label("FullyConnected");
    const int dims_count = weights_shape.DimensionsCount();
    const int input_rows = weights_shape.Dims(dims_count - 1);

    // cpu_backend_gemm::MatrixParams<float> rhs_params;
    // rhs_params.order = cpu_backend_gemm::Order::kColMajor;
    int rhs_params_rows = input_rows;
    int rhs_params_cols = input_shape.FlatSize() / input_rows;
    // rhs_params.cache_policy =
    //     cpu_backend_gemm::DefaultCachePolicy(params.rhs_cacheable);
    TFLITE_DCHECK_EQ(input_shape.FlatSize(), rhs_params_rows * rhs_params_cols);

    // cpu_backend_gemm::MatrixParams<float> lhs_params;
    // lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
    int lhs_params_cols = weights_shape.Dims(dims_count - 1);
    int lhs_params_rows = FlatSizeSkipDim(weights_shape, dims_count - 1);
    // lhs_params.cache_policy =
    //     cpu_backend_gemm::DefaultCachePolicy(params.lhs_cacheable);

    // cpu_backend_gemm::MatrixParams<float> dst_params;
    // dst_params.order = cpu_backend_gemm::Order::kColMajor;
    int dst_params_rows = output_shape.Dims(output_shape.DimensionsCount() - 1);
    int dst_params_cols = FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);

    // cpu_backend_gemm::GemmParams<float, float> gemm_params;
    // gemm_params.bias = optional_bias_data;
    // gemm_params.clamp_min = params.float_activation_min;
    // gemm_params.clamp_max = params.float_activation_max;

    unsigned in_len;
    unsigned in1_len;
    unsigned out_len;
    unsigned in_size;
    unsigned out_size;
    unsigned size;

    if (rhs_params_cols == 0 || lhs_params_cols == 0 || lhs_params_rows == 0) {
      fprintf(stderr, "  config error!!! \n");
      fprintf(stderr, "      rhs_params_cols = %d\n", rhs_params_cols);
      fprintf(stderr, "      lhs_params_cols = %d\n", lhs_params_cols);
      fprintf(stderr, "      lhs_params_rows = %d\n", lhs_params_rows);
      return kTfLiteOk;
      // return kTfLiteError;
    }

#ifdef ESP_RISCV
    // token_t* acc_buf;
    acc_buf = (token_t*)esp_alloc(50000000);
    cfg_gemm[0].hw_buf = acc_buf;

    fprintf(stderr, "\n\n-------------------\n");

    // calculate test parameters
    gemm_init_parameters(0, 0, 0, 1, rhs_params_cols, lhs_params_cols, lhs_params_rows, &in_len, &in1_len, &out_len, &in_size, &out_size,
                         &size);

    print_gemm_cfg(&cfg_gemm[0], &gemm_cfg_000[0]);

    fprintf(stderr, "  Start accelerator execution\n");
    esp_run(cfg_gemm, 1);
    fprintf(stderr, "  Completed accelerator execution\n");

    esp_free(acc_buf);
#endif  // ESP_RISCV

    bool do_linux_test = 0;

    return kTfLiteOk;
  }

  TfLiteStatus doFineGrainedFC(TfLiteContext* context, TfLiteNode* node) {
    int mult_len;
    int add_len;

    fprintf(stderr, "[humu]: ================== doFineGrainedFC: \n");

    const TfLiteTensor* tensors = context->tensors;
    const TfLiteConvParams* conv_params = static_cast<const TfLiteConvParams*>(node->builtin_data);

    // [humu]: the parameters are wrong (output and input tensors have to switch???)
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
    const TfLiteTensor* filter;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

    ConvParams params;
    params.padding_type = PaddingType::kSame;
    // params.padding_values.width = data->padding.width;
    // params.padding_values.height = data->padding.height;
    params.stride_width = params.stride_width;
    params.stride_height = params.stride_height;
    // params.dilation_width_factor = params->dilation_width_factor;
    // params.dilation_height_factor = params->dilation_height_factor;
    // params.float_activation_min = output_activation_min;
    // params.float_activation_max = output_activation_max;

    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& filter_shape = GetTensorShape(filter);
    const RuntimeShape& output_shape = GetTensorShape(output);

    // setup in multithreaded_conv.h Conv()
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const PaddingType padding = params.padding_type;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;

    const int batches2 = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth2 = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth2 = MatchingDim(filter_shape, 0, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    fprintf(stderr, "[humu]: doFCFineGrained: batches2 = %d\n", batches2);
    fprintf(stderr, "[humu]: doFCFineGrained: output_depth2 = %d\n", output_depth2);
    fprintf(stderr, "[humu]: doFCFineGrained: input_depth2 = %d\n", input_depth2);

    int batches = 50;
    int output_depth = 100;
    int accum_depth = 100;
    fprintf(stderr, "[humu]:      doFCFineGrained: batches = %d\n", batches);
    fprintf(stderr, "[humu]:      doFCFineGrained: output_depth = %d\n", output_depth);
    fprintf(stderr, "[humu]:      doFCFineGrained: accum_depth = %d\n", accum_depth);

    // int batches = 100;
    // int output_depth = 100;
    // int accum_depth = input_depth;  // 50;
    float input_data[batches * accum_depth];
    float weights_data[output_depth * accum_depth];
    float bias_data[output_depth];
    float output_data[batches * output_depth], total_1[batches * output_depth];
    float output_data_2[batches * output_depth];
    float output_data_3[batches * output_depth];
    float output_data_4[batches * output_depth];
    // int output_activation_min=0, output_activation_max=0;
    init_array(input_data, batches * accum_depth);
    init_array(weights_data, output_depth * accum_depth);
    init_array(bias_data, output_depth);
    init_array_0(output_data, batches * output_depth);
    init_array_0(output_data_2, batches * output_depth);
    init_array_0(output_data_3, batches * output_depth);
    init_array_0(output_data_4, batches * output_depth);
    // int bias_data = 1;
    int print_all = 0;
    int errors = 0;

    // 1 - original
    /*
    fprintf(stderr, "[humu]: fine-grained #1 original\n");

    for (int b = 0; b < batches; ++b) {
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        float total = 0.f;
        for (int d = 0; d < accum_depth; ++d) {
          total += input_data[b * accum_depth + d] * weights_data[out_c * accum_depth + d];
        }
        float bias_value = 0.0f;
        if (bias_data) {
          bias_value = bias_data[out_c];
        }
        output_data[out_c + output_depth * b] =
            ActivationFunctionWithMinMax(total + bias_value, output_activation_min, output_activation_max);

        if (print_all) fprintf(stderr, "total_1 is: %f batch: %d out_c: %d\n", output_data[out_c + output_depth * b], b, out_c);

        // debugging
        total_1[out_c + output_depth * b] = output_data[out_c + output_depth * b];
      }
    }
    */

    // 2 - basic accumulation level
    /*
    fprintf(stderr, "[humu]: fine-grained #2 basic accumulation level\n");

    float mult[accum_depth];
    float total_2[batches * output_depth];
    init_array_0(total_2, batches * output_depth);
    for (int b = 0; b < batches; ++b) {
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        float total = 0.f;

        // Run with mult accelerator - sending input_data[0:accum_depth] and weights_data[0:accum_depth]
        for (int d = 0; d < accum_depth; ++d) {
          mult[d] = input_data[b * accum_depth + d] *
            weights_data[out_c * accum_depth + d];
        }
        // acc_buf = (token_t*)esp_alloc(accum_depth * 3);
        // cfg_tf_mult3[0].hw_buf = acc_buf;

        // tf_mult3_cfg_000[0].tf_length = accum_depth;
        // tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
        // tf_mult3_cfg_000[0].tf_src_dst_offset_1 = accum_depth;
        // tf_mult3_cfg_000[0].tf_src_dst_offset_2 = accum_depth + accum_depth;
        // tf_mult3_cfg_000[0].chunk_size = 4096;

        // // print_tf_mult3_cfg(&cfg_tf_mult3[0], &tf_mult3_cfg_000[0]);

        // load_buffer(acc_buf, b * accum_depth, input_data, out_c * accum_depth, weights_data, accum_depth);

        // esp_run_no_print(cfg_tf_mult3, NACC);

        // store_buffer(acc_buf, 0, mult, accum_depth);

        // esp_free(acc_buf);

        // Accumulate
        for (int d = 0; d < accum_depth; ++d) {
          total_2[out_c + output_depth * b] += mult[d];
        }
      }
      if (bias_data) {
        // Run with Add accelerator
        for (int out_c = 0; out_c < output_depth; ++out_c) {
          total_2[out_c + output_depth * b] += bias_data[out_c];
        }
        // acc_buf = (token_t*)esp_alloc(output_depth * 3);
        // cfg_tf_add3[0].hw_buf = acc_buf;

        // tf_add3_cfg_000[0].tf_length = output_depth;
        // tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
        // tf_add3_cfg_000[0].tf_src_dst_offset_1 = output_depth;
        // tf_add3_cfg_000[0].tf_src_dst_offset_2 = output_depth + output_depth;
        // tf_add3_cfg_000[0].chunk_size = 4096;

        // // print_tf_add3_cfg(&cfg_tf_add3[0], &tf_add3_cfg_000[0]);

        // load_buffer(acc_buf, output_depth * b, total_2, 0, bias_data, output_depth);

        // esp_run_no_print(cfg_tf_add3, NACC);

        // store_buffer(acc_buf, output_depth * b, total_2, output_depth);

        // esp_free(acc_buf);
      }

      // Activation
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        int temp = out_c + output_depth * b;

        output_data_2[out_c + output_depth * b] =
            ActivationFunctionWithMinMax(total_2[out_c + output_depth * b], output_activation_min, output_activation_max);

        if (print_all) fprintf(stderr, "total_2 is: %f batch: %d out_c: %d\n", output_data_2[out_c + output_depth * b], b, out_c);

        // debugging
        total_2[temp] = output_data_2[temp];
        if (abs(total_2[temp] - total_1[temp]) / total_1[temp] > 0.01) {
          // if (errors < 20) {
          //   fprintf(stderr, "mismatch: total_1[%d] = %f, total_2[%d] = %f\n", temp, total_1[temp], temp, total_2[temp]);
          // }
          errors++;
        }
      }
    }
    */

    // 3 - output level
    /*
    fprintf(stderr, "[humu]: fine-grained #3 output level\n");
    mult_len = output_depth * accum_depth;
    add_len = output_depth;

    float mult_batch[accum_depth * output_depth];
    float total_3[batches * output_depth];
    init_array_0(total_3, batches * output_depth);

    for (int b = 0; b < batches; ++b) {
      // Initialize vector for inputs
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        for (int d = 0; d < accum_depth; ++d) {
          mult_batch[d + out_c * accum_depth] = input_data[b * accum_depth + d];
        }
      }

      // Run mult accelerator
      for (int i = 0; i < output_depth * accum_depth; i++) {
        mult_batch[i] *= weights_data[i];
      }
      // acc_buf = (token_t*)esp_alloc(mult_len * 3);
      // cfg_tf_mult3[0].hw_buf = acc_buf;

      // tf_mult3_cfg_000[0].tf_length = mult_len;
      // tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
      // tf_mult3_cfg_000[0].tf_src_dst_offset_1 = mult_len;
      // tf_mult3_cfg_000[0].tf_src_dst_offset_2 = mult_len + mult_len;

      // // print_tf_mult3_cfg(&cfg_tf_mult3[0], &tf_mult3_cfg_000[0]);

      // load_buffer(acc_buf, 0, mult_batch, 0, weights_data, mult_len);

      // esp_run_no_print(cfg_tf_mult3, NACC);

      // store_buffer(acc_buf, 0, mult_batch, mult_len);

      // esp_free(acc_buf);

      // Accumulate output
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        for (int d = 0; d < accum_depth; ++d) {
          total_3[out_c + output_depth * b] += mult_batch[out_c * accum_depth + d];
        }
      }

      if (bias_data) {
        // replace with Add accelerator
        for (int out_c = 0; out_c < output_depth; ++out_c) {
          total_3[out_c + output_depth * b] += bias_data[out_c];
        }
        // acc_buf = (token_t*)esp_alloc(add_len * 3);
        // cfg_tf_add3[0].hw_buf = acc_buf;

        // tf_add3_cfg_000[0].tf_length = add_len;
        // tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
        // tf_add3_cfg_000[0].tf_src_dst_offset_1 = add_len;
        // tf_add3_cfg_000[0].tf_src_dst_offset_2 = add_len + add_len;

        // // print_tf_add3_cfg(&cfg_tf_add3[0], &tf_add3_cfg_000[0]);

        // load_buffer(acc_buf, output_depth * b, total_3, 0, bias_data, add_len);

        // esp_run_no_print(cfg_tf_add3, NACC);

        // store_buffer(acc_buf, output_depth * b, total_3, add_len);

        // esp_free(acc_buf);
      }

      // Activation
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        int temp = out_c + output_depth * b;

        output_data_3[out_c + output_depth * b] =
            ActivationFunctionWithMinMax(total_3[out_c + output_depth * b], output_activation_min, output_activation_max);

        if (print_all) fprintf(stderr, "total_3 is: %f batch: %d out_c: %d\n", output_data_3[out_c + output_depth * b], b, out_c);

        // debugging
        total_3[temp] = output_data_3[temp];
        if (abs(total_3[temp] - total_1[temp]) / total_1[temp] > 0.01) {
          // if (errors < 20) {
          //   fprintf(stderr, "mismatch: total_1[%d] = %f, total_3[%d] = %f\n", temp, total_1[temp], temp, total_3[temp]);
          // }
          errors++;
        }
      }
    }
    */

    // 4 - batch level
    /*
    fprintf(stderr, "[humu]: fine-grained #4 batch level\n");
    mult_len = batches * accum_depth * output_depth;
    add_len = batches * output_depth;

    acc_buf_4 = (token_t*)esp_alloc(mult_len * 3);
    cfg_tf_mult3[0].hw_buf = acc_buf_4;
    tf_mult3_cfg_000[0].tf_length = mult_len;
    tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
    tf_mult3_cfg_000[0].tf_src_dst_offset_1 = mult_len;
    tf_mult3_cfg_000[0].tf_src_dst_offset_2 = mult_len + mult_len;
    tf_mult3_cfg_000[0].chunk_size = 4096;

    // acc_buf_4 = (token_t*)esp_alloc(add_len * 3);
    cfg_tf_add3[0].hw_buf = acc_buf_4;
    tf_add3_cfg_000[0].tf_length = mult_len;
    tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
    tf_add3_cfg_000[0].tf_src_dst_offset_1 = mult_len;
    tf_add3_cfg_000[0].tf_src_dst_offset_2 = mult_len + mult_len;
    tf_add3_cfg_000[0].chunk_size = 4096;

    float mult_above_batch[mult_len];
    float above_batch_weights_data[mult_len];
    float above_batch_bias_data[add_len];
    float total_4[add_len];
    init_array_0(total_4, add_len);

    // Initialize vector for inputs
    for (int b = 0; b < batches; ++b) {
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        for (int d = 0; d < accum_depth; ++d) {
          mult_above_batch[d + out_c * accum_depth + b * accum_depth * output_depth] = input_data[b * accum_depth + d];
        }
      }
    }

    // Initialize vector for weights
    for (int b = 0; b < batches; ++b) {
      for (int i = 0; i < output_depth * accum_depth; i++) {
        above_batch_weights_data[i + b * output_depth * accum_depth] = weights_data[i];
      }
    }

    // Run mult accelerator
    // for (int j = 0; j < mult_len; ++j) {
    //   mult_above_batch[j] *= above_batch_weights_data[j];
    // }

    print_tf_mult3_cfg(&cfg_tf_mult3[0], &tf_mult3_cfg_000[0]);

    // load_buffer(acc_mult_buf, 0, above_batch_weights_data, 0, mult_above_batch, mult_len);

    esp_run(cfg_tf_add3, NACC);  // esp_run(cfg_tf_mult3, NACC);

    // store_buffer(acc_mult_buf, 0, mult_above_batch, mult_len);

    tf_add3_cfg_000[0].tf_length = add_len;
    tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
    tf_add3_cfg_000[0].tf_src_dst_offset_1 = add_len;
    tf_add3_cfg_000[0].tf_src_dst_offset_2 = add_len + add_len;

    // Accumulate output vector
    for (int b = 0; b < batches; ++b) {
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        for (int d = 0; d < accum_depth; ++d) {
          total_4[out_c + output_depth * b] += mult_above_batch[out_c * accum_depth + d + b * accum_depth * output_depth];
        }
      }
    }

    if (bias_data) {
      // Initialize vector for bias
      for (int b = 0; b < batches; ++b) {
        for (int i = 0; i < output_depth; i++) {
          above_batch_bias_data[i + b * output_depth] = bias_data[i];
        }
      }
      // Run with Add accelerator
      // for (int i = 0; i < add_len; ++i) {
      //   total_4[i] += above_batch_bias_data[i];
      // }

      // print_tf_add3_cfg(&cfg_tf_add3[0], &tf_add3_cfg_000[0]);

      // load_buffer(acc_add_buf, 0, total_4, 0, above_batch_bias_data, add_len);

      esp_run(cfg_tf_add3, NACC);

      // store_buffer(acc_add_buf, 0, total_4, add_len);
    }

    // Activation
    for (int b = 0; b < batches; ++b) {
      for (int out_c = 0; out_c < output_depth; ++out_c) {
        int temp = out_c + output_depth * b;

        output_data_4[temp] = ActivationFunctionWithMinMax(total_4[temp], output_activation_min, output_activation_max);

        if (print_all) fprintf(stderr, "total_4 is: %f batch: %d out_c: %d\n", output_data_4[temp], b, out_c);

        // debugging
        total_4[temp] = output_data_4[temp];
        if (abs(total_4[temp] - total_1[temp]) / total_1[temp] > 0.01) {
          // if (errors < 20) {
          //   fprintf(stderr, "mismatch: total_1[%d] = %f, total_4[%d] = %f\n", temp, total_1[temp], temp, total_4[temp]);
          // }
          errors++;
        }
      }
    }
    esp_free(acc_buf_4);

    */

    // esp_free(acc_mult_buf);

    fprintf(stderr, "Number of errors: %d \n", errors);

    return kTfLiteOk;
  }

  /*
    // Computes the result of addition of 'input_tensor_1' and 'input_tensor_2'
    // and store the result in 'output_tensor'.
    TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code, const TfLiteTensor* input_tensor_1,
                               const TfLiteTensor* input_tensor_2, TfLiteTensor* output_tensor) {
      fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, builtin_code = %d\n", builtin_code);

      if (NumElements(input_tensor_1) != NumElements(input_tensor_2) || NumElements(input_tensor_1) != NumElements(output_tensor)) {
        return kTfLiteDelegateError;
      }
      // This code assumes no activation, and no broadcasting needed (both inputs
      // have the same size).
      auto* input_1 = GetTensorData<float>(input_tensor_1);
      auto* input_2 = GetTensorData<float>(input_tensor_2);
      auto* output = GetTensorData<float>(output_tensor);

      fprintf(stderr, "\n\n\n[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));

      for (int i = 0; i < NumElements(input_tensor_1); ++i) {
        if (builtin_code == kTfLiteBuiltinAdd) {
          // fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult Add %d\n", i);
          output[i] = input_1[i] + input_2[i];
        }
        if (builtin_code == kTfLiteBuiltinSub) {
          // fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult Sub %d\n", i);
          output[i] = input_1[i] - input_2[i];
        }
      }
      return kTfLiteOk;

      fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult SHOULDN'T SEE THIS!!!\n");

      // This code assumes no activation, and no broadcasting needed (both inputs have the same size).
      // auto* input_1 = GetTensorData<float>(input_tensor_1);
      // auto* input_2 = GetTensorData<float>(input_tensor_2);
      // auto* output = GetTensorData<float>(output_tensor);

      //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
      //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
      //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: NumElements(output_tensor) = %d\n",  NumElements(output_tensor));

      if (builtin_code == kTfLiteBuiltinAdd) {
        //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: kTfLiteBuiltinAdd\n");

        if (NumElements(input_tensor_1) != NumElements(output_tensor) && NumElements(input_tensor_2) != NumElements(output_tensor)) {
          //  fprintf(stderr, "[humu]: NumElements(input_1): %d, NumElements(input_2): %d, NumElements(output):
          //  %d\n",NumElements(input_tensor_1), NumElements(input_tensor_2), NumElements(output_tensor));
          return kTfLiteDelegateError;
        }

        if (NumElements(input_tensor_1) != NumElements(input_tensor_2)) {
          if (NumElements(input_tensor_1) == 1) {
            // set Acc configs
            int num_add_run = 1;
            int len = NumElements(input_tensor_2);
            if (len > 16384) {
              num_add_run = (int)(len / 16384) + 1;
              len = 16384;
            }
            tf_add3_cfg_000[0].tf_length = len;
            tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
            tf_add3_cfg_000[0].tf_src_dst_offset_1 = len;
            tf_add3_cfg_000[0].tf_src_dst_offset_2 = len + len;

            print_tf_add3_cfg(&cfg_tf_add3[0], &tf_add3_cfg_000[0]);

            // token_t* acc_buf;
            acc_buf = (token_t*)esp_alloc(5000000);
            cfg_tf_add3[0].hw_buf = acc_buf;
            for (int i = 0; i < num_add_run; i++) {
              esp_run_no_print(cfg_tf_add3, NACC);
              // fprintf(stderr, "[humu]: tf_add3: esp_run()\n");
            }
            esp_free(acc_buf);

            return kTfLiteOk;
          }

          if (NumElements(input_tensor_2) == 1) {
            // set Acc configs
            int num_add_run = 1;
            int len = NumElements(input_tensor_1);
            if (len > 16384) {
              num_add_run = (int)(len / 16384) + 1;
              len = 16384;
            }
            tf_add3_cfg_000[0].tf_length = len;
            tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
            tf_add3_cfg_000[0].tf_src_dst_offset_1 = len;
            tf_add3_cfg_000[0].tf_src_dst_offset_2 = len + len;

            print_tf_add3_cfg(&cfg_tf_add3[0], &tf_add3_cfg_000[0]);

            // token_t* acc_buf;
            acc_buf = (token_t*)esp_alloc(5000000);
            cfg_tf_add3[0].hw_buf = acc_buf;
            for (int i = 0; i < num_add_run; i++) {
              esp_run_no_print(cfg_tf_add3, NACC);
              //  fprintf(stderr, "[humu]]: tf_add3: esp_run()\n");
            }
            esp_free(acc_buf);

            return kTfLiteOk;
          }
          return kTfLiteDelegateError;
        }

        if (NumElements(input_tensor_1) == NumElements(input_tensor_2)) {
          // NumElements(input_tensor_1) == NumElements(input_tensor_2) == NumElements(output_tensor)
          // for (int i = 0; i < NumElements(input_tensor_1); ++i) {
          // fprintf(stderr, "[humu]: debug 2, 0\n");
          // output[i] = input_1[i] + input_2[i];
          // float temp = input_1[i] + input_2[i];
          // float temp = i;
          // fprintf(stderr, "[humu]: debug 2, %f\n", temp);
          // }

          // set Acc configs

          int num_add_run = 1;
          int len = NumElements(input_tensor_1);
          if (len > 16384) {
            num_add_run = (int)(len / 16384) + 1;
            len = 16384;
          }
          tf_add3_cfg_000[0].tf_length = len;
          tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
          tf_add3_cfg_000[0].tf_src_dst_offset_1 = len;
          tf_add3_cfg_000[0].tf_src_dst_offset_2 = len + len;

          print_tf_add3_cfg(&cfg_tf_add3[0], &tf_add3_cfg_000[0]);

          // token_t* acc_buf;
          acc_buf = (token_t*)esp_alloc(5000000);
          cfg_tf_add3[0].hw_buf = acc_buf;
          for (int i = 0; i < num_add_run; i++) {
            esp_run_no_print(cfg_tf_add3, NACC);
            //  fprintf(stderr, "[humu]]: tf_add3: esp_run()\n");
          }
          esp_free(acc_buf);

          return kTfLiteOk;
        }
      }

      if (builtin_code == kTfLiteBuiltinSub) {
        //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult:
        //  kTfLiteBuiltinSub\n");

        //  fprintf(stderr, "[humu]]: XNNPACK-ESP ComputeResult,
        //  NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
        //  fprintf(stderr, "[humu]]: XNNPACK-ESP ComputeResult,
        //  NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
        //  fprintf(stderr, "[humu]]: XNNPACK-ESP ComputeResult,
        //  NumElements(output_tensor) = %d\n", NumElements(output_tensor));

        if (NumElements(input_tensor_1) != NumElements(output_tensor) && NumElements(input_tensor_2) != NumElements(output_tensor)) {
          //  fprintf(stderr, "[humu]]: NumElements(input_1): %d,
          //  NumElements(input_2): %d, NumElements(output):
          //  %d\n",NumElements(input_tensor_1), NumElements(input_tensor_2),
          //  NumElements(output_tensor));
          return kTfLiteDelegateError;
        }

        if (NumElements(input_tensor_1) != NumElements(input_tensor_2)) {
          if (NumElements(input_tensor_1) == 1) {
            //   for (int i = 0; i < NumElements(input_tensor_2); i++) {
            //     output[i] = input_1[0] + input_2[i];
            //   }

            // set Acc configs
            int num_add_run = 1;
            int len = NumElements(input_tensor_2);
            if (len > 16384) {
              num_add_run = (int)(len / 16384) + 1;
              len = 16384;
            }
            tf_sub3_cfg_000[0].tf_length = len;
            tf_sub3_cfg_000[0].tf_src_dst_offset_0 = 0;
            tf_sub3_cfg_000[0].tf_src_dst_offset_1 = len;
            tf_sub3_cfg_000[0].tf_src_dst_offset_2 = len + len;

            //  fprintf(stderr, "[humu]]: tf_sub3_cfg_000[0].tf_length = %d\n",
            //  tf_sub3_cfg_000[0].tf_length); fprintf(stderr, "[humu]]:
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_0 = %d\n",
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_0); fprintf(stderr, "[humu]]:
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_1 = %d\n",
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_1); fprintf(stderr, "[humu]]:
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_2 = %d\n",
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_2);

            // token_t* acc_buf;
            acc_buf = (token_t*)esp_alloc(5000000);
            cfg_tf_sub3[0].hw_buf = acc_buf;
            for (int i = 0; i < num_add_run; i++) {
              esp_run_no_print(cfg_tf_sub3, NACC);
              //  fprintf(stderr, "[humu]]: tf_sub3: esp_run()\n");
            }
            esp_free(acc_buf);

            return kTfLiteOk;
          }

          if (NumElements(input_tensor_2) == 1) {
            // for (int i = 0; i < NumElements(input_tensor_1); i++) {
            // fprintf(stderr, "[humu]: debug 0, %d\n", i);
            // fprintf(stderr, "[humu]: debug ff, %f\n", input_1[i]);
            // fprintf(stderr, "[humu]: debug ff, %f\n", input_2[0]);
            // fprintf(stderr, "[humu]: debug ff, %f\n", output[i]);
            //   float temp = input_1[i] + input_2[0];
            //   // output[i] = input_1[i] + input_2[0];
            // fprintf(stderr, "[humu]: debug 1, %f\n", temp);

            // }

            // set Acc configs
            int num_add_run = 1;
            int len = NumElements(input_tensor_1);
            if (len > 16384) {
              num_add_run = (int)(len / 16384) + 1;
              len = 16384;
            }
            tf_sub3_cfg_000[0].tf_length = len;
            tf_sub3_cfg_000[0].tf_src_dst_offset_0 = 0;
            tf_sub3_cfg_000[0].tf_src_dst_offset_1 = len;
            tf_sub3_cfg_000[0].tf_src_dst_offset_2 = len + len;

            //  fprintf(stderr, "[humu]]: tf_sub3_cfg_000[0].tf_length = %d\n",
            //  tf_sub3_cfg_000[0].tf_length); fprintf(stderr, "[humu]]:
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_0 = %d\n",
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_0); fprintf(stderr, "[humu]]:
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_1 = %d\n",
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_1); fprintf(stderr, "[humu]]:
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_2 = %d\n",
            //  tf_sub3_cfg_000[0].tf_src_dst_offset_2);

            // token_t* acc_buf;
            acc_buf = (token_t*)esp_alloc(5000000);
            cfg_tf_sub3[0].hw_buf = acc_buf;
            for (int i = 0; i < num_add_run; i++) {
              esp_run_no_print(cfg_tf_sub3, NACC);
              //  fprintf(stderr, "[humu]]: tf_sub3: esp_run()\n");
            }
            esp_free(acc_buf);

            return kTfLiteOk;
          }
          return kTfLiteDelegateError;
        }

        if (NumElements(input_tensor_1) == NumElements(input_tensor_2)) {
          // NumElements(input_tensor_1) == NumElements(input_tensor_2) ==
          // NumElements(output_tensor) for (int i = 0; i <
          // NumElements(input_tensor_1); ++i) { fprintf(stderr, "[humu]: debug 2,
          // 0\n");
          //   // output[i] = input_1[i] + input_2[i];
          //   // float temp = input_1[i] + input_2[i];
          //   float temp = i;
          // fprintf(stderr, "[humu]: debug 2, %f\n", temp);
          // }

          // set Acc configs

          int num_add_run = 1;
          int len = NumElements(input_tensor_1);
          if (len > 16384) {
            num_add_run = (int)(len / 16384) + 1;
            len = 16384;
          }
          tf_sub3_cfg_000[0].tf_length = len;
          tf_sub3_cfg_000[0].tf_src_dst_offset_0 = 0;
          tf_sub3_cfg_000[0].tf_src_dst_offset_1 = len;
          tf_sub3_cfg_000[0].tf_src_dst_offset_2 = len + len;

          //  fprintf(stderr, "[humu]]: tf_sub3_cfg_000[0].tf_length = %d\n",
          //  tf_sub3_cfg_000[0].tf_length); fprintf(stderr, "[humu]]:
          //  tf_sub3_cfg_000[0].tf_src_dst_offset_0 = %d\n",
          //  tf_sub3_cfg_000[0].tf_src_dst_offset_0); fprintf(stderr, "[humu]]:
          //  tf_sub3_cfg_000[0].tf_src_dst_offset_1 = %d\n",
          //  tf_sub3_cfg_000[0].tf_src_dst_offset_1); fprintf(stderr, "[humu]]:
          //  tf_sub3_cfg_000[0].tf_src_dst_offset_2 = %d\n",
          //  tf_sub3_cfg_000[0].tf_src_dst_offset_2);

          // token_t* acc_buf;
          acc_buf = (token_t*)esp_alloc(5000000);
          cfg_tf_sub3[0].hw_buf = acc_buf;
          for (int i = 0; i < num_add_run; i++) {
            esp_run_no_print(cfg_tf_sub3, NACC);
            //  fprintf(stderr, "[humu]]: tf_sub3: esp_run()\n");
          }
          esp_free(acc_buf);

          return kTfLiteOk;
        }
      }

      if (builtin_code == kTfLiteBuiltinMul) {
        //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult:
        //  kTfLiteBuiltinMul\n");

        //  fprintf(stderr, "[humu]]: XNNPACK-ESP ComputeResult,
        //  NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
        //  fprintf(stderr, "[humu]]: XNNPACK-ESP ComputeResult,
        //  NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
        //  fprintf(stderr, "[humu]]: XNNPACK-ESP ComputeResult,
        //  NumElements(output_tensor) = %d\n", NumElements(output_tensor));

        if (NumElements(input_tensor_1) != NumElements(output_tensor) && NumElements(input_tensor_2) != NumElements(output_tensor)) {
          //  fprintf(stderr, "[humu]]: NumElements(input_1): %d,
          //  NumElements(input_2): %d, NumElements(output):
          //  %d\n",NumElements(input_tensor_1), NumElements(input_tensor_2),
          //  NumElements(output_tensor));
          return kTfLiteDelegateError;
        }

        if (NumElements(input_tensor_1) != NumElements(input_tensor_2)) {
          if (NumElements(input_tensor_1) == 1) {
            //   for (int i = 0; i < NumElements(input_tensor_2); i++) {
            //     output[i] = input_1[0] + input_2[i];
            //   }

            // set Acc configs
            int num_add_run = 1;
            int len = NumElements(input_tensor_2);
            if (len > 16384) {
              num_add_run = (int)(len / 16384) + 1;
              len = 16384;
            }
            tf_mult3_cfg_000[0].tf_length = len;
            tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
            tf_mult3_cfg_000[0].tf_src_dst_offset_1 = len;
            tf_mult3_cfg_000[0].tf_src_dst_offset_2 = len + len;

            //  fprintf(stderr, "[humu]]: tf_mult3_cfg_000[0].tf_length = %d\n",
            //  tf_mult3_cfg_000[0].tf_length); fprintf(stderr, "[humu]]:
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_0 = %d\n",
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_0); fprintf(stderr,
            //  "[humu]]: tf_mult3_cfg_000[0].tf_src_dst_offset_1 = %d\n",
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_1); fprintf(stderr,
            //  "[humu]]: tf_mult3_cfg_000[0].tf_src_dst_offset_2 = %d\n",
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_2);

            // token_t* acc_buf;
            acc_buf = (token_t*)esp_alloc(5000000);
            cfg_tf_mult3[0].hw_buf = acc_buf;
            for (int i = 0; i < num_add_run; i++) {
              esp_run_no_print(cfg_tf_mult3, NACC);
              //  fprintf(stderr, "[humu]]: tf_mult3: esp_run()\n");
            }
            esp_free(acc_buf);

            return kTfLiteOk;
          }

          if (NumElements(input_tensor_2) == 1) {
            // for (int i = 0; i < NumElements(input_tensor_1); i++) {
            // fprintf(stderr, "[humu]: debug 0, %d\n", i);
            // fprintf(stderr, "[humu]: debug ff, %f\n", input_1[i]);
            // fprintf(stderr, "[humu]: debug ff, %f\n", input_2[0]);
            // fprintf(stderr, "[humu]: debug ff, %f\n", output[i]);
            //   float temp = input_1[i] + input_2[0];
            //   // output[i] = input_1[i] + input_2[0];
            // fprintf(stderr, "[humu]: debug 1, %f\n", temp);

            // }

            // set Acc configs
            int num_add_run = 1;
            int len = NumElements(input_tensor_1);
            if (len > 16384) {
              num_add_run = (int)(len / 16384) + 1;
              len = 16384;
            }
            tf_mult3_cfg_000[0].tf_length = len;
            tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
            tf_mult3_cfg_000[0].tf_src_dst_offset_1 = len;
            tf_mult3_cfg_000[0].tf_src_dst_offset_2 = len + len;

            //  fprintf(stderr, "[humu]]: tf_mult3_cfg_000[0].tf_length = %d\n",
            //  tf_mult3_cfg_000[0].tf_length); fprintf(stderr, "[humu]]:
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_0 = %d\n",
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_0); fprintf(stderr,
            //  "[humu]]: tf_mult3_cfg_000[0].tf_src_dst_offset_1 = %d\n",
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_1); fprintf(stderr,
            //  "[humu]]: tf_mult3_cfg_000[0].tf_src_dst_offset_2 = %d\n",
            //  tf_mult3_cfg_000[0].tf_src_dst_offset_2);

            // token_t* acc_buf;
            acc_buf = (token_t*)esp_alloc(5000000);
            cfg_tf_mult3[0].hw_buf = acc_buf;
            for (int i = 0; i < num_add_run; i++) {
              esp_run_no_print(cfg_tf_mult3, NACC);
              //  fprintf(stderr, "[humu]]: tf_mult3: esp_run()\n");
            }
            esp_free(acc_buf);

            return kTfLiteOk;
          }
          return kTfLiteDelegateError;
        }

        if (NumElements(input_tensor_1) == NumElements(input_tensor_2)) {
          // NumElements(input_tensor_1) == NumElements(input_tensor_2) ==
          // NumElements(output_tensor) for (int i = 0; i <
          // NumElements(input_tensor_1); ++i) { fprintf(stderr, "[humu]: debug 2,
          // 0\n");
          //   // output[i] = input_1[i] + input_2[i];
          //   // float temp = input_1[i] + input_2[i];
          //   float temp = i;
          // fprintf(stderr, "[humu]: debug 2, %f\n", temp);
          // }

          // set Acc configs

          int num_add_run = 1;
          int len = NumElements(input_tensor_1);
          if (len > 16384) {
            num_add_run = (int)(len / 16384) + 1;
            len = 16384;
          }
          tf_mult3_cfg_000[0].tf_length = len;
          tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
          tf_mult3_cfg_000[0].tf_src_dst_offset_1 = len;
          tf_mult3_cfg_000[0].tf_src_dst_offset_2 = len + len;

          //  fprintf(stderr, "[humu]]: tf_mult3_cfg_000[0].tf_length = %d\n",
          //  tf_mult3_cfg_000[0].tf_length); fprintf(stderr, "[humu]]:
          //  tf_mult3_cfg_000[0].tf_src_dst_offset_0 = %d\n",
          //  tf_mult3_cfg_000[0].tf_src_dst_offset_0); fprintf(stderr, "[humu]]:
          //  tf_mult3_cfg_000[0].tf_src_dst_offset_1 = %d\n",
          //  tf_mult3_cfg_000[0].tf_src_dst_offset_1); fprintf(stderr, "[humu]]:
          //  tf_mult3_cfg_000[0].tf_src_dst_offset_2 = %d\n",
          //  tf_mult3_cfg_000[0].tf_src_dst_offset_2);

          // token_t* acc_buf;
          acc_buf = (token_t*)esp_alloc(5000000);
          cfg_tf_mult3[0].hw_buf = acc_buf;
          for (int i = 0; i < num_add_run; i++) {
            esp_run_no_print(cfg_tf_mult3, NACC);
            //  fprintf(stderr, "[humu]]: tf_mult3: esp_run()\n");
          }
          esp_free(acc_buf);

          return kTfLiteOk;
        }
      }

      return kTfLiteOk;
    }


  */

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_;
  std::vector<std::vector<int>> outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;

  int counter_Eval;
  token_t* acc_buf_4;
  token_t* acc_buf;

  const TfLiteXNNPackDelegateOptions options_;
};

// ------------------------------------------------------------------
// XNNPackDelegate represents the Delegate capabilities:
// which operations are supported (ADD) for now,
// and creating a kernel which encapsulates the delegated graph.
// ------------------------------------------------------------------
class XNNPackDelegate : public SimpleDelegateInterface {
 public:
  explicit XNNPackDelegate(const TfLiteXNNPackDelegateOptions& options) : options_(options) {
    printf("[humu]: -------- XNNPackDelegate()\n");
  }
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration, const TfLiteNode* node, TfLiteContext* context) const override {
    printf("[humu]: -------- IsNodeSupportedByDelegate()\n");

    if (
        // ====================================================================
        // [humu]: uncomment the ones that you want to enable
        // ====================================================================
        // registration->builtin_code != kTfLiteBuiltinAdd
        // registration->builtin_code != kTfLiteBuiltinSub &&
        // registration->builtin_code != kTfLiteBuiltinMul
        // registration->builtin_code != kTfLiteBuiltinFullyConnected
        // registration->builtin_code != kTfLiteBuiltinDepthwiseConv2d &&
        registration->builtin_code != kTfLiteBuiltinConv2d
        // [humu]: place holder for formatter
    ) {
      return false;
    }
    printf("[humu]: IsNodeSupportedByDelegate(), debug 1\n");

    // This delegate only supports float32, int32, int8, uint8
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type == kTfLiteFloat32) {
        printf("[humu]: input type: kTfLiteFloat32\n");
      } else if (tensor.type == kTfLiteInt32) {
        printf("[humu]: input type: kTfLiteInt32\n");
      } else if (tensor.type == kTfLiteInt8) {
        printf("[humu]: input type: kTfLiteInt8\n");
      } else if (tensor.type == kTfLiteUInt8) {
        printf("[humu]: input type: kTfLiteUInt8\n");
      } else {
        printf("[humu]: input type: unsupported\n");
        return false;
      }
    }
    printf("[humu]: IsNodeSupportedByDelegate(), debug 2\n");

    return true;
    // return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    printf("[humu]: -------- Initialize()\n");
    return kTfLiteOk;
  }

  const char* Name() const override {
    static constexpr char kName[] = "XNNPackDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface() override {
    return std::make_unique<XNNPackDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const TfLiteXNNPackDelegateOptions options_;
};

}  // namespace xnnpack_test
}  // namespace tflite

TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault() {
  printf("[humu]: -------- TfLiteXNNPackDelegateOptionsDefault()\n");

  TfLiteXNNPackDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this xnnpack test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteXNNPackDelegateCreate(const TfLiteXNNPackDelegateOptions* options) {
  printf("[humu]: -------- TfLiteXNNPackDelegateCreate()\n");

  std::unique_ptr<tflite::xnnpack_test::XNNPackDelegate> xnnpack(
      new tflite::xnnpack_test::XNNPackDelegate(options ? *options : TfLiteXNNPackDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(xnnpack));
  // return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(xnnpack), options->flags);
}

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate) {
  printf("[humu]: -------- TfLiteXNNPackDelegateDelete()\n");
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
