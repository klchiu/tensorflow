#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

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

// [humu]: include this header file for using ESP APIs
#define __FIXED
#define BITWIDTH 32
#include "tensorflow/esp_libs/cfg_tf_add3.h"
#include "tensorflow/esp_libs/cfg_tf_sub3.h"
#include "tensorflow/esp_libs/cfg_tf_mult3.h"

#include "tensorflow/esp_libs/cfg_conv2d.h"
#include "tensorflow/esp_libs/conv2d_helper.h"
#include "tensorflow/esp_libs/esp_api_include.h"
// [humu]: end for including ESP APIs

namespace tflite {
namespace {
TfLiteRegistration GetDelegateKernelRegistration(
    SimpleDelegateInterface* delegate) {
  TfLiteRegistration kernel_registration{};
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = delegate->Name();
  kernel_registration.version = 1;
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<SimpleDelegateKernelInterface*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    if (params == nullptr) {
      TF_LITE_KERNEL_LOG(context, "NULL TfLiteDelegateParams passed.");
      return nullptr;
    }
    auto* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(params->delegate->data_);
    std::unique_ptr<SimpleDelegateKernelInterface> delegate_kernel(
        delegate->CreateDelegateKernelInterface());
    if (delegate_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return delegate_kernel.release();
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Delegate kernel was not initialized");
      return kTfLiteError;
    }
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    return delegate_kernel->Prepare(context, node);
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    TFLITE_DCHECK(delegate_kernel != nullptr);
    return delegate_kernel->Eval(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context,
                             TfLiteDelegate* base_delegate) {
  auto* delegate =
      reinterpret_cast<SimpleDelegateInterface*>(base_delegate->data_);
  auto delegate_options = delegate->DelegateOptions();
  if (delegate_options.max_delegated_partitions <= 0)
    delegate_options.max_delegated_partitions = std::numeric_limits<int>::max();

  TF_LITE_ENSURE_STATUS(delegate->Initialize(context));
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    return delegate->IsNodeSupportedByDelegate(registration, node, context);
  };
  // TODO(b/149484598): Update to have method that gets all supported nodes.
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  std::vector<int> supported_nodes = helper.GetNodesOfFirstNLargestPartitions(
      delegate_options.max_delegated_partitions,
      delegate_options.min_nodes_per_partition);

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "%s delegate: %d nodes delegated out of %d nodes with "
                       "%d partitions.\n",
                       delegate->Name(), supported_nodes.size(),
                       helper.num_total_nodes(), helper.num_partitions());
  TfLiteRegistration delegate_kernel_registration =
      GetDelegateKernelRegistration(delegate);

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, delegate_kernel_registration,
      BuildTfLiteIntArray(supported_nodes).get(), base_delegate);
}
}  // namespace

TfLiteDelegate* TfLiteDelegateFactory::CreateSimpleDelegate(
    std::unique_ptr<SimpleDelegateInterface> simple_delegate, int64_t flag) {
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
  SimpleDelegateInterface* simple_delegate =
      reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
  delete simple_delegate;
  delete delegate;
}

namespace xnnpack_test {

// ------------------------------------------------------------------
// XNNPack delegate kernel.
// Init(): called once for one-time initialization
// Prepare(): called for each different instance of this node
// Eval(): called for inference
// ------------------------------------------------------------------
class XNNPackDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit XNNPackDelegateKernel(const TfLiteXNNPackDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
  //  fprintf(stderr, "[humu]: XNNPACK-ESP Init()\n");

    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
    //  fprintf(stderr, "[humu]: XNNPACK-ESP Init(), loop i = %d\n", i);

      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }

    return kTfLiteOk;

    // return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  // [humu]: #define TF_LITE_ENSURE_EQ(context, a, b)
  // Check whether the value `a == b` is true, and if not return kTfLiteError
  // from the current function, while also reporting the location of the error.
  // `a` and `b` may be evaluated more than once, so no side effects or
  // extremely expensive computations should be done.

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    // return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  //  fprintf(stderr, "[humu]: XNNPACK-ESP Prepare()\n");
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // Evaluate the delegated graph.
    // Here we loop over all the delegated nodes.
    // We know that all the nodes are either ADD or SUB operations and the
    // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
    // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
    // outputs for node
    // ''i''. Note, that it is intentional we have simple implementation as this
    // is for demonstration.
  //  fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), inputs_.size() = %d\n", inputs_.size());

    auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
    // OpData* data = reinterpret_cast<OpData*>(node->user_data);

    for (int i = 0; i < inputs_.size(); ++i) {
      // Get the node input tensors.
      // Add/Sub operation accepts 2 inputs.
      auto& input_tensor_1 = context->tensors[inputs_[i][0]];
      auto& input_tensor_2 = context->tensors[inputs_[i][1]];
      auto& output_tensor = context->tensors[outputs_[i][0]];

    //  fprintf(stderr, "[humu]: XNNPACK-ESP Eval(), builtin_code = %d\n", builtin_code_[i]);

      if (builtin_code_[i] == kTfLiteBuiltinConv2d) {
        doConv2dAcc(context, node);
      }
      else if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) {
        doConv2dAcc(context, node);
      }
      else{
      TF_LITE_ENSURE_EQ(
          context,
          ComputeResult(context, builtin_code_[i], &input_tensor_1,
                        &input_tensor_2, &output_tensor),
          kTfLiteOk);
      }

    }
    return kTfLiteOk;

    // return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  TfLiteStatus doConv2dAcc(TfLiteContext* context, TfLiteNode* node) {
  //  fprintf(stderr, "[humu]: ================== doConv2dAcc: \n");

    const TfLiteTensor* tensors = context->tensors;
    const TfLiteConvParams* conv_params =
        static_cast<const TfLiteConvParams*>(node->builtin_data);

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

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

  //  fprintf(stderr, "[humu]: acc_config --  stride_width: %d\n", stride_width);
  //  fprintf(stderr, "[humu]: acc_config --  stride_height: %d\n",stride_height);
  //  fprintf(stderr, "[humu]: acc_config --  pad_width: %d\n", pad_width);
  //  fprintf(stderr, "[humu]: acc_config --  pad_height: %d\n", pad_height);
  //  fprintf(stderr, "[humu]: acc_config --  output_activation_min: %d\n", output_activation_min);
  //  fprintf(stderr, "[humu]: acc_config --  output_activation_max: %d\n", output_activation_max);
  //  fprintf(stderr, "[humu]: acc_config --  batches: %d\n", batches);
  //  fprintf(stderr, "[humu]: acc_config --  input_depth: %d\n", input_depth);
  //  fprintf(stderr, "[humu]: acc_config --  output_depth: %d\n", output_depth);
  //  fprintf(stderr, "[humu]: acc_config --  input_height: %d\n", input_height);
  //  fprintf(stderr, "[humu]: acc_config --  input_width: %d\n", input_width);
  //  fprintf(stderr, "[humu]: acc_config --  filter_height: %d\n", filter_height);
  //  fprintf(stderr, "[humu]: acc_config --  filter_width: %d\n", filter_width);
  //  fprintf(stderr, "[humu]: acc_config --  output_height: %d\n", output_height);
  //  fprintf(stderr, "[humu]: acc_config --  output_width: %d\n", output_width);

    bool do_conv2d_sw = (filter_height == 1 && filter_width == 1 &&
                               input_height == 1 && input_width == 1);

    if(output_height == 1 && output_width == 1 && input_height == 1 && input_width == 1){
          do_conv2d_sw = 1;
      }

    ///*
    if(do_conv2d_sw){
       fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d\n", do_conv2d_sw);


    }
    else{ // !do_conv2d_sw
      //  fprintf(stderr, "[humu]: doConv2dAcc: do_conv2d_sw = %d\n", do_conv2d_sw);


 // set ACC parameters
      conv2d_cfg_000[0].n_channels = output_depth;
      conv2d_cfg_000[0].feature_map_height = output_height;
      conv2d_cfg_000[0].feature_map_width = output_width;
      conv2d_cfg_000[0].n_filters = output_depth;
      conv2d_cfg_000[0].filter_dim = 1;
      if (padding == tflite::PaddingType::kSame) {
        conv2d_cfg_000[0].is_padded = 1;
      } else {
        conv2d_cfg_000[0].is_padded = 0;
      }
      conv2d_cfg_000[0].stride = 1;                  // should be the same as stride_cols
      conv2d_cfg_000[0].do_relu = 0;    // this function doesn't do relu (?)
      conv2d_cfg_000[0].pool_type = 0;  // this function doesn't do pooling (?)
      conv2d_cfg_000[0].batch_size = batches;


      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, n_channels = %d\n", conv2d_cfg_000[0].n_channels);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, feature_map_height = %d\n", conv2d_cfg_000[0].feature_map_height);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, feature_map_width = %d\n", conv2d_cfg_000[0].feature_map_width);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, n_filters = %d\n", conv2d_cfg_000[0].n_filters);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, filter_dim = %d\n", conv2d_cfg_000[0].filter_dim);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, is_padded = %d\n", conv2d_cfg_000[0].is_padded);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, stride = %d\n", conv2d_cfg_000[0].stride);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, do_relu = %d\n", conv2d_cfg_000[0].do_relu);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, pool_type = %d\n", conv2d_cfg_000[0].pool_type);
      //  fprintf(stderr, "[humu]: doConv2dAcc: conv2d_cfg_000, batch_size = %d\n", conv2d_cfg_000[0].batch_size);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_conv2d[0].hw_buf = acc_buf;

      //  fprintf(stderr, "[humu]: doConv2dAcc: esp_run()\n");
        esp_run_no_print(cfg_conv2d, NACC);


        esp_free(acc_buf);
    }
    // */
    return kTfLiteOk;
  }

  // Computes the result of addition of 'input_tensor_1' and 'input_tensor_2'
  // and store the result in 'output_tensor'.
  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
  //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, builtin_code = %d\n", builtin_code);

    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
    auto* input_1 = GetTensorData<float>(input_tensor_1);
    auto* input_2 = GetTensorData<float>(input_tensor_2);
    auto* output = GetTensorData<float>(output_tensor);

  //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
  //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
  //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: NumElements(output_tensor) = %d\n",  NumElements(output_tensor));


    if (builtin_code == kTfLiteBuiltinAdd) {
      //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: kTfLiteBuiltinAdd\n");

     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(output_tensor) = %d\n", NumElements(output_tensor));


      if (NumElements(input_tensor_1) != NumElements(output_tensor) && NumElements(input_tensor_2) != NumElements(output_tensor)) {
       fprintf(stderr, "[humu]: NumElements(input_1): %d, NumElements(input_2): %d, NumElements(output): %d\n",NumElements(input_tensor_1), NumElements(input_tensor_2), NumElements(output_tensor));
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
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_add3_cfg_000[0].tf_length = len;
        tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_add3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_add3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_length = %d\n", tf_add3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_add3: esp_run()\n");
        }
        esp_free(acc_buf);



          return kTfLiteOk;
        }

        if (NumElements(input_tensor_2) == 1) {
           
          // for (int i = 0; i < NumElements(input_tensor_1); i++) {
          //  fprintf(stderr, "[humu]: debug 0, %d\n", i);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", input_1[i]);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", input_2[0]);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", output[i]);
          //   float temp = input_1[i] + input_2[0];
          //   // output[i] = input_1[i] + input_2[0];
          //  fprintf(stderr, "[humu]: debug 1, %f\n", temp);

          // }

        // set Acc configs
        int num_add_run = 1;
        int len = NumElements(input_tensor_1);
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_add3_cfg_000[0].tf_length = len;
        tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_add3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_add3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_length = %d\n", tf_add3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_add3: esp_run()\n");
        }
        esp_free(acc_buf);




          return kTfLiteOk;
        }
        return kTfLiteDelegateError;
      }
      
      if (NumElements(input_tensor_1) == NumElements(input_tensor_2)) {
      // NumElements(input_tensor_1) == NumElements(input_tensor_2) == NumElements(output_tensor)
        // for (int i = 0; i < NumElements(input_tensor_1); ++i) {
        //  fprintf(stderr, "[humu]: debug 2, 0\n");
        //   // output[i] = input_1[i] + input_2[i];
        //   // float temp = input_1[i] + input_2[i];
        //   float temp = i;
        //  fprintf(stderr, "[humu]: debug 2, %f\n", temp);
        // }


        // set Acc configs

        int num_add_run = 1;
        int len = NumElements(input_tensor_1);
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_add3_cfg_000[0].tf_length = len;
        tf_add3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_add3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_add3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_length = %d\n", tf_add3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_add3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_add3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_add3: esp_run()\n");
        }
        esp_free(acc_buf);

        return kTfLiteOk;
      }
    }





    if (builtin_code == kTfLiteBuiltinSub) {
      //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: kTfLiteBuiltinSub\n");

     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(output_tensor) = %d\n", NumElements(output_tensor));


      if (NumElements(input_tensor_1) != NumElements(output_tensor) && NumElements(input_tensor_2) != NumElements(output_tensor)) {
       fprintf(stderr, "[humu]: NumElements(input_1): %d, NumElements(input_2): %d, NumElements(output): %d\n",NumElements(input_tensor_1), NumElements(input_tensor_2), NumElements(output_tensor));
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
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_sub3_cfg_000[0].tf_length = len;
        tf_sub3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_sub3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_sub3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_length = %d\n", tf_sub3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_sub3: esp_run()\n");
        }
        esp_free(acc_buf);



          return kTfLiteOk;
        }

        if (NumElements(input_tensor_2) == 1) {
           
          // for (int i = 0; i < NumElements(input_tensor_1); i++) {
          //  fprintf(stderr, "[humu]: debug 0, %d\n", i);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", input_1[i]);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", input_2[0]);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", output[i]);
          //   float temp = input_1[i] + input_2[0];
          //   // output[i] = input_1[i] + input_2[0];
          //  fprintf(stderr, "[humu]: debug 1, %f\n", temp);

          // }

        // set Acc configs
        int num_add_run = 1;
        int len = NumElements(input_tensor_1);
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_sub3_cfg_000[0].tf_length = len;
        tf_sub3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_sub3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_sub3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_length = %d\n", tf_sub3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_sub3: esp_run()\n");
        }
        esp_free(acc_buf);




          return kTfLiteOk;
        }
        return kTfLiteDelegateError;
      }
      
      if (NumElements(input_tensor_1) == NumElements(input_tensor_2)) {
      // NumElements(input_tensor_1) == NumElements(input_tensor_2) == NumElements(output_tensor)
        // for (int i = 0; i < NumElements(input_tensor_1); ++i) {
        //  fprintf(stderr, "[humu]: debug 2, 0\n");
        //   // output[i] = input_1[i] + input_2[i];
        //   // float temp = input_1[i] + input_2[i];
        //   float temp = i;
        //  fprintf(stderr, "[humu]: debug 2, %f\n", temp);
        // }


        // set Acc configs

        int num_add_run = 1;
        int len = NumElements(input_tensor_1);
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_sub3_cfg_000[0].tf_length = len;
        tf_sub3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_sub3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_sub3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_length = %d\n", tf_sub3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_sub3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_sub3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_sub3: esp_run()\n");
        }
        esp_free(acc_buf);

        return kTfLiteOk;
      }
    }






    if (builtin_code == kTfLiteBuiltinMul) {
      //  fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: kTfLiteBuiltinMul\n");

     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_1) = %d\n", NumElements(input_tensor_1));
     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(input_tensor_2) = %d\n", NumElements(input_tensor_2));
     fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult, NumElements(output_tensor) = %d\n", NumElements(output_tensor));


      if (NumElements(input_tensor_1) != NumElements(output_tensor) && NumElements(input_tensor_2) != NumElements(output_tensor)) {
       fprintf(stderr, "[humu]: NumElements(input_1): %d, NumElements(input_2): %d, NumElements(output): %d\n",NumElements(input_tensor_1), NumElements(input_tensor_2), NumElements(output_tensor));
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
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_mult3_cfg_000[0].tf_length = len;
        tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_mult3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_mult3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_length = %d\n", tf_mult3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_mult3: esp_run()\n");
        }
        esp_free(acc_buf);



          return kTfLiteOk;
        }

        if (NumElements(input_tensor_2) == 1) {
           
          // for (int i = 0; i < NumElements(input_tensor_1); i++) {
          //  fprintf(stderr, "[humu]: debug 0, %d\n", i);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", input_1[i]);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", input_2[0]);
          //  fprintf(stderr, "[humu]: debug ff, %f\n", output[i]);
          //   float temp = input_1[i] + input_2[0];
          //   // output[i] = input_1[i] + input_2[0];
          //  fprintf(stderr, "[humu]: debug 1, %f\n", temp);

          // }

        // set Acc configs
        int num_add_run = 1;
        int len = NumElements(input_tensor_1);
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_mult3_cfg_000[0].tf_length = len;
        tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_mult3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_mult3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_length = %d\n", tf_mult3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_mult3: esp_run()\n");
        }
        esp_free(acc_buf);




          return kTfLiteOk;
        }
        return kTfLiteDelegateError;
      }
      
      if (NumElements(input_tensor_1) == NumElements(input_tensor_2)) {
      // NumElements(input_tensor_1) == NumElements(input_tensor_2) == NumElements(output_tensor)
        // for (int i = 0; i < NumElements(input_tensor_1); ++i) {
        //  fprintf(stderr, "[humu]: debug 2, 0\n");
        //   // output[i] = input_1[i] + input_2[i];
        //   // float temp = input_1[i] + input_2[i];
        //   float temp = i;
        //  fprintf(stderr, "[humu]: debug 2, %f\n", temp);
        // }


        // set Acc configs

        int num_add_run = 1;
        int len = NumElements(input_tensor_1);
        if (len > 16384){
          num_add_run = (int)(len / 16384) + 1;
          len = 16384;
        }
        tf_mult3_cfg_000[0].tf_length = len;
        tf_mult3_cfg_000[0].tf_src_dst_offset_0 = 0;
		    tf_mult3_cfg_000[0].tf_src_dst_offset_1 = len;
		    tf_mult3_cfg_000[0].tf_src_dst_offset_2 = len+len;

       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_length = %d\n", tf_mult3_cfg_000[0].tf_length);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_0 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_0);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_1 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_1);
       fprintf(stderr, "[humu]: tf_mult3_cfg_000[0].tf_src_dst_offset_2 = %d\n", tf_mult3_cfg_000[0].tf_src_dst_offset_2);

        token_t* acc_buf;
        acc_buf = (token_t*)esp_alloc(5000000);
        cfg_tf_add3[0].hw_buf = acc_buf;
        for(int i = 0 ;i < num_add_run; i++){
          esp_run(cfg_tf_add3, NACC);
         fprintf(stderr, "[humu]: tf_mult3: esp_run()\n");
        }
        esp_free(acc_buf);

        return kTfLiteOk;
      }
    }







    if (builtin_code == kTfLiteBuiltinFullyConnected) {
      fprintf(stderr, "[humu]: XNNPACK-ESP ComputeResult: kTfLiteBuiltinFullyConnected\n");
      // void* buf = NULL;
      // esp_dummy(4);
    }

    return kTfLiteOk;
  }

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_, outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;

  const TfLiteXNNPackDelegateOptions options_;
};

// ------------------------------------------------------------------
// XNNPackDelegate represents the Delegate capabilities:
// which operations are supported (ADD) for now,
// and creating a kernel which encapsulates the delegated graph.
// ------------------------------------------------------------------
class XNNPackDelegate : public SimpleDelegateInterface {
 public:
  explicit XNNPackDelegate(const TfLiteXNNPackDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports Add and Sub ops.
    if (
      // registration->builtin_code != kTfLiteBuiltinAdd &&
        registration->builtin_code != kTfLiteBuiltinSub &&
        registration->builtin_code != kTfLiteBuiltinFullyConnected &&
        registration->builtin_code != kTfLiteBuiltinDepthwiseConv2d &&
        registration->builtin_code != kTfLiteBuiltinConv2d)
      return false;

    // This delegate only supports float32 types.
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteFloat32) return false;
    }

    return true;
    // return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "XNNPackDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
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
  TfLiteXNNPackDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this xnnpack test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options) {
  std::unique_ptr<tflite::xnnpack_test::XNNPackDelegate> xnnpack(
      new tflite::xnnpack_test::XNNPackDelegate(
          options ? *options : TfLiteXNNPackDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(xnnpack));
  // return
  // tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(xnnpack),
  // options->flags);
}

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
