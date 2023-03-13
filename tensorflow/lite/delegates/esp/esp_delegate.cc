#include "tensorflow/lite/delegates/esp/esp_delegate.h"

#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include "tensorflow/lite/builtin_ops.h"

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "tensorflow/lite/logger.h"

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"



#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace esp_test {

// ------------------------------------------------------------------
// Esp delegate kernel.
// Init(): called once for one-time initialization
// Prepare(): called for each different instance of this node
// Eval(): called for inference
// ------------------------------------------------------------------
class EspDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit EspDelegateKernel(const EspDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    
     // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
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
// Check whether the value `a == b` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
// `a` and `b` may be evaluated more than once, so no side effects or
// extremely expensive computations should be done.


  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    // return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
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

    for (int i = 0; i < inputs_.size(); ++i) {
      // Get the node input tensors.
      // Add/Sub operation accepts 2 inputs.
      auto& input_tensor_1 = context->tensors[inputs_[i][0]];
      auto& input_tensor_2 = context->tensors[inputs_[i][1]];
      auto& output_tensor = context->tensors[outputs_[i][0]];
      TF_LITE_ENSURE_EQ(
          context,
          ComputeResult(context, builtin_code_[i], &input_tensor_1,
                        &input_tensor_2, &output_tensor),
          kTfLiteOk);
    }
    return kTfLiteOk;
    
    // return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:

 // Computes the result of addition of 'input_tensor_1' and 'input_tensor_2'
  // and store the result in 'output_tensor'.
  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1) != NumElements(input_tensor_2) ||
        NumElements(input_tensor_1) != NumElements(output_tensor)) {
      return kTfLiteDelegateError;
    }
    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
    auto* input_1 = GetTensorData<float>(input_tensor_1);
    auto* input_2 = GetTensorData<float>(input_tensor_2);
    auto* output = GetTensorData<float>(output_tensor);
    for (int i = 0; i < NumElements(input_tensor_1); ++i) {
      if (builtin_code == kTfLiteBuiltinAdd){
        output[i] = input_1[i] + input_2[i];
        printf("[humu]: ESP ComputeResult: kTfLiteBuiltinAdd\n");
      }
      if (builtin_code == kTfLiteBuiltinSub){
        output[i] = input_1[i] - input_2[i];
        printf("[humu]: ESP ComputeResult: kTfLiteBuiltinSub\n");
      }

      if (builtin_code == kTfLiteBuiltinReshape){
        printf("[humu]: ESP ComputeResult: kTfLiteBuiltinReshape\n");
      }

      if (builtin_code == kTfLiteBuiltinConv2d){
        printf("[humu]: ESP ComputeResult: kTfLiteBuiltinConv2d\n");
      }

      if (builtin_code == kTfLiteBuiltinFullyConnected){
        printf("[humu]: ESP ComputeResult: kTfLiteBuiltinFullyConnected\n");
      }

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




  const EspDelegateOptions options_;
};

// ------------------------------------------------------------------
// EspDelegate represents the Delegate capabilities:
// which operations are supported (ADD) for now,
// and creating a kernel which encapsulates the delegated graph.
// ------------------------------------------------------------------
class EspDelegate : public SimpleDelegateInterface {
 public:
  explicit EspDelegate(const EspDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports Add and Sub ops.
    if (registration->builtin_code != kTfLiteBuiltinAdd &&
        registration->builtin_code != kTfLiteBuiltinSub &&
        registration->builtin_code != kTfLiteBuiltinReshape &&
        registration->builtin_code != kTfLiteBuiltinConv2d &&
        registration->builtin_code != kTfLiteBuiltinFullyConnected )
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
    static constexpr char kName[] = "EspDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<EspDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const EspDelegateOptions options_;
};

}  // namespace esp_test
}  // namespace tflite

EspDelegateOptions TfLiteEspDelegateOptionsDefault() {
  EspDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this esp test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteEspDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteEspDelegateCreate(const EspDelegateOptions* options) {
  std::unique_ptr<tflite::esp_test::EspDelegate> esp(
      new tflite::esp_test::EspDelegate(
          options ? *options : TfLiteEspDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(esp));
}

// Destroys a delegate created with `TfLiteEspDelegateCreate` call.
void TfLiteEspDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
