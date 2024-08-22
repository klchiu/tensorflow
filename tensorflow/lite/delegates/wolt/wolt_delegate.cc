/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/delegates/wolt/wolt_delegate.h"

#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include "tensorflow/lite/builtin_ops.h"




namespace tflite {
namespace wolt_test {

// Wolt delegate kernel.
class WoltDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit WoltDelegateKernel(const WoltDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const WoltDelegateOptions options_;
};

// WoltDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class WoltDelegate : public SimpleDelegateInterface {
 public:
  explicit WoltDelegate(const WoltDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    
    // Only supports Add and Sub ops.
    if (kTfLiteBuiltinAdd != registration->builtin_code &&
        kTfLiteBuiltinSub != registration->builtin_code){
          return false;
        }

    // This delegate only supports float32 types.
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteFloat32)  return false;
    }
    return true;
    
    // return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "WoltDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<WoltDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const WoltDelegateOptions options_;
};

}  // namespace wolt_test
}  // namespace tflite

WoltDelegateOptions TfLiteWoltDelegateOptionsDefault() {
  WoltDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this wolt test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteWoltDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteWoltDelegateCreate(const WoltDelegateOptions* options) {
  std::unique_ptr<tflite::wolt_test::WoltDelegate> wolt(
      new tflite::wolt_test::WoltDelegate(
          options ? *options : TfLiteWoltDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(wolt));
}

// Destroys a delegate created with `TfLiteWoltDelegateCreate` call.
void TfLiteWoltDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
