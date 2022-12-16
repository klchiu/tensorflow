#include "tensorflow/lite/delegates/esp/esp_delegate.h"

#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace esp_test {

// Esp delegate kernel.
class EspDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit EspDelegateKernel(const EspDelegateOptions& options)
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
  const EspDelegateOptions options_;
};

// EspDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class EspDelegate : public SimpleDelegateInterface {
 public:
  explicit EspDelegate(const EspDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    return options_.allowed_builtin_code == registration->builtin_code;
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
