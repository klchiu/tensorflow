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
#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/esp/esp_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"

namespace tflite {
namespace delegates {
class EspPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
    return TfLiteDelegatePtr(TfLiteEspDelegateCreate(&options_),
                             TfLiteEspDelegateDelete);
  }
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override { return 0; }
  static std::unique_ptr<DelegatePluginInterface> New(
      const TFLiteSettings& acceleration) {
    return std::make_unique<EspPlugin>(acceleration);
  }
  explicit EspPlugin(const TFLiteSettings& tflite_settings)
      : options_(TfLiteEspDelegateOptionsDefault()) {
    const auto* esp_settings = tflite_settings.esp_settings();
    if (esp_settings) {
      options_.num_threads = esp_settings->num_threads();
      options_.flags = esp_settings->flags();
    }
  }

 private:
  TfLiteEspDelegateOptions options_;
};

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(EspPlugin, EspPlugin::New);

}  // namespace delegates
}  // namespace tflite
