// This file implements the C API Delegate Plugin for the ESP Delegate.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/Esp_plugin.h"

#include <memory>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/esp/esp_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

extern "C" {

static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  auto options(TfLiteEspDelegateOptionsDefault());
  const auto* esp_settings = tflite_settings->esp_settings();
  if (esp_settings) {
    options.num_threads = esp_settings->num_threads();
  }
  return TfLiteEspDelegateCreate(&options);
}

static void DestroyDelegate(TfLiteDelegate* delegate) {
  TfLiteEspDelegateDelete(delegate);
}

static int DelegateErrno(TfLiteDelegate* from_delegate) { return 0; }

static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};

const TfLiteDelegatePlugin* TfLiteEspDelegatePluginCApi() {
  return &kPluginCApi;
}

}  // extern "C"
