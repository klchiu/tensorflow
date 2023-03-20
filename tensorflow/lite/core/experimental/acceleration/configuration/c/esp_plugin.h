// NOLINTBEGIN(whitespace/line_length)
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/experimental/acceleration/configuration/c/esp_plugin.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// NOLINTEND(whitespace/line_length)
#ifndef TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_ESP_PLUGIN_H_
#define TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_ESP_PLUGIN_H_

// This header file is for the delegate plugin for ESP.
//
// For the C++ delegate plugin interface, the ESP delegate plugin is added
// to the DelegatePluginRegistry by the side effect of a constructor for a
// static object, so there's no public API needed for this plugin, other than
// the API of tflite::delegates::DelegatePluginRegistry, which is declared in
// delegate_registry.h.
//
// But to provide a C API to access the ESP delegate plugin, we do expose
// some functions, which are declared below.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/delegate_plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

// C API for the ESP delegate plugin.
// Returns a pointer to a statically allocated table of function pointers.
const TfLiteDelegatePlugin* TfLiteEspDelegatePluginCApi();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_ESP_PLUGIN_H_
