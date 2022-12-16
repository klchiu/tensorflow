#include <string>
#include <utility>

#include "tensorflow/lite/delegates/esp/esp_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class EspDelegateProvider : public DelegateProvider {
 public:
  EspDelegateProvider() {
    default_params_.AddParam("use_esp_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "EspDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(EspDelegateProvider);

std::vector<Flag> EspDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_esp_delegate", params,
                                              "use the esp delegate.")};
  return flags;
}

void EspDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_esp_delegate", "Use esp test delegate",
                 verbose);
}

TfLiteDelegatePtr EspDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_esp_delegate")) {
    auto default_options = TfLiteEspDelegateOptionsDefault();
    return TfLiteEspDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
EspDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_esp_delegate"));
}
}  // namespace tools
}  // namespace tflite
