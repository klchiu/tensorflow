// Some very simple unit tests of the C API Delegate Plugin for the
// ESP Delegate.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/esp_plugin.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/esp/esp_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {

class EspTest : public testing::Test {
 public:
  static constexpr int kNumThreadsForTest = 7;
  void SetUp() override {
    // Construct a FlatBuffer that contains
    // TFLiteSettings { EspSettings { num_threads: kNumThreadsForTest } }.
    EspSettingsBuilder esp_settings_builder(flatbuffer_builder_);
    esp_settings_builder.add_num_threads(kNumThreadsForTest);
    flatbuffers::Offset<EspSettings> esp_settings =
        esp_settings_builder.Finish();
    TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
    tflite_settings_builder.add_esp_settings(esp_settings);
    flatbuffers::Offset<TFLiteSettings> tflite_settings =
        tflite_settings_builder.Finish();
    flatbuffer_builder_.Finish(tflite_settings);
    settings_ = flatbuffers::GetRoot<TFLiteSettings>(
        flatbuffer_builder_.GetBufferPointer());
  }
  ~EspTest() override {}

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *settings_;
};

constexpr int EspTest::kNumThreadsForTest;

TEST_F(EspTest, CanCreateAndDestroyDelegate) {
  TfLiteDelegate *delegate =
      TfLiteEspDelegatePluginCApi()->create(settings_);
  EXPECT_NE(delegate, nullptr);
  TfLiteEspDelegatePluginCApi()->destroy(delegate);
}

TEST_F(EspTest, CanGetDelegateErrno) {
  TfLiteDelegate *delegate =
      TfLiteEspDelegatePluginCApi()->create(settings_);
  int error_number =
      TfLiteEspDelegatePluginCApi()->get_delegate_errno(delegate);
  EXPECT_EQ(error_number, 0);
  TfLiteEspDelegatePluginCApi()->destroy(delegate);
}

TEST_F(EspTest, SetsCorrectThreadCount) {
  TfLiteDelegate *delegate =
      TfLiteEspDelegatePluginCApi()->create(settings_);
  pthreadpool_t threadpool =
      static_cast<pthreadpool_t>(TfLiteEspDelegateGetThreadPool(delegate));
  int thread_count = pthreadpool_get_threads_count(threadpool);
  EXPECT_EQ(thread_count, kNumThreadsForTest);
  TfLiteEspDelegatePluginCApi()->destroy(delegate);
}
}  // namespace tflite
