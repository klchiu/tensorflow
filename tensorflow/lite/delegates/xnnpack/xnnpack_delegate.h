#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus



namespace tflite {

using TfLiteDelegateUniquePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleDelegateKernelInterface {
 public:
  virtual ~SimpleDelegateKernelInterface() {}

  // Initializes a delegated subgraph.
  // The nodes in the subgraph are inside TfLiteDelegateParams->nodes_to_replace
  virtual TfLiteStatus Init(TfLiteContext* context,
                            const TfLiteDelegateParams* params) = 0;

  // Will be called by the framework. Should handle any needed preparation
  // for the subgraph e.g. allocating buffers, compiling model.
  // Returns status, and signalling any errors.
  virtual TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) = 0;

  // Actual subgraph inference should happen on this call.
  // Returns status, and signalling any errors.
  // NOTE: Tensor data pointers (tensor->data) can change every inference, so
  // the implementation of this method needs to take that into account.
  virtual TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) = 0;
};

// Pure Interface that clients should implement.
// The Interface represents a delegate's capabilities and provides a factory
// for SimpleDelegateKernelInterface.
//
// Clients should implement the following methods:
// - IsNodeSupportedByDelegate
// - Initialize
// - Name
// - CreateDelegateKernelInterface
// - DelegateOptions
class SimpleDelegateInterface {
 public:
  // Properties of a delegate.  These are used by TfLiteDelegateFactory to
  // help determine how to partition the graph, i.e. which nodes each delegate
  // will get applied to.
  struct Options {
    // Maximum number of delegated subgraph, values <=0 means unlimited.
    int max_delegated_partitions = 0;

    // The minimum number of nodes allowed in a delegated graph, values <=0
    // means unlimited.
    int min_nodes_per_partition = 0;
  };

  virtual ~SimpleDelegateInterface() {}

  // Returns true if 'node' is supported by the delegate. False otherwise.
  virtual bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context) const = 0;

  // Initialize the delegate before finding and replacing TfLite nodes with
  // delegate kernels, for example, retrieving some TFLite settings from
  // 'context'.
  virtual TfLiteStatus Initialize(TfLiteContext* context) = 0;

  // Returns a name that identifies the delegate.
  // This name is used for debugging/logging/profiling.
  virtual const char* Name() const = 0;

  // Returns instance of an object that implements the interface
  // SimpleDelegateKernelInterface.
  // An instance of SimpleDelegateKernelInterface represents one subgraph to
  // be delegated.
  // Caller takes ownership of the returned object.
  virtual std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() = 0;

  // Returns SimpleDelegateInterface::Options which has delegate properties
  // relevant for graph partitioning.
  virtual SimpleDelegateInterface::Options DelegateOptions() const = 0;
};

// Factory class that provides static methods to deal with SimpleDelegate
// creation and deletion.
class TfLiteDelegateFactory {
 public:
  // Creates TfLiteDelegate from the provided SimpleDelegateInterface.
  // The returned TfLiteDelegate should be deleted using DeleteSimpleDelegate.
  // A simple usage of the flags bit mask:
  // CreateSimpleDelegate(..., kTfLiteDelegateFlagsAllowDynamicTensors |
  // kTfLiteDelegateFlagsRequirePropagatedShapes)
  static TfLiteDelegate* CreateSimpleDelegate(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate,
      int64_t flags = kTfLiteDelegateFlagsNone);

  // Deletes 'delegate' the passed pointer must be the one returned
  // from CreateSimpleDelegate.
  // This function will destruct the SimpleDelegate object too.
  static void DeleteSimpleDelegate(TfLiteDelegate* delegate);

  // A convenient function wrapping the above two functions and returning a
  // std::unique_ptr type for auto memory management.
  inline static TfLiteDelegateUniquePtr Create(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate) {
    return TfLiteDelegateUniquePtr(
        CreateSimpleDelegate(std::move(simple_delegate)), DeleteSimpleDelegate);
  }
};

}  // namespace tflite







// Enable XNNPACK acceleration for signed quantized 8-bit inference.
// This includes operators with channel-wise quantized weights.
#define TFLITE_XNNPACK_DELEGATE_FLAG_QS8 0x00000001
// Enable XNNPACK acceleration for unsigned quantized 8-bit inference.
#define TFLITE_XNNPACK_DELEGATE_FLAG_QU8 0x00000002
// Force FP16 inference for FP32 operators.
#define TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 0x00000004



typedef struct {
  // Number of threads to use in the thread pool.
  // 0 or negative value means no thread pool used.
  int32_t num_threads;
  // Bitfield with any combination of the following binary options:
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QS8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QU8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
  uint32_t flags;
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} TfLiteXNNPackDelegateOptions;

// Returns a structure with the default delegate options.
TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate*  TfLiteXNNPackDelegateCreate(const TfLiteXNNPackDelegateOptions* options);

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteXNNPackDelegateCreateUnique(const TfLiteXNNPackDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteXNNPackDelegateCreate(options), TfLiteXNNPackDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
