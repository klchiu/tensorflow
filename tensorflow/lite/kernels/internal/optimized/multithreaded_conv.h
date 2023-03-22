/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_MULTITHREADED_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_MULTITHREADED_CONV_H_

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"



#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <stdio.h>


// [humu]: include this header file for using ESP APIs
#define __FIXED
#define BITWIDTH 32
#include "tensorflow/esp_libs/cfg_conv2d.h"
#include "tensorflow/esp_libs/conv2d_helper.h"
#include "tensorflow/esp_libs/esp_api_include.h"

static int humu_counter = 0;

namespace tflite {
namespace multithreaded_ops {

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenMatrix;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenMatrix;

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenTensor;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, EigenMatrix out, ConstEigenMatrix in0,
      ConstEigenMatrix in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

template <class T>
class EigenTensorConvFunctor {
 private:
  Eigen::PaddingType RuntimePadding2EigenPadding(PaddingType padding) {
    switch (padding) {
      case PaddingType::kValid:
        return Eigen::PADDING_VALID;
      case PaddingType::kSame:
        return Eigen::PADDING_SAME;
      case PaddingType::kNone:
        assert(false);  // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME;  // Prevent compiler warning about missing
                                 // return
  }

 public:
  void operator()(const Eigen::ThreadPoolDevice& device, const T* input_data,
                  int input_batches, int input_height, int input_width,
                  int input_depth, const T* filter_data, int filter_height,
                  int filter_width, int filter_count, int stride_rows,
                  int stride_cols, int pad_width, int pad_height,
                  PaddingType padding, T* output_data, int output_height,
                  int output_width) {
    // [humu]: invoke the accelerator here
    printf("-- humu_counter = %d\n", humu_counter);
    humu_counter += 1;
   int buf_i = 0;
      int b, c, x, y, cin, cout, index;




    if (humu_counter > 0) {

      // printf("const T* input_data   = %d\n", *input_data);
      // printf("int input_batches     = %d\n", input_batches);
      // printf("int input_height      = %d\n", input_height);
      // printf("int input_width       = %d\n", input_width);
      // printf("int input_depth       = %d\n", input_depth);
      // printf("const T* filter_data  = %d\n", *filter_data);
      // printf("int filter_height     = %d\n", filter_height);
      // printf("int filter_width      = %d\n", filter_width);
      // printf("int filter_count      = %d\n", filter_count);
      // printf("int stride_rows       = %d\n", stride_rows);
      // printf("int stride_cols       = %d\n", stride_cols);
      // printf("int pad_width         = %d\n", pad_width);
      // printf("int pad_height        = %d\n", pad_height);
      // printf("PaddingType padding   = %d\n", padding);
      // printf("T* output_data        = %d\n", *output_data);
      // printf("int output_height     = %d\n", output_height);
      // printf("int output_width      = %d\n", output_width);

int input_size2 = input_batches * input_height * input_width * input_depth;
int output_size2 = output_height * output_width;
int filter_size2 = filter_height * filter_width * filter_count;

  //  printf("-- input_size2 = %d\n", input_size2);
  //  printf("-- output_size2 = %d\n", output_size2);
  //  printf("-- filter_size2 = %d\n", filter_size2);


// for (x = 0 ; x < input_size2; x++){
//   printf("-- input_data[%d] = %f\n", x, input_data[x]);
// }
// for (x = 0 ; x < filter_size2; x++){
//   printf("-- filter_data[%d] = %f\n", x, filter_data[x]);
// }


      // // void *buf = NULL;
      // // esp_dummy(buf);

      token_t* acc_buf;
      acc_buf = (token_t*)esp_alloc(MAX_SIZE);
      cfg_000[0].hw_buf = acc_buf;

      // set parameters
      conv2d_cfg_000[0].n_channels = input_depth;
      conv2d_cfg_000[0].feature_map_height = input_height;
      conv2d_cfg_000[0].feature_map_width = input_width;
      conv2d_cfg_000[0].n_filters = filter_count;
      conv2d_cfg_000[0].filter_dim =
          filter_height;  // should be the same as filter_width
      if (padding == tflite::PaddingType::kSame) {
        conv2d_cfg_000[0].is_padded = 1;
      } else {
        conv2d_cfg_000[0].is_padded = 0;
      }
      conv2d_cfg_000[0].stride =
          stride_rows;                  // should be the same as stride_cols
      conv2d_cfg_000[0].do_relu = 0;    // this function doesn't do relu (?)
      conv2d_cfg_000[0].pool_type = 0;  // this function doesn't do pooling (?)
      conv2d_cfg_000[0].batch_size = input_batches;

      // setup buffer

      int32_t output_h;
      int32_t output_pool_h;
      int32_t pad_dim = 0;

      if (padding == tflite::PaddingType::kSame) {
        pad_dim = filter_height / 2;
      }

      int32_t pool_type = 0;  // this function doesn't do pooling (?)

      output_h = (input_height + 2 * pad_dim - ((filter_height - 1) + 1)) /
                     stride_rows +
                 1;
      output_pool_h = pool_type ? output_h / 2 : output_h;

      // Input data and golden output (aligned to DMA_WIDTH makes your life
      // easier)
      int32_t in_len = round_up(
          input_batches *
              round_up(
                  input_depth * round_up(input_height * input_width, DMA_RATIO),
                  DMA_RATIO),
          DMA_RATIO);
      int32_t weights_len =
          round_up(filter_count * input_depth * filter_height * filter_height,
                   DMA_RATIO);
      int32_t bias_len = round_up(filter_count, DMA_RATIO);
      int32_t out_len = round_up(
          input_batches *
              round_up(filter_count *
                           round_up(output_pool_h * output_pool_h, DMA_RATIO),
                       DMA_RATIO),
          DMA_RATIO);


  //  printf("-- output_h = %d\n", output_h);
  //  printf("-- output_pool_h = %d\n", output_pool_h);
  //  printf("-- pad_dim = %d\n", pad_dim);
  //  printf("-- in_len = %d\n", in_len);
  //  printf("-- weights_len = %d\n", weights_len);
  //  printf("-- bias_len = %d\n", bias_len);
  //  printf("-- out_len = %d\n", out_len);



      // *in_size        = *in_len * sizeof(token_t);
      // *weights_size   = *weights_len * sizeof(token_t);
      // *bias_size      = *bias_len * sizeof(token_t);
      // *out_size       = *out_len * sizeof(token_t);
      // *weights_offset = *in_len;
      // *bias_offset    = *in_len + *weights_len;
      // *out_offset     = *in_len + *weights_len + *bias_len;
      // *size           = *in_size + *weights_size + *bias_size + *out_size;


      // load input
      for (b = 0; b < input_batches; b++) {
        for (c = 0; c < input_depth; c++) {    
          for (x = 0; x < input_width; x++) {
            for (y = 0; y < input_height; y++) {    
              index = b * input_height * input_width * input_depth 
                        + y * input_width * input_depth 
                        + x * input_depth + c;
              acc_buf[buf_i] = float2fx(input_data[index], FX_IL);
              buf_i++;
      }}}}

  //  printf("-- buf_i = %d\n", buf_i);


    // load weight
    for (cout = 0; cout < filter_count; cout++) {
      for (cin = 0; cin < input_depth; cin++) {
        for (x = 0; x < filter_width; x++) {
          for (y = 0; y < filter_height; y++) {
              index = cin * filter_height * filter_width * filter_count
                    + y * filter_width * filter_count
                    + x * filter_count + cout;
              acc_buf[buf_i] = float2fx(filter_data[index], FX_IL);
              buf_i++;
      }}}}
        //  printf("-- buf_i = %d\n", buf_i);


      // bias offset
      float bb = 0.0;
      for (cout = 0; cout < filter_count; cout++) {
        acc_buf[buf_i] = float2fx(bb, FX_IL);
        buf_i++;
      }
        //  printf("-- buf_i = %d\n", buf_i);


      esp_run_no_print(cfg_000, NACC);


      // store output
      for (b = 0; b < input_batches; b++) {
        for (y = 0; y < output_height; y++) {    
          for (x = 0; x < output_width; x++) {
            for (c = 0; c < filter_count; c++) {
              index = b * filter_count * output_width * output_height
                        + c * output_width * output_height
                        + x * output_height + y;
                      output_data[index] = fx2float(acc_buf[buf_i], FX_IL);
              buf_i++;
      }}}}
        //  printf("-- buf_i = %d\n", buf_i);


 
      esp_free(acc_buf);

          if(humu_counter == 1){
          for (x = 0 ; x < 100; x++){
            // printf("acc -- output_data[%d] = %f\n", x, output_data[x]);
          }
        }

    }

    if (humu_counter < 0) {
      const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1 &&
                                  stride_rows == 1 && stride_cols == 1);

      if (is_1x1_kernel) {
        // For 1x1 kernel, the 2D convolution is reduced to matrix
        // multiplication.
        printf("[humu]: multithreaded_conv.h: EigenTensorConvFunctor 0\n");

   const int conv_width = output_height * output_width;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, input_batches * conv_width, filter_count);
      ConstEigenMatrix input(input_data, input_batches * conv_width,
                             input_depth);
      ConstEigenMatrix filter(filter_data, input_depth, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
      } else if (filter_height == input_height && filter_width == input_width &&
                 pad_width == 0 && pad_height == 0) {
        // If the input data and filter have the same height/width,
        // the 2D convolution is reduced to matrix multiplication.
        printf("[humu]: multithreaded_conv.h: EigenTensorConvFunctor 1\n");

        const int k =  // Length of reduction dimension.
            filter_width * filter_height * input_depth;
        Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
        dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
        EigenMatrix output(output_data, input_batches, filter_count);
        ConstEigenMatrix input(input_data, input_batches, k);
        ConstEigenMatrix filter(filter_data, k, filter_count);
        MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                        filter, dim_pair);
      } else {
        printf("[humu]: multithreaded_conv.h: EigenTensorConvFunctor 2\n");

        EigenTensor output(output_data, input_batches, output_height,
                           output_width, filter_count);
        ConstEigenTensor input(input_data, input_batches, input_height,
                               input_width, input_depth);
        ConstEigenTensor filter(filter_data, filter_height, filter_width,
                                input_depth, filter_count);
        output.device(device) =
            Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                      RuntimePadding2EigenPadding(padding));




        if(humu_counter == 1){
          FILE *log_input;
          log_input = fopen("log_input.txt", "w");
          FILE *log_filter;
          log_filter = fopen("log_filter.txt", "w");
          FILE *log_output;
          log_output = fopen("log_output.txt", "w");

          x = 0;
               for (b = 0; b < input_batches; b++) {
        for (c = 0; c < input_depth; c++) {    
          for (x = 0; x < input_width; x++) {
            for (y = 0; y < input_height; y++) {   
            // fprintf(log_input, "input_data[%d] = %f\n", x, input_data[x]);
            fprintf(log_input, "%f\n", x, input_data[x]);
            x++;
            }}}}
          
x = 0;
  for (cout = 0; cout < filter_count; cout++) {
      for (cin = 0; cin < input_depth; cin++) {
        for (x = 0; x < filter_width; x++) {
          for (y = 0; y < filter_height; y++) {        
                // fprintf(log_filter, "filter_data[%d] = %f\n", x, filter_data[x]);
            fprintf(log_filter, "%f\n", x, filter_data[x]);
            x++;
          }}}}


x = 0;
  for (b = 0; b < input_batches; b++) {
        for (y = 0; y < output_height; y++) {    
          for (x = 0; x < output_width; x++) {
            for (c = 0; c < filter_count; c++) {            // fprintf(log_output, "output_data[%d] = %f\n", x, output_data[x]);
            fprintf(log_output, "%f\n", x, output_data[x]);
            x++;
          }}}}

          fclose(log_input);
          fclose(log_filter);
          fclose(log_output);


        }


      }
    }
  }
};

inline void Conv(const Eigen::ThreadPoolDevice& device,
                 const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
  // Nest profiling under "Conv", to aggregate with other kernels.
  ruy::profiler::ScopeLabel label("Conv");
  ruy::profiler::ScopeLabel inner_label("Multithreaded EigenTensor");

  // printf("[humu]: multithreaded_conv.h: Conv 0\n");

  // im2col data should not be generated for the multi-thread supporting case.
  TFLITE_DCHECK(!im2col_data);
  (void)im2col_shape;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const PaddingType padding = params.padding_type;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(device, input_data, batches, input_height, input_width,
               input_depth, filter_data, filter_height, filter_width,
               output_depth, stride_height, stride_width, pad_height, pad_width,
               padding, output_data, output_height, output_width);

  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max, bias_shape, bias_data,
      output_shape, output_data);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_MULTITHREADED_CONV_H_
