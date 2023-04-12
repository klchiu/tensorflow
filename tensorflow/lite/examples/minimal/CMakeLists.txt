#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  # message("[humu]: examples/minimal/CMakeLists: debug 0")
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

if(TFLITE_ENABLE_ESP)
  message("[humu]: CMakeLists(minimal) TFLITE_ENABLE_ESP = ON")
  list(APPEND TFLITE_MINIMAL_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/esp_delegate_provider.cc
  )
else()
  message("[humu]: CMakeLists(minimal) TFLITE_ENABLE_ESP = OFF")
  set(TFLITE_LABEL_IMAGE_CC_OPTIONS "-DTFLITE_WITHOUT_ESP")
endif()  # TFLITE_ENABLE_ESP


if(ESP_RISCV)
  add_definitions(-DESP_RISCV)
  message("[humu]: CMakeLists(minimal) add definition")
endif()


add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 17)
add_executable(minimal
  minimal.cc
)

if(ESP_RISCV)
  # message("[humu]: CMakeLists(minimal) ESP_RISCV = ON")
  target_link_libraries(minimal
  tensorflow-lite
  # [humu]: include these static libraries for using ESP APIs
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/esp_libs/libmonitors.a
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/esp_libs/libesp.a
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/esp_libs/libutils.a
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/esp_libs/libtest.a
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/esp_libs/libcontig.a
  )
else()
  # message("[humu]: CMakeLists(minimal) ESP_RISCV = OFF")
  target_link_libraries(minimal
  tensorflow-lite
  )
endif() # ESP_RISCV MATCHES "ON"
