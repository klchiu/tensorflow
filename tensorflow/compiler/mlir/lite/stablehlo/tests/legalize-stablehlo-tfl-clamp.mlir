// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}
}

// CHECK:       module {
// CHECK-NEXT:    func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
// CHECK-NEXT:      %0 = "tfl.custom"(%arg0, %arg0, %arg0) {custom_code = "mhlo.clamp", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:      return %0 : tensor<2xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
