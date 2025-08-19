// RUN: mlir-loop --no-alias --print-transformed-ir %s 2>&1 | grep fma | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I",
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64" = {"unroll","vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:                 %7 = vector.fma %5, %3, %6 : vector<64xf32>
// CHECK-NEXT:            %18 = vector.fma %16, %14, %17 : vector<64xf32>
// CHECK-NEXT:            %29 = vector.fma %27, %25, %28 : vector<64xf32>
// CHECK-NEXT:            %40 = vector.fma %38, %36, %39 : vector<64xf32>
// CHECK-NEXT:            %51 = vector.fma %49, %47, %50 : vector<64xf32>
// CHECK-NEXT:            %62 = vector.fma %60, %58, %61 : vector<64xf32>
// CHECK-NEXT:            %73 = vector.fma %71, %69, %72 : vector<64xf32>
// CHECK-NEXT:            %84 = vector.fma %82, %80, %83 : vector<64xf32>
