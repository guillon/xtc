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
// CHECK:                 %8 = vector.fma %6, %4, %7 : vector<64xf32>
// CHECK-NEXT:            %19 = vector.fma %17, %15, %18 : vector<64xf32>
// CHECK-NEXT:            %30 = vector.fma %28, %26, %29 : vector<64xf32>
// CHECK-NEXT:            %41 = vector.fma %39, %37, %40 : vector<64xf32>
// CHECK-NEXT:            %52 = vector.fma %50, %48, %51 : vector<64xf32>
// CHECK-NEXT:            %63 = vector.fma %61, %59, %62 : vector<64xf32>
// CHECK-NEXT:            %74 = vector.fma %72, %70, %73 : vector<64xf32>
// CHECK-NEXT:            %85 = vector.fma %83, %81, %84 : vector<64xf32>
