// RUN: mlir-loop --no-alias --print-transformed-ir %s 2>&1 | grep "vector\." | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I" = {"parallelize"},
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}

// CHECK:                 %1 = vector.transfer_read %subview_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %2 = vector.transfer_read %subview_10[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %3 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %4 = vector.extract %2[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %5 = vector.extract %1[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %6 = vector.broadcast %5 : f32 to vector<64xf32>
// CHECK-NEXT:            %7 = vector.extract %3[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %8 = vector.fma %6, %4, %7 : vector<64xf32>
// CHECK-NEXT:            %9 = vector.insert %8, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %9, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %12 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %13 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %14 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %15 = vector.extract %13[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %16 = vector.extract %12[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %17 = vector.broadcast %16 : f32 to vector<64xf32>
// CHECK-NEXT:            %18 = vector.extract %14[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %19 = vector.fma %17, %15, %18 : vector<64xf32>
// CHECK-NEXT:            %20 = vector.insert %19, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %20, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %23 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %24 = vector.transfer_read %subview_15[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %25 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %26 = vector.extract %24[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %27 = vector.extract %23[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %28 = vector.broadcast %27 : f32 to vector<64xf32>
// CHECK-NEXT:            %29 = vector.extract %25[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %30 = vector.fma %28, %26, %29 : vector<64xf32>
// CHECK-NEXT:            %31 = vector.insert %30, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %31, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %34 = vector.transfer_read %subview_16[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %35 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %36 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %37 = vector.extract %35[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %38 = vector.extract %34[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %39 = vector.broadcast %38 : f32 to vector<64xf32>
// CHECK-NEXT:            %40 = vector.extract %36[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %41 = vector.fma %39, %37, %40 : vector<64xf32>
// CHECK-NEXT:            %42 = vector.insert %41, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %42, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %45 = vector.transfer_read %subview_18[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %46 = vector.transfer_read %subview_19[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %47 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %48 = vector.extract %46[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %49 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %50 = vector.broadcast %49 : f32 to vector<64xf32>
// CHECK-NEXT:            %51 = vector.extract %47[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %52 = vector.fma %50, %48, %51 : vector<64xf32>
// CHECK-NEXT:            %53 = vector.insert %52, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %53, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %56 = vector.transfer_read %subview_20[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %57 = vector.transfer_read %subview_21[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %58 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %59 = vector.extract %57[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %60 = vector.extract %56[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %61 = vector.broadcast %60 : f32 to vector<64xf32>
// CHECK-NEXT:            %62 = vector.extract %58[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %63 = vector.fma %61, %59, %62 : vector<64xf32>
// CHECK-NEXT:            %64 = vector.insert %63, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %64, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %67 = vector.transfer_read %subview_22[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %68 = vector.transfer_read %subview_23[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %69 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %70 = vector.extract %68[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %71 = vector.extract %67[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %72 = vector.broadcast %71 : f32 to vector<64xf32>
// CHECK-NEXT:            %73 = vector.extract %69[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %74 = vector.fma %72, %70, %73 : vector<64xf32>
// CHECK-NEXT:            %75 = vector.insert %74, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %75, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %78 = vector.transfer_read %subview_24[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %79 = vector.transfer_read %subview_25[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %80 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %81 = vector.extract %79[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %82 = vector.extract %78[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %83 = vector.broadcast %82 : f32 to vector<64xf32>
// CHECK-NEXT:            %84 = vector.extract %80[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %85 = vector.fma %83, %81, %84 : vector<64xf32>
// CHECK-NEXT:            %86 = vector.insert %85, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %86, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
