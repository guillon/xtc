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

// CHECK:                 %0 = vector.transfer_read %subview_8[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %1 = vector.transfer_read %subview_9[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %2 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %3 = vector.extract %1[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %4 = vector.extract %0[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %5 = vector.broadcast %4 : f32 to vector<64xf32>
// CHECK-NEXT:            %6 = vector.extract %2[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %7 = vector.fma %5, %3, %6 : vector<64xf32>
// CHECK-NEXT:            %8 = vector.insert %7, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %8, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %11 = vector.transfer_read %subview_11[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %12 = vector.transfer_read %subview_12[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %13 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %14 = vector.extract %12[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %15 = vector.extract %11[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %16 = vector.broadcast %15 : f32 to vector<64xf32>
// CHECK-NEXT:            %17 = vector.extract %13[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %18 = vector.fma %16, %14, %17 : vector<64xf32>
// CHECK-NEXT:            %19 = vector.insert %18, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %19, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %22 = vector.transfer_read %subview_13[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %23 = vector.transfer_read %subview_14[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %24 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %25 = vector.extract %23[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %26 = vector.extract %22[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %27 = vector.broadcast %26 : f32 to vector<64xf32>
// CHECK-NEXT:            %28 = vector.extract %24[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %29 = vector.fma %27, %25, %28 : vector<64xf32>
// CHECK-NEXT:            %30 = vector.insert %29, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %30, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %33 = vector.transfer_read %subview_15[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %34 = vector.transfer_read %subview_16[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %35 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %36 = vector.extract %34[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %37 = vector.extract %33[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %38 = vector.broadcast %37 : f32 to vector<64xf32>
// CHECK-NEXT:            %39 = vector.extract %35[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %40 = vector.fma %38, %36, %39 : vector<64xf32>
// CHECK-NEXT:            %41 = vector.insert %40, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %41, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %44 = vector.transfer_read %subview_17[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %45 = vector.transfer_read %subview_18[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %46 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %47 = vector.extract %45[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %48 = vector.extract %44[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %49 = vector.broadcast %48 : f32 to vector<64xf32>
// CHECK-NEXT:            %50 = vector.extract %46[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %51 = vector.fma %49, %47, %50 : vector<64xf32>
// CHECK-NEXT:            %52 = vector.insert %51, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %52, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %55 = vector.transfer_read %subview_19[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %56 = vector.transfer_read %subview_20[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %57 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %58 = vector.extract %56[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %59 = vector.extract %55[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %60 = vector.broadcast %59 : f32 to vector<64xf32>
// CHECK-NEXT:            %61 = vector.extract %57[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %62 = vector.fma %60, %58, %61 : vector<64xf32>
// CHECK-NEXT:            %63 = vector.insert %62, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %63, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %66 = vector.transfer_read %subview_21[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %67 = vector.transfer_read %subview_22[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %68 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %69 = vector.extract %67[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %70 = vector.extract %66[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %71 = vector.broadcast %70 : f32 to vector<64xf32>
// CHECK-NEXT:            %72 = vector.extract %68[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %73 = vector.fma %71, %69, %72 : vector<64xf32>
// CHECK-NEXT:            %74 = vector.insert %73, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %74, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %77 = vector.transfer_read %subview_23[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %78 = vector.transfer_read %subview_24[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %79 = vector.transfer_read %subview_4[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %80 = vector.extract %78[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %81 = vector.extract %77[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %82 = vector.broadcast %81 : f32 to vector<64xf32>
// CHECK-NEXT:            %83 = vector.extract %79[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %84 = vector.fma %82, %80, %83 : vector<64xf32>
// CHECK-NEXT:            %85 = vector.insert %84, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %85, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
