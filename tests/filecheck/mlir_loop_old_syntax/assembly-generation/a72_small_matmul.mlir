// RUN: mlir-loop --old-syntax %s --no-alias --arch aarch64 --cpu cortex-a72 --print-assembly | filecheck %s

func.func @myfun(
  %A: memref<4x4xf32>,
  %B: memref<4x4xf32>,
  %C: memref<4x4xf32>
) {
  linalg.matmul
    ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
    outs(%C : memref<4x4xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	ldp	s2, s0, [x1, #56]
// CHECK-NEXT:  	ldp	s4, s1, [x1, #40]
// CHECK-NEXT:  	ldp	s6, s3, [x1, #24]
// CHECK-NEXT:  	ldp	s7, s5, [x1, #8]
// CHECK-NEXT:  	ldp	s17, s16, [x1, #32]
// CHECK-NEXT:  	ldp	s21, s18, [x1, #16]
// CHECK-NEXT:  	ldp	s23, s20, [x1]
// CHECK-NEXT:  	ldp	s22, s19, [x1, #48]
// CHECK-NEXT:  	ldp	s24, s25, [x0]
// CHECK-NEXT:  	ldp	s26, s27, [x2]
// CHECK-NEXT:  	fmadd	s26, s24, s23, s26
// CHECK-NEXT:  	ldp	s28, s29, [x2, #8]
// CHECK-NEXT:  	fmadd	s27, s24, s20, s27
// CHECK-NEXT:  	fmadd	s28, s24, s7, s28
// CHECK-NEXT:  	fmadd	s24, s24, s5, s29
// CHECK-NEXT:  	fmadd	s26, s25, s21, s26
// CHECK-NEXT:  	fmadd	s27, s25, s18, s27
// CHECK-NEXT:  	fmadd	s28, s25, s6, s28
// CHECK-NEXT:  	fmadd	s24, s25, s3, s24
// CHECK-NEXT:  	ldp	s25, s29, [x0, #8]
// CHECK-NEXT:  	fmadd	s26, s25, s17, s26
// CHECK-NEXT:  	fmadd	s27, s25, s16, s27
// CHECK-NEXT:  	fmadd	s28, s25, s4, s28
// CHECK-NEXT:  	fmadd	s24, s25, s1, s24
// CHECK-NEXT:  	fmadd	s25, s29, s22, s26
// CHECK-NEXT:  	fmadd	s26, s29, s19, s27
// CHECK-NEXT:  	fmadd	s27, s29, s2, s28
// CHECK-NEXT:  	fmadd	s24, s29, s0, s24
// CHECK-NEXT:  	ldp	s28, s29, [x0, #16]
// CHECK-NEXT:  	stp	s25, s26, [x2]
// CHECK-NEXT:  	ldp	s25, s26, [x2, #16]
// CHECK-NEXT:  	fmadd	s25, s28, s23, s25
// CHECK-NEXT:  	fmadd	s25, s29, s21, s25
// CHECK-NEXT:  	stp	s27, s24, [x2, #8]
// CHECK-NEXT:  	ldp	s24, s27, [x0, #24]
// CHECK-NEXT:  	fmadd	s25, s24, s17, s25
// CHECK-NEXT:  	fmadd	s25, s27, s22, s25
// CHECK-NEXT:  	fmadd	s26, s28, s20, s26
// CHECK-NEXT:  	fmadd	s26, s29, s18, s26
// CHECK-NEXT:  	fmadd	s26, s24, s16, s26
// CHECK-NEXT:  	fmadd	s26, s27, s19, s26
// CHECK-NEXT:  	stp	s25, s26, [x2, #16]
// CHECK-NEXT:  	ldp	s25, s26, [x2, #24]
// CHECK-NEXT:  	fmadd	s25, s28, s7, s25
// CHECK-NEXT:  	fmadd	s25, s29, s6, s25
// CHECK-NEXT:  	fmadd	s25, s24, s4, s25
// CHECK-NEXT:  	fmadd	s25, s27, s2, s25
// CHECK-NEXT:  	fmadd	s26, s28, s5, s26
// CHECK-NEXT:  	fmadd	s26, s29, s3, s26
// CHECK-NEXT:  	fmadd	s24, s24, s1, s26
// CHECK-NEXT:  	fmadd	s24, s27, s0, s24
// CHECK-NEXT:  	stp	s25, s24, [x2, #24]
// CHECK-NEXT:  	ldp	s24, s25, [x0, #32]
// CHECK-NEXT:  	ldp	s26, s27, [x2, #32]
// CHECK-NEXT:  	fmadd	s26, s24, s23, s26
// CHECK-NEXT:  	fmadd	s26, s25, s21, s26
// CHECK-NEXT:  	ldp	s28, s29, [x0, #40]
// CHECK-NEXT:  	fmadd	s26, s28, s17, s26
// CHECK-NEXT:  	fmadd	s26, s29, s22, s26
// CHECK-NEXT:  	fmadd	s27, s24, s20, s27
// CHECK-NEXT:  	fmadd	s27, s25, s18, s27
// CHECK-NEXT:  	fmadd	s27, s28, s16, s27
// CHECK-NEXT:  	fmadd	s27, s29, s19, s27
// CHECK-NEXT:  	stp	s26, s27, [x2, #32]
// CHECK-NEXT:  	ldp	s26, s27, [x2, #40]
// CHECK-NEXT:  	fmadd	s26, s24, s7, s26
// CHECK-NEXT:  	fmadd	s24, s24, s5, s27
// CHECK-NEXT:  	fmadd	s26, s25, s6, s26
// CHECK-NEXT:  	fmadd	s26, s28, s4, s26
// CHECK-NEXT:  	fmadd	s26, s29, s2, s26
// CHECK-NEXT:  	fmadd	s24, s25, s3, s24
// CHECK-NEXT:  	fmadd	s24, s28, s1, s24
// CHECK-NEXT:  	fmadd	s24, s29, s0, s24
// CHECK-NEXT:  	stp	s26, s24, [x2, #40]
// CHECK-NEXT:  	ldp	s24, s25, [x0, #48]
// CHECK-NEXT:  	ldp	s26, s27, [x2, #48]
// CHECK-NEXT:  	fmadd	s23, s24, s23, s26
// CHECK-NEXT:  	fmadd	s21, s25, s21, s23
// CHECK-NEXT:  	ldp	s23, s26, [x0, #56]
// CHECK-NEXT:  	fmadd	s17, s23, s17, s21
// CHECK-NEXT:  	fmadd	s17, s26, s22, s17
// CHECK-NEXT:  	fmadd	s20, s24, s20, s27
// CHECK-NEXT:  	fmadd	s18, s25, s18, s20
// CHECK-NEXT:  	fmadd	s16, s23, s16, s18
// CHECK-NEXT:  	fmadd	s16, s26, s19, s16
// CHECK-NEXT:  	stp	s17, s16, [x2, #48]
// CHECK-NEXT:  	ldp	s16, s17, [x2, #56]
// CHECK-NEXT:  	fmadd	s7, s24, s7, s16
// CHECK-NEXT:  	fmadd	s6, s25, s6, s7
// CHECK-NEXT:  	fmadd	s4, s23, s4, s6
// CHECK-NEXT:  	fmadd	s2, s26, s2, s4
// CHECK-NEXT:  	fmadd	s4, s24, s5, s17
// CHECK-NEXT:  	fmadd	s3, s25, s3, s4
// CHECK-NEXT:  	fmadd	s1, s23, s1, s3
// CHECK-NEXT:  	fmadd	s0, s26, s0, s1
// CHECK-NEXT:  	stp	s2, s0, [x2, #56]
// CHECK-NEXT:  	ret
