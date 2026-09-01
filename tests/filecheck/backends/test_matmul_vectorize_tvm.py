# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 4, 32, 256, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.strip_mine("j", {"j0": 24})
sch.interchange(["i", "j", "k", "j0"])
sch.vectorize(["j0"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_vectorize_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)

module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x256xfloat32
# CHECK-NEXT:    - %1 : 256x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x256xfloat32, 256x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((4, 256), "float32"), _1: T.Buffer((256, 32), "float32"), C: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for i, j in T.grid(4, 32):
# CHECK-NEXT:              C_1 = T.Buffer((128,), data=C.data)
# CHECK-NEXT:              C_1[i * 32 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(256):
# CHECK-NEXT:                  cse_var_1: T.int32 = i * 32 + j
# CHECK-NEXT:                  _0_1 = T.Buffer((1024,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((8192,), data=_1.data)
# CHECK-NEXT:                  C_1[cse_var_1] = C_1[cse_var_1] + _0_1[i * 256 + k] * _1_1[k * 32 + j]
# CHECK-NEXT:  O = obj['C']
# CHECK-NEXT:  i, j, = O.op.axis
# CHECK-NEXT:  k, = O.op.reduce_axis
# CHECK-NEXT:  j, j0 = sch[O].split(j, factor=24)
# CHECK-NEXT:  j0, __v_j0 = sch[O].split(j0, factor=8)
# CHECK-NEXT:  sch[O].reorder(i, j, k, j0, __v_j0)
# CHECK-NEXT:  sch[O].unroll(j0)
# CHECK-NEXT:  sch[O].vectorize(__v_j0)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((4, 256), "float32"), _1: T.Buffer((256, 32), "float32"), C: T.Buffer((4, 32), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for i, j_outer in T.grid(4, 2):
# CHECK-NEXT:              C_1 = T.Buffer((128,), data=C.data)
# CHECK-NEXT:              C_1[i * 32 + j_outer * 24:i * 32 + j_outer * 24 + 8] = T.Broadcast(T.float32(0.0), 8)
# CHECK-NEXT:              if T.likely(j_outer < 1):
# CHECK-NEXT:                  C_1[i * 32 + j_outer * 24 + 8:i * 32 + j_outer * 24 + 8 + 8] = T.Broadcast(T.float32(0.0), 8)
# CHECK-NEXT:              if T.likely(j_outer < 1):
# CHECK-NEXT:                  C_1[i * 32 + j_outer * 24 + 16:i * 32 + j_outer * 24 + 16 + 8] = T.Broadcast(T.float32(0.0), 8)
# CHECK-NEXT:              for k in range(256):
# CHECK-NEXT:                  cse_var_2: T.int32 = j_outer * 24
# CHECK-NEXT:                  cse_var_1: T.int32 = i * 32 + cse_var_2
# CHECK-NEXT:                  _0_1 = T.Buffer((1024,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((8192,), data=_1.data)
# CHECK-NEXT:                  C_1[cse_var_1:cse_var_1 + 8] = C_1[cse_var_1:cse_var_1 + 8] + T.Broadcast(_0_1[i * 256 + k], 8) * _1_1[k * 32 + cse_var_2:k * 32 + cse_var_2 + 8]
# CHECK-NEXT:                  if T.likely(j_outer < 1):
# CHECK-NEXT:                      cse_var_3: T.int32 = cse_var_1 + 8
# CHECK-NEXT:                      C_1[cse_var_3:cse_var_3 + 8] = C_1[cse_var_3:cse_var_3 + 8] + T.Broadcast(_0_1[i * 256 + k], 8) * _1_1[k * 32 + cse_var_2 + 8:k * 32 + cse_var_2 + 8 + 8]
# CHECK-NEXT:                  if T.likely(j_outer < 1):
# CHECK-NEXT:                      cse_var_4: T.int32 = cse_var_1 + 16
# CHECK-NEXT:                      C_1[cse_var_4:cse_var_4 + 8] = C_1[cse_var_4:cse_var_4 + 8] + T.Broadcast(_0_1[i * 256 + k], 8) * _1_1[k * 32 + cse_var_2 + 16:k * 32 + cse_var_2 + 16 + 8]
# CHECK-NEXT:  CODE: 0
