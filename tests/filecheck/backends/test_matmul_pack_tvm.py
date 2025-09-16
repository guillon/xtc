# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 64, 64, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 8, "i2": 4})
sch.tile("j", {"j1": 32, "j2": 16})
sch.tile("k", {"k1": 16})
sch.interchange(["j", "k", "i", "j1", "i1", "k1", "i2", "j2"])
sch.buffer_at("j")
sch.pack_at("k", 1, pad=True)
sch.vectorize(["j2"])
sch.unroll({"i3": 4})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_tvm_pack",
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
# CHECK-NEXT:    - %0 : 64x64xfloat32
# CHECK-NEXT:    - %1 : 64x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 64x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [64x64xfloat32, 64x64xfloat32] -> [64x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((64, 64), "float32"), _1: T.Buffer((64, 64), "float32"), C: T.Buffer((64, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          for i, j in T.grid(64, 64):
# CHECK-NEXT:              C_1 = T.Buffer((4096,), data=C.data)
# CHECK-NEXT:              C_1[i * 64 + j] = T.float32(0.0)
# CHECK-NEXT:              for k in range(64):
# CHECK-NEXT:                  cse_var_2: T.int32 = i * 64
# CHECK-NEXT:                  cse_var_1: T.int32 = cse_var_2 + j
# CHECK-NEXT:                  _0_1 = T.Buffer((4096,), data=_0.data)
# CHECK-NEXT:                  _1_1 = T.Buffer((4096,), data=_1.data)
# CHECK-NEXT:                  C_1[cse_var_1] = C_1[cse_var_1] + _0_1[cse_var_2 + k] * _1_1[k * 64 + j]
# CHECK-NEXT:  INPS = list(obj.values())[:-1]
# CHECK-NEXT:  O = obj['C']
# CHECK-NEXT:  O_W0 = sch.cache_write(O, "global")
# CHECK-NEXT:  I_R1 = sch.cache_read(INPS[1], "local", [O_W0])
# CHECK-NEXT:  i, j, = O.op.axis
# CHECK-NEXT:  k, = O.op.reduce_axis
# CHECK-NEXT:  j, j_ = sch[O].split(j, factor=32)
# CHECK-NEXT:  i_ = i
# CHECK-NEXT:  sch[O].reorder(j, j_, i_)
# CHECK-NEXT:  sch[O_W0].compute_at(sch[O], j)
# CHECK-NEXT:  i, j, = O_W0.op.axis
# CHECK-NEXT:  k, = O_W0.op.reduce_axis
# CHECK-NEXT:  j1 = j
# CHECK-NEXT:  i, i1 = sch[O_W0].split(i, factor=8)
# CHECK-NEXT:  k, k1 = sch[O_W0].split(k, factor=16)
# CHECK-NEXT:  i1, i2 = sch[O_W0].split(i1, factor=4)
# CHECK-NEXT:  j1, j2 = sch[O_W0].split(j1, factor=16)
# CHECK-NEXT:  sch[O_W0].reorder(k, i, j1, i1, k1, i2, j2)
# CHECK-NEXT:  sch[I_R1].compute_at(sch[O_W0], k)
# CHECK-NEXT:  sch[I_R1].storage_align(I_R1.op.axis[-2], factor=1024, offset=16)
# CHECK-NEXT:  sch[O_W0].vectorize(j2)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tir as T
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func
# CHECK-NEXT:      def main(_0: T.Buffer((64, 64), "float32"), _1: T.Buffer((64, 64), "float32"), C: T.Buffer((64, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
# CHECK-NEXT:          C_global = T.allocate([2048], "float32", "global")
# CHECK-NEXT:          _1_local = T.allocate([16640], "float32", "local")
# CHECK-NEXT:          for j_outer in range(2):
# CHECK-NEXT:              C_global_1 = T.Buffer((2048,), data=C_global)
# CHECK-NEXT:              for i_c_outer_init, j_c_outer_init, i_c_inner_outer_init, i_c_inner_inner_init in T.grid(8, 2, 2, 4):
# CHECK-NEXT:                  C_global_1[i_c_outer_init * 256 + i_c_inner_outer_init * 128 + i_c_inner_inner_init * 32 + j_c_outer_init * 16:i_c_outer_init * 256 + i_c_inner_outer_init * 128 + i_c_inner_inner_init * 32 + j_c_outer_init * 16 + 16] = T.Broadcast(T.float32(0.0), 16)
# CHECK-NEXT:              for k_outer in range(4):
# CHECK-NEXT:                  _1_local_1 = T.Buffer((16640,), data=_1_local, scope="local")
# CHECK-NEXT:                  for ax0, ax1 in T.grid(16, 32):
# CHECK-NEXT:                      _1_1 = T.Buffer((4096,), data=_1.data)
# CHECK-NEXT:                      _1_local_1[ax0 * 1040 + ax1] = _1_1[k_outer * 1024 + ax0 * 64 + j_outer * 32 + ax1]
# CHECK-NEXT:                  for i_c_outer, j_c_outer, i_c_inner_outer, k_inner, i_c_inner_inner in T.grid(8, 2, 2, 16, 4):
# CHECK-NEXT:                      cse_var_2: T.int32 = j_c_outer * 16
# CHECK-NEXT:                      cse_var_1: T.int32 = i_c_outer * 256 + i_c_inner_outer * 128 + i_c_inner_inner * 32 + cse_var_2
# CHECK-NEXT:                      _0_1 = T.Buffer((4096,), data=_0.data)
# CHECK-NEXT:                      C_global_1[cse_var_1:cse_var_1 + 16] = C_global_1[cse_var_1:cse_var_1 + 16] + T.Broadcast(_0_1[i_c_outer * 512 + i_c_inner_outer * 256 + i_c_inner_inner * 64 + k_outer * 16 + k_inner], 16) * _1_local_1[k_inner * 1040 + cse_var_2:k_inner * 1040 + cse_var_2 + 16]
# CHECK-NEXT:              for j_inner, i in T.grid(32, 64):
# CHECK-NEXT:                  C_1 = T.Buffer((4096,), data=C.data)
# CHECK-NEXT:                  C_1[i * 64 + j_outer * 32 + j_inner] = C_global_1[i * 32 + j_inner]
# CHECK-NEXT:  CODE: 0
