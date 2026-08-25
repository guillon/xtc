# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

I, J, K, dtype = 14, 14, 14, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="pad_matmul_unpad") as gb:
    p1 = O.pad(a, padding=2, name="A_pad")
    p2 = O.pad(b, padding=2, name="B_pad")
    m_pad = O.matmul(p1, p2, name="matmul_padded")
    O.unpad(m_pad, padding=2, name="C")
graph = gb.graph
print(graph)

impl = Backend(graph)
sch = impl.get_scheduler(default_node="matmul_padded")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_int_matmul_unpad_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       graph:
# CHECK-NEXT:    name: pad_matmul_unpad
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 14x14xfloat32
# CHECK-NEXT:    - %1 : 14x14xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %5 : 14x14xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad(%0, padding=(2, 2), constant_value=0) {name = 'A_pad'} : [14x14xfloat32] -> [18x18xfloat32]
# CHECK-NEXT:    - %3: pad(%1, padding=(2, 2), constant_value=0) {name = 'B_pad'} : [14x14xfloat32] -> [18x18xfloat32]
# CHECK-NEXT:    - %4: matmul(%2, %3) {name = 'matmul_padded'} : [18x18xfloat32, 18x18xfloat32] -> [18x18xfloat32]
# CHECK-NEXT:    - %5: unpad(%4, padding=(2, 2)) {name = 'C'} : [18x18xfloat32] -> [14x14xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def pad_matmul_unpad(_0: T.Buffer((14, 14), "float32"), _1: T.Buffer((14, 14), "float32"), C: T.Buffer((14, 14), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          A_pad = T.sblock_alloc_buffer((18, 18))
# CHECK-NEXT:          B_pad = T.sblock_alloc_buffer((18, 18))
# CHECK-NEXT:          matmul_padded = T.sblock_alloc_buffer((18, 18))
# CHECK-NEXT:          for i0, i1 in T.grid(18, 18):
# CHECK-NEXT:              with T.sblock("A_pad"):
# CHECK-NEXT:                  v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
# CHECK-NEXT:                  T.reads(_0[v_i0 - 2, v_i1 - 2])
# CHECK-NEXT:                  T.writes(A_pad[v_i0, v_i1])
# CHECK-NEXT:                  A_pad[v_i0, v_i1] = T.if_then_else(2 <= v_i0 and v_i0 < 16 and 2 <= v_i1 and v_i1 < 16, _0[v_i0 - 2, v_i1 - 2], T.float32(0.0))
# CHECK-NEXT:          for i0, i1 in T.grid(18, 18):
# CHECK-NEXT:              with T.sblock("B_pad"):
# CHECK-NEXT:                  v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
# CHECK-NEXT:                  T.reads(_1[v_i0 - 2, v_i1 - 2])
# CHECK-NEXT:                  T.writes(B_pad[v_i0, v_i1])
# CHECK-NEXT:                  B_pad[v_i0, v_i1] = T.if_then_else(2 <= v_i0 and v_i0 < 16 and 2 <= v_i1 and v_i1 < 16, _1[v_i0 - 2, v_i1 - 2], T.float32(0.0))
# CHECK-NEXT:          for i, j, k in T.grid(18, 18, 18):
# CHECK-NEXT:              with T.sblock("matmul_padded"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(A_pad[v_i, v_k], B_pad[v_k, v_j])
# CHECK-NEXT:                  T.writes(matmul_padded[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      matmul_padded[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  matmul_padded[v_i, v_j] = matmul_padded[v_i, v_j] + A_pad[v_i, v_k] * B_pad[v_k, v_j]
# CHECK-NEXT:          for i0, i1 in T.grid(14, 14):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
# CHECK-NEXT:                  T.reads(matmul_padded[v_i0 + 2, v_i1 + 2])
# CHECK-NEXT:                  T.writes(C[v_i0, v_i1])
# CHECK-NEXT:                  C[v_i0, v_i1] = matmul_padded[v_i0 + 2, v_i1 + 2]
# CHECK-NEXT:  O = sch.get_sblock("matmul_padded")
# CHECK-NEXT:  i, j, k, = sch.get_loops(O)
# CHECK-NEXT:  sch.reorder(i, j, k)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def pad_matmul_unpad(_0: T.Buffer((14, 14), "float32"), _1: T.Buffer((14, 14), "float32"), C: T.Buffer((14, 14), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          A_pad = T.sblock_alloc_buffer((18, 18))
# CHECK-NEXT:          B_pad = T.sblock_alloc_buffer((18, 18))
# CHECK-NEXT:          matmul_padded = T.sblock_alloc_buffer((18, 18))
# CHECK-NEXT:          for i0, i1 in T.grid(18, 18):
# CHECK-NEXT:              with T.sblock("A_pad"):
# CHECK-NEXT:                  v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
# CHECK-NEXT:                  T.reads(_0[v_i0 - 2, v_i1 - 2])
# CHECK-NEXT:                  T.writes(A_pad[v_i0, v_i1])
# CHECK-NEXT:                  A_pad[v_i0, v_i1] = T.if_then_else(2 <= v_i0 and v_i0 < 16 and 2 <= v_i1 and v_i1 < 16, _0[v_i0 - 2, v_i1 - 2], T.float32(0.0))
# CHECK-NEXT:          for i0, i1 in T.grid(18, 18):
# CHECK-NEXT:              with T.sblock("B_pad"):
# CHECK-NEXT:                  v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
# CHECK-NEXT:                  T.reads(_1[v_i0 - 2, v_i1 - 2])
# CHECK-NEXT:                  T.writes(B_pad[v_i0, v_i1])
# CHECK-NEXT:                  B_pad[v_i0, v_i1] = T.if_then_else(2 <= v_i0 and v_i0 < 16 and 2 <= v_i1 and v_i1 < 16, _1[v_i0 - 2, v_i1 - 2], T.float32(0.0))
# CHECK-NEXT:          for i, j, k in T.grid(18, 18, 18):
# CHECK-NEXT:              with T.sblock("matmul_padded"):
# CHECK-NEXT:                  v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
# CHECK-NEXT:                  T.reads(A_pad[v_i, v_k], B_pad[v_k, v_j])
# CHECK-NEXT:                  T.writes(matmul_padded[v_i, v_j])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      matmul_padded[v_i, v_j] = T.float32(0.0)
# CHECK-NEXT:                  matmul_padded[v_i, v_j] = matmul_padded[v_i, v_j] + A_pad[v_i, v_k] * B_pad[v_k, v_j]
# CHECK-NEXT:          for i0, i1 in T.grid(14, 14):
# CHECK-NEXT:              with T.sblock("C"):
# CHECK-NEXT:                  v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
# CHECK-NEXT:                  T.reads(matmul_padded[v_i0 + 2, v_i1 + 2])
# CHECK-NEXT:                  T.writes(C[v_i0, v_i1])
# CHECK-NEXT:                  C[v_i0, v_i1] = matmul_padded[v_i0 + 2, v_i1 + 2]
# CHECK-NEXT:  CODE: 0
