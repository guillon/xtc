# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 3, 3, 3, 1, 1, "float32"
a = O.tensor((N, H + R - 1, W + S - 1, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="conv2d_nhwc_mini") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_mini_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x10x10x3xfloat32
# CHECK-NEXT:    - %1 : 3x3x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x8x8x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(1, 1)) {name = 'O'} : [1x10x10x3xfloat32, 3x3x3x16xfloat32] -> [1x8x8x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_mini(_0: T.Buffer((1, 10, 10, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), O: T.Buffer((1, 8, 8, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 8, 8, 16, 3, 3, 3):
# CHECK-NEXT:              with T.sblock("O"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(_0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      O[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  O[v_b, v_h, v_w, v_f] = O[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  O = sch.get_sblock("O")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  sch.reorder(b, h, w, f, r, s, c)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_mini(_0: T.Buffer((1, 10, 10, 3), "float32"), _1: T.Buffer((3, 3, 3, 16), "float32"), O: T.Buffer((1, 8, 8, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 8, 8, 16, 3, 3, 3):
# CHECK-NEXT:              with T.sblock("O"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(_0[v_b, v_h + v_r, v_w + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      O[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  O[v_b, v_h, v_w, v_f] = O[v_b, v_h, v_w, v_f] + _0[v_b, v_h + v_r, v_w + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  CODE: 0
