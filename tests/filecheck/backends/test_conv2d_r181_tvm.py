# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend
from xtc.artifacts import get_operation

op = get_operation("conv2d", "ResNet18_01")
N, H, W, F, R, S, C = [op["dims"][k] for k in ["n", "h", "w", "f", "r", "s", "c"]]
SH, SW = [op["params"][k] for k in ["SH", "SW"]]
dtype = "float32"

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_r181") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="O")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("w", {"w1": 4})
sch.tile("f", {"f1": 16})
sch.interchange(["b", "h", "w", "f", "r", "s", "c", "w1", "f1"])
sch.vectorize(["f1"])
sch.unroll({"w1": 4, "c": 3})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="conv2d_nhwc_r181_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_nhwc_r181
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x230x230x3xfloat32
# CHECK-NEXT:    - %1 : 7x7x3x64xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x112x112x64xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(2, 2)) {name = 'O'} : [1x230x230x3xfloat32, 7x7x3x64xfloat32] -> [1x112x112x64xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_r181(_0: T.Buffer((1, 230, 230, 3), "float32"), _1: T.Buffer((7, 7, 3, 64), "float32"), O: T.Buffer((1, 112, 112, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 112, 112, 64, 7, 7, 3):
# CHECK-NEXT:              with T.sblock("O"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(_0[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      O[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  O[v_b, v_h, v_w, v_f] = O[v_b, v_h, v_w, v_f] + _0[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  O = sch.get_sblock("O")
# CHECK-NEXT:  b, h, w, f, r, s, c, = sch.get_loops(O)
# CHECK-NEXT:  w, w1, = sch.split(w, factors=[None, 4])
# CHECK-NEXT:  f, f1, = sch.split(f, factors=[None, 16])
# CHECK-NEXT:  c, __u_c, = sch.split(c, factors=[None, 3])
# CHECK-NEXT:  sch.reorder(b, h, w, f, r, s, c, __u_c, w1, f1)
# CHECK-NEXT:  sch.unroll(w1)
# CHECK-NEXT:  sch.unroll(__u_c)
# CHECK-NEXT:  sch.vectorize(f1)
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def conv2d_nhwc_r181(_0: T.Buffer((1, 230, 230, 3), "float32"), _1: T.Buffer((7, 7, 3, 64), "float32"), O: T.Buffer((1, 112, 112, 64), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          for b, h, w_0, f_0, r, s, c_0 in T.grid(1, 112, 28, 4, 7, 7, 1):
# CHECK-NEXT:              for c_1 in T.unroll(3):
# CHECK-NEXT:                  for w_1 in T.unroll(4):
# CHECK-NEXT:                      for f_1 in T.vectorized(16):
# CHECK-NEXT:                          with T.sblock("O"):
# CHECK-NEXT:                              v_b, v_h = T.axis.remap("SS", [b, h])
# CHECK-NEXT:                              v_w = T.axis.spatial(112, w_0 * 4 + w_1)
# CHECK-NEXT:                              v_f = T.axis.spatial(64, f_0 * 16 + f_1)
# CHECK-NEXT:                              v_r, v_s = T.axis.remap("RR", [r, s])
# CHECK-NEXT:                              v_c = T.axis.reduce(3, c_0 * 3 + c_1)
# CHECK-NEXT:                              T.reads(_0[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c], _1[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                              T.writes(O[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                              with T.init():
# CHECK-NEXT:                                  O[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                              O[v_b, v_h, v_w, v_f] = O[v_b, v_h, v_w, v_f] + _0[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c] * _1[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  CODE: 0
