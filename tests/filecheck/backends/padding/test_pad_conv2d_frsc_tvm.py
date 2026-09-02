# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_tvm

import xtc.graphs.xtc.op as O
from xtc.backends.tvm import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((F, R, S, C), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad")
    t = O.transpose(b, axes=(1, 2, 3, 0))
    O.conv2d(p, t, stride=(SH, SW), name="conv")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="pad_conv2d_nhwc_frsc_mini_tvm",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       graph:
# CHECK-NEXT:    name: pad_conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:    - %1 : 16x5x5x3xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4 : 1x4x4x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:    - %3: transpose(%1, axes=(1, 2, 3, 0)) : [16x5x5x3xfloat32] -> [5x5x3x16xfloat32]
# CHECK-NEXT:    - %4: conv2d(%2, %3, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  # from tvm.script import ir as I
# CHECK-NEXT:  # from tvm.script import tirx as T
# CHECK-NEXT:  # from tvm.tirx.layout import Axis
# CHECK-NEXT:  
# CHECK-NEXT:  @I.ir_module
# CHECK-NEXT:  class Module:
# CHECK-NEXT:      @T.prim_func(s_tir=True)
# CHECK-NEXT:      def pad_conv2d_nhwc_mini(_0: T.Buffer((1, 8, 8, 3), "float32"), _1: T.Buffer((16, 5, 5, 3), "float32"), conv: T.Buffer((1, 4, 4, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          pad = T.sblock_alloc_buffer((1, 12, 12, 3))
# CHECK-NEXT:          _3 = T.sblock_alloc_buffer((1200,))
# CHECK-NEXT:          T_reshape = T.sblock_alloc_buffer((5, 5, 3, 16))
# CHECK-NEXT:          for i0, i1, i2, i3 in T.grid(1, 12, 12, 3):
# CHECK-NEXT:              with T.sblock("pad"):
# CHECK-NEXT:                  v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
# CHECK-NEXT:                  T.reads(_0[v_i0, v_i1 - 2, v_i2 - 2, v_i3])
# CHECK-NEXT:                  T.writes(pad[v_i0, v_i1, v_i2, v_i3])
# CHECK-NEXT:                  pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(2 <= v_i1 and v_i1 < 10 and 2 <= v_i2 and v_i2 < 10, _0[v_i0, v_i1 - 2, v_i2 - 2, v_i3], T.float32(0.0))
# CHECK-NEXT:          for i0 in range(1200):
# CHECK-NEXT:              with T.sblock("%3"):
# CHECK-NEXT:                  v_i0 = T.axis.spatial(1200, i0)
# CHECK-NEXT:                  T.reads(_1[v_i0 % 16, v_i0 % 1200 // 240, v_i0 % 240 // 48, v_i0 % 48 // 16])
# CHECK-NEXT:                  T.writes(_3[v_i0])
# CHECK-NEXT:                  _3[v_i0] = _1[v_i0 % 16, v_i0 % 1200 // 240, v_i0 % 240 // 48, v_i0 % 48 // 16]
# CHECK-NEXT:          for ax0, ax1, ax2, ax3 in T.grid(5, 5, 3, 16):
# CHECK-NEXT:              with T.sblock("T_reshape"):
# CHECK-NEXT:                  v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
# CHECK-NEXT:                  T.reads(_3[(v_ax0 * 240 + v_ax1 * 48 + v_ax2 * 16 + v_ax3) % 1200])
# CHECK-NEXT:                  T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
# CHECK-NEXT:                  T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = _3[(v_ax0 * 240 + v_ax1 * 48 + v_ax2 * 16 + v_ax3) % 1200]
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 4, 4, 16, 5, 5, 3):
# CHECK-NEXT:              with T.sblock("conv"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c], T_reshape[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      conv[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  conv[v_b, v_h, v_w, v_f] = conv[v_b, v_h, v_w, v_f] + pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c] * T_reshape[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  O = sch.get_sblock("conv")
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
# CHECK-NEXT:      def pad_conv2d_nhwc_mini(_0: T.Buffer((1, 8, 8, 3), "float32"), _1: T.Buffer((16, 5, 5, 3), "float32"), conv: T.Buffer((1, 4, 4, 16), "float32")):
# CHECK-NEXT:          T.func_attr({"tirx.noalias": True})
# CHECK-NEXT:          # with T.sblock("root"):
# CHECK-NEXT:          pad = T.sblock_alloc_buffer((1, 12, 12, 3))
# CHECK-NEXT:          _3 = T.sblock_alloc_buffer((1200,))
# CHECK-NEXT:          T_reshape = T.sblock_alloc_buffer((5, 5, 3, 16))
# CHECK-NEXT:          for i0, i1, i2, i3 in T.grid(1, 12, 12, 3):
# CHECK-NEXT:              with T.sblock("pad"):
# CHECK-NEXT:                  v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
# CHECK-NEXT:                  T.reads(_0[v_i0, v_i1 - 2, v_i2 - 2, v_i3])
# CHECK-NEXT:                  T.writes(pad[v_i0, v_i1, v_i2, v_i3])
# CHECK-NEXT:                  pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(2 <= v_i1 and v_i1 < 10 and 2 <= v_i2 and v_i2 < 10, _0[v_i0, v_i1 - 2, v_i2 - 2, v_i3], T.float32(0.0))
# CHECK-NEXT:          for i0 in range(1200):
# CHECK-NEXT:              with T.sblock("%3"):
# CHECK-NEXT:                  v_i0 = T.axis.spatial(1200, i0)
# CHECK-NEXT:                  T.reads(_1[v_i0 % 16, v_i0 % 1200 // 240, v_i0 % 240 // 48, v_i0 % 48 // 16])
# CHECK-NEXT:                  T.writes(_3[v_i0])
# CHECK-NEXT:                  _3[v_i0] = _1[v_i0 % 16, v_i0 % 1200 // 240, v_i0 % 240 // 48, v_i0 % 48 // 16]
# CHECK-NEXT:          for ax0, ax1, ax2, ax3 in T.grid(5, 5, 3, 16):
# CHECK-NEXT:              with T.sblock("T_reshape"):
# CHECK-NEXT:                  v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
# CHECK-NEXT:                  T.reads(_3[(v_ax0 * 240 + v_ax1 * 48 + v_ax2 * 16 + v_ax3) % 1200])
# CHECK-NEXT:                  T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
# CHECK-NEXT:                  T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = _3[(v_ax0 * 240 + v_ax1 * 48 + v_ax2 * 16 + v_ax3) % 1200]
# CHECK-NEXT:          for b, h, w, f, r, s, c in T.grid(1, 4, 4, 16, 5, 5, 3):
# CHECK-NEXT:              with T.sblock("conv"):
# CHECK-NEXT:                  v_b, v_h, v_w, v_f, v_r, v_s, v_c = T.axis.remap("SSSSRRR", [b, h, w, f, r, s, c])
# CHECK-NEXT:                  T.reads(pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c], T_reshape[v_r, v_s, v_c, v_f])
# CHECK-NEXT:                  T.writes(conv[v_b, v_h, v_w, v_f])
# CHECK-NEXT:                  with T.init():
# CHECK-NEXT:                      conv[v_b, v_h, v_w, v_f] = T.float32(0.0)
# CHECK-NEXT:                  conv[v_b, v_h, v_w, v_f] = conv[v_b, v_h, v_w, v_f] + pad[v_b, v_h * 2 + v_r, v_w * 2 + v_s, v_c] * T_reshape[v_r, v_s, v_c, v_f]
# CHECK-NEXT:  CODE: 0
