# RUN: python %s --mlir 2>&1 | filecheck %s
# RUN: python %s --mlir --descript 2>&1 | filecheck %s
# RUN: python %s --tvm 2>&1 | filecheck %s
# RUN: python %s --tvm --descript 2>&1 | filecheck %s
# REQUIRES: module_tvm

import sys
import xtc.graphs.xtc.op as O
from importlib import import_module
from xtc.artifacts import get_operation
from xtc.schedules.descript import descript_scheduler

backend = "mlir"
descript = False
if len(sys.argv) > 1:
    assert sys.argv[1][:2] == "--"
    backend = sys.argv[1][2:]
    if len(sys.argv) > 2 and sys.argv[2] == "--descript":
        descript = True

backend = import_module(f"xtc.backends.{backend}")

# A reduced version of Yolo9000_12
N, H, W, F, R, S, C = 1, 34, 34, 128, 3, 3, 32
SH, SW = 1, 1
dtype = "float32"
abstract_dims = ["b", "h", "w", "f", "r", "s", "c"]
abstract_sizes = dict(zip(abstract_dims, [N, H, W, F, R, S, C]))

a = O.tensor((N, H + R - 1, W + S - 1, C), dtype)
b = O.tensor((R, S, C, F), dtype)

with O.graph(name="conv2d_nhwc_yolo9k2") as gb:
    O.conv2d(a, b, stride=(SH, SW), name="C")

graph = gb.graph
print(graph)

impl = backend.Backend(graph)
sch = impl.get_scheduler()

if descript:
    descript_spec = {
        "b": {},
        "w[0:20]" : {
            "s": {},
            "w#10": {},
            "h": {},
            "r": {},
            "c": {},
            "h#17": {},
            "c#16": {},
            "c#2": {"unroll" : 2},
            "w#5": {"unroll" : 5},
            "f": {"unroll" : 64},
            "f#16": {"vectorize" : True}
        },
        "w[20:34]" : {
            "w#14": {},
            "s": {},
            "h": {},
            "r": {},
            "c": {},
            "h#34": {},
            "h#17": {},
            "c#16": {},
            "c#2": {"unroll" : 2},
            "w#7": {"unroll" : 7},
            "f": {"unroll" : 64},
            "f#16": {"vectorize" : True}
        }
    }

    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=abstract_dims,
        abstract_dim_sizes=abstract_sizes,
        spec=descript_spec,
    )
else:
    sch.split("w", {"wl": 0, "wh": 20})
    sch.interchange(["b", "wl", "wh"])
    sch.strip_mine("w", {"w0": 10, "w1": 5}, root="./wl")
    sch.strip_mine("h", {"h0": 17}, root="./wl")
    sch.strip_mine("c", {"c0": 16, "c1": 2}, root="./wl")
    sch.strip_mine("f", {"f0": 16}, root="./wl")
    sch.interchange(["w", "s", "w0", "h", "r", "c", "h0", "c0", "c1", "w1", "f", "f0"], root="./wl")
    sch.unroll({"c1": 2, "w1": 5, "f": 64}, root="./wl")
    sch.vectorize(["f0"], root="./wl")
    sch.strip_mine("w", {"w0": 14, "w1": 7}, root="./wh")
    sch.strip_mine("h", {"h0": 34, "h1": 17}, root="./wh")
    sch.strip_mine("c", {"c0": 16, "c1": 2}, root="./wh")
    sch.strip_mine("f", {"f0": 16}, root="./wh")
    sch.interchange(["w", "w0", "s", "h", "r", "c", "h0", "h1", "c0", "c1", "w1", "f", "f0"], root="./wh")
    sch.unroll({"c1": 2, "w1": 7, "f": 64}, root="./wh")
    sch.vectorize(["f0"], root="./wh")

loop_nest = sch.get_loop_nest()
print(loop_nest.root_node.pretty_print())
loop_nest.check()

res = impl.evaluate(
    sch.schedule(),
)
print("VALID:", isinstance(res, float))
# CHECK:       graph:
# CHECK-NEXT:    name: conv2d_nhwc_yolo9k2
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x36x36x32xfloat32
# CHECK-NEXT:    - %1 : 3x3x32x128xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 1x34x34x128xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: conv2d(%0, %1, stride=(1, 1)) {name = 'C'} : [1x36x36x32xfloat32, 3x3x32x128xfloat32] -> [1x34x34x128xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  loop b
# CHECK-NEXT:    split(w, 0, 20)
# CHECK-NEXT:      loop w
# CHECK-NEXT:        loop s
# CHECK-NEXT:          tile(w, 10)
# CHECK-NEXT:            loop h
# CHECK-NEXT:              loop r
# CHECK-NEXT:                loop c
# CHECK-NEXT:                  tile(h, 17)
# CHECK-NEXT:                    tile(c, 16)
# CHECK-NEXT:                      tile(c, 2)  // unroll(2)
# CHECK-NEXT:                        tile(w, 5)  // unroll(5)
# CHECK-NEXT:                          loop f  // unroll(64)
# CHECK-NEXT:                            tile(f, 16)  // vectorized
# CHECK-NEXT:                              ...
# CHECK-NEXT:    split(w, 20, ...)
# CHECK-NEXT:      loop w
# CHECK-NEXT:        tile(w, 14)
# CHECK-NEXT:          loop s
# CHECK-NEXT:            loop h
# CHECK-NEXT:              loop r
# CHECK-NEXT:                loop c
# CHECK-NEXT:                  tile(h, 34)
# CHECK-NEXT:                    tile(h, 17)
# CHECK-NEXT:                      tile(c, 16)
# CHECK-NEXT:                        tile(c, 2)  // unroll(2)
# CHECK-NEXT:                          tile(w, 7)  // unroll(7)
# CHECK-NEXT:                            loop f  // unroll(64)
# CHECK-NEXT:                              tile(f, 16)  // vectorized
# CHECK-NEXT:                                ...
# CHECK-NEXT:  VALID: True
