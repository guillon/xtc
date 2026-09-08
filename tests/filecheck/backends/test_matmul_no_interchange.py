# RUN: python %s --mlir 2>&1 | filecheck %s
# RUN: python %s --tvm 2>&1 | filecheck %s

import sys
import xtc.graphs.xtc.op as O
from importlib import import_module

backend = "mlir"
descript = False
if len(sys.argv) > 1:
    assert sys.argv[1][:2] == "--"
    backend = sys.argv[1][2:]

backend = import_module(f"xtc.backends.{backend}")

I, J, K, dtype = 4, 32, 256, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = backend.Backend(graph)

sch = impl.get_scheduler()
sch.strip_mine("i", {"i1": 2})
sch.strip_mine("j", {"j1": 16})
sch.unroll({"i1": 2})

loop_nest = sch.get_loop_nest()
print(loop_nest.root_node.pretty_print())
loop_nest.check()

res = impl.evaluate(
    sch.schedule(),
)
print("VALID:", isinstance(res, float))

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
# CHECK-NEXT:  loop i
# CHECK-NEXT:    tile(i, 2)  // unroll(2)
# CHECK-NEXT:      loop j
# CHECK-NEXT:        tile(j, 16)
# CHECK-NEXT:          loop k
# CHECK-NEXT:            ...
# CHECK-NEXT:  VALID: True
