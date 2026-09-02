# RUN: python %s --mlir 2>&1 | filecheck %s
# RUN: python %s --mlir --descript 2>&1 | filecheck %s

import sys
import xtc.graphs.xtc.op as O
from importlib import import_module
from xtc.schedules.descript import descript_scheduler

backend = "mlir"
descript = False
if len(sys.argv) > 1:
    assert sys.argv[1][:2] == "--"
    backend = sys.argv[1][2:]
    if len(sys.argv) > 2 and sys.argv[2] == "--descript":
        descript = True

backend = import_module(f"xtc.backends.{backend}")

I, J, K, dtype = 64, 256, 256, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")
graph = gb.graph

impl = backend.Backend(graph)
sch = impl.get_scheduler()

if descript:
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["I", "J", "K"],
        spec={
            "K": {},
            "I[:2]": {
                "J": {},
            },
            "I[2:]": {
                "J": {"vectorize": True},
            }
        }
    )
else:
    sch.set_dims(["I", "J", "K"])
    sch.split("I", {"I_lo": 0, "I_hi": 2})
    sch.interchange(["K", "I_lo", "I_hi"])

    sch.interchange(["I", "J"], root="./I_lo")

    sch.interchange(["I", "J"], root="./I_hi")
    sch.vectorize(["J"], root="./I_hi")

loop_nest = sch.get_loop_nest()
print(loop_nest.root_node.pretty_print())
loop_nest.check()

res = impl.evaluate(
    sch.schedule(),
)
print("VALID:", isinstance(res, float))

# CHECK:       loop K
# CHECK-NEXT:    split(I, 0, 2)
# CHECK-NEXT:      loop I
# CHECK-NEXT:        loop J
# CHECK-NEXT:          ...
# CHECK-NEXT:    split(I, 2, ...)
# CHECK-NEXT:      loop I
# CHECK-NEXT:        loop J  // vectorized
# CHECK-NEXT:          ...
# CHECK-NEXT:  VALID: True
