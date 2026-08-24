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

if backend == "tvm":
    backend_kwargs = {"tir_schedule": True}
else:
    backend_kwargs = {}

backend = import_module(f"xtc.backends.{backend}")

I, J, K, dtype = 64, 192, 256, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")
graph = gb.graph

impl = backend.Backend(graph, **backend_kwargs)
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
                "J[:64]": {
                },
                "J[64:]": {
                    "J#128": {"vectorize": True},
                }
            }
        }
    )
else:
    sch.set_dims(["I", "J", "K"])
    sch.split("I", {"I_lo": 0, "I_hi": 2})

    sch.interchange(["K", "I_lo", "I_hi"])
    sch.interchange(["I", "J"], root="./I_lo")

    sch.split("J", {"J_lo": 0, "J_hi": 64}, root="./I_hi")
    sch.interchange(["I", "J_lo","J_hi"], root="./I_hi")
    sch.interchange(["J"], root="./I_hi/J_lo")
    sch.tile("J", {"J0": 128}, root="./I_hi/J_hi")
    sch.interchange(["J", "J0"], root="./I_hi/J_hi")
    sch.vectorize(["J0"], root="./I_hi/J_hi")

loop_nest = sch.get_loop_nest()
print(loop_nest.root_node.pretty_print())
loop_nest.check()

# TODO: for now mlir backend fails to generate
# res = impl.evaluate(
#     sch.schedule(),
# )
# print("VALID:", isinstance(res, float))

# CHECK:       loop K
# CHECK-NEXT:    split(I, 0, 2)
# CHECK-NEXT:      loop I
# CHECK-NEXT:        loop J
# CHECK-NEXT:          ...
# CHECK-NEXT:    split(I, 2, ...)
# CHECK-NEXT:      loop I
# CHECK-NEXT:        split(J, 0, 64)
# CHECK-NEXT:          loop J
# CHECK-NEXT:            ...
# CHECK-NEXT:        split(J, 64, ...)
# CHECK-NEXT:          loop J
# CHECK-NEXT:            tile(J, 128)  // vectorized
# CHECK-NEXT:              ...
