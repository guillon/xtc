# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

# fusion is only compatible with splitting if its above the split
import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="small_pad2d_conv2d") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad")
    O.conv2d(p, b, stride=(SH, SW), name="conv")

graph = gb.graph

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
# split above fuse dim
sch.split("h", {"h0": 0, "h1":3})
sch.interchange(["b", "h0", "h1"])
sch.interchange(["w","r","s", "c", "f"],"./h0")
sch.interchange(["w","r","s", "c", "f"],"./h1")
sch.fuse_producer_at("w",0)
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="invalid_split_then_fuse_mlir_tensor",
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
# XFAIL: *
