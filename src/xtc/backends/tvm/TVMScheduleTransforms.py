#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, cast

import tvm
import tvm.s_tir


def loop_partition_rebased(
    sch: tvm.s_tir.Schedule,
    loop: Any,
    factors: list[int | None],
) -> tuple[tvm.s_tir.Schedule, list[Any]]:
    """Partition a loop and rebase each resulting loop to start at zero.

    TVM's loop_partition preserves the original loop coordinates, while several
    schedule primitives, including split, require a zero loop minimum.  Rebuild
    the partition loops with a zero minimum and add the old minimum to every use
    of their induction variable.

    Rebuilding the IR invalidates schedule RVs, so child blocks are reacquired
    by name and returned with the new schedule.
    """
    working_on = sch.func_working_on
    assert working_on is not None

    partitions = sch.loop_partition(loop, cast(Any, factors))
    children = [sch.get_child_blocks(partition)[0] for partition in partitions]
    child_names = [cast(str, cast(Any, sch.get(child)).name_hint) for child in children]
    partition_vars = [
        cast(Any, sch.get(partition)).loop_var for partition in partitions
    ]

    def postorder(node: Any) -> Any:
        if not isinstance(node, tvm.tirx.For):
            return node
        if not any(node.loop_var.same_as(var) for var in partition_vars):
            return node
        if isinstance(node.min, tvm.tirx.IntImm) and node.min.value == 0:
            return node

        old_var = node.loop_var
        new_var = tvm.ir.Var(f"{old_var.name}_zero", old_var.ty)
        body = tvm.tirx.stmt_functor.substitute(
            node.body,
            {old_var: new_var + node.min},
        )
        return tvm.tirx.For(
            new_var,
            cast(Any, 0),
            node.extent,
            node.kind,
            body,
            node.thread_binding,
            node.annotations,
            node.step,
            node.span,
        )

    old_mod = cast(Any, sch.mod)
    old_func = old_mod[working_on]
    new_body = tvm.tirx.stmt_functor.ir_transform(
        old_func.body,
        None,
        postorder,
        ["tirx.For"],
    )

    new_mod = tvm.IRModule(
        old_mod.functions,
        attrs=old_mod.attrs,
        global_infos=old_mod.global_infos,
    )
    new_mod.update_func(working_on, old_func.with_body(new_body))

    new_sch = tvm.s_tir.Schedule(new_mod)
    new_sch.work_on(working_on.name_hint)
    new_children = [new_sch.get_sblock(name) for name in child_names]
    return new_sch, new_children
