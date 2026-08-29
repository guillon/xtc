#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.graph import Graph
from xtc.graphs.xtc.graph import XTCGraph

from .TVMOps import TVMBaseExpr, TVMOperation
from .TVMScheduler import TVMScheduler
from .TVMCompiler import TVMCompiler

__all__ = [
    "TVMBackend",
]


class TVMBackend(itf.back.Backend):
    def __init__(
        self,
        source_op: TVMBaseExpr | Graph,
        dims: dict[str, int] | None = None,
        parallel_dims: list[str] | None = None,
        reduction_dims: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._tir_schedule = kwargs.get("tir_schedule", False)
        self._graph: Graph | None = None
        self._tvm_base: TVMBaseExpr
        if isinstance(source_op, XTCGraph):
            graph = source_op
            self._graph = graph
            self._tvm_base = TVMBaseExpr.from_graph(graph)
            self._ops = self._tvm_base._operations
            self._payload_name = self._graph.name
        else:
            assert isinstance(source_op, TVMOperation)
            assert dims is not None
            self._tvm_base = source_op
            assert source_op.name is not None
            self._ops = {source_op.name: source_op}
            self._payload_name = source_op.name
            assert tuple(dims.keys()) == source_op.operator.dims(), (
                f"incompatible dims names: {tuple(dims.keys())} != "
                f"{source_op.operator.dims()}"
            )
            op_parallel_dims = source_op.operator.dims("P")
            op_reduction_dims = source_op.operator.dims("R")
            if parallel_dims is not None:
                assert tuple(parallel_dims) == op_parallel_dims, (
                    f"incompatible parallel dims names: {tuple(parallel_dims)} != "
                    f"{op_parallel_dims}"
                )
            if reduction_dims is not None:
                assert tuple(reduction_dims) == op_reduction_dims, (
                    f"incompatible reduction dims names: {tuple(reduction_dims)} != "
                    f"{op_reduction_dims}"
                )

    @override
    def get_scheduler(self, **kwargs: Any) -> itf.schd.Scheduler:
        return TVMScheduler(self, **kwargs)

    @override
    def get_compiler(self, **kwargs: Any) -> itf.comp.Compiler:
        return TVMCompiler(self, **kwargs)

    @property
    @override
    def payload_name(self) -> str:
        return self._payload_name

    @property
    @override
    def graph(self) -> itf.graph.Graph:
        assert self._graph is not None
        return self._graph
