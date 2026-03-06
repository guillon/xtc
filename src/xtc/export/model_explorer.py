#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from xtc.itf.data import TensorType
from xtc.itf.graph import Graph, Node


def _jsonable_attr(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_attr(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable_attr(v) for k, v in value.items()}
    return repr(value)


def _tensor_type_to_dict(tensor_type: TensorType | None) -> dict[str, Any] | None:
    if tensor_type is None:
        return None
    shape = None if tensor_type.shape is None else list(tensor_type.shape)
    return {
        "shape": shape,
        "dtype": tensor_type.dtype,
        "ndim": tensor_type.ndim,
    }


def _ensure_inferred_types(graph: Graph) -> None:
    if graph.inputs_types is None:
        return
    needs_inference = any(node.inputs_types is None for node in graph.nodes.values())
    if needs_inference:
        graph.forward_types(graph.inputs_types)


def _node_to_dict(
    node: Node,
    graph_inputs: set[str],
    graph_outputs: set[str],
    include_access_maps: bool,
) -> dict[str, Any]:
    op_name = "unknown"
    attrs: dict[str, Any] = {}
    dims: dict[str, int | str] = {}
    dims_parallel: list[str] = []
    dims_reduction: list[str] = []
    accesses_maps: Any = None

    if node.inputs_types is not None and node.outputs_types is not None:
        op = node.operation
        op_name = op.name
        attrs = {k: _jsonable_attr(v) for k, v in op.attrs.items()}
        dims = dict(op.dims)
        dims_parallel = list(op.dims_kind("P"))
        dims_reduction = list(op.dims_kind("R"))
        if include_access_maps:
            accesses_maps = _jsonable_attr(op.accesses_maps)

    data: dict[str, Any] = {
        "id": node.uid,
        "name": node.name,
        "op": op_name,
        "inputs": list(node.inputs),
        "outputs": list(node.outputs),
        "input_types": (
            None
            if node.inputs_types is None
            else [_tensor_type_to_dict(t) for t in node.inputs_types]
        ),
        "output_types": (
            None
            if node.outputs_types is None
            else [_tensor_type_to_dict(t) for t in node.outputs_types]
        ),
        "attrs": attrs,
        "dims": dims,
        "dims_parallel": dims_parallel,
        "dims_reduction": dims_reduction,
        "is_graph_input": node.uid in graph_inputs,
        "is_graph_output": node.uid in graph_outputs,
    }
    if include_access_maps:
        data["accesses_maps"] = accesses_maps
    return data


def graph_to_model_explorer(
    graph: Graph,
    *,
    include_access_maps: bool = True,
) -> dict[str, Any]:
    """Export an XTC graph to a model-explorer-friendly JSON structure."""
    _ensure_inferred_types(graph)

    graph_inputs = set(graph.inputs)
    graph_outputs = set(graph.outputs)

    nodes = [
        _node_to_dict(
            node=node,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            include_access_maps=include_access_maps,
        )
        for node in graph.nodes.values()
    ]
    edges = []
    for node in graph.nodes.values():
        for input_index, producer_uid in enumerate(node.inputs):
            edges.append(
                {
                    "source": producer_uid,
                    "target": node.uid,
                    "input_index": input_index,
                    "tensor": producer_uid,
                }
            )

    return {
        "format": "xtc.model_explorer.v1",
        "name": graph.name,
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "nodes": nodes,
        "edges": edges,
    }


def save_model_explorer_json(
    graph: Graph,
    out_file: str | Path,
    *,
    include_access_maps: bool = True,
    indent: int | None = 2,
) -> None:
    payload = graph_to_model_explorer(
        graph=graph,
        include_access_maps=include_access_maps,
    )
    Path(out_file).write_text(
        json.dumps(payload, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
