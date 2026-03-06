#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import argparse
from pathlib import Path

from xtc.export import save_model_explorer_json
from xtc.itf.graph import Graph


def build_graph(args: argparse.Namespace) -> Graph:
    import xtc.graphs.xtc.op as O

    dtype = args.dtype
    operator = args.operator
    dims = args.dims

    if operator == "matmul":
        if len(dims) != 3:
            raise ValueError("matmul expects 3 dims: i j k")
        i, j, k = dims
        a = O.tensor((i, k), dtype, name="A")
        b = O.tensor((k, j), dtype, name="B")
        with O.graph(name=args.name) as gb:
            O.matmul(a, b, name="C")
        return gb.graph

    if operator == "relu":
        if len(dims) != 1:
            raise ValueError("relu expects 1 dim: n")
        (n,) = dims
        inp = O.tensor((n,), dtype, name="I")
        with O.graph(name=args.name) as gb:
            O.relu(inp, threshold=0, name="O")
        return gb.graph

    if operator == "conv2d":
        if len(dims) != 7:
            raise ValueError("conv2d expects 7 dims: n h w f r s c")
        n, h, w, f, r, s, c = dims
        a = O.tensor((n, h + r - 1, w + s - 1, c), dtype, name="A")
        b = O.tensor((r, s, c, f), dtype, name="B")
        with O.graph(name=args.name) as gb:
            O.conv2d(a, b, stride=(1, 1), name="O")
        return gb.graph

    raise ValueError(f"unsupported operator: {operator}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an XTC graph as model-explorer JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operator",
        choices=["matmul", "relu", "conv2d"],
        required=True,
        help="Operator graph to instantiate before export.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        required=True,
        help="Operator dimensions. For matmul: i j k. For relu: n. For conv2d: n h w f r s c.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Input/output dtype used to build the graph.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Graph name.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--include-access-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include operation access maps.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_graph(args)
    out_file = Path(args.out)
    save_model_explorer_json(
        graph=graph,
        out_file=out_file,
        include_access_maps=args.include_access_maps,
        indent=args.indent,
    )
    print(out_file)


if __name__ == "__main__":
    main()

