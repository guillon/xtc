#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from .loop_names import basename
from .plain_schedule import PlainNodeSchedule
from .loop_nest import LoopInfo, LoopNest, LoopNestNode, SplitOrigin


class LoopNestBuilder:
    @staticmethod
    def from_plain_node_schedule(node_sched: PlainNodeSchedule) -> LoopNest:
        dims = node_sched.dims[:]

        loop_nest = LoopNest(abstract_dims=dims)
        root_node = loop_nest.build_root_node(node_sched.node_name)

        # Assign splits to root_node first, stripping the root prefix from names
        for axis, axis_splits in node_sched.splits.items():
            root_node.splits[axis] = {basename(k): v for k, v in axis_splits.items()}

        # Build mapper to get splits_info
        mapper = LoopInfo.build_from_node(root_node)

        def populate_node(node: LoopNestNode, perm: list[str]) -> None:
            """Populate node with data for loops in its permutation."""
            perm_set = set(perm)
            node.interchange = [basename(n) for n in perm]
            for axis, axis_tiles in node_sched.tiles.items():
                for tile_name, size in axis_tiles.items():
                    if tile_name in perm_set:
                        if axis not in node.tiles:
                            node.tiles[axis] = {}
                        node.tiles[axis][basename(tile_name)] = size
            node.vectorize = [
                basename(v) for v in node_sched.vectorization if v in perm_set
            ]
            node.parallelize = [
                basename(p) for p in node_sched.parallelization if p in perm_set
            ]
            node.unroll = {
                basename(k): v for k, v in node_sched.unrolling.items() if k in perm_set
            }
            # TODO: loop nest supports only one buffer per axis
            node.buffer_at = {
                basename(k): v[0]
                for k, v in node_sched.write_buffers.items()
                if k in perm_set
            }
            # TODO: loop nest supports only one pack per axis
            node.pack_at = {
                basename(k): v[0]
                for k, v in node_sched.packed_buffers.items()
                if k in perm_set
            }
            # TODO: loop nest supports only one fuse per axis
            node.fuse_producer_at = {
                basename(k): v for k, v in node_sched.fused if k in perm_set
            }
            # TODO: loop nest supports only one fuse consumer per axis
            node.fuse_consumer_at = [
                basename(k) for k in node_sched.fused_consumers if k in perm_set
            ]

        # Process each root in permutation
        for root, perm in node_sched.permutation.items():
            root_name = basename(root)
            if root_name in mapper.splits_info:
                # This root is a split - create child node
                axis, start, end = mapper.splits_info[root_name]
                child = LoopNestNode(
                    root=root_name,
                    tiles={d: {} for d in dims},
                    split_origin=SplitOrigin(axis=axis, start=start, end=end),
                )
                populate_node(child, perm)
                root_node.add_child(child)
            else:
                # This is the main root
                populate_node(root_node, perm)

        return loop_nest
