#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing_extensions import override
from dataclasses import dataclass, asdict
from pprint import pformat
from copy import deepcopy

from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.loop_names import make_loop_name


@dataclass(frozen=True)
class PlainNodeSchedule:
    node_name: str
    node_ident: str
    dims: list[str]
    loop_stamps: list[str]
    splits: dict[str, dict[str, int]]
    tiles: dict[str, dict[str, int]]
    permutation: dict[str, list[str]]
    vectorization: list[str]
    parallelization: list[str]
    unrolling: dict[str, int]
    packed_buffers: dict[str, list[tuple[int, str | None, bool]]]
    write_buffers: dict[str, list[str | None]]
    memory_mesh: dict[str, int]
    processor_mesh: dict[str, int]
    distribution: dict[str, str]
    distributed_buffers: dict[str, dict]
    fused: list[tuple[str, int]]
    fused_consumers: list[str]

    @override
    def __str__(self):
        return pformat(asdict(self))


class PlainNodeScheduler:
    def __init__(
        self,
        node_name: str,
        node_ident: str,
        dims: list[str],
        loop_stamps: list[str] = [],
    ) -> None:
        self.node_name = node_name
        self.node_ident = node_ident
        self.dims = dims[:]
        self.loop_stamps = loop_stamps[:]
        self.splits: dict[str, dict[str, int]] = {}
        self.tiles: dict[str, dict[str, int]] = {k: {} for k in self.dims}
        self.permutation: dict[str, list[str]] = {}
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}
        self.packed_buffers: dict[str, list[tuple[int, str | None, bool]]] = {}
        self.write_buffers: dict[str, list[str | None]] = {}
        self.memory_mesh: dict[str, int] = {}
        self.processor_mesh: dict[str, int] = {}
        self.distribution: dict[str, str] = {}
        self.distributed_buffers: dict[str, dict] = {}
        self.fused: list[tuple[str, int]] = []
        self.fused_consumers: list[str] = []

    def get_plain_schedule(self) -> PlainNodeSchedule:
        if not self.permutation:
            self.permutation[DEFAULT_ROOT] = self.get_default_interchange(DEFAULT_ROOT)

        for fuse_axis in self.fused:
            assert fuse_axis[0] in self.permutation[next(iter(self.permutation))], (
                "Fusion must be to an axis in the base root not the result of a split."
            )

        return PlainNodeSchedule(
            node_name=self.node_name,
            node_ident=self.node_ident,
            dims=deepcopy(self.dims),
            loop_stamps=deepcopy(self.loop_stamps),
            tiles=deepcopy(self.tiles),
            splits=deepcopy(self.splits),
            permutation=deepcopy(self.permutation),
            vectorization=deepcopy(self.vectorization),
            parallelization=deepcopy(self.parallelization),
            unrolling=deepcopy(self.unrolling),
            memory_mesh=deepcopy(self.memory_mesh),
            packed_buffers=deepcopy(self.packed_buffers),
            write_buffers=deepcopy(self.write_buffers),
            processor_mesh=deepcopy(self.processor_mesh),
            distribution=deepcopy(self.distribution),
            distributed_buffers=deepcopy(self.distributed_buffers),
            fused=deepcopy(self.fused),
            fused_consumers=deepcopy(self.fused_consumers),
        )

    @override
    def __str__(self) -> str:
        return str(self.get_plain_schedule())

    def get_default_interchange(self, root: str) -> list[str]:
        ret = [make_loop_name(root, d) for d in self.dims]
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for _, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                ret.append(dim_name)
        return ret

    def set_dims(self, dims: list[str]) -> None:
        assert len(dims) == len(self.dims)
        self.dims = dims[:]
        self.tiles = {k: {} for k in self.dims}

    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        segments_renamed = {
            make_loop_name(root, key): val for key, val in segments.items()
        }
        self.splits[dim] = segments_renamed
        for s in segments_renamed:
            self.tiles[s] = {}

    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT):
        for d, s in tiles.items():
            tile_name = make_loop_name(root, d)
            self.tiles[dim][tile_name] = s

    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT):
        self.permutation[root] = [make_loop_name(root, a) for a in permutation]

    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.vectorization += [make_loop_name(root, a) for a in axes]

    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT):
        self.parallelization = [make_loop_name(root, a) for a in axes]

    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT):
        for dim, ufactor in unrolls.items():
            self.unrolling[make_loop_name(root, dim)] = ufactor

    def buffer_at(
        self,
        axis: str,
        mtype: str | None = None,
        root: str = DEFAULT_ROOT,
    ) -> None:
        axis_key = make_loop_name(root, axis)
        if axis_key not in self.write_buffers.keys():
            self.write_buffers[axis_key] = [mtype]
        else:
            self.write_buffers[axis_key].append(mtype)

    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ):
        axis_key = make_loop_name(root, axis)
        if axis_key not in self.packed_buffers.keys():
            self.packed_buffers[axis_key] = [(input_idx, mtype, pad)]
        else:
            self.packed_buffers[axis_key].append((input_idx, mtype, pad))

    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        axis_key = make_loop_name(root, axis)
        self.fused.append((axis_key, input_idx))

    def fuse_consumer_at(self, axis: str, root: str = DEFAULT_ROOT) -> None:
        axis_key = make_loop_name(root, axis)
        self.fused_consumers.append(axis_key)
