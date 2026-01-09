#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from dataclasses import dataclass, field
import re
from xtc.itf.schd.scheduler import Scheduler


@dataclass
class LoopsDimsMapper:
    tiles_to_axis: dict[str, str]
    splits_to_axis: dict[str, str]
    dims: list[str]

    @property
    def loops_to_axis(self) -> dict[str, str]:
        loops_to_axis = (
            self.tiles_to_axis | self.splits_to_axis | dict(zip(self.dims, self.dims))
        )
        return loops_to_axis

    @staticmethod
    def build_from_slices(slices: list["LoopNestSlice"]) -> "LoopsDimsMapper":
        tiles_to_axis = {}
        splits_to_axis = {}
        dims = set()
        for slice in slices:
            tiles_to_axis.update(LoopsDimsMapper._get_subloops_to_axis(slice.tiles))
            splits_to_axis.update(LoopsDimsMapper._get_subloops_to_axis(slice.splits))
        refined_loops = list(tiles_to_axis) + list(splits_to_axis)
        for slice in slices:
            dims.update(
                [loop for loop in slice.interchange if loop not in refined_loops]
            )
            dims.update(tiles_to_axis.values())
            dims.update(splits_to_axis.values())
        return LoopsDimsMapper(tiles_to_axis, splits_to_axis, list(dims))

    @staticmethod
    def _get_subloops_to_axis(subloops: dict[str, dict[str, Any]]) -> dict[str, str]:
        loop_to_axis: dict[str, str] = {}
        for axis_name, subloops in subloops.items():
            for loop_name in subloops:
                loop_to_axis[loop_name] = axis_name
        return loop_to_axis


@dataclass
class LoopNestSlice:
    root: str
    tiles: dict[str, dict[str, int]]
    splits: dict[str, dict[str, int]] = field(default_factory=dict)
    interchange: list[str] = field(default_factory=list)
    vectorize: list[str] = field(default_factory=list)
    parallelize: list[str] = field(default_factory=list)
    unroll: dict[str, int] = field(default_factory=dict)

    @property
    def splits_to_sizes(self) -> dict[str, int]:
        splits_to_sizes: dict[str, int] = {}
        for axis in self.splits:
            last_start = None
            for loop_name, start in reversed(self.splits[axis].items()):
                if last_start is not None:
                    size_of_split = last_start - start
                    splits_to_sizes[loop_name] = size_of_split
                last_start = start
        return splits_to_sizes

    @property
    def tiles_to_sizes(self) -> dict[str, int]:
        tiles_to_sizes: dict[str, int] = {}
        for tiles in self.tiles.values():
            for loop, size in tiles.items():
                tiles_to_sizes[loop] = size
        return tiles_to_sizes


@dataclass
class LoopNest:
    abstract_dims: list[str]
    slices: list[LoopNestSlice] = field(default_factory=list)

    @property
    def empty(self):
        return not self.slices

    def build_slice(self, root: str) -> LoopNestSlice:
        slice = LoopNestSlice(root=root, tiles={a: {} for a in self.abstract_dims})
        self.slices = [slice] + self.slices
        return slice

    def check(self):
        self._check_use_defined_dims()
        self._check_vectorization_consistency()
        self._check_tiling_consistency()
        self._check_sizes()

    def _check_use_defined_dims(self):
        mapper = LoopsDimsMapper.build_from_slices(self.slices)
        for dim in self.abstract_dims:
            if dim not in mapper.dims:
                raise RuntimeError(f"{dim} defined but never used")

    def _check_vectorization_consistency(self):
        for sched in self.slices:
            vect_above = False
            for loop_name in sched.interchange:
                if loop_name in sched.vectorize:
                    vect_above = True
                elif vect_above:
                    raise RuntimeError(
                        f"Inner loop {loop_name} isn't vectorized but an outer one is."
                    )

    def _check_tiling_consistency(self) -> None:
        mapper = LoopsDimsMapper.build_from_slices(self.slices)
        seen_axes: dict[str, int | None] = {}
        for sched in self.slices:
            for loop_name in sched.interchange:
                if loop_name in mapper.dims:
                    seen_axes[loop_name] = None
                elif loop_name in mapper.tiles_to_axis:
                    axis = mapper.tiles_to_axis[loop_name]
                    size = sched.tiles_to_sizes[loop_name]
                    if axis not in seen_axes:
                        raise RuntimeError(
                            f"""
                            `{axis}#{size}`: {axis} has not been materialized yet.
                            """
                        )
                    seen_axes[axis] = sched.tiles[axis][loop_name]

    def _check_sizes(self):
        mapper = LoopsDimsMapper.build_from_slices(self.slices)
        current_size_of_split: dict[str, int | None] = {}
        for sched in self.slices:
            current_size_of_tile: dict[str, int] = {}

            for loop_name in sched.interchange:
                axis = mapper.loops_to_axis[loop_name]
                current_sizes = (
                    {d: None for d in mapper.dims}
                    | current_size_of_split
                    | current_size_of_tile
                )
                if loop_name in mapper.dims:
                    if loop_name not in current_size_of_split:
                        current_size_of_split[loop_name] = None
                elif loop_name in mapper.tiles_to_axis:
                    loop_size = sched.tiles[axis][loop_name]
                    LoopNest._must_be_smaller_routine(
                        new_size=loop_size,
                        current_sizes=current_sizes,
                        loop_name=loop_name,
                        axis=axis,
                    )
                    current_size_of_tile[axis] = loop_size
                elif (
                    loop_name in mapper.splits_to_axis
                    and loop_name in sched.splits_to_sizes
                ):
                    loop_size = sched.splits_to_sizes[loop_name]
                    LoopNest._must_be_smaller_routine(
                        new_size=loop_size,
                        current_sizes=current_sizes,
                        loop_name=loop_name,
                        axis=axis,
                    )
                    current_size_of_split[axis] = loop_size

    @staticmethod
    def _must_be_smaller_routine(
        new_size: int, current_sizes: dict[str, int | None], loop_name: str, axis: str
    ):
        old_size = current_sizes[axis]
        if old_size is not None and new_size > old_size:
            raise RuntimeError(
                f"""
                Inner loop {loop_name} on axis {axis} must be smaller than outer loop.
                """
            )


def descript_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    spec: dict[str, dict],
):
    descript = Descript(scheduler=scheduler, abstract_axis=abstract_axis)
    descript.apply(node_name=node_name, spec=spec)


@dataclass(frozen=True)
class Descript:
    scheduler: Scheduler
    abstract_axis: list[str]

    def apply(self, node_name: str, spec: dict[str, dict]):
        flat_schedules = self._flatten_schedule(root=node_name, spec=spec, head=[])
        flat_schedules.check()

        self.scheduler.set_dims(self.abstract_axis)
        for schedule in flat_schedules.slices:
            root = schedule.root

            for d, s in schedule.splits.items():
                self.scheduler.split(d, s, root=root)

            for d, s in schedule.tiles.items():
                self.scheduler.tile(d, s, root=root)

            self.scheduler.interchange(schedule.interchange, root=root)
            self.scheduler.vectorize(schedule.vectorize, root=root)
            self.scheduler.parallelize(schedule.parallelize, root=root)
            self.scheduler.unroll(schedule.unroll, root=root)

    def _flatten_schedule(
        self, root: str, spec: dict[str, dict], head: list[str]
    ) -> LoopNest:
        recursive_scheds = LoopNest(abstract_dims=self.abstract_axis)
        sched = recursive_scheds.build_slice(root)
        # State of the schedule
        sizes: dict[str, int] = {}
        previous_cut: dict[str, int | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = head
        # Processing the schedule
        for declaration, val in spec.items():
            # Splits
            if ":" in declaration:
                axis_name, x, y = parse_split_declaration(declaration)
                self._check_axis_existence(axis_name)

                # The only declaration where y (the cut) is None is the
                # last one, so it cannot be the previous one.
                cut = previous_cut[axis_name]

                # When x (the starting point of the slice), is not
                # specified, it is the previous cut
                if x is None:
                    x = cut
                assert x is not None

                self._check_splitting_intervals(declaration, axis_name, cut, x, y)

                # Update the previous cut
                previous_cut[axis_name] = y
                # Save the cutting points of the new dimensions
                if not axis_name in sched.splits:
                    sched.splits[axis_name] = {}
                new_dim_index = len(sched.splits[axis_name])
                new_dim_name = f"{axis_name}[{new_dim_index}]"
                new_root_name = f"{root}/{new_dim_name}"
                sched.splits[axis_name][new_dim_name] = x
                interchange.append(new_dim_name)
                # Fetch the schedule associated with the new dimension
                next_schedule = val
                assert isinstance(next_schedule, dict)
                inner_scheds = self._flatten_schedule(
                    spec=next_schedule, root=new_root_name, head=[axis_name]
                )
                recursive_scheds.slices += inner_scheds.slices
                continue

            # Tiles
            elif "#" in declaration:
                axis_name, tile_size = declaration.split("#")
                self._check_axis_existence(axis_name)
                try:
                    loop_size = int(tile_size)
                except:
                    raise RuntimeError(
                        f"`{declaration}`: {tile_size} is not an integer."
                    )

                if loop_size <= 0:
                    raise RuntimeError(
                        f"`{declaration}`: tile sizes should be strictly positive."
                    )

                tile_num = len(sched.tiles[axis_name])
                loop_name = f"{axis_name}{tile_num}"
                sched.tiles[axis_name][loop_name] = loop_size
                sizes[loop_name] = loop_size
                interchange.append(loop_name)

            elif declaration in self.abstract_axis:
                if declaration in interchange:
                    raise RuntimeError(
                        f"""
                        Axis {declaration} is scheduled twice (or more).
                        """
                    )
                loop_name = declaration
                interchange.append(loop_name)

            else:
                self._unknown_axis_error(declaration)

            annotate(loop_name=loop_name, sizes=sizes, annotations=val, sched=sched)

        # Check if the last cut of each axis is either 0 or None.
        # None correspond to "until the end of the loop". 0 is the
        # default value, if it has 0 then it means the axis isn't splitted.
        # Any other value means the split is let in a partial state.
        for axis, cut in previous_cut.items():
            if cut is not None and cut != 0:
                raise RuntimeError(f"Splitting of {axis} unachieved (stops at {cut}).")

        sched.interchange = interchange
        return recursive_scheds

    def _check_splitting_intervals(
        self,
        decl: str,
        axis_name: str,
        cut: int | None,
        x: int | None,
        y: int | None,
    ):
        if cut is None:
            raise RuntimeError(
                f"""
                {decl}: {axis_name} already covered.
                """
            )

        assert isinstance(cut, int)
        assert isinstance(x, int)

        if x > cut:
            raise RuntimeError(
                f"""
                {decl}: splitting doesn't fully cover {axis_name} (jumps from {cut} to {x}).
                """
            )
        elif x < cut:
            raise RuntimeError(
                f"""
                {decl}: the segment begins at {x} but the previous one ends at {cut}.
                """
            )

        assert x is not None

        if y is not None and x >= y:
            raise RuntimeError(
                f"""
                {decl}: the ending point should be greater than the starting point.
                """
            )

    def _unknown_axis_error(self, axis: str):
        raise RuntimeError(
            f"""
            Axis {axis} is not a defined axis (defined axis: {self.abstract_axis}).
            """
        )

    def _check_axis_existence(self, axis: str):
        if axis not in self.abstract_axis:
            self._unknown_axis_error(axis)


def annotate(
    loop_name: str,
    sizes: dict[str, int],
    annotations: dict[str, Any],
    sched: LoopNestSlice,
):
    for instr, param in annotations.items():
        assert isinstance(instr, str)
        assert isinstance(param, int | None)
        match instr:
            case "unroll":
                hd = f'`{{"unroll" = {param}}}`'
                if param is None and loop_name not in sizes:
                    raise RuntimeError(
                        f"""
                        {hd}: {loop_name}'s size being unknown, an unroll factor is needed.
                        """
                    )
                elif param is not None and param <= 0:
                    raise RuntimeError(
                        f"""
                        {hd}: unroll parameter should be strictly positive.
                        """
                    )
                elif param and loop_name in sizes and sizes[loop_name] < param:
                    raise RuntimeError(
                        f"""
                        {hd}: unroll factor should be smaller than {sizes[loop_name]}.
                        """
                    )
                sched.unroll[loop_name] = sizes[loop_name] if param is None else param

            case "vectorize":
                if param is not None:
                    raise RuntimeError(
                        f"""
                        `{{\"vectorize\" = {param}}}`: parameterized vectorization not implemented.
                        """
                    )
                sched.vectorize.append(loop_name)

            case "parallelize":
                if param is not None:
                    raise RuntimeError(
                        f"""
                        `{{\"parallelize\" = {param}}}`: parameterized parallelization not implemented.
                        """
                    )

                sched.parallelize.append(loop_name)

            case _:
                raise RuntimeError(f"Unknown annotation on {loop_name}: {instr}")


def parse_split_declaration(declaration: str) -> Tuple[str, int | None, int | None]:
    pattern = r"^(.*)\[(?:(-\d+|\d*)?):(?:(-\d+|\d*)?)\]$"
    match = re.match(pattern, declaration)
    if not match:
        raise RuntimeError(f"Wrong format {declaration}")

    prefix, x_str, y_str = match.groups()
    x = int(x_str) if x_str else None
    y = int(y_str) if y_str else None
    return prefix, x, y
