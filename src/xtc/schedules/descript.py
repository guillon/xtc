#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from dataclasses import dataclass
import re
from xtc.itf.schd.scheduler import Scheduler

SchedDict = dict[str, Any]


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
        flat_schedules = self._flatten_schedule(root=node_name, spec=spec)
        for schedule in flat_schedules:
            root = schedule["root"]
            self.scheduler.interchange(schedule["interchange"], root=root)
            for d, s in schedule["splits"].items():
                self.scheduler.split(d, s, root=root)
            for d, s in schedule["tiles"].items():
                self.scheduler.tile(d, s, root=root)
            self.scheduler.vectorize(schedule["vectorize"], root=root)
            self.scheduler.parallelize(schedule["parallelize"], root=root)
            self.scheduler.unroll(schedule["unroll"], root=root)

    def _flatten_schedule(
        self,
        root: str,
        spec: dict[str, dict],
    ) -> list[SchedDict]:
        recursive_scheds: list[SchedDict] = []
        sched: SchedDict = {
            "root": root,
            "splits": {},
            "tiles": {a: {} for a in self.abstract_axis},
            "interchange": [],
            "vectorize": [],
            "parallelize": [],
            "unroll": {},
        }
        # State of the schedule
        sizes: dict[str, int | None] = {}
        previous_cut: dict[str, int | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = []
        # Processing the schedule
        for declaration, val in spec.items():
            # Splits
            if ":" in declaration:
                axis_name, x, y = parse_split_declaration(declaration)
                # The only declaration where y (the cut) is None is the
                # last one, so it cannot be the previous one.
                assert previous_cut[axis_name] is not None
                # When x (the starting point of the slice), is not
                # specified, it is the previous cut
                if x is None:
                    x = previous_cut[axis_name]
                assert x is not None
                # Update the previous cut
                previous_cut[axis_name] = y
                # Save the cutting points of the new dimensions
                if not axis_name in sched["splits"]:
                    sched["splits"][axis_name] = {}
                new_dim_index = len(sched["splits"][axis_name])
                new_dim_name = f"{axis_name}[{new_dim_index}]"
                new_root_name = f"{root}/{new_dim_name}"
                sched["splits"][axis_name][new_dim_name] = x
                interchange.append(new_dim_name)
                # Fetch the schedule associated with the new dimension
                next_schedule = val
                assert isinstance(next_schedule, dict)
                inner_scheds = self._flatten_schedule(
                    spec=next_schedule, root=new_root_name
                )
                recursive_scheds += inner_scheds
                continue

            # Tiles
            elif "#" in declaration:
                axis_name, tile_size = declaration.split("#")
                loop_size = int(tile_size)
                tile_num = len(sched["tiles"][axis_name])
                loop_name = f"{axis_name}{tile_num}"
                sched["tiles"][axis_name][loop_name] = loop_size
                sizes[loop_name] = loop_size
                interchange.append(loop_name)

            elif declaration in self.abstract_axis:
                loop_name = declaration
                interchange.append(loop_name)

            else:
                assert False

            annotate(loop_name=loop_name, sizes=sizes, annotations=val, sched=sched)
        #
        sched["interchange"] = interchange
        return [sched] + recursive_scheds


def annotate(
    loop_name: str,
    sizes: dict[str, int | None],
    annotations: dict[str, Any],
    sched: dict[str, Any],
):
    for instr, param in annotations.items():
        assert isinstance(instr, str)
        assert isinstance(param, int | None)
        match instr:
            case "unroll":
                ufactor = sizes[loop_name] if param is None else param
                assert isinstance(ufactor, int)
                sched["unroll"][loop_name] = ufactor
            case "vectorize":
                sched["vectorize"].append(loop_name)
            case "parallelize":
                sched["parallelize"].append(loop_name)
            case _:
                raise Exception(f"Unknown annotation on {loop_name}: {instr}")


def parse_split_declaration(declaration: str) -> Tuple[str, int | None, int | None]:
    pattern = r"^(.*)\[(?:(\d*)?):(?:(\d*)?)\]$"
    match = re.match(pattern, declaration)
    if not match:
        raise ValueError("Wrong format.")
    prefix, x_str, y_str = match.groups()
    x = int(x_str) if x_str else None
    y = int(y_str) if y_str else None
    return prefix, x, y
