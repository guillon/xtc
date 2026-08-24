#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
__all__ = ["make_loop_name", "basename", "parent_name", "path_names", "rooted_name"]

_LOOP_SEP = "/"
_LOOP_ROOT = "."


def make_loop_name(root: str, name: str) -> str:
    return f"{root}{_LOOP_SEP}{name}"


def basename(loop_name: str) -> str:
    return loop_name.split(_LOOP_SEP)[-1]


def parent_name(loop_name: str) -> str:
    return loop_name.rsplit(_LOOP_SEP, 1)[0]


def path_names(loop_name: str) -> list[str]:
    return loop_name.split(_LOOP_SEP)


def rooted_name(loop_name: str, root: str) -> str:
    assert root != _LOOP_ROOT
    if loop_name == _LOOP_ROOT:
        return root
    prefix = f"{_LOOP_ROOT}{_LOOP_SEP}"
    if loop_name.startswith(prefix):
        return make_loop_name(root, loop_name[len(prefix) :])
    return loop_name
