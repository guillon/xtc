#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import argparse
import difflib

import tomlkit


ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES_FILE = ROOT / "dependencies.toml"
PYPROJECT_FILE = ROOT / "pyproject.toml"


def resolve_group(
    name: str,
    groups: dict[str, Any],
    stack: tuple[str, ...] = (),
) -> tuple[bool, list[str]]:
    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise ValueError(f"Circular dependency group inclusion: {cycle}")

    try:
        group = groups[name]
    except KeyError:
        raise ValueError(f"Unknown dependency group: {name!r}") from None

    dependencies: list[str] = []

    for included_group in group.get("include", []):
        dependencies.extend(
            resolve_group(
                included_group,
                groups,
                (*stack, name),
            )[1]
        )

    dependencies.extend(group.get("dependencies", []))

    # Preserve order while removing duplicates.
    dependencies = list(dict.fromkeys(dependencies))
    publish = group.get("publish", True)
    return publish, dependencies


def pyproject_dumps() -> str:
    dependencies_doc = tomlkit.parse(DEPENDENCIES_FILE.read_text())
    pyproject = tomlkit.parse(PYPROJECT_FILE.read_text())

    project = pyproject["project"]

    # Base dependencies
    project["dependencies"] = dependencies_doc.get("dependencies", [])

    # Optional dependencies
    groups = dependencies_doc.get("groups", {})
    optional_dependencies = tomlkit.table()

    for name in groups:
        publish, dependencies = resolve_group(name, groups)
        if publish:
            optional_dependencies[name] = dependencies

    project["optional-dependencies"] = optional_dependencies

    return tomlkit.dumps(pyproject)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update pyproject.toml from dependencies.toml"
    )
    parser.add_argument("--check", action="store_true", help="check only")

    args = parser.parse_args()

    pyproject_str = pyproject_dumps()

    tmp_file = Path(f"{PYPROJECT_FILE}.tmp")
    tmp_file.write_text(pyproject_str)
    try:
        if args.check:
            diff = list(
                difflib.unified_diff(
                    PYPROJECT_FILE.read_text().splitlines(keepends=True),
                    tmp_file.read_text().splitlines(keepends=True),
                    fromfile=str(PYPROJECT_FILE),
                    tofile=str(tmp_file),
                )
            )
            if diff:
                print("".join(diff), file=sys.stderr)
                raise SystemExit(f"ERROR: output differs for {PYPROJECT_FILE}")
        else:
            tmp_file.replace(PYPROJECT_FILE)
    finally:
        tmp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
