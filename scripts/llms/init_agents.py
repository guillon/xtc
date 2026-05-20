#!/usr/bin/env python3
"""
Generates AGENTS.md or CLAUDE.md from a markdown file by:
- Replacing the main title with a format-specific header
- Removing specified sections
"""

import sys
import re
from typing import Literal

DocFormat = Literal["agents", "claude"]

DOC_CONFIG: dict[DocFormat, tuple[str, str]] = {
    "agents": (
        "# AGENTS.md",
        "This file provides guidance to LLM agents when working with code in this repository.",
    ),
    "claude": (
        "# CLAUDE.md",
        "This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.",
    ),
}


def count_h1(lines: list[str]) -> int:
    """Count level-1 headings, ignoring code blocks."""
    in_code = False
    count = 0
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^# [^#]", line):
            count += 1
    return count


def transform_markdown(lines: list[str], skip_sections: list[str]) -> list[str]:
    """Transform markdown: skip H1 and filter out specified sections."""
    result = []
    in_code = False
    skip = False
    found_h1 = False

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code

        if not in_code:
            # Level-1 heading: enable output but skip this line
            if re.match(r"^# [^#]", line):
                found_h1 = True
                continue

            # Other headings: check if section should be skipped
            if re.match(r"^##+ ", line):
                skip = any(section in line for section in skip_sections if section)

        if found_h1 and not skip:
            result.append(line)

    return result


def main() -> int:
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <agents|claude> <markdown-file> [sections-to-skip...]",
            file=sys.stderr,
        )
        return 1

    doc_format = sys.argv[1]
    if doc_format not in DOC_CONFIG:
        print(
            f"Error: unknown format {doc_format!r}, expected 'agents' or 'claude'",
            file=sys.stderr,
        )
        return 1

    file_path = sys.argv[2]
    skip_sections = sys.argv[3:]

    with open(file_path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    h1_count = count_h1(lines)
    if h1_count != 1:
        print(
            f"Error: expected exactly one level-1 heading, found {h1_count}",
            file=sys.stderr,
        )
        return 1

    title, intro = DOC_CONFIG[doc_format]  # type: ignore[index]
    print(title)
    print()
    print(intro)
    print()

    for line in transform_markdown(lines, skip_sections):
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
