#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from pathlib import Path


def relative_to(file: str | Path, directory: str | Path) -> Path:
    """Return `file` relative to `directory` if contained within it, otherwise absolute."""
    file = Path(file).resolve()
    directory = Path(directory).resolve()
    try:
        return file.relative_to(directory)
    except ValueError:
        return file
