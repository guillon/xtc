#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import importlib.metadata
from typing import Any


def on_config(config: Any) -> Any:
    version = importlib.metadata.version("xtc-tools")
    config["site_name"] = f"XTC {version}"
    config.setdefault("extra", {})["xtc_version"] = version
    return config
