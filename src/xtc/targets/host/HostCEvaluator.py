#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from typing_extensions import override
import tempfile
from pathlib import Path
import shutil
import sys

from xtc_build import (
    BuildContext,
    ExternalArchive,
    ExternalSharedLibrary,
    Object,
    SharedLibrary,
)

import xtc.itf as itf
import xtc.targets.host as host
from xtc.utils.ext_tools import cc_opts


__all__ = [
    "HostCEvaluator",
    "HostCExecutor",
]


class HostCEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        assert module.file_type == "csrc", "must pass a csrc module to a HostCEvaluator"
        self._module = module
        self._build_shlib_module()
        self._shlib_evaluator = host.HostEvaluator(self._shlib_module, **kwargs)

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        return self._shlib_evaluator.evaluate()

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module

    def _compile_to_shlib(self, build_dir: Path, shlib_name: str) -> Path:
        sources = [self._module.file_name, *self._module.csrcs]
        includes = [
            *(Path(header).parent for header in self._module.headers),
            *(Path(path) for path in self._module.headers_path),
        ]
        includes = list(dict.fromkeys(includes))
        objects = [
            Object(
                name=f"{index:04d}_{Path(source).stem}",
                source=source,
                inputs=self._module.headers,
                includes=includes,
                compile_flags=cc_opts,
                pic=True,
            )
            for index, source in enumerate(sources)
        ]
        archives = [ExternalArchive(path, pic=True) for path in self._module.arlibs]
        libraries = [ExternalSharedLibrary(path) for path in self._module.shlibs]
        link_flags = [*cc_opts]
        if sys.platform == "darwin":
            link_flags.extend(["-undefined", "dynamic_lookup"])
        library = SharedLibrary(
            shlib_name,
            objects=objects,
            archives=archives,
            libraries=libraries,
            link_flags=link_flags,
        )
        return library.build(BuildContext(build_dir=build_dir))

    def _build_shlib_module(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        c_stem = Path(self._module.file_name).stem
        shlib_name = f"{c_stem}_eval"
        shlib_path = self._compile_to_shlib(Path(self.tmp_dir.name), shlib_name)
        self._shlib_module: host.HostModule = host.HostModule(
            shlib_name,
            self._module.payload_name,
            str(shlib_path),
            "shlib",
            bare_ptr=self._module._bare_ptr,
            graph=self._module._graph,
        )

    def __del__(self):
        shutil.rmtree(self.tmp_dir.name, ignore_errors=True)


class HostCExecutor(itf.exec.Executor):
    def __init__(self, module: "host.HostModule", **kwargs: Any) -> None:
        self._evaluator = HostCEvaluator(
            module=module,
            repeat=1,
            min_repeat_ms=0,
            number=0,
            **kwargs,
        )

    @override
    def execute(self) -> int:
        _, code, _ = self._evaluator.evaluate()
        return code

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._evaluator.module
