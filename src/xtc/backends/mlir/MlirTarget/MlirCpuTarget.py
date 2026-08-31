#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from typing import Any
from typing_extensions import override
import subprocess
import os
import sys
import tempfile
import shutil
from pathlib import Path

from xtc.utils.host_tools import disassemble
from xtc.utils.ext_tools import (
    get_shlib_extension,
    shared_lib_opts,
    exe_opts,
    runtime_libs,
    system_libs,
    cc_bin,
)

from xtc.targets.host import HostModule
import xtc.itf as itf
from xtc.itf.graph import Graph

from mlir.ir import OpResult

from .MlirTarget import MlirTarget
from ..MlirProgram import RawMlirProgram


class MlirCpuTarget(MlirTarget):
    """Common base for the CPU targets (LLVM and C).

    It drives the host-side compilation (lowering -> object file -> shared lib /
    executable) that is identical across CPU targets. Concrete targets only
    provide `_lower_and_compile_object`, which lowers the MLIR module and
    compiles it down to an object file using the target-specific toolchain.
    """

    @override
    def arch(self) -> str:
        return "cpu"

    @abstractmethod
    def _lower_and_compile_object(
        self,
        mlir_program: RawMlirProgram,  # Modified in place
        dump_tmp_file: str,
        dump_base: str,
        obj_dump_file: str,
    ) -> list[str]:
        """Lower the module and compile it to ``obj_dump_file``.

        Args:
            mlir_program: the program to lower (modified in place)
            dump_tmp_file: path prefix for target-specific intermediate files
            dump_base: base name used for ``save_temps`` dumps
            obj_dump_file: object file the target must produce

        Returns:
            The intermediate files to remove when ``save_temps`` is off.
        """
        ...

    @override
    def generate_code_for_target(
        self,
        mlir_program: RawMlirProgram,  # Will be modified in place
        **kwargs: Any,
    ) -> None:
        temp_dir = None
        dump_file = kwargs.get("dump_file", None)
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/payload"
        if self._config.save_temps:
            assert dump_file is not None, "TODO: save_temp requires dump_file"
            dump_tmp_dir = Path(self._config.save_temps_dir)
            os.makedirs(self._config.save_temps_dir, exist_ok=True)
        else:
            dump_tmp_dir = Path(dump_file).parent
        dump_base = Path(dump_file).name

        dump_tmp_file = f"{dump_tmp_dir}/{dump_base}"
        obj_dump_file = f"{dump_tmp_file}.o"
        exe_c_file = f"{dump_tmp_file}.main.c"
        so_dump_file = f"{dump_file}.{get_shlib_extension()}"
        exe_dump_file = f"{dump_file}.out"

        # Target-specific lowering + compilation down to an object file.
        intermediate_files = self._lower_and_compile_object(
            mlir_program, dump_tmp_file, dump_base, obj_dump_file
        )

        if self._config.print_assembly:
            disassembly = disassemble(
                obj_dump_file,
                function=self._config.to_disassemble,
                arch=self._config.arch,
                color=self._config.color,
                visualize_jumps=self._config.visualize_jumps,
            )
            print(disassembly, file=sys.stderr)

        payload_objs = [obj_dump_file, *self.shared_libs]
        payload_path = [*self.shared_path]
        if self._config.shared_lib:
            shared_cmd = [
                *self.cmd_cc,
                *shared_lib_opts,
                obj_dump_file,
                "-o",
                so_dump_file,
                *self.shared_libs,
                *self.shared_path,
            ]
            shlib_process = self.execute_command(cmd=shared_cmd)
            assert shlib_process.returncode == 0

            payload_objs = [so_dump_file]
            payload_path = ["-Wl,-rpath,$ORIGIN"]

        if self._config.executable:
            exe_cmd = [
                *self.cmd_cc,
                *exe_opts,
                exe_c_file,
                "-o",
                exe_dump_file,
                *payload_objs,
                *payload_path,
            ]
            with open(exe_c_file, "w") as outf:
                outf.write("extern void entry(void); int main() { entry(); return 0; }")
            exe_process = self.execute_command(cmd=exe_cmd)
            assert exe_process.returncode == 0

        if not self._config.save_temps:
            for path in (obj_dump_file, exe_c_file, *intermediate_files):
                Path(path).unlink(missing_ok=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir)

    @override
    def create_module(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> itf.comp.Module:
        return HostModule(name, payload_name, file_name, file_type, graph, **kwargs)

    @override
    def has_custom_vectorize(self) -> bool:
        return False

    @override
    def apply_custom_vectorize(self, handle: OpResult) -> None:
        return

    def dump_ir(self, mlir_program: RawMlirProgram, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(mlir_program.mlir_module), file=sys.stderr)

    @property
    def cmd_cc(self):
        return [cc_bin]

    @property
    def shared_libs(self):
        return system_libs + [
            f"{self._config.mlir_install_dir}/lib/{lib}" for lib in runtime_libs
        ]

    @property
    def shared_path(self):
        return [f"-Wl,-rpath,{self._config.mlir_install_dir}/lib/"]

    def _save_temp(self, fname: str, content: Any) -> None:
        if not self._config.save_temps:
            return
        os.makedirs(self._config.save_temps_dir, exist_ok=True)
        with open(f"{self._config.save_temps_dir}/{fname}", "w") as outf:
            outf.write(str(content))

    def execute_command(
        self,
        cmd: list[str],
        input_pipe: str | None = None,
        pipe_stdoutput: bool = True,
    ) -> subprocess.CompletedProcess:
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        if self._config.debug:
            print(f"> exec: {pretty_cmd}", file=sys.stderr)

        if input_pipe and pipe_stdoutput:
            result = subprocess.run(
                cmd, input=input_pipe, stdout=subprocess.PIPE, text=True
            )
        elif input_pipe and not pipe_stdoutput:
            result = subprocess.run(cmd, input=input_pipe, text=True)
        elif not input_pipe and pipe_stdoutput:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        else:
            result = subprocess.run(cmd, text=True)
        return result
