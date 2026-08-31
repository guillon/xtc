#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..graph.graph import Graph
from ..schd.scheduler import Scheduler
from ..schd.schedule import Schedule
from ..comp.compiler import Compiler


class Backend(ABC):
    """An abstract implementation of specific Graph implementation.

    A Backend is constructed from an input Graph and provides backend-specific
    implementations of the graph operations. It serves as a bridge between the abstract
    graph representation and concrete backend implementations (e.g., MLIR, TVM, JIR).

    The Implementer provides access to associated Scheduler and Compiler instances
    for applying transformations and generating executable code.
    """

    @property
    @abstractmethod
    def payload_name(self) -> str:
        """Name of the payload (entry point) implemented by this backend."""
        ...

    @abstractmethod
    def get_scheduler(self, **kwargs: Any) -> Scheduler:
        """Returns the scheduler associated with this implementation.

        Args:
            kwargs: scheduler configuration

        Returns:
            The scheduler for applying transformations
        """
        ...

    @abstractmethod
    def get_compiler(self, **kwargs: Any) -> Compiler:
        """Returns the compiler associated with this implementation.

        Args:
            kwargs: compiler configuration

        Returns:
            The compiler for generating executable code
        """
        ...

    @property
    @abstractmethod
    def graph(self) -> Graph:
        """Returns the graph being implemented.

        Returns:
            The source graph for this implementation
        """
        ...

    def evaluate(
        self,
        schedule: Schedule,
        compiler_args: dict[str, Any] | None = None,
        evaluate_args: dict[str, Any] | None = None,
    ) -> float | str:
        """Compile and evaluate the given schedule as a shared library.

        Args:
            schedule: the schedule to compile and run
            compiler_args: extra keyword arguments forwarded to the compiler
            evaluate_args: extra keyword arguments forwarded to the evaluator

        Returns:
            The best (minimum) measured runtime on success, or the error
            message when the run failed.
        """
        compiler_args = compiler_args or {}
        evaluate_args = evaluate_args or {}
        with tempfile.TemporaryDirectory() as dirname:
            libpath = Path(dirname) / f"payload_{self.payload_name}"
            compiler = self.get_compiler(
                dump_file=str(libpath),
                shared_lib=True,
                **compiler_args,
            )
            module = compiler.compile(schedule)
            evaluator = module.get_evaluator(
                validate=True,
                **evaluate_args,
            )
            results, code, error_msg = evaluator.evaluate()
        return min(results) if code == 0 else error_msg
