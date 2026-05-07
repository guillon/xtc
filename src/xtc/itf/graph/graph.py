#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence, Mapping
from .node import Node
from ..data import TensorType, Tensor


class Graph(ABC):
    """An abstract representation of a dataflow graph over Tensor types.

    A Graph is a directed acyclic graph (DAG) over Node objects with input Tensors
    and output Tensors. From given input Tensor dimensions and type, all node inputs,
    outputs and graph outputs Tensor dimensions and types can be inferred.

    A Graph can be evaluated by interpretation or compiled through various backends
    (mlir, tvm, jir) using corresponding Backend, Scheduler, and Compiler classes.

    Nodes in the graph are keyed by their uid which is globally unique.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of this graph. May be non-unique or empty.

        Returns:
            The graph's name
        """
        ...

    @property
    @abstractmethod
    def nodes(self) -> Mapping[str, Node]:
        """Returns a dictionary of all nodes in the graph, keyed by node uid.

        Returns:
            Dictionary mapping node names to Node objects
        """
        ...

    @property
    @abstractmethod
    def inputs(self) -> Sequence[str]:
        """Returns the list of input tensor uids for this graph.

        Returns:
            List of input tensor uids
        """
        ...

    @property
    @abstractmethod
    def outputs(self) -> Sequence[str]:
        """Returns the list of output tensor uids for the graph.

        Returns:
            List of output tensor uids
        """
        ...

    @property
    @abstractmethod
    def inputs_types(self) -> Sequence[TensorType] | None:
        """Returns the list of inputs tensor types

        Returns None when no input tensor type was given
        and forward_types was not called.

        Returns:
            List of input tensor types or None if undef
        """
        ...

    @property
    def outputs_types(self) -> Sequence[TensorType] | None:
        """Returns the list of outputs tensor types

        Returns None when no input tensor type was given
        and forward_types was not called.

        Returns:
            List of output tensor types or None if undef
        """
        ...

    @property
    @abstractmethod
    def inputs_nodes(self) -> Sequence[Node]:
        """Returns the list of inputs nodes.

        The list is such that:
        - nodes appear in the same order as outputs
        - nodes appear only once in the list, hence
          the size may differe from the inputs size

        Returns:
            List of input nodes
        """
        ...

    @property
    @abstractmethod
    def outputs_nodes(self) -> Sequence[Node]:
        """Returns the list of output nodes.

        The list is such that:
        - nodes appear in the same order as outputs
        - nodes appear only once in the list, hence
          the size may differe from the outputs size

        Returns:
            List of output nodes
        """
        ...

    @abstractmethod
    def forward_types(self, inputs_types: Sequence[TensorType]) -> Sequence[TensorType]:
        """Infers output tensor types from input tensor types.

        Args:
            inputs_types: List of input tensor types

        Returns:
            List of inferred output tensor types
        """
        ...

    @abstractmethod
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[Tensor]:
        """Evaluate the graph with input tensors to produce output tensors.

        Args:
            inputs: List of input tensors

        Returns:
            List of output tensors
        """
        ...

    @abstractmethod
    def ops_count(self) -> int:
        """Returns an estimate of the operation count for
        the whole graph. Used for peak performance estimates.

        This may not return an overestimate, only a lower bound.

        For instance on a graph with matmul->relu
        with dimensions i, j, k, this returns i*j*k + i*j.

        Returns:
            Estimated graph ops count
        """
        ...

    @abstractmethod
    def ops_dtype(self) -> str:
        """Returns the datatype string for the estimated
        ops count.

        This is possibly an approximation when the graph
        contains multiple nodes type. In this case the
        ops_count may return for instance a lower count
        and ensure they match the dtype.

        Returns:
            Data type string of the main graph ops
        """
        ...
