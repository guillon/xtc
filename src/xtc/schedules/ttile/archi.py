#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

from enum import Enum
from dataclasses import dataclass


class Architype_enum(Enum):
    INTEL = 1
    ARM = 2
    AMD = 3


@dataclass
class Archi:
    name: str
    archi_type: Architype_enum

    # Register level
    vector_size: int  # In octets (since we do not know yet the datasize)
    num_vect_reg: int  # Number of vector register

    # Cache properties
    cache_l1_size: int
    cache_l1_assoc: int

    cache_l2_size: int
    cache_l2_assoc: int

    cache_l3_size: int
    cache_l3_assoc: int

    # Parallel above L2
    num_core: int


# ====================================================================

# Currently used architectures

# Intel(R) Xeon(R) Gold 6230R CPU
pinocchio_machine = Archi(
    "Pinocchio", Architype_enum.INTEL, 64, 32, 32768, 8, 1048576, 16, 37486592, 11, 52
)

# Intel(R) Core(TM) i5-8365U CPU
laptop_guillaume_machine = Archi(
    "Laptop_Gui", Architype_enum.INTEL, 32, 16, 32768, 8, 262144, 4, 6291456, 12, 8
)

# Note: add other machines here
