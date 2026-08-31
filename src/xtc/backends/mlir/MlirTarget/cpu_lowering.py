#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#


def cpu_frontend_lowering(mlir_extensions: list[str], *, uplift_fma: bool) -> list[str]:
    """Shared CPU lowering prefix for the LLVM and C targets.

    Lowers linalg/vector down to scf, before the target-specific transformations.

    Args:
        mlir_extensions: extensions required by the program
        uplift_fma: whether to uplift ``mul``+``add`` to ``fma``

    Returns:
        The ordered list of pass names for the shared front-end lowering.
    """
    pipeline = ["canonicalize", "cse", "sccp"]
    if "sdist" in mlir_extensions:
        pipeline += [
            "sdist-lower-distribution",
            "convert-sdist-to-std",
            "cse",
            "canonicalize",
            "convert-sdist-utils-to-std",
        ]
    if uplift_fma:
        pipeline.append("math-uplift-to-fma")
    pipeline += [
        # From complex control to the soup of basic blocks
        "expand-strided-metadata",
        "convert-linalg-to-loops",
        "lower-affine",
        "func.func(lower-vector-mask)",
        "convert-vector-to-scf{full-unroll=true}",
        "scf-forall-to-parallel",
        "convert-scf-to-openmp",
        "canonicalize",
        "cse",
        "sccp",
    ]
    return pipeline
