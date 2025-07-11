#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os

from typing import List, Dict, Tuple
from xtc.schedules.ttile.scheme import (
    AtomType,
    Atom,
    get_sizes_scheme,
    get_unlambda_sizes_scheme,
    build_scheme_from_str,
)
from xtc.schedules.ttile.computation import (
    Computation,
    Computation_spec,
    compute_number_ops,
    get_parallel_dims,
    get_reduction_dims,
)
from xtc.schedules.ttile.archi import Archi

from xtc.utils.cpu import cpu_peak_time

# BEGIN of potential manual modifications
temp_file_name = "/tmp/__temp_xdsl_transf_input"
temp_file_measurement_name = "/tmp/__temp_xdsl_transf_time"
# END of potential manual modifications


# Convert an input "Ttile-style" scheme into an input of xdsl-transform
#   Also contain the interfacing to launch xdsl-transform on it


# ===========================================================================

# 1) Conversion fo a Scheme into an input for "xdsl-transform"


# [Aux function] Build the attributes of an operations, and place them inside a mlir AST node.
# Attributes asked on an operation node to make "xdsl-transform" work:
#   - parallel/reduction dims (inherent properties of the computation)
#   - tile_name/sizes : need to assign names to each intermediate tile level
#   - interchange: deducable from the scheme (careful about the order: outer to inner, not inner to outer)
#   - vectorize : get the corresponding name
#   - unroll : get the corresponding names
def convert_scheme_to_xdsl_transform_input(
    scheme: List[Atom], comp: Computation
) -> Tuple[
    List[str],
    List[str],
    dict[str, List[str]],
    List[Tuple[str, int]],
    List[str],
    List[str],
    List[str],
    List[Tuple[str, int]],
]:
    parallel_dims = get_parallel_dims(comp)
    reduction_dims = get_reduction_dims(comp)

    # Dict: [dim_name] -> [number of time it appears in a scheme]
    dimname_vect = None
    d_num_loops = dict()
    for atom in scheme:
        if atom.type in [AtomType.ULAMBDA, AtomType.TLAMBDA, AtomType.SEQ]:
            raise ValueError(
                f"convert_scheme_to_xdsl_transform_input : atomType {atom.type} is not supported."
            )

        # No loop variable for hoisting
        if atom.type in [AtomType.HOIST]:
            continue

        # Keep track of the single vectorization dim
        if atom.type == AtomType.VECT:
            assert dimname_vect == None
            dimname_vect = atom.dim

        # Counting the number of loops for each dimension
        if atom.dim not in d_num_loops:
            d_num_loops[atom.dim] = 1
        else:
            d_num_loops[atom.dim] += 1

    # Special case where the vect dim does not appear in the rest of the scheme:
    #   We still need to have the dimension name somewhere, even if no tilename var will be created
    if (dimname_vect != None) and (dimname_vect not in d_num_loops):
        d_num_loops[dimname_vect] = 1

    # DEBUG
    # print(f"{d_num_loops=}")

    # Tiles names convention: [dimname][num, starting by 1], the loop
    dtiles_name = dict()
    for dim in d_num_loops:  # Last variable to be used is the dimension name itself
        dtiles_name[dim] = [dim]
    # This first variable will be filter out for the final output
    for dim in d_num_loops:
        for i in range(d_num_loops[dim] - 1):
            dtiles_name[dim].append(f"{dim}{i + 1}")

    # DEBUG
    # print(f"{dtiles_name=}")

    # Tile sizes + interchange
    ldim_vectorize = []
    l_dim_parallelized = []
    ltilesizes = []
    llooporder = []  # Note: append in reverse order
    lunroll = []

    d_num_loops_encountered = (
        dict()
    )  # To keep track the number of time a dim is encountered
    for dim in d_num_loops:
        d_num_loops_encountered[dim] = 1

    # For each atom of the scheme...
    for i in range(len(scheme)):
        atom = scheme[i]

        if atom.type == AtomType.HOIST:
            continue

        # Partial tile management: like for TILE/UNROLL, but we have a size instead of a ratio
        if atom.type == AtomType.TILE_PARTIAL:
            current_dim = atom.dim
            current_size = atom.size

            current_loop_var = dtiles_name[current_dim][
                -d_num_loops_encountered[current_dim]
            ]
            d_num_loops_encountered[current_dim] += 1

            if current_loop_var not in d_num_loops.keys():
                ltilesizes.append((current_loop_var, current_size))

            llooporder.append(current_loop_var)
            continue

        # atom.type must be TILE, UNROLL
        assert atom.type in [
            AtomType.TILE,
            AtomType.UNROLL,
            AtomType.TILE_PARAL,
            AtomType.VECT,
        ]
        current_dim = atom.dim
        current_ratio = atom.ratio

        current_loop_var = dtiles_name[current_dim][
            -d_num_loops_encountered[current_dim]
        ]
        d_num_loops_encountered[current_dim] += 1

        if atom.type == AtomType.VECT:
            ldim_vectorize.append(current_loop_var)

        if atom.type == AtomType.TILE_PARAL:
            l_dim_parallelized.append(current_loop_var)

        if atom.type == AtomType.UNROLL:
            # Unroll argument of vectorized dimensions are ignored by xdsl-transform
            # NOTE: This specification was cancelled later on, due to limitation of the vectorization of llvm.
            # if (current_loop_var not in ldim_vectorize):

            lunroll.append((current_loop_var, current_ratio))

        # If current_loop_var is not the last one (which has the name of the dim),
        #   we have to add the size of the current tile to ltilesizes.
        # This is done by ocmputing the sizes of the sub-scheme finishing with the current atom
        if current_loop_var not in d_num_loops.keys():
            d_lsize = get_sizes_scheme(scheme[: i + 1])
            assert len(d_lsize[current_dim]) == 1
            size_current_dim = d_lsize[current_dim][0]

            ltilesizes.append((current_loop_var, size_current_dim))

        # Loop order is inner to outer here.
        llooporder.append(current_loop_var)

    # Ttile scheme are innermost to outermost loops. xdsl-transform is outer to inner.
    llooporder.reverse()

    # Now, we just need to filter out from dtiles_name the original dimension sizes
    #   These are always the first element of each list
    ndtiles_name = dict()
    for dim in dtiles_name:
        nl = dtiles_name[dim][1:]
        if nl != []:
            ndtiles_name[dim] = nl
    dtiles_name = ndtiles_name

    # DEBUG
    """
  print(f"parallel_dims = {parallel_dims}")
  print(f"reduction_dims = {reduction_dims}")
  print(f"dtiles_name = {dtiles_name}")
  print(f"ltilesizes = {ltilesizes}")
  print(f"llooporder = {llooporder}")
  print(f"ldim_vectorize = {ldim_vectorize}")
  print(f"lunroll = {lunroll}")
  """

    # At that point, all informations are computed for the interface with xdsl-transform
    return (
        parallel_dims,
        reduction_dims,
        dtiles_name,
        ltilesizes,
        llooporder,
        ldim_vectorize,
        l_dim_parallelized,
        lunroll,
    )


# [Aux function] Because xdsl-transform input changed in the meantimes, merge the infos of 2 datastructures
def merge_dtiles_name_ltilesizes(
    dtiles_name: dict[str, List[str]], ltilesizes: List[Tuple[str, int]]
) -> dict[str, dict[str, int]]:
    dd_tilenamesize = dict()
    for orig_dim in dtiles_name.keys():
        d_tilenamesize = dict()

        ltiles_name = dtiles_name[orig_dim]
        for var_name in ltiles_name:
            # Find the name back in ltilesizes + their associated size
            size_var_name = None
            for ltuple in ltilesizes:
                if ltuple[0] == var_name:
                    size_var_name = ltuple[1]
                    break
            assert (
                size_var_name != None
            )  # If this triggers, then issue in ltilesizes/dtiles_name

            # Commit
            d_tilenamesize[var_name] = size_var_name

        # Commit
        dd_tilenamesize[orig_dim] = d_tilenamesize

    return dd_tilenamesize


# [Aux function - New syntax] Convert a Ttile scheme into a loop.schedule string argument
def convert_scheme_to_loopschedule(comp: Computation, machine: Archi, scheme) -> str:
    # One atom = one element in loop.schedule
    # => We generate their corresponding string, then we will assemble them (in the correct order) later
    lstr_lpsch_atom = []
    ldim_alreadyseen = []
    for i in range(len(scheme)):
        atom = scheme[-i - 1]

        # Not supported (yet?) by xtc
        assert atom.type not in {AtomType.ULAMBDA, AtomType.TLAMBDA, AtomType.SEQ}

        if atom.type == AtomType.HOIST:
            continue

        # Vectorization
        if atom.type == AtomType.VECT:
            dim = atom.dim
            size = atom.ratio  # Vectorization must be the innermost part of a loop
            if dim in ldim_alreadyseen:
                str_tilesize = f"#{size}"
            else:
                ldim_alreadyseen.append(dim)
                str_tilesize = ""
            str_atom = f'"{dim}{str_tilesize}" = ' + '{"vectorize"}'

            lstr_lpsch_atom.append(str_atom)

        elif atom.type in {AtomType.UNROLL, AtomType.TILE, AtomType.TILE_PARAL}:
            bunroll = atom.type == AtomType.UNROLL
            bparal = atom.type == AtomType.TILE_PARAL

            dim = atom.dim
            d_lsizes = get_sizes_scheme(scheme[0 : (len(scheme) - i - 1)])
            if dim in d_lsizes:
                lsize = d_lsizes[dim]
                assert len(lsize) == 1
                size = lsize[0] * atom.ratio
            else:
                size = 1 * atom.ratio
            if dim in ldim_alreadyseen:
                str_tilesize = f"#{size}"
            else:
                ldim_alreadyseen.append(dim)
                str_tilesize = ""
            str_atom = f'"{dim}{str_tilesize}"'
            if bunroll or bparal:
                str_atom += " = {"
                if bunroll:
                    str_atom += '"unroll"'
                if bparal:
                    str_atom += '"parallel"'
                str_atom += "}"
            lstr_lpsch_atom.append(str_atom)

        elif atom.type == AtomType.TILE_PARTIAL:
            dim = atom.dim
            size = atom.size
            if dim in ldim_alreadyseen:
                str_tilesize = f"#{size}"
            else:
                ldim_alreadyseen.append(dim)
                str_tilesize = ""
            str_atom = f'"{dim}{str_tilesize}"'
            lstr_lpsch_atom.append(str_atom)
        else:
            raise ValueError(
                "convert_scheme_to_loopschedule : Unrecognized atom type {atom.type}"
            )

    # DEBUG
    # print(lstr_lpsch_atom)

    # Assembly!
    str_loop_sched = ""
    for i in range(len(lstr_lpsch_atom)):
        str_lpsch_atom = lstr_lpsch_atom[i]
        if i > 0:
            str_loop_sched += ", "
        str_loop_sched += str_lpsch_atom

    # DEBUG
    # print(str_loop_sched)

    return str_loop_sched


# Replicate the "avx2_matmult.mlir" from xdsl-transform
# MATMULT: linalg.matmul
#     https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop
def build_xdsl_module_string_matmul(
    comp: Computation,
    machine: Archi,
    scheme: List[Atom],
    dsizes: dict[str, int],
    b_initOut: bool = False,
) -> str:
    # Recover the individual problem sizes
    default_sizes = {"i": 1, "j": 1, "k": 1}
    dsizes = default_sizes | dsizes
    size_i = dsizes["i"]
    size_j = dsizes["j"]
    size_k = dsizes["k"]

    if comp.elem_size == 4:
        fXX = "f32"
    elif comp.elem_size == 8:
        fXX = "f64"
    else:
        raise ValueError(
            f"build_xdsl_module_op_matmul :: unknown type of float of size {comp.elem_size} octets"
        )

    num_elem_per_vec = int(machine.vector_size / comp.elem_size)  # Both are in octets

    # Model: "avx2_matmul.mlir"
    str_output = f"func.func @myfun(\n"
    str_output += f"  %A: memref<{size_i}x{size_k}x{fXX}>,\n"
    str_output += f"  %B: memref<{size_k}x{size_j}x{fXX}>,\n"
    str_output += f"  %C: memref<{size_i}x{size_j}x{fXX}>\n"
    str_output += ") {\n"

    # Initialization of the output
    if b_initOut:
        str_output += f"  %cst = arith.constant 0.000 : {fXX}\n"
        str_output += f"  linalg.fill\n"
        str_output += "    {\n"
        str_output += '      loop.dims = ["i", "j"],\n'
        # str_output += f"      loop.parallel_dims = [\"i\",\"j\"],\n"
        # str_output += f"      loop.reduction_dims = [],\n"
        str_output += "      loop.tiles_names = {"
        str_output += f'"j" = ["j1"]'
        str_output += "},\n"
        str_output += "      loop.tiles_sizes = {"
        str_output += f"j1 = {num_elem_per_vec}"
        str_output += "},\n"
        str_output += f'      loop.interchange = ["i","j","j1"],\n'
        str_output += f'      loop.vectorize = ["j1"]\n'
        # Bad idea to add a loop parallelize on i here  (tank the perfs)
        str_output += "    }\n"
        str_output += f"    ins(%cst : {fXX})\n"
        str_output += f"    outs(%C : memref<{size_i}x{size_j}x{fXX}>)\n"

    str_output += f"  linalg.matmul\n"
    str_output += "    {\n"

    # This is the part where we plug the scheme in the text
    str_output += '      loop.dims = ["i","j","k"],\n'
    str_loop_sched = convert_scheme_to_loopschedule(comp, machine, scheme)
    str_output += "      loop.schedule = {"
    str_output += f"{str_loop_sched}"
    str_output += "}\n"
    str_output += "    }\n"
    str_output += f"    ins(%A, %B : memref<{size_i}x{size_k}x{fXX}>, memref<{size_k}x{size_j}x{fXX}>)\n"
    str_output += f"    outs(%C : memref<{size_i}x{size_j}x{fXX}>)\n"
    str_output += f"  return\n"
    str_output += "}\n"

    return str_output


# [Aux functions] Replace the F/C/X/Y/H/W notation to F/C/H/W/R/S for conv2D
def _rename_xyhw_to_hwrs(dim_name: str) -> str:
    if dim_name == "x":
        return "h"
    elif dim_name == "y":
        return "w"
    elif dim_name == "h":
        return "r"
    elif dim_name == "w":
        return "s"
    else:
        return dim_name


def subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme: List[Atom]) -> List[Atom]:
    nscheme = []
    for atom in scheme:
        atom.dim = _rename_xyhw_to_hwrs(atom.dim)
        nscheme.append(atom)
    return nscheme


def subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes: dict[str, int]) -> dict[str, int]:
    dsizes_renamed = dict()
    for k in dsizes:
        nk = _rename_xyhw_to_hwrs(k)
        dsizes_renamed[nk] = dsizes[k]
    return dsizes_renamed


# Same than above, but for convolution, for xdsl-transform
# CONV: linalg.conv_2d_nhwc_hwcf
#     https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nhwc_hwcf-linalgconv2dnhwchwcfop
def build_xdsl_module_string_conv(
    comp, machine, scheme, dsizes, b_initOut=False
) -> str:
    # There are several naming notation for the conv2D image/kernel dimensions:
    #   X / Y + H / W
    # or
    #   H / W + R / S
    # => Preprocessing the first set into the second one, to uniform notations
    b_conv_dim_renaming = False
    for atom in scheme:
        # Ignore the atom with no dims
        if atom in [AtomType.HOIST]:
            continue
        if atom.dim in ["x", "y"]:
            b_conv_dim_renaming = True
    if b_conv_dim_renaming:
        scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
        dsizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes)

    if "strx" in dsizes:
        assert dsizes["strx"] == 1  # TODO: Stride not managed yet.
    if "stry" in dsizes:
        assert dsizes["stry"] == 1  # TODO: Stride not managed yet.

    # Recover the individual problem sizes
    size_n = 1  # Batch dim == 1

    default_sizes = {"n": 1, "f": 1, "c": 1, "h": 1, "w": 1, "r": 1, "s": 1}
    dsizes = default_sizes | dsizes

    # Output/Input feature dimensions
    size_f = dsizes["f"]
    size_c = dsizes["c"]
    size_h = dsizes["h"]
    size_w = dsizes["w"]
    size_r = dsizes["r"]
    size_s = dsizes["s"]

    if comp.elem_size == 4:
        fXX = "f32"
    elif comp.elem_size == 8:
        fXX = "f64"
    else:
        raise ValueError(
            f"build_xdsl_module_op_matmul :: unknown type of float of size {comp.elem_size} octets"
        )

    num_elem_per_vec = int(machine.vector_size / comp.elem_size)  # Both are in octets

    # Let's do it!
    str_output = f"func.func @myfun(\n"
    str_output += f"  %I: memref<{size_n}x{(size_h + size_r - 1)}x{(size_w + size_s - 1)}x{size_c}x{fXX}>,\n"
    str_output += f"  %K: memref<{size_r}x{size_s}x{size_c}x{size_f}x{fXX}>,\n"
    str_output += f"  %O: memref<{size_n}x{size_h}x{size_w}x{size_f}x{fXX}>\n"
    str_output += ") {\n"

    # Initialization of the output
    if b_initOut:
        str_output += f"  %cst = arith.constant 0.000 : {fXX}\n"
        str_output += f"  linalg.fill\n"
        str_output += "    {\n"
        str_output += '      loop.dims = ["n","h","w","f"],\n'
        # str_output += f"      loop.parallel_dims = [\"n\",\"h\",\"w\",\"f\"],\n"
        # str_output += f"      loop.reduction_dims = [],\n"
        str_output += "      loop.tiles_names = {"
        str_output += f'"f" = ["f1"]'
        str_output += "},\n"
        str_output += "      loop.tiles_sizes = {"
        str_output += f"f1 = {num_elem_per_vec}"
        str_output += "},\n"
        str_output += f'      loop.interchange = ["n","h","w","f","f1"],\n'
        str_output += f'      loop.vectorize = ["f1"]\n'
        # Note: this is a bad idea to add a loop.parallelize on n/h here
        str_output += "    }\n"
        str_output += f"    ins(%cst : {fXX})\n"
        str_output += (
            f"    outs(%O : memref<{size_n}x{size_h}x{size_w}x{size_f}x{fXX}>)\n"
        )

    # Version direct operator (not available yet in xdsl)
    # str_output += f"  linalg.conv_2d_nhwc_hwcf {\n"

    # [MOD] Version of "conv_2d_nhwc_hwcf" with a generic
    str_output += "  linalg.generic {\n"
    str_output += "      indexing_maps = [\n"
    str_output += "        affine_map<(n,h,w,f,r,s,c) -> (n,h+r,w+s,c)>,\n"
    str_output += "        affine_map<(n,h,w,f,r,s,c) -> (r,s,c,f)>,\n"
    str_output += "        affine_map<(n,h,w,f,r,s,c) -> (n,h,w,f)>],\n"
    str_output += (
        '      iterator_types = ["parallel", "parallel", "parallel", "parallel",\n'
    )
    str_output += '          "reduction", "reduction", "reduction"]\n'
    str_output += "    }\n"
    str_output += f"    ins(%I, %K : memref<{size_n}x{(size_h + size_r - 1)}x{(size_w + size_s - 1)}x{size_c}x{fXX}>,"
    str_output += f" memref<{size_r}x{size_s}x{size_c}x{size_f}x{fXX}>)\n"
    str_output += f"    outs(%O : memref<{size_n}x{size_h}x{size_w}x{size_f}x{fXX}>)\n"
    str_output += "     attrs = {\n"

    # This is the part where we plug the scheme in the text
    str_output += '      loop.dims = ["n","h","w","f","r","s","c"],\n'
    str_loop_sched = convert_scheme_to_loopschedule(comp, machine, scheme)
    str_output += "      loop.schedule = {"
    str_output += f"{str_loop_sched}"
    str_output += "}\n"
    str_output += "     }\n"

    # [MOD] Version of "conv_2d_nhwc_hwcf" with a generic
    str_output += "  {\n"
    str_output += f"    ^bb0(%0: {fXX}, %1: {fXX}, %2: {fXX}) :\n"
    str_output += f"      %3 = arith.mulf %0, %1 : {fXX}\n"
    str_output += f"      %4 = arith.addf %2, %3 : {fXX}\n"
    str_output += f"      linalg.yield %4 : {fXX}\n"
    str_output += "  }\n"
    str_output += f"  return\n"
    str_output += "}\n"

    return str_output


# ===========================================================================

# 2) Launch scheme execution & measurement through xdsl-transform "mlir-loops"
# Interface with Hugo's "mlir-loop" that generates transform dialect, must go through MLIR compilation pipeline


# Launch through the "mlir_loop.py" Python script - old syntax (since April 25)
def launch_and_measure_scheme(
    comp: Computation, machine: Archi, scheme: List[Atom], dsizes: dict[str, int]
):
    # Recover the text of the input
    if comp.spec == Computation_spec.MATMULT:
        str_input_xdsltransf = build_xdsl_module_string_matmul(
            comp, machine, scheme, dsizes
        )
    elif comp.spec == Computation_spec.CONV:
        str_input_xdsltransf = build_xdsl_module_string_conv(
            comp, machine, scheme, dsizes
        )
    else:
        raise ValueError("launch_and_measure_scheme : Unknown computation spec.")

    # Write it inside a temporary file
    finput = open(temp_file_name, "w")  # Fresh file
    finput.write(str_input_xdsltransf)
    finput.close()

    # Number of elements inside a vector
    vector_size_in_elem = int(machine.vector_size / comp.elem_size)

    # Launch mlir-loop (from xdsl-transform) on this input, with the right options
    cmd_xdsl_transf = f"mlir-loop {temp_file_name}"
    # cmd_xdsl_transf += f" --llvm-dir={llvm_build_installation}"
    cmd_xdsl_transf += (
        f" --evaluate --no-alias --init-zero --vectors-size {vector_size_in_elem} "
    )
    cmd_xdsl_transf += f" > {temp_file_measurement_name}"

    # DEBUG
    # print(cmd_xdsl_transf)

    os.system(cmd_xdsl_transf)

    # Note: possible to subst "mlir_loop" by "mlir_loop.py" if add the correct 2 lines at the end of its script.

    # Recover the time
    ftime_meas = open(temp_file_measurement_name, "r")
    res_measurement = dict()
    for line in ftime_meas:
        res_measurement["time"] = float(line)
    ftime_meas.close()

    # Get the peak_perf
    # xdsl_transform got utils.py :: cpu_peak_time(ops: int, dtype: str, threads: int = 1)
    num_ops = compute_number_ops(comp, dsizes)
    if comp.elem_size == 4:
        dtype = "float32"
    elif comp.elem_size == 8:
        dtype = "float64"
    else:
        raise ValueError(
            f"launch_and_measure_scheme :: unknown type of float of size {comp.elem_size} octets"
        )
    peak_time = cpu_peak_time(num_ops, dtype)

    # DEBUG
    # print(f"measurement = {res_measurement['time']}")
    # print(f"{peak_time=}")

    # Compute peak_perf
    peak_perf = peak_time / res_measurement["time"]
    res_measurement["peak_perf"] = peak_perf

    return res_measurement
