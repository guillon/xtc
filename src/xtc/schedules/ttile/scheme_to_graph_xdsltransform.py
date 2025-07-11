#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

# Interfacing with Christophe graph interface
# Inspired from the "test_conv2d_r181_mlir.py" file from xtc
from typing import List, Dict, Any

from xtc.schedules.ttile.scheme import (
    build_scheme_from_str,
    Atom,
    AtomType,
    convert_scheme_to_str,
)
from xtc.schedules.ttile.scheme import get_sizes_scheme
from xtc.schedules.ttile.scheme import (
    stringify_lambda_choice,
    recover_all_branchid_from_stringified,
)
from xtc.schedules.ttile.scheme import (
    recover_branchid_from_stringified,
    remove_dim_from_stringified,
    get_list_dims_str_lambda_loc,
)

from xtc.schedules.ttile.computation import (
    Computation,
    Computation_spec,
    compute_number_ops,
)
from xtc.schedules.ttile.computation import get_ldims_computation

from xtc.schedules.ttile.archi import Archi

from xtc.schedules.ttile.scheme_to_xdsltransform import (
    convert_scheme_to_xdsl_transform_input,
    merge_dtiles_name_ltilesizes,
)
from xtc.schedules.ttile.scheme_to_xdsltransform import (
    subst_dimname_xyhw_to_hwrs_conv2D_scheme,
    subst_dimname_xyhw_to_hwrs_conv2D_dsizes,
)

# Backends
import xtc.graphs.xtc.op as O
from xtc.itf.back.backend import Backend
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as MlirBackend
from xtc.backends.tvm import TVMBackend as TVMBackend

from xtc.schedules.descript import descript_scheduler

from xtc.utils.cpu import cpu_peak_time

# ===========================================================================

# ===========================================
# 1A) For the xtc format with descriptor


# [Aux function] Factorisation of the generation of a string for an atom with a ratio
def _get_spec_atom_ratio(
    dim_atom: str, size_atom: int, b_lastatom_dim: bool, b_unroll: bool, b_paral: bool
) -> str:
    # If we are at the top level, we directly use the dim name
    if not b_lastatom_dim:
        str_size = f"#{size_atom}"
    else:
        str_size = ""

    str_spec_atom = f'"{dim_atom}{str_size}": '

    # Attribute
    if b_unroll:
        str_unroll = f'"unroll" : {size_atom}'
    else:
        str_unroll = ""

    if b_unroll and b_paral:
        str_comma = ", "
    else:
        str_comma = ""

    if b_paral:
        str_paral = '"parallelize": None'
    else:
        str_paral = ""

    str_spec_atom = str_spec_atom + "{" + str_unroll + str_comma + str_paral + "},\n"
    return str_spec_atom


# [Main function] Generate a string corresponding to the schedule descriptor, from the Ttile specification
#
# Inputs:
#  - scheme: the Ttile scheme (str)
#  - comp: the considered computation
#  - machine: the considered architecture
#  - d_probsize : [dim_name --> int]
# Output:
#  - str_descr_sched, ready to be "eval()"
def get_descr_sched(scheme: List[Atom], comp: Computation, machine: Archi):
    ldims = get_ldims_computation(comp)

    # Dictionnary: [lambda_loc] --> string of spec (to be "eval(...)")
    d_current_str_desc = dict()
    d_current_str_desc[""] = ""

    # Size (note: computation will be similar to "scheme::get_sizes_scheme")
    d_lsizes = dict()
    for dim in ldims:
        d_lsizes[dim] = [1]

    # Quick pass to know if an atom is the last one
    lb_lastatom_dim = []
    l_encountered_dim = []
    for i in range(len(scheme)):
        atom = scheme[len(scheme) - i - 1]

        # Case where there are no "atom.dim"
        if atom.type == AtomType.HOIST:
            lb_lastatom_dim.append(False)
            continue

        if atom.dim not in l_encountered_dim:
            lb_lastatom_dim.append(True)
            l_encountered_dim.append(atom.dim)
        else:
            lb_lastatom_dim.append(False)
    lb_lastatom_dim.reverse()

    # DEBUG
    # print(f"{lb_lastatom_dim=}")

    # For each atom, inner to outer
    for i_atom in range(len(scheme)):
        atom = scheme[i_atom]

        # DEBUG
        # print(f"Current atom = {atom}")
        # print(f"  - {d_lsizes=}")
        # print(f"  - {d_current_str_desc=}")

        dim_atom = atom.dim

        if atom.type == AtomType.VECT:
            # Vector size (in elements)
            size_vec = int(machine.vector_size / comp.elem_size)
            str_spec_atom = f'"{dim_atom}#{size_vec}": ' + '{"vectorize" : None},\n'

            # We check that we are the first one
            assert "" in d_current_str_desc.keys()
            assert d_current_str_desc[""] == ""

            # Commit
            d_current_str_desc[""] = str_spec_atom
            d_lsizes[dim_atom] = [size_vec]

        elif atom.type == AtomType.HOIST:
            continue  # Skip

        elif atom.type in [
            AtomType.UNROLL,
            AtomType.TILE,
            AtomType.TILE_PARAL,
            AtomType.TILE_PARTIAL,
        ]:
            # Lambda loc stays the same

            # Gather infos
            if atom.type != AtomType.TILE_PARTIAL:
                ratio = atom.ratio
                b_unroll = atom.type == AtomType.UNROLL
                b_paral = atom.type == AtomType.TILE_PARAL

                # We update d_lsizes[dim_atom]
                lsize_prev = d_lsizes[dim_atom]
                for i in range(len(lsize_prev)):
                    lsize_prev[i] = lsize_prev[i] * atom.ratio
                # Note: no need to update due to pointer magic in Python :)
            else:
                # We have a size (instead of a ratio) specified
                lsize_prev = d_lsizes[dim_atom]
                for i in range(len(lsize_prev)):
                    lsize_prev[i] = atom.size

            # For each lambda loc...
            for str_lambda_loc in d_current_str_desc.keys():
                str_desc = d_current_str_desc[str_lambda_loc]

                # Note: ibr=0 if the dimension is not inside str_lambda_loc
                ibr = recover_branchid_from_stringified(str_lambda_loc, dim_atom)
                size_atom = lsize_prev[ibr]

                str_spec_atom = _get_spec_atom_ratio(
                    dim_atom, size_atom, lb_lastatom_dim[i_atom], b_unroll, b_paral
                )

                # Commit
                n_str_desc = str_spec_atom + str_desc
                d_current_str_desc[str_lambda_loc] = n_str_desc

            # Note: d_current_str_desc was correctly updated - no need for a commit here

        elif atom.type in [AtomType.ULAMBDA, AtomType.TLAMBDA]:
            # Lambda_loc might change: we check this
            str_lambda_loc_rand = list(d_current_str_desc.keys())[0]
            ldim_prev_lambda_loc = get_list_dims_str_lambda_loc(str_lambda_loc_rand)

            if dim_atom in ldim_prev_lambda_loc:
                # Case where the lambda on this dim is not the first one (lambda_loc are already ok)

                # Gather infos
                b_unroll = atom.type == AtomType.ULAMBDA
                b_paral = False

                # We update d_lsizes[dim_atom]
                lsize_prev = d_lsizes[dim_atom]
                for i in range(len(lsize_prev)):
                    lsize_prev[i] = lsize_prev[i] * atom.lratios[i]
                # Note: no need to update due to pointer magic in Python :)

                # For each lambda loc, that stays the same
                for str_lambda_loc in d_current_str_desc.keys():
                    str_desc = d_current_str_desc[str_lambda_loc]

                    ibr = recover_branchid_from_stringified(str_lambda_loc, dim_atom)
                    size_atom = lsize_prev[ibr]

                    str_spec_atom = _get_spec_atom_ratio(
                        dim_atom, size_atom, lb_lastatom_dim[i_atom], b_unroll, b_paral
                    )

                    # Commit
                    n_str_desc = str_spec_atom + str_desc
                    d_current_str_desc[str_lambda_loc] = n_str_desc
            else:
                n_branches = len(atom.lratios)

                # The lambda is new: we need to adapt & duplicate the str
                assert len(d_lsizes[dim_atom]) == 1
                size_prev = d_lsizes[dim_atom][0]

                # Update the size in this dimension
                nlsize = []
                for i in range(n_branches):
                    nlsize.append(size_prev * atom.lratios[i])
                d_lsizes[dim_atom] = nlsize

                # Create the new lambda loc
                n_d_current_str_desc = dict()
                for str_lambda_loc_prev in d_current_str_desc.keys():
                    d_lambda_loc_prev = recover_all_branchid_from_stringified(
                        str_lambda_loc_prev
                    )

                    for i_br in range(n_branches):
                        d_curr_lambda = d_lambda_loc_prev.copy()
                        d_curr_lambda[dim_atom] = i_br
                        str_lambda_loc = stringify_lambda_choice(ldims, d_curr_lambda)

                        # Schedule
                        str_desc = d_current_str_desc[str_lambda_loc_prev]
                        size_atom = nlsize[i_br]

                        str_spec_atom = _get_spec_atom_ratio(
                            dim_atom,
                            size_atom,
                            lb_lastatom_dim[i_atom],
                            b_unroll,
                            b_paral,
                        )

                        n_str_desc = str_spec_atom + str_desc

                        # Commit
                        n_d_current_str_desc[str_lambda_loc] = n_str_desc

                # Update
                d_current_str_desc = n_d_current_str_desc

        elif atom.type == AtomType.SEQ:
            # The lambda loc reduces

            # Computing the new size list for this dim
            lsizes_dim_prev = d_lsizes[dim_atom]  # Keeping it in memory
            n_branches = len(lsizes_dim_prev)

            # Updating the d_lsizes
            n_size_dim = sum(lsizes_dim_prev)
            d_lsizes[dim_atom] = [n_size_dim]

            # Regrouping the entry of d_current_str_desc by new lambda
            d_lbranches: dict[str, List[str]] = dict()
            for str_lambda_loc_prev in d_current_str_desc.keys():
                str_lambda_loc = remove_dim_from_stringified(
                    str_lambda_loc_prev, dim_atom, ldims
                )
                i_br = recover_branchid_from_stringified(str_lambda_loc_prev, dim_atom)

                if str_lambda_loc in d_lbranches.keys():
                    d_lbranches[str_lambda_loc][i_br] = d_current_str_desc[
                        str_lambda_loc_prev
                    ]
                else:
                    nl_str_desc = [""] * n_branches
                    nl_str_desc[i_br] = d_current_str_desc[str_lambda_loc_prev]
                    d_lbranches[str_lambda_loc] = nl_str_desc

            # Creating the new d_current_str_desc from d_lbranches
            d_current_str_desc = dict()
            for str_lambda_loc in d_lbranches.keys():
                lbranches = d_lbranches[str_lambda_loc]

                str_desc = ""
                start_branch = 0
                for i_br in range(len(lbranches)):
                    str_br = lbranches[i_br]
                    end_branch = start_branch + lsizes_dim_prev[i_br]

                    str_desc = (
                        str_desc
                        + f'"{dim_atom}[{start_branch}:{end_branch}]" :'
                        + "{"
                        + str_br
                        + "},\n"
                    )

                    start_branch = end_branch
                str_desc = str_desc + "\n"

                d_current_str_desc[str_lambda_loc] = str_desc
        else:
            raise ValueError(f"get_descr_sched :: Unrecognized operator {atom.type}")

    # At the end, only "" should remain
    str_descr_sched = d_current_str_desc[""]
    str_descr_sched = "{\n" + str_descr_sched + "}"
    return str_descr_sched


# ===========================================
# 1B) For the xtc format without descriptor


# Schedule to be specified here (convert Ttile to TVM-like instructions)
# Note: "xrc/itf/schd/scheduler.py"
def build_schedule_from_ttile(
    impl: Backend,
    comp: Computation,
    machine: Archi,
    scheme: List[Atom],
    b_no_descript_sched: bool = False,
):
    sch = impl.get_scheduler()

    if b_no_descript_sched:  # Version with no descriptor
        # - tile_name/sizes : need to assign names to each intermediate tile level
        # - interchange: deducable from the scheme (careful about the order: outer to inner, not inner to outer)
        # - vectorize : get the corresponding name
        # - unroll : get the corresponding names
        linfos = convert_scheme_to_xdsl_transform_input(scheme, comp)
        dtiles_name = linfos[2]
        ltilesizes = linfos[3]
        llooporder = linfos[4]
        ldim_vectorize = linfos[5]
        l_dim_parallelized = linfos[6]
        lunroll = linfos[7]

        dd_tilenamesize = merge_dtiles_name_ltilesizes(dtiles_name, ltilesizes)

        """ Example of output:
		dtiles_name = {'f': ['f1', 'f2'], 'x': ['x1', 'x2'], 'y': ['y1'], 'c': ['c1']}
		ltilesizes = [['f2', 16], ['f1', 64], ['x2', 2], ['y1', 2], ['c1', 4], ['x1', 14]]
		llooporder = ['h', 'w', 'x', 'y', 'x1', 'f', 'c', 'c1', 'y1', 'x2', 'f1']
		ldim_vectorize = ['f2']
		l_dim_parallelized = ['h']
		lunroll = [['f1', 4], ['x2', 2], ['y1', 2], ['c1', 4]]

		dd_tilenamesize = {'j': {'j1': 4096, 'j2': 32}, 'i': {'i1': 8}, 'k': {'k1': 1024}}
		"""

        # Convert lunroll into dunroll
        dunroll = dict()
        for elem in lunroll:
            dunroll[elem[0]] = elem[1]

        # DEBUG
        if False:
            print("Scheduling specification:")
            print(f"- Tiling: {dd_tilenamesize}")
            print(f"- Interchange: {llooporder}")
            print(f"- Parallelization: {l_dim_parallelized}")
            print(f"- Vectorization: {ldim_vectorize}")
            print(f"- Unrolling: {dunroll}")

        # Tiling (per dimensions)
        for k in dd_tilenamesize:
            dval = dd_tilenamesize[k]
            sch.tile(k, dval)

        # Interchange
        sch.interchange(llooporder)

        # Parallelization
        sch.parallelize(l_dim_parallelized)

        # Split
        # sch.split(dim: str, segments: dict[str,int])

        # Vectorization
        sch.vectorize(ldim_vectorize)

        # Unrolling
        sch.unroll(dunroll)
    else:  # Version with descriptor
        # ldims: the dims that appears in the scheme
        ldims = []
        for atom in scheme:
            if atom.dim not in ldims:
                ldims.append(atom.dim)

        name_op = str(comp.spec)

        str_descr_sched = get_descr_sched(scheme, comp, machine)
        spec_schedule = eval(str_descr_sched)

        # DEBUG
        # print(f"{spec_schedule=}")
        # print(f"{ldims=}")

        descript_scheduler(
            scheduler=sch, node_name=name_op, abstract_axis=ldims, spec=spec_schedule
        )

    # And run it!
    sched = sch.schedule()
    return sched


# ===========================================================================
# 2) Main launch function


# Launch scheme execution & measurement through xdsl-transform script (higher level)
def launch_and_measure_scheme_graph_interf(
    comp: Computation,
    machine: Archi,
    scheme: List[Atom],
    dsizes: dict[str, int],
    backend: str,
    b_no_descript_sched: bool = False,
) -> dict[str, float]:
    # 1) Computation - described as a graph
    dtype = f"float{comp.elem_size * 8}"

    if comp.spec == Computation_spec.CONV:
        scheme = subst_dimname_xyhw_to_hwrs_conv2D_scheme(scheme)
        dsizes = subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes)

        # Getting the sizes
        default_sizes = {
            "n": 1,
            "f": 1,
            "c": 1,
            "h": 1,
            "w": 1,
            "r": 1,
            "s": 1,
            "strx": 1,
            "stry": 1,
        }
        dsizes = default_sizes | dsizes

        size_n = dsizes["n"]
        size_f = dsizes["f"]
        size_c = dsizes["c"]
        size_h = dsizes["h"]
        size_w = dsizes["w"]
        size_r = dsizes["r"]
        size_s = dsizes["s"]

        assert dsizes["strx"] == 1  # TODO: For now...
        assert dsizes["stry"] == 1

        # Specifying the computation
        a = O.tensor((size_n, size_h + size_r - 1, size_w + size_s - 1, size_c), dtype)
        b = O.tensor((size_r, size_s, size_c, size_f), dtype)
        with O.graph(name="ttile_xdsl_conv2D_graph") as gb:
            O.conv2d(a, b, stride=(dsizes["strx"], dsizes["stry"]), name="Out")
    elif comp.spec == Computation_spec.MATMULT:
        # Getting the sizes
        default_sizes = {"i": 1, "j": 1, "k": 1}
        dsizes = default_sizes | dsizes
        size_i = dsizes["i"]
        size_j = dsizes["j"]
        size_k = dsizes["k"]

        # Specifying the computation
        a = O.tensor((size_i, size_k), dtype)
        b = O.tensor((size_k, size_j), dtype)
        with O.graph(name="matmul") as gb:
            O.matmul(a, b, name="C")
    else:
        raise ValueError(f"Unrecognized computation {comp}")

    # Build the graph
    graph = gb.graph

    # DEBUG
    # print(graph)

    # Backend
    if backend == "mlir":
        impl_backend: MlirBackend | TVMBackend = MlirBackend(
            graph, always_vectorize=True, no_alias=True
        )
    elif backend == "tvm":
        impl_backend = TVMBackend(graph, always_vectorize=True, no_alias=True)
    else:
        raise ValueError(f"Unrecognized backend: {backend}")

    # Schedule
    sched = build_schedule_from_ttile(
        impl_backend, comp, machine, scheme, b_no_descript_sched=b_no_descript_sched
    )

    # Compiler & Execute it
    compiler_args: dict[str, Any] = dict()
    evaluate_args: dict[str, Any] = dict()

    # For debugging
    # compiler_args["print_source_ir"] = True

    time = impl_backend.evaluate(sched, compiler_args, evaluate_args)

    """
	comp = impl_backend.get_compiler(
	    shared_lib=True,
	    dump_file=f"compiled_file_{backend}",
	    print_source_ir=True,
	    print_transformed_ir=True,
	)
	module = comp.compile(sched)
	executor = module.get_executor(validate=True)
	res_errorcode = executor.execute()
	"""

    res_measurement = dict()
    res_measurement["time"] = float(time)

    # Get the peak_perf
    # xdsl_transform got utils.py :: cpu_peak_time(ops: int, dtype: str, threads: int = 1)
    num_ops = compute_number_ops(comp, dsizes)
    peak_time = cpu_peak_time(num_ops, dtype)

    # DEBUG
    # print(f"measurement = {res_measurement['time']}")
    # print(f"{peak_time=}")

    # Compute peak_perf
    peak_perf = peak_time / res_measurement["time"]
    res_measurement["peak_perf"] = peak_perf

    return res_measurement
