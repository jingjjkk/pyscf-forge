#!/usr/bin/env python
"""Analytic nonadiabatic couplings for NTTDA ``deltaS=-1``.

The Hellmann--Feynman/ETF numerator is obtained by polarizing the trusted
analytic excitation-gradient functional.  The remaining moving-CSF
connection is evaluated from an explicitly spin-adapted auxiliary
wavefunction (AWF).  The same AWF supplies a deliberately slow,
cross-geometry overlap finite-difference reference.
"""

import copy
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pyscf import dft, gto, lib
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger


_COEFFICIENT_CUTOFF = 1e-14


def _as_x_array(xy):
    """Return the TDA X amplitude from a PySCF-style ``xy`` tuple."""
    if isinstance(xy, (tuple, list)):
        return np.asarray(xy[0])
    return np.asarray(xy)


def _make_xy(x):
    return np.asarray(x), 0


def _combine_xy(xy_i, xy_j, sign=1.0):
    return _make_xy(_as_x_array(xy_i) + sign * _as_x_array(xy_j))


def _scale_xy(xy, factor):
    return _make_xy(_as_x_array(xy) * factor)


def _dot_amplitude(xy_i, xy_j):
    return np.vdot(_as_x_array(xy_i), _as_x_array(xy_j))


def _orbital_partition(tdobj):
    """Return closed/open/virtual indices for an NTTDA ROKS reference."""
    occ = np.asarray(tdobj._scf.mo_occ)
    if occ.ndim != 1:
        raise ValueError("NTTDA NAC requires spatial ROKS orbitals")
    closed = np.flatnonzero(occ == 2)
    open_ = np.flatnonzero(occ == 1)
    virtual = np.flatnonzero(occ == 0)
    if len(open_) < 2:
        raise ValueError("NTTDA deltaS=-1 NAC requires reference spin S >= 1")
    return closed, open_, virtual


def _state_add(state, determinant, value):
    if abs(value) < _COEFFICIENT_CUTOFF:
        return
    value = state.get(determinant, 0.0) + value
    if abs(value) < _COEFFICIENT_CUTOFF:
        state.pop(determinant, None)
    else:
        state[determinant] = value


def _state_scale(state, factor):
    return {
        determinant: factor * value
        for determinant, value in state.items()
        if abs(factor * value) >= _COEFFICIENT_CUTOFF
    }


def _state_combine(parts):
    output = {}
    for factor, state in parts:
        if not factor:
            continue
        for determinant, value in state.items():
            _state_add(output, determinant, factor * value)
    return output


def _state_norm(state):
    return math.sqrt(float(sum(abs(value) ** 2 for value in state.values())))


def _apply_one_body(state, create, annihilate):
    """Apply ``a_create^dagger a_annihilate`` to a determinant expansion."""
    output = {}
    lower_annihilate = (1 << annihilate) - 1
    lower_create = (1 << create) - 1
    for determinant, value in state.items():
        if not (determinant >> annihilate) & 1:
            continue
        phase = -1 if (determinant & lower_annihilate).bit_count() % 2 else 1
        intermediate = determinant ^ (1 << annihilate)
        if (intermediate >> create) & 1:
            continue
        if (intermediate & lower_create).bit_count() % 2:
            phase = -phase
        _state_add(output, intermediate | (1 << create), phase * value)
    return output


def _freeze_state(state):
    return tuple(sorted(state.items()))


def _thaw_state(state):
    return dict(state)


@lru_cache(maxsize=None)
def _reference_component(nclosed, nopen, nvirtual, lowering_power):
    """Normalized ``(S_-)^lowering_power |S,S>`` determinant expansion."""
    if lowering_power not in (0, 1, 2):
        raise ValueError("only M=S, S-1, S-2 reference components are needed")
    nmo = nclosed + nopen + nvirtual
    determinant = 0
    for orbital in range(nclosed):
        determinant |= 1 << orbital
        determinant |= 1 << (nmo + orbital)
    for orbital in range(nclosed, nclosed + nopen):
        determinant |= 1 << orbital
    state = {determinant: 1.0}
    for _ in range(lowering_power):
        state = _state_combine([
            (1.0, _apply_one_body(state, nmo + orbital, orbital))
            for orbital in range(nmo)
        ])
    norm = _state_norm(state)
    if norm == 0:
        raise ValueError("spin-lowered reference component has zero norm")
    return _freeze_state(_state_scale(state, 1.0 / norm))


def _tensor_component(state, target, source, projection, nmo):
    if projection == -1:
        return _apply_one_body(state, nmo + target, source)
    if projection == 0:
        return _state_combine((
            (1.0 / math.sqrt(2.0),
             _apply_one_body(state, target, source)),
            (-1.0 / math.sqrt(2.0),
             _apply_one_body(state, nmo + target, nmo + source)),
        ))
    if projection == 1:
        return _state_scale(
            _apply_one_body(state, target, nmo + source), -1.0,
        )
    raise ValueError("rank-one spin tensor projection must be -1, 0, or 1")


@lru_cache(maxsize=None)
def _spin_adapted_configuration(
        nclosed, nopen, nvirtual, target, source):
    """One normalized NTTDA ``S_t=S-1`` tensor configuration."""
    nmo = nclosed + nopen + nvirtual
    spin = 0.5 * nopen
    coefficients = (
        math.sqrt((2.0 * spin - 1.0) / (2.0 * spin + 1.0)),
        -math.sqrt(
            (2.0 * spin - 1.0) / (spin * (2.0 * spin + 1.0))
        ),
        1.0 / math.sqrt(spin * (2.0 * spin + 1.0)),
    )
    state = _state_combine([
        (
            coefficient,
            _tensor_component(
                _thaw_state(_reference_component(
                    nclosed, nopen, nvirtual, lowering_power,
                )),
                target,
                source,
                projection,
                nmo,
            ),
        )
        for coefficient, projection, lowering_power in zip(
            coefficients, (-1, 0, 1), (0, 1, 2),
        )
    ])

    source_is_closed = source < nclosed
    target_is_open = target < nclosed + nopen
    if source_is_closed and target_is_open:
        normalization = math.sqrt(2.0 * spin / (2.0 * spin + 1.0))
    elif source_is_closed and not target_is_open:
        normalization = 1.0
    elif not source_is_closed and target_is_open:
        normalization = math.sqrt(
            (2.0 * spin - 1.0) / (2.0 * spin + 1.0)
        )
    else:
        normalization = math.sqrt(2.0 * spin / (2.0 * spin + 1.0))
    return _freeze_state(_state_scale(state, normalization))


@dataclass(frozen=True)
class AuxiliaryWavefunction:
    """Spin-adapted NTTDA AWF in an alpha-first determinant convention."""

    nmo: int
    nalpha: int
    nbeta: int
    coefficients: dict

    @property
    def norm(self):
        return float(sum(abs(value) ** 2 for value in self.coefficients.values()))


@dataclass(frozen=True)
class CSFConnectionComponents:
    """AO and relaxed-orbital pieces of the moving-AWF connection."""

    ao: np.ndarray
    orbital: np.ndarray
    total: np.ndarray
    residual: float


def build_spin_adapted_awf(tdobj, xy, cutoff=_COEFFICIENT_CUTOFF):
    """Expand one NTTDA ``deltaS=-1`` root in spin-adapted determinants."""
    if getattr(tdobj, "deltaS", None) != -1:
        raise NotImplementedError("NTTDA AWF currently supports only deltaS=-1")
    closed, open_, virtual = _orbital_partition(tdobj)
    nclosed, nopen, nvirtual = len(closed), len(open_), len(virtual)
    nmo = nclosed + nopen + nvirtual
    amplitude = _as_x_array(xy)
    expected = (nclosed + nopen, nopen + nvirtual)
    if amplitude.size != expected[0] * expected[1]:
        raise ValueError(
            "deltaS=-1 amplitude has shape %s; expected %s" %
            (amplitude.shape, expected)
        )
    amplitude = amplitude.reshape(expected)
    state = {}
    for source_local, source in enumerate(range(nclosed + nopen)):
        for target_local, target in enumerate(range(nclosed, nmo)):
            weight = amplitude[source_local, target_local]
            if abs(weight) < cutoff:
                continue
            configuration = _thaw_state(_spin_adapted_configuration(
                nclosed, nopen, nvirtual, target, source,
            ))
            for determinant, coefficient in configuration.items():
                _state_add(state, determinant, weight * coefficient)
    return AuxiliaryWavefunction(
        nmo=nmo,
        nalpha=nclosed + nopen - 1,
        nbeta=nclosed + 1,
        coefficients=state,
    )


def _spin_summed_rdm1(left, right):
    """Return ``gamma[p,q] = <left|sum_s p_s^dag q_s|right>``."""
    if left.nmo != right.nmo:
        raise ValueError("interstate RDM requires equal orbital dimensions")
    nmo = left.nmo
    gamma = np.zeros(
        (nmo, nmo),
        dtype=np.result_type(
            *left.coefficients.values(), *right.coefficients.values(), float,
        ),
    )
    for determinant, right_coefficient in right.coefficients.items():
        for spin_offset in (0, nmo):
            for source in range(nmo):
                annihilate = spin_offset + source
                if not (determinant >> annihilate) & 1:
                    continue
                phase = (
                    -1 if (determinant & ((1 << annihilate) - 1)).bit_count() % 2
                    else 1
                )
                intermediate = determinant ^ (1 << annihilate)
                for target in range(nmo):
                    create = spin_offset + target
                    if (intermediate >> create) & 1:
                        continue
                    create_phase = (
                        -1 if (intermediate & ((1 << create) - 1)).bit_count() % 2
                        else 1
                    )
                    bra = intermediate | (1 << create)
                    left_coefficient = left.coefficients.get(bra)
                    if left_coefficient is not None:
                        gamma[target, source] += (
                            np.conjugate(left_coefficient)
                            * right_coefficient * phase * create_phase
                        )
    return gamma


def interstate_rdm1(tdobj, xy_i, xy_j):
    """Spin-summed MO-basis AWF transition density for two NTTDA roots."""
    return _spin_summed_rdm1(
        build_spin_adapted_awf(tdobj, xy_i),
        build_spin_adapted_awf(tdobj, xy_j),
    )


@lru_cache(maxsize=65536)
def _occupied_orbitals(determinant, nmo):
    alpha = tuple(p for p in range(nmo) if (determinant >> p) & 1)
    beta = tuple(p for p in range(nmo) if (determinant >> (nmo + p)) & 1)
    return alpha, beta


def _awf_overlap_from_mo(left, right, mo_overlap):
    if left.nalpha != right.nalpha or left.nbeta != right.nbeta:
        return 0.0
    if mo_overlap.shape != (left.nmo, right.nmo):
        raise ValueError("MO overlap has incompatible dimensions")
    determinant_cache = {}
    value = 0.0j
    for left_det, left_coefficient in left.coefficients.items():
        left_alpha, left_beta = _occupied_orbitals(left_det, left.nmo)
        for right_det, right_coefficient in right.coefficients.items():
            right_alpha, right_beta = _occupied_orbitals(right_det, right.nmo)
            key = (left_alpha, right_alpha, left_beta, right_beta)
            det_overlap = determinant_cache.get(key)
            if det_overlap is None:
                alpha = mo_overlap[np.ix_(left_alpha, right_alpha)]
                beta = mo_overlap[np.ix_(left_beta, right_beta)]
                det_overlap = np.linalg.det(alpha) * np.linalg.det(beta)
                determinant_cache[key] = det_overlap
            value += (
                np.conjugate(left_coefficient)
                * right_coefficient * det_overlap
            )
    return np.real_if_close(value)


def awf_overlap(td_left, xy_left, td_right, xy_right, ao_overlap=None):
    """Cross-geometry overlap of two spin-adapted NTTDA auxiliary states."""
    left = build_spin_adapted_awf(td_left, xy_left)
    right = build_spin_adapted_awf(td_right, xy_right)
    if ao_overlap is None:
        ao_overlap = gto.intor_cross(
            "int1e_ovlp", td_left.mol, td_right.mol,
        )
    coefficient_left = np.asarray(td_left._scf.mo_coeff)
    coefficient_right = np.asarray(td_right._scf.mo_coeff)
    mo_overlap = coefficient_left.conj().T @ ao_overlap @ coefficient_right
    return _awf_overlap_from_mo(left, right, mo_overlap)


def _ao_csf_connection(td_nac, gamma, atmlst=None):
    """Antisymmetric AO-basis part of the moving-AWF connection."""
    tdobj = td_nac.base
    mol = tdobj.mol
    mo_coeff = np.asarray(tdobj._scf.mo_coeff)
    density = mo_coeff @ gamma @ mo_coeff.conj().T
    overlap_derivative = tdobj._scf.nuc_grad_method().get_ovlp(mol)
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    offsets = mol.offset_nr_by_atom()
    result = np.zeros(
        (len(atmlst), 3), dtype=np.result_type(density, float),
    )
    for index, atom in enumerate(atmlst):
        p0, p1 = offsets[atom][2:]
        result[index] -= 0.5 * lib.einsum(
            "xpq,pq->x", overlap_derivative[:, p0:p1], density[p0:p1],
        )
        result[index] += 0.5 * lib.einsum(
            "xqp,pq->x", overlap_derivative[:, p0:p1], density[:, p0:p1],
        )
    return np.real_if_close(result)


def _orbital_csf_connection(td_nac, gamma, atmlst=None):
    """ROKS orbital-response part of the moving-AWF connection.

    The scalar source is the antisymmetric interstate one-particle density.
    An adjoint ROKS Hessian solve avoids forming nuclear orbital responses for
    every Cartesian coordinate explicitly.
    """
    from pyscf.grad.nttda.delta_s_minus_one import (
        spin_fock_direct_dft,
        spin_fock_direct_hf,
    )
    from pyscf.grad.nttda.roks import (
        _orbital_gradient,
        make_hessian_transpose_action,
        pack_m_matrix,
        solve_zvector,
        zvector_adjoint_matrix,
        zvector_probe_densities,
    )

    tdobj = td_nac.base
    antisymmetric = 0.5 * (gamma - gamma.T)
    action, pairs = make_hessian_transpose_action(tdobj)
    rhs = pack_m_matrix(antisymmetric, pairs)
    if np.max(np.abs(rhs), initial=0.0) < 1e-15:
        if atmlst is None:
            atmlst = range(tdobj.mol.natm)
        zeros = np.zeros((len(tuple(atmlst)), 3))
        return zeros, 0.0
    zvector = solve_zvector(
        action,
        pairs,
        tdobj,
        rhs,
        tolerance=td_nac.cphf_conv_tol,
        max_cycle=td_nac.cphf_max_cycle,
    )
    adjoint = zvector_adjoint_matrix(tdobj, pairs, zvector)
    residual = float(np.max(np.abs(
        pack_m_matrix(adjoint, pairs) - rhs,
    )))
    probe_alpha, probe_beta = zvector_probe_densities(
        tdobj, pairs, zvector,
    )
    xctype = tdobj._scf._numint._xc_type(tdobj._scf.xc)
    if xctype == "HF":
        fock_contraction = spin_fock_direct_hf(
            td_nac,
            tdobj,
            probe_alpha,
            probe_beta,
            atmlst=atmlst,
        )
    else:
        fock_contraction = spin_fock_direct_dft(
            td_nac,
            tdobj,
            probe_alpha,
            probe_beta,
            atmlst=atmlst,
        )
    orbital = _orbital_gradient(
        tdobj,
        antisymmetric,
        adjoint,
        fock_contraction,
        atmlst=atmlst,
    )
    return np.real_if_close(orbital), residual


def nac_csf_components(td_nac, x_y_i, x_y_j, atmlst=None):
    """Return all analytic moving-CSF/AWF connection components."""
    gamma = interstate_rdm1(td_nac.base, x_y_i, x_y_j)
    ao = _ao_csf_connection(td_nac, gamma, atmlst=atmlst)
    orbital, residual = _orbital_csf_connection(
        td_nac, gamma, atmlst=atmlst,
    )
    return CSFConnectionComponents(
        ao=ao,
        orbital=orbital,
        total=np.real_if_close(ao + orbital),
        residual=residual,
    )


def nac_csf(td_nac, x_y_i, x_y_j, atmlst=None):
    """Complete analytic moving-CSF/AWF contribution."""
    return nac_csf_components(
        td_nac, x_y_i, x_y_j, atmlst=atmlst,
    ).total


def get_hf_interstate_numerator(
        td_nac, x_y_i, x_y_j, atmlst=None, verbose=logger.INFO):
    r"""Return the relaxed-orbital numerator ``X_I^T A^[R] X_J``."""
    if td_nac.base.deltaS != -1:
        raise NotImplementedError("NTTDA NAC currently supports only deltaS=-1")
    gradient = td_nac._gradient_driver(verbose=verbose)
    plus = gradient.grad_elec(
        _combine_xy(x_y_i, x_y_j, 1.0), atmlst=atmlst,
    )
    minus = gradient.grad_elec(
        _combine_xy(x_y_i, x_y_j, -1.0), atmlst=atmlst,
    )
    return 0.25 * (plus - minus)


def _root_overlap_matrix(reference, displaced):
    ao_overlap = gto.intor_cross(
        "int1e_ovlp", reference.mol, displaced.mol,
    )
    output = np.empty((len(reference.xy), len(displaced.xy)))
    for i, xy_i in enumerate(reference.xy):
        for j, xy_j in enumerate(displaced.xy):
            output[i, j] = np.real(awf_overlap(
                reference, xy_i, displaced, xy_j,
                ao_overlap=ao_overlap,
            ))
    return output


def _align_displaced_roots(reference, displaced, threshold, required=()):
    """Reorder and phase displaced roots by maximum spin-adapted AWF overlap."""
    from scipy.optimize import linear_sum_assignment

    overlaps = _root_overlap_matrix(reference, displaced)
    rows, columns = linear_sum_assignment(-np.abs(overlaps))
    assignment = dict(zip(rows.tolist(), columns.tolist()))
    for root in required:
        if root not in assignment:
            raise RuntimeError("displaced NTTDA calculation did not return root %d" % root)
        value = abs(overlaps[root, assignment[root]])
        if value < threshold:
            raise RuntimeError(
                "NTTDA state-tracking overlap %.6f for root %d is below %.6f" %
                (value, root + 1, threshold)
            )

    ordered_xy = []
    ordered_e = []
    ordered_converged = []
    used = set()
    converged = getattr(displaced, "converged", None)
    for root in range(min(len(reference.xy), len(displaced.xy))):
        if root not in assignment:
            continue
        column = assignment[root]
        used.add(column)
        overlap = overlaps[root, column]
        phase = 1.0 if overlap >= 0 else -1.0
        ordered_xy.append(_scale_xy(displaced.xy[column], phase))
        ordered_e.append(displaced.e[column])
        if isinstance(converged, (tuple, list, np.ndarray)):
            ordered_converged.append(converged[column])
    for column in range(len(displaced.xy)):
        if column in used:
            continue
        ordered_xy.append(displaced.xy[column])
        ordered_e.append(displaced.e[column])
        if isinstance(converged, (tuple, list, np.ndarray)):
            ordered_converged.append(converged[column])
    displaced.xy = ordered_xy
    displaced.e = np.asarray(ordered_e)
    if ordered_converged:
        displaced.converged = np.asarray(ordered_converged)
    return overlaps


def _copy_td_snapshot(tdobj):
    snapshot = copy.copy(tdobj)
    snapshot_mf = copy.copy(tdobj._scf)
    snapshot_mf.mol = tdobj.mol.copy()
    snapshot_mf.mo_coeff = np.array(tdobj._scf.mo_coeff, copy=True)
    snapshot_mf.mo_occ = np.array(tdobj._scf.mo_occ, copy=True)
    snapshot._scf = snapshot_mf
    snapshot.mol = snapshot_mf.mol
    snapshot.xy = copy.deepcopy(tdobj.xy)
    snapshot.e = np.array(tdobj.e, copy=True)
    return snapshot


class NonAdiabaticCouplings(rhf_grad.GradientsBase):
    """Analytic NTTDA ``deltaS=-1`` excited-state NAC driver.

    ``use_etfs=False`` includes the moving-CSF contribution.  With
    ``use_etfs=True`` only the Hellmann--Feynman/ETF term is retained.
    ``ediff=False`` returns the energy-scaled numerator; ``ediff=True``
    returns the derivative coupling in inverse length units.
    """

    _keys = rhf_grad.GradientsBase._keys | {
        "state_I", "state_J", "ediff", "use_etfs", "gap_tol",
        "cphf_conv_tol", "cphf_max_cycle", "fixed_grid", "step",
        "root_overlap_tol",
    }

    def __init__(self, tdobj):
        super().__init__(tdobj)
        self.state_I = None
        self.state_J = None
        self.ediff = False
        self.use_etfs = False
        self.gap_tol = 1e-10
        self.cphf_conv_tol = 1e-12
        self.cphf_max_cycle = None
        self.fixed_grid = isinstance(tdobj._scf, dft.KohnShamDFT)
        self.step = 2e-4
        self.root_overlap_tol = 0.5
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        self.nttda_nac_details = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("******** NTTDA deltaS=-1 NAC ********")
        log.info("state_I = %s", self.state_I)
        log.info("state_J = %s", self.state_J)
        log.info("ediff = %s", self.ediff)
        log.info("use_etfs = %s", self.use_etfs)
        log.info("fixed_grid = %s", self.fixed_grid)
        return self

    def _gradient_driver(self, verbose=None):
        gradient = self.base.Gradients()
        gradient.verbose = self.verbose if verbose is None else verbose
        gradient.cphf_conv_tol = self.cphf_conv_tol
        gradient.cphf_max_cycle = self.cphf_max_cycle
        gradient.fixed_grid = self.fixed_grid
        return gradient

    def _phase_match(self, xy, previous, label):
        if previous is not None:
            overlap = _dot_amplitude(previous, xy)
            if np.real(overlap) < 0:
                logger.debug(self, "Flipping NTTDA NAC phase for state %s", label)
                xy = _scale_xy(xy, -1.0)
        return xy

    def reset_phase(self):
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        return self

    def _validate_states(self, state_i, state_j):
        if self.base.deltaS != -1:
            raise NotImplementedError("NTTDA NAC currently supports only deltaS=-1")
        if self.base.xy is None:
            self.base.run()
        nstates = len(self.base.xy)
        if not 1 <= state_i <= nstates:
            raise ValueError("state_I must be in [1, %d]" % nstates)
        if not 1 <= state_j <= nstates:
            raise ValueError("state_J must be in [1, %d]" % nstates)
        if state_i == state_j:
            raise ValueError("NTTDA NAC requires two distinct excited states")

    def compute_nac(
            self, state_I=None, state_J=None, atmlst=None, ediff=None,
            use_etfs=None):
        state_i = self.state_I if state_I is None else state_I
        state_j = self.state_J if state_J is None else state_J
        if state_i is None or state_j is None:
            raise RuntimeError("state_I and state_J must be specified")
        ediff = self.ediff if ediff is None else ediff
        use_etfs = self.use_etfs if use_etfs is None else use_etfs
        self._validate_states(state_i, state_j)

        xy_i = self._phase_match(
            self.base.xy[state_i - 1], self.x_y_I_prev, state_i,
        )
        xy_j = self._phase_match(
            self.base.xy[state_j - 1], self.x_y_J_prev, state_j,
        )
        self.x_y_I_prev = copy.deepcopy(xy_i)
        self.x_y_J_prev = copy.deepcopy(xy_j)
        gap = self.base.e[state_j - 1] - self.base.e[state_i - 1]

        hf_numerator = get_hf_interstate_numerator(
            self, xy_i, xy_j, atmlst=atmlst, verbose=self.verbose,
        )
        if use_etfs:
            csf = np.zeros_like(hf_numerator)
            csf_components = None
            numerator = hf_numerator
        else:
            csf_components = nac_csf_components(
                self, xy_i, xy_j, atmlst=atmlst,
            )
            csf = csf_components.total
            numerator = hf_numerator + gap * csf
        self.nttda_nac_details = {
            "hf_numerator": hf_numerator,
            "csf": csf,
            "csf_ao": None if csf_components is None else csf_components.ao,
            "csf_orbital": (
                None if csf_components is None else csf_components.orbital
            ),
            "csf_residual": (
                None if csf_components is None else csf_components.residual
            ),
            "full_numerator": numerator,
            "gap": gap,
            "state_I": state_i,
            "state_J": state_j,
            "use_etfs": use_etfs,
        }
        if ediff:
            if abs(gap) < self.gap_tol:
                raise ZeroDivisionError(
                    "NTTDA state gap %.6e is below gap_tol %.6e" %
                    (gap, self.gap_tol)
                )
            return np.real_if_close(numerator / gap)
        return np.real_if_close(numerator)

    def kernel(
            self, state_I=None, state_J=None, atmlst=None, ediff=None,
            use_etfs=None):
        if state_I is not None:
            self.state_I = state_I
        if state_J is not None:
            self.state_J = state_J
        if atmlst is not None:
            self.atmlst = atmlst
        if ediff is not None:
            self.ediff = ediff
        if use_etfs is not None:
            self.use_etfs = use_etfs
        if atmlst is None:
            atmlst = self.atmlst
        if self.verbose >= logger.INFO:
            self.dump_flags()
        self.nac = self.compute_nac(
            state_I=self.state_I,
            state_J=self.state_J,
            atmlst=atmlst,
            ediff=self.ediff,
            use_etfs=self.use_etfs,
        )
        return self.nac

    def _displaced_td(self, coordinates):
        from pyscf.grad.nttda import _copy_td_settings, _displaced_reference
        from pyscf.sftda.nttda import NTTDA

        mol = self.mol.copy()
        mol.set_geom_(coordinates, unit="Bohr")
        reference = _displaced_reference(
            self.base._scf, mol, self.fixed_grid,
        )
        reference.kernel(dm0=self.base._scf.make_rdm1())
        if not reference.converged:
            raise RuntimeError("displaced ROKS reference did not converge")
        tdobj = _copy_td_settings(self.base, NTTDA(reference))
        tdobj.kernel()
        return tdobj

    def finite_difference(
            self, state_I=None, state_J=None, atmlst=None, step=None):
        """Balanced spin-adapted AWF-overlap finite-difference NAC."""
        state_i = self.state_I if state_I is None else state_I
        state_j = self.state_J if state_J is None else state_J
        if state_i is None or state_j is None:
            raise RuntimeError("state_I and state_J must be specified")
        self._validate_states(state_i, state_j)
        if step is None:
            step = self.step
        if atmlst is None:
            atmlst = self.atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = tuple(atmlst)
        coordinates = self.mol.atom_coords()
        result = np.zeros((len(atmlst), 3))
        required = (state_i - 1, state_j - 1)
        for index, atom in enumerate(atmlst):
            for xyz in range(3):
                plus_coordinates = coordinates.copy()
                minus_coordinates = coordinates.copy()
                plus_coordinates[atom, xyz] += step
                minus_coordinates[atom, xyz] -= step
                plus = self._displaced_td(plus_coordinates)
                minus = self._displaced_td(minus_coordinates)
                _align_displaced_roots(
                    self.base, plus, self.root_overlap_tol, required=required,
                )
                _align_displaced_roots(
                    self.base, minus, self.root_overlap_tol, required=required,
                )
                minus_plus = awf_overlap(
                    minus, minus.xy[state_i - 1],
                    plus, plus.xy[state_j - 1],
                )
                plus_minus = awf_overlap(
                    plus, plus.xy[state_i - 1],
                    minus, minus.xy[state_j - 1],
                )
                result[index, xyz] = np.real(
                    (minus_plus - plus_minus) / (4.0 * step)
                )
        return result

    def as_scanner(self):
        if isinstance(self, _NACScanner):
            return self
        return _NACScanner(self)


class _NACScanner(NonAdiabaticCouplings):
    def __init__(self, nac):
        self.__dict__.update(nac.__dict__)

    def __call__(self, mol):
        tdobj = self.base
        mf = tdobj._scf
        if np.allclose(mf.mol.atom_coords(), mol.atom_coords()):
            return self.kernel()
        snapshot = _copy_td_snapshot(tdobj)
        density = mf.make_rdm1()
        mf.reset(mol)
        mf.kernel(dm0=density)
        if not mf.converged:
            raise RuntimeError("scanner ROKS reference did not converge")
        tdobj.mol = mol
        tdobj.kernel()
        required = ()
        if self.state_I is not None and self.state_J is not None:
            required = (self.state_I - 1, self.state_J - 1)
        _align_displaced_roots(
            snapshot, tdobj, self.root_overlap_tol, required=required,
        )
        return self.kernel()


NAC = NonAdiabaticCouplings


__all__ = [
    "AuxiliaryWavefunction",
    "CSFConnectionComponents",
    "NAC",
    "NonAdiabaticCouplings",
    "awf_overlap",
    "build_spin_adapted_awf",
    "get_hf_interstate_numerator",
    "interstate_rdm1",
    "nac_csf",
    "nac_csf_components",
]
