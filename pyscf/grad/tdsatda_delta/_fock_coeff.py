from dataclasses import dataclass

import numpy as np

from pyscf import lib

from ._blocks import (
    _sasf_orbitals,
    _mo_pair_dm,
    make_sasf_blocks,
)
from ._fock_basis import make_fock_basis


@dataclass
class SASFFockCoefficients:
    '''Coefficient matrices for the SASF Fock-like ``delta_A`` terms.

    The scalar contraction represented by these matrices is

        E = Tr[T_s_cc F_s(CC)] + Tr[T_s_vv F_s(VV)]
          + Tr[T_s_cv F_s(CV)]
          + Tr[T_b_vo F_beta(VO)] + Tr[T_b_co F_beta(CO)]
          + Tr[T_a_oc F_alpha(OC)] + Tr[T_a_vo F_alpha(VO)]

    where rows/columns follow the labels in each field name.
    '''

    t_s_cc: np.ndarray
    t_s_vv: np.ndarray
    t_s_cv: np.ndarray
    t_b_vo: np.ndarray
    t_b_co: np.ndarray
    t_a_oc: np.ndarray
    t_a_vo: np.ndarray
    trace_oo: float
    si: float


def sasf_fock_coefficients(tdobj, xy):
    '''Build all coefficient matrices for SASF Fock-like ``delta_A`` terms.

    These coefficients are the code counterpart of the seven ``T`` matrices in
    ``pyscf/sftda/derivations_sasf.md``.  They are independent of AO integral
    derivatives and can be tested directly against ``sasf_fock_action``.
    '''
    b = make_sasf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')

    tr_oo = float(np.trace(b.x_oo))
    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1
    chi = 1.0 / np.sqrt(2 * si * (2 * si - 1))

    t_s_cc = (
        lib.einsum('ia,ja->ji', b.x_cv, b.x_cv) / si
        + lib.einsum('iu,ju->ji', b.x_co, b.x_co) * 2 / (2 * si - 1)
    )
    t_s_vv = (
        lib.einsum('ia,ib->ab', b.x_cv, b.x_cv) / si
        + lib.einsum('ua,ub->ab', b.x_ov, b.x_ov) * 2 / (2 * si - 1)
    )
    t_s_cv = (gamma * (1 + 1 / si)) * tr_oo * b.x_cv

    t_b_vo = (
        2 * eta * lib.einsum('ia,iv->av', b.x_cv, b.x_co)
        + 2 * zeta * lib.einsum('ua,uv->av', b.x_ov, b.x_oo)
    )
    t_b_co = 2 * chi * tr_oo * b.x_co

    t_a_oc = (
        -2 * eta * lib.einsum('ia,va->vi', b.x_cv, b.x_ov)
        - 2 * zeta * lib.einsum('iu,vu->vi', b.x_co, b.x_oo)
    )
    t_a_vo = -2 * chi * tr_oo * b.x_ov.T

    return SASFFockCoefficients(
        t_s_cc=t_s_cc,
        t_s_vv=t_s_vv,
        t_s_cv=t_s_cv,
        t_b_vo=t_b_vo,
        t_b_co=t_b_co,
        t_a_oc=t_a_oc,
        t_a_vo=t_a_vo,
        trace_oo=tr_oo,
        si=si,
    )


def sasf_fock_probe_densities(tdobj, xy):
    '''Return AO probe densities for SASF Fock-like ``delta_A`` terms.

    The returned tuple ``(dm_alpha, dm_beta)`` satisfies

        E_delta_fock = Tr[dm_alpha F_alpha] + Tr[dm_beta F_beta]

    for the current molecular orbitals.  Non-symmetric transition densities are
    intentionally preserved; callers should symmetrize only when contracting
    with a symmetric response kernel that expects density variations.
    '''
    coeff = sasf_fock_coefficients(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)

    dm_s = np.zeros((tdobj.mol.nao, tdobj.mol.nao))
    dm_s += _mo_pair_dm(orbcs, coeff.t_s_cc, orbcs)
    dm_s += _mo_pair_dm(orbvs, coeff.t_s_vv, orbvs)
    dm_s += _mo_pair_dm(orbcs, coeff.t_s_cv, orbvs)

    dm_b = np.zeros_like(dm_s)
    dm_b += _mo_pair_dm(orbvs, coeff.t_b_vo, orbos)
    dm_b += _mo_pair_dm(orbcs, coeff.t_b_co, orbos)

    dm_a = np.zeros_like(dm_s)
    dm_a += _mo_pair_dm(orbos, coeff.t_a_oc, orbcs)
    dm_a += _mo_pair_dm(orbvs, coeff.t_a_vo, orbos)

    return dm_a - 0.5 * dm_s, dm_b + 0.5 * dm_s


def sasf_fock_coefficient_energy(tdobj, xy):
    '''Evaluate the SASF Fock-like correction energy from coefficient matrices.'''
    coeff = sasf_fock_coefficients(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    fbasis = make_fock_basis(tdobj._scf)
    fock0 = fbasis.fock0
    fockz = fbasis.fockz

    focks_cc = -orbcs.conj().T @ fockz @ orbcs
    focks_vv = -orbvs.conj().T @ fockz @ orbvs
    focks_cv = -orbcs.conj().T @ fockz @ orbvs
    fockb_vo = orbvs.conj().T @ (fock0 - fockz) @ orbos
    fockb_co = orbcs.conj().T @ (fock0 - fockz) @ orbos
    focka_oc = orbos.conj().T @ (fock0 + fockz) @ orbcs
    focka_vo = orbvs.conj().T @ (fock0 + fockz) @ orbos

    e = 0.0
    e += lib.einsum('ji,ji', coeff.t_s_cc, focks_cc)
    e += lib.einsum('ab,ab', coeff.t_s_vv, focks_vv)
    e += lib.einsum('ia,ia', coeff.t_s_cv, focks_cv)
    e += lib.einsum('av,av', coeff.t_b_vo, fockb_vo)
    e += lib.einsum('iu,iu', coeff.t_b_co, fockb_co)
    e += lib.einsum('vi,vi', coeff.t_a_oc, focka_oc)
    e += lib.einsum('au,au', coeff.t_a_vo, focka_vo)
    return float(e)
