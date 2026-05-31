import numpy as np

from pyscf import ao2mo
from pyscf import lib

from ._blocks import (
    _sasf_orbitals,
    _hybrid_coefficients,
    make_sasf_blocks,
)


def _sasf_hf_exchange_energy_with_coeff(tdobj, xy, coeff, omega=None):
    mf = tdobj._scf
    mol = mf.mol
    b = make_sasf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)

    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1

    def general(orbs):
        if omega is None or omega == 0:
            return ao2mo.general(mol, orbs, compact=False)
        with mol.with_range_coulomb(omega):
            return ao2mo.general(mol, orbs, compact=False)

    e = 0.0
    eri = general([orbos, orbcs, orbcs, orbos]).reshape(
        len(b.osidx), len(b.csidx), len(b.csidx), len(b.osidx)
    )
    e -= lib.einsum('iu,jv,uijv', b.x_co, b.x_co, eri) * coeff / (2 * si - 1)

    eri = general([orbvs, orbos, orbos, orbvs]).reshape(
        len(b.vsidx), len(b.osidx), len(b.osidx), len(b.vsidx)
    )
    e -= lib.einsum('ua,vb,auvb', b.x_ov, b.x_ov, eri) * coeff / (2 * si - 1)

    eri = general([orbvs, orbos, orbcs, orbcs]).reshape(
        len(b.vsidx), len(b.osidx), len(b.csidx), len(b.csidx)
    )
    e -= 2 * coeff * eta * lib.einsum('ia,jv,avji', b.x_cv, b.x_co, eri)

    eri = general([orbvs, orbvs, orbos, orbcs]).reshape(
        len(b.vsidx), len(b.vsidx), len(b.osidx), len(b.csidx)
    )
    e -= 2 * coeff * eta * lib.einsum('ia,vb,abvi', b.x_cv, b.x_ov, eri)

    eri1 = general([orbos, orbcs, orbos, orbvs]).reshape(
        len(b.osidx), len(b.csidx), len(b.osidx), len(b.vsidx)
    )
    eri2 = general([orbos, orbvs, orbos, orbcs]).reshape(
        len(b.osidx), len(b.vsidx), len(b.osidx), len(b.csidx)
    )
    e += 2 * coeff * lib.einsum('iu,vb,uivb', b.x_co, b.x_ov, eri1) / (2 * si - 1)
    e -= 2 * coeff * lib.einsum('iu,vb,ubvi', b.x_co, b.x_ov, eri2) / (2 * si - 1)

    eri = general([orbvs, orbos, orbos, orbcs]).reshape(
        len(b.vsidx), len(b.osidx), len(b.osidx), len(b.csidx)
    )
    e -= 2 * coeff * (gamma - 1) * lib.einsum('ia,wv,avwi', b.x_cv, b.x_oo, eri)

    eri = general([orbos, orbos, orbos, orbcs]).reshape(
        len(b.osidx), len(b.osidx), len(b.osidx), len(b.csidx)
    )
    e -= 2 * coeff * zeta * lib.einsum('iu,wv,uvwi', b.x_co, b.x_oo, eri)

    eri = general([orbvs, orbos, orbos, orbos]).reshape(
        len(b.vsidx), len(b.osidx), len(b.osidx), len(b.osidx)
    )
    e -= 2 * coeff * zeta * lib.einsum('ua,wv,avwu', b.x_ov, b.x_oo, eri)
    return float(e)


def sasf_hf_exchange_coefficient_energy(tdobj, xy):
    '''Evaluate SASF HF exchange-like correction energy by independent blocks.'''
    mf = tdobj._scf
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if not hybrid:
        return 0.0
    e = _sasf_hf_exchange_energy_with_coeff(tdobj, xy, hyb)
    if omega != 0:
        e += _sasf_hf_exchange_energy_with_coeff(tdobj, xy, alpha - hyb, omega=omega)
    return e
