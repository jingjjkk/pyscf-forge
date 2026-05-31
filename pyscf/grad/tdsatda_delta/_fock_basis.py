"""Fock matrix basis helpers for SATDA delta-gradient blocks."""

from dataclasses import dataclass

from pyscf import dft


@dataclass
class SATDAFockBasis:
    """AO and MO Fock matrices in the ``fock0/fockz`` basis.

    The convention matches ``pyscf.sftda.satda``:

        F_alpha = F0 + Fz
        F_beta  = F0 - Fz
        F_spin  = 0.5 * (F_beta - F_alpha) = -Fz
    """

    fock0: object
    fockz: object
    fock0_mo: object
    fockz_mo: object


def alpha_from_0z(fock0, fockz):
    return fock0 + fockz


def beta_from_0z(fock0, fockz):
    return fock0 - fockz


def spin_from_0z(fock0, fockz):
    del fock0
    return -fockz


def fock0z_from_alpha_beta(fock_alpha, fock_beta):
    return 0.5 * (fock_alpha + fock_beta), 0.5 * (fock_alpha - fock_beta)


def fock_by_spin(spin, fock0, fockz):
    if spin == 'alpha':
        return alpha_from_0z(fock0, fockz)
    if spin == 'beta':
        return beta_from_0z(fock0, fockz)
    if spin == 'spin':
        return spin_from_0z(fock0, fockz)
    raise ValueError('Unknown Fock spin label %s' % spin)


def make_fock_basis(mf, mo_coeff=None, max_memory=None):
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    fock = mf.get_fock()
    if (isinstance(mf, dft.KohnShamDFT)
            and mf._numint._xc_type(mf.xc) != 'HF'):
        from pyscf.sftda.satda import gen_rohf_response_sf

        if max_memory is None:
            max_memory = mf.max_memory
        _, fockz = gen_rohf_response_sf(
            mf, mo_coeff=mo_coeff, mo_occ=mf.mo_occ,
            hermi=0, max_memory=max_memory,
        )
        fock0 = fock.focka - fockz
    else:
        fock0, fockz = fock0z_from_alpha_beta(fock.focka, fock.fockb)
    return SATDAFockBasis(
        fock0=fock0,
        fockz=fockz,
        fock0_mo=mo_coeff.conj().T @ fock0 @ mo_coeff,
        fockz_mo=mo_coeff.conj().T @ fockz @ mo_coeff,
    )
