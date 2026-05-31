from dataclasses import dataclass

import numpy as np

from pyscf import lib


@dataclass
class SASFBlocks:
    csidx: np.ndarray
    osidx: np.ndarray
    vsidx: np.ndarray
    x_co: np.ndarray
    x_cv: np.ndarray
    x_oo: np.ndarray
    x_ov: np.ndarray
    si: float


def _sasf_orbitals(tdobj):
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    return csidx, osidx, vsidx, orbcs, orbos, orbvs


def make_sasf_blocks(tdobj, xy):
    mf = tdobj._scf
    mo_occ = mf.mo_occ
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    ncs = len(csidx)
    nos = len(osidx)
    x = np.asarray(xy[0])
    if x.shape != (ncs + nos, nos + len(vsidx)):
        raise ValueError(
            'SASF X amplitude shape %s incompatible with C/O/V dimensions '
            '(%d, %d)' % (x.shape, ncs + nos, nos + len(vsidx))
        )
    return SASFBlocks(
        csidx=csidx,
        osidx=osidx,
        vsidx=vsidx,
        x_co=x[:ncs, :nos],
        x_cv=x[:ncs, nos:],
        x_oo=x[ncs:, :nos],
        x_ov=x[ncs:, nos:],
        si=(mf.mol.nelec[0] - mf.mol.nelec[1]) * 0.5,
    )


def _split_sasf_amplitudes(tdobj, x):
    csidx, osidx, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    ncs = len(csidx)
    nos = len(osidx)
    nvs = len(vsidx)
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.reshape(1, *x.shape)
    if x.shape[1:] != (ncs + nos, nos + nvs):
        raise ValueError(
            'SASF amplitude shape %s incompatible with C/O/V dimensions '
            '(%d, %d)' % (x.shape[1:], ncs + nos, nos + nvs)
        )
    return (
        x[:, :ncs, :nos],
        x[:, :ncs, nos:],
        x[:, ncs:, :nos],
        x[:, ncs:, nos:],
    )


def _mo_pair_dm(c_left, mat, c_right):
    return c_left @ mat @ c_right.conj().T


def _normalized_x(tdobj, root):
    x = np.asarray(tdobj.xy[root][0]).ravel()
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise RuntimeError('SA-SF-TDA root has near-zero amplitude norm')
    return x / norm


def _amplitude_overlap(x_ref, tdobj, root):
    x = _normalized_x(tdobj, root)
    return abs(np.dot(x_ref.conj(), x))


def _hybrid_coefficients(mf):
    from pyscf import dft
    if isinstance(mf, dft.KohnShamDFT):
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
        hybrid = mf._numint.libxc.is_hybrid_xc(mf.xc)
        return hybrid, hyb, omega, alpha
    return True, 1.0, 0.0, 0.0


def roks_spaces(tdobj):
    mo_occ = tdobj._scf.mo_occ
    return (
        np.where(mo_occ == 2)[0],
        np.where(mo_occ == 1)[0],
        np.where(mo_occ == 0)[0],
    )


def pack_roks_kappa(zco, zcv, zov):
    return np.hstack((zco.ravel(), zcv.ravel(), zov.ravel()))


def unpack_roks_kappa(vec, nc, no, nv):
    nco = no * nc
    ncv = nv * nc
    zco = vec[:nco].reshape(no, nc)
    zcv = vec[nco:nco + ncv].reshape(nv, nc)
    zov = vec[nco + ncv:].reshape(nv, no)
    return zco, zcv, zov
