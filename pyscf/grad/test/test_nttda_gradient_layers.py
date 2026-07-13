#!/usr/bin/env python
"""Layered M-matrix and frozen-orbital checks for NTTDA gradients."""

import unittest
from pathlib import Path

import numpy as np

import pyscf


ROOT = Path(__file__).resolve().parents[3]
PYSCF_PATH = str(ROOT / "pyscf")
if PYSCF_PATH not in pyscf.__path__:
    pyscf.__path__.insert(0, PYSCF_PATH)

import pyscf.grad
import pyscf.sftda
from pyscf import dft, gto

for path_list, path in (
        (pyscf.grad.__path__, ROOT / "pyscf" / "grad"),
        (pyscf.sftda.__path__, ROOT / "pyscf" / "sftda")):
    value = str(path)
    if value in path_list:
        path_list.remove(value)
    path_list.insert(0, value)

from pyscf.grad.nttda.delta_s_minus_one import (  # noqa: E402
    grad_elec as lowering_grad_elec,
    spin_lowering_ledger_scalar,
)
from pyscf.grad.nttda.delta_s_zero import (  # noqa: E402
    grad_elec as same_spin_grad_elec,
    same_spin_ledger_scalar,
)
from pyscf.sftda.nttda import NTTDA  # noqa: E402


FUNCTIONALS = ("HF", "SVWN", "PBE", "TPSS", "M06-2X", "CAM-B3LYP")


def make_molecule():
    return gto.M(
        atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
        basis="sto-3g",
        spin=2,
        unit="Bohr",
        verbose=0,
    )


def make_reference(mol, xc):
    mf = dft.ROKS(mol).set(
        xc=xc,
        conv_tol=1e-14,
        conv_tol_grad=1e-11,
        max_cycle=200,
        verbose=0,
    )
    mf.grids.level = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("ROKS reference did not converge")
    return mf


def exact_state_two(mf, delta_s, nobeta):
    tdobj = NTTDA(mf).set(
        deltaS=delta_s,
        nobeta=nobeta,
        nstates=3,
        max_memory=mf.max_memory,
        verbose=0,
    )
    if delta_s == 0:
        vind, diagonal = tdobj.gen_vind_sc()
    else:
        vind, diagonal = tdobj.gen_vind_sfd()
    size = diagonal.size
    rows = np.asarray(vind(np.eye(size))).reshape(size, size)
    if np.max(np.abs(rows - rows.T)) >= 1e-10:
        raise AssertionError("NTTDA action is not symmetric")
    energies, vectors = np.linalg.eigh(0.5 * (rows + rows.T))
    if delta_s == -1:
        vectors = vectors[:, np.abs(energies) > 1e-8]
    vector = vectors[:, 1]
    if delta_s == -1:
        nc = np.count_nonzero(mf.mo_occ == 2)
        no = np.count_nonzero(mf.mo_occ == 1)
        nv = np.count_nonzero(mf.mo_occ == 0)
        vector = vector.reshape(nc + no, no + nv)
    return tdobj, (vector, 0)


def gradient_components(mf, tdobj, xy, atmlst):
    builder = lowering_grad_elec if tdobj.deltaS == -1 else same_spin_grad_elec
    return builder(mf.nuc_grad_method(), tdobj, xy, atmlst=atmlst)


def channel_scalar(tdobj, xy):
    if tdobj.deltaS == -1:
        return spin_lowering_ledger_scalar(tdobj, xy)
    return same_spin_ledger_scalar(tdobj, xy)


def frozen_scalar_at(base_mf, tdobj, xy, coords):
    mol = base_mf.mol.copy()
    mol.set_geom_(coords, unit="Bohr")
    mf = dft.ROKS(mol).set(xc=base_mf.xc, verbose=0)
    mf.grids.coords = np.array(base_mf.grids.coords, copy=True)
    mf.grids.weights = np.array(base_mf.grids.weights, copy=True)
    mf.grids.non0tab = None
    mf.mo_coeff = np.array(base_mf.mo_coeff, copy=True)
    mf.mo_occ = np.array(base_mf.mo_occ, copy=True)
    mf.mo_energy = np.array(base_mf.mo_energy, copy=True)
    displaced = NTTDA(mf).set(
        deltaS=tdobj.deltaS,
        nobeta=tdobj.nobeta,
        max_memory=tdobj.max_memory,
        verbose=0,
    )
    return channel_scalar(displaced, xy)


class GradientLayerChecks(unittest.TestCase):
    def test_full_m_matrix_for_both_channels_and_fock_modes(self):
        rng = np.random.default_rng(103)
        for delta_s in (-1, 0):
            for xc in FUNCTIONALS:
                mf = make_reference(make_molecule(), xc)
                for nobeta in (False, True):
                    tdobj, xy = exact_state_two(mf, delta_s, nobeta)
                    result = gradient_components(mf, tdobj, xy, atmlst=())
                    nmo = mf.mo_coeff.shape[1]
                    perturbation = rng.normal(size=(nmo, nmo))
                    original = np.array(mf.mo_coeff, copy=True)
                    step = 1e-6
                    values = []
                    try:
                        for sign in (1.0, -1.0):
                            mf.mo_coeff = original @ (
                                np.eye(nmo) + sign * step * perturbation
                            )
                            values.append(channel_scalar(tdobj, xy))
                    finally:
                        mf.mo_coeff = original
                    finite_difference = (values[0] - values[1]) / (2 * step)
                    analytic = np.trace(result.m_matrix.T @ perturbation)
                    with self.subTest(
                            delta_s=delta_s, xc=xc, nobeta=nobeta):
                        self.assertLess(abs(analytic - finite_difference), 1e-8)

    def test_direct_derivative_for_representative_functionals(self):
        cases = (
            (0, "HF", False),
            (0, "SVWN", False),
            (0, "PBE", False),
            (0, "TPSS", False),
            (0, "M06-2X", False),
            (0, "M06-2X", True),
            (0, "CAM-B3LYP", False),
            (-1, "HF", False),
            (-1, "PBE", False),
            (-1, "TPSS", False),
            (-1, "M06-2X", True),
        )
        step = 1e-4
        for delta_s, xc, nobeta in cases:
            mf = make_reference(make_molecule(), xc)
            tdobj, xy = exact_state_two(mf, delta_s, nobeta)
            result = gradient_components(
                mf, tdobj, xy, atmlst=range(mf.mol.natm),
            )
            coords0 = mf.mol.atom_coords()
            finite_difference = np.zeros_like(coords0)
            for atom in range(mf.mol.natm):
                for xyz in range(3):
                    coords_plus = coords0.copy()
                    coords_minus = coords0.copy()
                    coords_plus[atom, xyz] += step
                    coords_minus[atom, xyz] -= step
                    value_plus = frozen_scalar_at(
                        mf, tdobj, xy, coords_plus,
                    )
                    value_minus = frozen_scalar_at(
                        mf, tdobj, xy, coords_minus,
                    )
                    finite_difference[atom, xyz] = (
                        (value_plus - value_minus) / (2 * step)
                    )
            error = np.max(np.abs(result.direct - finite_difference))
            threshold = 1e-6 if xc == "HF" else 3e-6
            with self.subTest(delta_s=delta_s, xc=xc, nobeta=nobeta):
                self.assertLess(error, threshold)


if __name__ == "__main__":
    unittest.main()
