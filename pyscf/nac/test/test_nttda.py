#!/usr/bin/env python
"""Acceptance tests for analytic NTTDA ``deltaS=-1`` NACs."""

import unittest
from pathlib import Path

import numpy as np
import pyscf


ROOT = Path(__file__).resolve().parents[3]
PYSCF_PATH = str(ROOT / "pyscf")
if PYSCF_PATH not in pyscf.__path__:
    pyscf.__path__.insert(0, PYSCF_PATH)

import pyscf.grad  # noqa: E402
import pyscf.nac  # noqa: E402
import pyscf.sftda  # noqa: E402

for package, path in (
        (pyscf.grad, ROOT / "pyscf" / "grad"),
        (pyscf.nac, ROOT / "pyscf" / "nac"),
        (pyscf.sftda, ROOT / "pyscf" / "sftda")):
    path = str(path)
    if path not in package.__path__:
        package.__path__.insert(0, path)

from pyscf import dft, gto  # noqa: E402
from pyscf.nac.nttda import (  # noqa: E402
    awf_overlap,
    build_spin_adapted_awf,
    get_hf_interstate_numerator,
    nac_csf_components,
)
from pyscf.sftda.nttda import NTTDA  # noqa: E402


class NTTDANACTests(unittest.TestCase):
    @staticmethod
    def molecule():
        return gto.M(
            atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )

    def make_td(self, xc="HF", nobeta=False):
        mf = dft.ROKS(self.molecule()).set(
            xc=xc,
            conv_tol=1e-12,
            conv_tol_grad=1e-10,
            max_cycle=200,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        self.assertTrue(mf.converged)
        tdobj = NTTDA(mf).set(
            deltaS=-1,
            nobeta=nobeta,
            nstates=3,
            conv_tol=1e-9,
            max_cycle=200,
            verbose=0,
        ).run()
        self.assertGreaterEqual(len(tdobj.xy), 3)
        return tdobj

    def test_spin_adapted_awf_metric(self):
        tdobj = self.make_td("HF")
        for xy in tdobj.xy[:3]:
            self.assertAlmostEqual(
                build_spin_adapted_awf(tdobj, xy).norm, 1.0, places=11,
            )
            self.assertAlmostEqual(
                float(awf_overlap(tdobj, xy, tdobj, xy)), 1.0, places=11,
            )
        self.assertLess(
            abs(awf_overlap(tdobj, tdobj.xy[0], tdobj, tdobj.xy[1])),
            1e-11,
        )

    def test_hf_symmetry_etf_and_diagonal_limit(self):
        tdobj = self.make_td("HF")
        nac = tdobj.NAC().set(verbose=0, cphf_conv_tol=1e-10)
        hf_12 = get_hf_interstate_numerator(
            nac, tdobj.xy[0], tdobj.xy[1], atmlst=[0], verbose=0,
        )
        hf_21 = get_hf_interstate_numerator(
            nac, tdobj.xy[1], tdobj.xy[0], atmlst=[0], verbose=0,
        )
        np.testing.assert_allclose(hf_12, hf_21, atol=1e-10)

        diagonal = get_hf_interstate_numerator(
            nac, tdobj.xy[0], tdobj.xy[0], atmlst=[0], verbose=0,
        )
        gradient = tdobj.Gradients().set(verbose=0).grad_elec(
            tdobj.xy[0], atmlst=[0],
        )
        np.testing.assert_allclose(diagonal, gradient, atol=1e-10)

        etf = nac.kernel(
            state_I=1, state_J=2, ediff=True, use_etfs=True,
        )
        self.assertLess(np.max(np.abs(np.sum(etf, axis=0))), 1e-8)

    def test_complete_hf_matches_awf_overlap_finite_difference(self):
        tdobj = self.make_td("HF")
        nac = tdobj.NAC().set(
            verbose=0,
            cphf_conv_tol=1e-10,
            root_overlap_tol=0.5,
        )
        analytic = nac.kernel(
            state_I=1,
            state_J=2,
            atmlst=[0],
            ediff=True,
            use_etfs=False,
        )
        finite_difference = nac.finite_difference(
            state_I=1, state_J=2, atmlst=[0], step=2e-4,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=2e-5, rtol=0,
        )
        details = nac.nttda_nac_details
        self.assertLess(details["csf_residual"], 1e-7)

    def test_complete_pbe_matches_awf_overlap_finite_difference(self):
        tdobj = self.make_td("PBE")
        nac = tdobj.NAC().set(
            verbose=0,
            cphf_conv_tol=1e-9,
            root_overlap_tol=0.5,
            fixed_grid=True,
        )
        analytic = nac.kernel(
            state_I=1,
            state_J=2,
            atmlst=[0],
            ediff=True,
            use_etfs=False,
        )
        finite_difference = nac.finite_difference(
            state_I=1, state_J=2, atmlst=[0], step=2e-4,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=3e-5, rtol=0,
        )

    def test_xc_families_nobeta_and_state_exchange(self):
        for xc, nobeta in (
                ("SVWN", False),
                ("TPSS", False),
                ("M06-2X", False),
                ("CAM-B3LYP", False),
                ("M06-2X", True)):
            with self.subTest(xc=xc, nobeta=nobeta):
                tdobj = self.make_td(xc, nobeta=nobeta)
                nac = tdobj.NAC().set(
                    verbose=0, cphf_conv_tol=1e-8, fixed_grid=True,
                )
                forward = nac.kernel(
                    state_I=1, state_J=2, atmlst=[0],
                    ediff=True, use_etfs=False,
                )
                nac.reset_phase()
                reverse = nac.kernel(
                    state_I=2, state_J=1, atmlst=[0],
                    ediff=True, use_etfs=False,
                )
                self.assertTrue(np.all(np.isfinite(forward)))
                np.testing.assert_allclose(forward, -reverse, atol=1e-9)

    def test_csf_components_and_input_validation(self):
        tdobj = self.make_td("HF")
        nac = tdobj.NAC().set(verbose=0)
        components_12 = nac_csf_components(
            nac, tdobj.xy[0], tdobj.xy[1], atmlst=[0],
        )
        components_21 = nac_csf_components(
            nac, tdobj.xy[1], tdobj.xy[0], atmlst=[0],
        )
        np.testing.assert_allclose(
            components_12.total, -components_21.total, atol=1e-10,
        )
        np.testing.assert_allclose(
            components_12.total,
            components_12.ao + components_12.orbital,
            atol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            nac.kernel(state_I=1, state_J=1)
        with self.assertRaisesRegex(ValueError, "state_J"):
            nac.kernel(state_I=1, state_J=99)
        tdobj.deltaS = 0
        with self.assertRaisesRegex(NotImplementedError, "deltaS=-1"):
            nac.kernel(state_I=1, state_J=2)

    def test_scanner_recomputes_and_tracks_roots(self):
        tdobj = self.make_td("HF")
        scanner = tdobj.NAC().set(
            state_I=1,
            state_J=2,
            ediff=True,
            use_etfs=False,
            root_overlap_tol=0.5,
            verbose=0,
        ).as_scanner()
        initial = scanner(self.molecule())
        displaced = self.molecule()
        coordinates = displaced.atom_coords()
        coordinates[2, 1] += 1e-3
        displaced.set_geom_(coordinates, unit="Bohr")
        updated = scanner(displaced)
        self.assertEqual(updated.shape, (displaced.natm, 3))
        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertGreater(np.max(np.abs(updated - initial)), 1e-8)


if __name__ == "__main__":
    unittest.main()
