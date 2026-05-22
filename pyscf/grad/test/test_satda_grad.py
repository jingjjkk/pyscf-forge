#!/usr/bin/env python

import unittest

import numpy as np

from pyscf import gto
from pyscf import lib
from pyscf.sftda.satda import SATDA


def normalized_x(td, root):
    x = np.asarray(td.xy[root][0]).ravel()
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise RuntimeError('SA-SF-TDA root has near-zero amplitude norm')
    return x / norm


def amplitude_overlap(x_ref, td, root):
    x = normalized_x(td, root)
    return abs(np.vdot(x_ref, x))


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = 'O 0 0 0; O 0 0 1.2075'
        mol.spin = 2
        mol.basis = 'sto-3g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def make_mf(self):
        return self.mol.ROKS(xc='HF').set(conv_tol=1e-10).run()

    def make_td(self, deltaS, nstates=4):
        mf = self.make_mf()
        td = SATDA(mf).set(deltaS=deltaS, nstates=nstates,
                           verbose=0, conv_tol=1e-8)
        td.kernel()
        self.assertTrue(np.all(td.converged))
        return td

    def run_satda_at(self, coords_bohr, deltaS, nstates, x_ref=None,
                     target_state=1):
        mol = self.mol.copy()
        mol.set_geom_(coords_bohr, unit='Bohr')
        mf = mol.ROKS(xc='HF').set(conv_tol=1e-10, verbose=0).run()
        td = SATDA(mf).set(deltaS=deltaS, nstates=nstates,
                           verbose=0, conv_tol=1e-8)
        td.kernel()
        self.assertTrue(np.all(td.converged))

        if x_ref is None:
            root = target_state - 1
            overlap = 1.0
        else:
            overlaps = np.array([amplitude_overlap(x_ref, td, i)
                                 for i in range(len(td.e))])
            root = int(np.argmax(overlaps))
            overlap = overlaps[root]
            self.assertGreater(overlap, 0.2)
        return mf.e_tot + td.e[root], root, overlap

    def independent_total_finite_diff(self, td, state, step):
        coords0 = self.mol.atom_coords()
        x_ref = normalized_x(td, state - 1)
        grad = np.zeros_like(coords0)

        for ia in range(self.mol.natm):
            for xyz in range(3):
                coords_p = coords0.copy()
                coords_m = coords0.copy()
                coords_p[ia, xyz] += step
                coords_m[ia, xyz] -= step
                e_p, _, _ = self.run_satda_at(
                    coords_p, td.deltaS, td.nstates, x_ref, state)
                e_m, _, _ = self.run_satda_at(
                    coords_m, td.deltaS, td.nstates, x_ref, state)
                grad[ia, xyz] = (e_p - e_m) / (2 * step)
        return grad

    def test_gradients_interface_deltaS_minus1_hf(self):
        td = self.make_td(deltaS=-1, nstates=4)
        grad = td.Gradients().set(verbose=0, root_overlap_tol=0.2).kernel(
            state=4, step=1e-3)
        self.assertEqual(grad.shape, (2, 3))
        self.assertAlmostEqual(abs(grad.sum(axis=0)).max(), 0, 6)

    def test_gradients_interface_deltaS_0_hf(self):
        td = self.make_td(deltaS=0, nstates=3)
        grad = td.Gradients().set(verbose=0, root_overlap_tol=0.2).kernel(
            state=1, step=1e-3)
        self.assertEqual(grad.shape, (2, 3))
        self.assertAlmostEqual(abs(grad.sum(axis=0)).max(), 0, 6)

    def test_state0_ground_state_gradient(self):
        td = self.make_td(deltaS=-1, nstates=2)
        grad = td.Gradients().set(verbose=0).kernel(state=0)
        ref = td._scf.nuc_grad_method().set(verbose=0).kernel()
        self.assertAlmostEqual(abs(grad - ref).max(), 0, 10)

    def test_analytic_not_implemented(self):
        td = self.make_td(deltaS=-1, nstates=2)
        with self.assertRaises(NotImplementedError):
            td.Gradients().set(verbose=0).kernel(state=1, method='analytic')

    def test_finite_diff_total_gradient_matches_independent_reference(self):
        td = self.make_td(deltaS=-1, nstates=4)
        state = 4
        step = 1e-3
        grad_interface = td.Gradients().set(
            verbose=0, root_overlap_tol=0.2).kernel(state=state, step=step)
        grad_ref = self.independent_total_finite_diff(td, state, step)
        self.assertAlmostEqual(abs(grad_interface - grad_ref).max(), 0, 8)


if __name__ == '__main__':
    unittest.main()
