#!/usr/bin/env python

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm

from pyscf import gto
from pyscf import lib
from pyscf.sftda.satda import SATDA
from pyscf.sftda.satda import gen_rohf_response_sf


def normalized_x(td, root):
    x = np.asarray(td.xy[root][0]).ravel()
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise RuntimeError('SA-SF-TDA root has near-zero amplitude norm')
    return x / norm


def amplitude_overlap(x_ref, td, root):
    x = normalized_x(td, root)
    return abs(np.vdot(x_ref, x))


def load_tdsatda_delta():
    pkg_dir = Path(__file__).resolve().parents[1] / 'tdsatda_delta'
    name = 'pyscf_forge_tdsatda_delta'
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(
        name, pkg_dir / '__init__.py',
        submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _GradShim:
    """Minimal gradient object for direct helper-level SATDA delta tests."""

    def __init__(self, td):
        self.base = td
        self.mol = td.mol
        self._mf_grad = td._scf.nuc_grad_method().set(verbose=0)

    def __getattr__(self, name):
        return getattr(self._mf_grad, name)


class _FrozenTD:
    def __init__(self, mf, nstates=1):
        self._scf = mf
        self.mol = mf.mol
        self.nstates = nstates


def _copy_grid_settings(src, dst):
    dst.grids.level = src.grids.level
    dst.grids.prune = src.grids.prune
    dst.grids.radi_method = src.grids.radi_method
    dst.grids.becke_scheme = src.grids.becke_scheme
    return dst


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

    def make_mf(self, xc='HF'):
        mf = self.mol.ROKS(xc=xc).set(conv_tol=1e-10)
        if xc.upper() != 'HF':
            mf.grids.level = 1
        return mf.run()

    def make_td(self, deltaS, nstates=4, xc='HF'):
        mf = self.make_mf(xc=xc)
        td = SATDA(mf).set(deltaS=deltaS, nstates=nstates,
                           verbose=0, conv_tol=1e-8)
        td.kernel()
        self.assertTrue(np.all(td.converged))
        return td

    def make_td_lda_unpruned(self, nstates=3):
        mf = self.mol.ROKS(xc='SVWN').set(conv_tol=1e-10, verbose=0)
        mf.grids.level = 1
        mf.grids.prune = None
        mf.kernel()
        td = SATDA(mf).set(deltaS=-1, nstates=nstates,
                           verbose=0, conv_tol=1e-8)
        td.kernel()
        self.assertTrue(np.all(td.converged))
        return td

    def run_satda_at(self, coords_bohr, deltaS, nstates, x_ref=None,
                     target_state=1, xc='HF'):
        mol = self.mol.copy()
        mol.set_geom_(coords_bohr, unit='Bohr')
        mf = mol.ROKS(xc=xc).set(conv_tol=1e-10, verbose=0)
        if xc.upper() != 'HF':
            mf.grids.level = 1
        mf.kernel()
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

    def test_analytic_experimental_deltaS_minus1_hf_matches_finite_diff(self):
        td = self.make_td(deltaS=-1, nstates=4)
        state = 2
        grad_analytic = td.Gradients().set(
            verbose=0, root_overlap_tol=0.2).kernel(
                state=state, method='analytic_experimental')
        grad_fd = td.Gradients().set(
            verbose=0, root_overlap_tol=0.2).kernel(
                state=state, step=2e-4, method='finite_diff')
        self.assertAlmostEqual(abs(grad_analytic - grad_fd).max(), 0, 3)

    def test_analytic_experimental_deltaS_minus1_lda_matches_finite_diff(self):
        td = self.make_td(deltaS=-1, nstates=3, xc='SVWN')
        state = 2
        grad_analytic = td.Gradients().set(
            verbose=0, root_overlap_tol=0.2).kernel(
                state=state, method='analytic_experimental')
        grad_fd = td.Gradients().set(
            verbose=0, root_overlap_tol=0.2).kernel(
                state=state, step=2e-4, method='finite_diff')
        self.assertAlmostEqual(abs(grad_analytic - grad_fd).max(), 0, 4)

    def test_analytic_experimental_deltaS_0_not_implemented(self):
        td = self.make_td(deltaS=0, nstates=3)
        with self.assertRaises(NotImplementedError):
            td.Gradients().set(verbose=0).kernel(
                state=1, method='analytic_experimental')

    def test_finite_diff_total_gradient_matches_independent_reference(self):
        td = self.make_td(deltaS=-1, nstates=4)
        state = 4
        step = 1e-3
        grad_interface = td.Gradients().set(
            verbose=0, root_overlap_tol=0.2).kernel(state=state, step=step)
        grad_ref = self.independent_total_finite_diff(td, state, step)
        self.assertAlmostEqual(abs(grad_interface - grad_ref).max(), 0, 8)

    def test_migrated_hf_delta_zvec_helper_smoke(self):
        satda_delta_gradient_zvec = (
            load_tdsatda_delta().satda_delta_gradient_zvec)

        td = self.make_td(deltaS=-1, nstates=4)
        xy = td.xy[1]
        grad, details = satda_delta_gradient_zvec(
            _GradShim(td), td, xy, atmlst=range(td.mol.natm))

        self.assertEqual(grad.shape, (td.mol.natm, 3))
        self.assertEqual(details['de_direct'].shape, grad.shape)
        self.assertEqual(details['de_orbital'].shape, grad.shape)
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_migrated_hf_delta_zvec_krylov_matches_dense(self):
        delta = load_tdsatda_delta()
        zsolver = importlib.import_module(delta.__name__ + '._zvec_solver')

        td = self.make_td(deltaS=-1, nstates=4)
        grad_obj = _GradShim(td)
        xy = td.xy[1]

        href, pairs, _ = zsolver.build_roks_hessian_reference(td)
        hnew, pairs_new, _ = zsolver.build_roks_hessian(td)
        self.assertEqual(pairs, pairs_new)
        self.assertAlmostEqual(abs(hnew - href).max(), 0, 10)

        action_t, pairs_t, _ = zsolver.make_roks_hessian_transpose_action(
            td, pairs)
        eye = np.eye(len(pairs))
        ht_action = np.column_stack([action_t(e) for e in eye])
        self.assertAlmostEqual(abs(ht_action - hnew.T).max(), 0, 10)

        grad_dense, details_dense = delta.satda_delta_gradient_zvec(
            grad_obj, td, xy, atmlst=range(td.mol.natm),
            hessian_solver='dense')
        grad_krylov, details_krylov = delta.satda_delta_gradient_zvec(
            grad_obj, td, xy, atmlst=range(td.mol.natm),
            hessian_solver='krylov')
        self.assertEqual(details_krylov['hessian_solver'], 'krylov')
        self.assertLess(details_krylov['zvec_residual'], 1e-8)
        self.assertAlmostEqual(abs(grad_krylov - grad_dense).max(), 0, 8)

    def test_migrated_lda_fock_basis_uses_satda_fockz(self):
        delta = load_tdsatda_delta()
        fock_basis = importlib.import_module(delta.__name__ + '._fock_basis')

        td = self.make_td_lda_unpruned(nstates=3)
        fbasis = fock_basis.make_fock_basis(td._scf)
        _, fockz_ref = gen_rohf_response_sf(
            td._scf, mo_coeff=td._scf.mo_coeff, mo_occ=td._scf.mo_occ,
            hermi=0, max_memory=td._scf.max_memory,
        )
        self.assertAlmostEqual(abs(fbasis.fockz - fockz_ref).max(), 0, 12)
        self.assertAlmostEqual(
            abs(fbasis.fock0 - (td._scf.get_fock().focka - fockz_ref)).max(),
            0, 12,
        )

    def test_migrated_lda_xc_m_matrix_matches_orbital_fd(self):
        delta = load_tdsatda_delta()
        xc_lda = importlib.import_module(delta.__name__ + '._xc_lda')
        zsolver = importlib.import_module(delta.__name__ + '._zvec_solver')

        td = self.make_td_lda_unpruned(nstates=3)
        xy = td.xy[1]
        pairs = zsolver._canonical_roks_pairs(td)
        m = xc_lda.lda_xc_m_matrix(td, xy)
        analytic = zsolver.pack_mvec(m, pairs)

        step = 1e-5
        mo0 = td._scf.mo_coeff
        fd = np.zeros_like(analytic)
        eye = np.eye(len(pairs))
        for ipair in range(len(pairs)):
            kappa = zsolver._anti_mo_from_roks_canonical_vec(
                mo0.shape[1], pairs, eye[ipair])
            mf_p = td._scf.copy()
            mf_m = td._scf.copy()
            mf_p.mo_coeff = mo0 @ expm(step * kappa)
            mf_m.mo_coeff = mo0 @ expm(-step * kappa)
            e_p = xc_lda.lda_xc_energy(_FrozenTD(mf_p, td.nstates), xy)
            e_m = xc_lda.lda_xc_energy(_FrozenTD(mf_m, td.nstates), xy)
            fd[ipair] = (e_p - e_m) / (2 * step)

        self.assertAlmostEqual(abs(analytic - fd).max(), 0, 6)

    def test_migrated_lda_xc_direct_matches_frozen_fd(self):
        delta = load_tdsatda_delta()
        xc_lda = importlib.import_module(delta.__name__ + '._xc_lda')

        td = self.make_td_lda_unpruned(nstates=3)
        xy = td.xy[1]
        grad_obj = _GradShim(td)
        analytic = xc_lda.lda_xc_direct_de(
            grad_obj, td, xy, atmlst=range(td.mol.natm))

        coords0 = td.mol.atom_coords()
        step = 1e-4
        fd = np.zeros_like(analytic)

        def energy_at(coords):
            mol = td.mol.copy()
            mol.set_geom_(coords, unit='Bohr')
            mf = mol.ROKS(xc=td._scf.xc).set(verbose=0)
            _copy_grid_settings(td._scf, mf)
            mf.mo_coeff = td._scf.mo_coeff
            mf.mo_occ = td._scf.mo_occ
            mf.max_memory = td._scf.max_memory
            return xc_lda.lda_xc_energy(_FrozenTD(mf, td.nstates), xy)

        for ia in range(td.mol.natm):
            for xyz in range(3):
                cp = coords0.copy()
                cm = coords0.copy()
                cp[ia, xyz] += step
                cm[ia, xyz] -= step
                fd[ia, xyz] = (energy_at(cp) - energy_at(cm)) / (2 * step)

        self.assertAlmostEqual(abs(analytic - fd).max(), 0, 5)

    def test_migrated_lda_total_delta_gradient_not_silently_enabled(self):
        delta = load_tdsatda_delta()
        td = self.make_td_lda_unpruned(nstates=3)
        with self.assertRaises(NotImplementedError):
            delta.satda_delta_gradient_zvec(
                _GradShim(td), td, td.xy[1], atmlst=range(td.mol.natm))


if __name__ == '__main__':
    unittest.main()
