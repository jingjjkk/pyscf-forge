#!/usr/bin/env python

import unittest

import numpy as np

from pyscf import gto
from pyscf import lib
from pyscf.sftda.satda import SATDA


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        C   0.000000   0.000000   0.000000
        O   0.000000   0.000000   1.206000
        H   0.000000   0.942000  -0.587000
        H   0.000000  -0.942000  -0.587000
        '''
        mol.spin = 2
        mol.basis = 'sto-3g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def make_mf(self):
        return self.mol.ROKS(xc='HF').set(conv_tol=1e-10, verbose=0).run()

    def make_td(self, deltaS=-1, nstates=3):
        td = SATDA(self.make_mf()).set(deltaS=deltaS, nstates=nstates,
                                      verbose=0, conv_tol=1e-8,
                                      max_cycle=100)
        td.kernel()
        self.assertTrue(np.all(td.converged))
        return td

    def assert_unreliable_satda_nac(self, cm):
        msg = str(cm.exception)
        self.assertIn('SATDA NAC is not enabled', msg)
        self.assertIn('ordinary spin-flip determinant overlap', msg)
        self.assertIn('diagonal excitation-gradient limit', msg)

    def test_finite_diff_deltaS_minus1_hf_disabled(self):
        td = self.make_td()
        with self.assertRaises(NotImplementedError) as cm:
            td.NAC().set(verbose=0, root_overlap_tol=0.2).kernel(
                state_I=2, state_J=3, atmlst=[1], method='finite_diff',
                ediff=True, step=5e-4)
        self.assert_unreliable_satda_nac(cm)

    def test_awf_overlap_disabled(self):
        td = self.make_td()
        nac_mod = td.NAC().__class__.__module__
        tdsatda = __import__(nac_mod, fromlist=['awf_overlap'])
        with self.assertRaises(NotImplementedError) as cm:
            tdsatda.awf_overlap(td, td.xy[1], td, td.xy[2])
        self.assert_unreliable_satda_nac(cm)

    def test_nac_csf_disabled(self):
        td = self.make_td()
        nac_mod = td.NAC().__class__.__module__
        tdsatda = __import__(nac_mod, fromlist=['nac_csf'])
        with self.assertRaises(NotImplementedError) as cm:
            tdsatda.nac_csf(td.NAC(), td.xy[1], td.xy[2], atmlst=[1])
        self.assert_unreliable_satda_nac(cm)

    def test_hf_interstate_numerator_disabled(self):
        td = self.make_td()
        nac_mod = td.NAC().__class__.__module__
        tdsatda = __import__(nac_mod, fromlist=['get_hf_interstate_numerator'])
        with self.assertRaises(NotImplementedError) as cm:
            tdsatda.get_hf_interstate_numerator(
                td.NAC(), td.xy[1], td.xy[2], atmlst=[1], verbose=0)
        self.assert_unreliable_satda_nac(cm)

    def test_analytic_experimental_deltaS_minus1_hf_not_enabled(self):
        td = self.make_td()
        with self.assertRaises(NotImplementedError) as cm:
            td.NAC().set(verbose=0, root_overlap_tol=0.2).kernel(
                state_I=2, state_J=3, atmlst=[1],
                method='analytic_experimental', ediff=True)
        self.assert_unreliable_satda_nac(cm)

    def test_analytic_experimental_deltaS_0_not_implemented(self):
        td = self.make_td(deltaS=0, nstates=2)
        with self.assertRaises(NotImplementedError):
            td.NAC().set(verbose=0).kernel(
                state_I=1, state_J=2, method='analytic_experimental')


if __name__ == '__main__':
    unittest.main()
