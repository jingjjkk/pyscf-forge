# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest import mock
import numpy as np
from pyscf import gto
from pyscf.sftda import nttda


REFS = {
    'HF': {
        True: {
            -1: np.array([-0.25588162251385593, 0.03179164805914653,
                          0.08216457752409057, 0.10984172789557056,
                          0.14436589862650057]),
            0: np.array([-0.021227306082553812, 0.03681224565830704,
                         0.055828674198227415, 0.10820095281084524,
                         0.13769598373589217]),
            1: np.array([0.2637303396826713, 0.3211458704926323,
                         0.3576719206041217, 0.4089546816647437,
                         0.48418822465431277]),
        },
        False: {
            -1: np.array([-0.25588162251385815, 0.03179164805915535,
                          0.08216457752408901, 0.10984172789556618,
                          0.14436589862650168]),
            0: np.array([-0.021227306082554027, 0.03681224565830669,
                         0.05582867419822887, 0.10820095281083697,
                         0.13769598373589134]),
            1: np.array([0.26373033968267307, 0.32114587049263676,
                         0.35767192060412084, 0.4089546816647443,
                         0.48418822465431444]),
        },
    },
    'SVWN': {
        True: {
            -1: np.array([-0.21165548027771708, 0.023016590669849873,
                          0.04408915291993318, 0.07135041294964697,
                          0.11506678142813023]),
            0: np.array([-0.0014098728485465067, 0.029604104153684127,
                         0.043162992283502455, 0.08689517072493848,
                         0.1786282801253175]),
            1: np.array([0.2619775484902895, 0.3145807011465694,
                         0.40080677626055994, 0.44868646268879425,
                         0.45213486508698836]),
        },
        False: {
            -1: np.array([-0.21170979048359168, 0.023046236405179832,
                          0.04399445674703403, 0.07151698987123586,
                          0.1149177949908444]),
            0: np.array([-0.0014102920144926014, 0.029534076286594643,
                         0.043327478728623726, 0.08675602648118973,
                         0.17868955257035488]),
            1: np.array([0.2621305574444208, 0.3146577468311684,
                         0.400854855533031, 0.4488089982219909,
                         0.45217145231139155]),
        },
    },
    'M062X': {
        True: {
            -1: np.array([-0.24815864298156592, 0.017580888798555155,
                          0.04750407979916185, 0.06869829978985995,
                          0.12080155100077003]),
            0: np.array([-0.006334558424146739, 0.026173936786711878,
                         0.03345446642434067, 0.09401057294611516,
                         0.15995688332197233]),
            1: np.array([0.27764573687858196, 0.32942929592876036,
                         0.39061540013417073, 0.4397100789480405,
                         0.4908735878545699]),
        },
        False: {
            -1: np.array([-0.24280053851498867, 0.011530030283297799,
                          0.05005354330269396, 0.06762698114448712,
                          0.12639763640154533]),
            0: np.array([-0.008184446338165025, 0.025150738879015422,
                         0.032777031664227074, 0.09911876211938828,
                         0.15953063092372488]),
            1: np.array([0.26880002289621757, 0.3280851476633962,
                         0.3822461897656717, 0.4389979233432141,
                         0.4818601324603382]),
        },
    },
    'CAM-B3LYP': {
        True: {
            -1: np.array([-0.22426403224297575, 0.022466394285794833,
                          0.05213215138974739, 0.08059986439783909,
                          0.120599597052892]),
            0: np.array([-0.004475899116949252, 0.034472834086202324,
                         0.04427544513019546, 0.09510982685230546,
                         0.17177368314729133]),
            1: np.array([0.2715321339948938, 0.3215785436368489,
                         0.38749208493006304, 0.43389268073125986,
                         0.47849030966408423]),
        },
        False: {
            -1: np.array([-0.22362676199942616, 0.02217598445976246,
                          0.0521859034687622, 0.08054218557201234,
                          0.12099781181846828]),
            0: np.array([-0.004488754315208394, 0.03433453230586325,
                         0.044209793487172355, 0.09543419827896533,
                         0.17172220940564042]),
            1: np.array([0.27064837890318744, 0.32121564431676974,
                         0.3866085753082913, 0.43353862650338093,
                         0.47757913242638816]),
        },
    },
}


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O                  0.64372820    0.14077399   -0.04477253
        O                 -0.64862595   -0.12779073   -0.05445498
        H                  1.16027512   -0.65947800    0.36730132
        H                 -1.12109306    0.55561188    0.42651873
        '''
        mol.charge = 0
        mol.spin = 2
        mol.basis = '631g'
        mol.symmetry = True
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def _check_functional(self, xc):
        mf = self.mol.ROKS(xc=xc).run()
        for nobeta, refs_by_delta_s in REFS[xc].items():
            for delta_s, ref in refs_by_delta_s.items():
                with self.subTest(xc=xc, nobeta=nobeta, deltaS=delta_s):
                    td = nttda.NTTDA(mf)
                    td.nstates = 5
                    td.deltaS = delta_s
                    td.nobeta = nobeta
                    td.verbose = 0
                    td.kernel()
                    self.assertTrue(np.all(td.converged))
                    np.testing.assert_allclose(td.e, ref, atol=1e-6, rtol=0)

    def test_hf_nttda(self):
        self._check_functional('HF')

    def test_svwn_nttda(self):
        self._check_functional('SVWN')

    def test_m062x_nttda(self):
        self._check_functional('M062X')

    def test_cam_b3lyp_nttda(self):
        self._check_functional('CAM-B3LYP')

    def test_mo_grid_fxc1_vind_matches_ao(self):
        rng = np.random.default_rng(12)
        for xc in ('BLYP', 'TPSS'):
            mf = self.mol.ROKS(xc=xc).run()
            for delta_s in (0, -1):
                with self.subTest(xc=xc, deltaS=delta_s):
                    td0 = nttda.NTTDA(mf)
                    td0.deltaS = delta_s
                    td0.verbose = 0
                    td1 = nttda.NTTDA(mf)
                    td1.deltaS = delta_s
                    td1.verbose = 0
                    with mock.patch.object(nttda, 'MO_GRID_FXC1', False):
                        vind0, hdiag0 = (td0.gen_vind_sc() if delta_s == 0
                                         else td0.gen_vind_sfd())
                    with mock.patch.object(nttda, 'MO_GRID_FXC1', True):
                        vind1, hdiag1 = (td1.gen_vind_sc() if delta_s == 0
                                         else td1.gen_vind_sfd())
                    np.testing.assert_allclose(hdiag1, hdiag0, atol=1e-10, rtol=0)
                    zs = rng.standard_normal((2, hdiag0.size))
                    np.testing.assert_allclose(vind1(zs), vind0(zs), atol=1e-9, rtol=0)


if __name__ == "__main__":
    print("Full Tests for noncollinear tensor TDA based on ROKS reference")
    unittest.main()
