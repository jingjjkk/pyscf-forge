#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

'''
Finite-difference nuclear gradients for SA-SF-TDA.

This module provides the PySCF-style gradient interface for
``pyscf.sftda.satda.SATDA``.  The analytical SA-SF-TDA gradient is not
implemented here; the finite-difference backend is intended as a reference
interface and validation tool.
'''

import numpy as np

from pyscf import dft
from pyscf import lib
from pyscf import scf
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger


def _normalized_x(tdobj, root):
    x = np.asarray(tdobj.xy[root][0]).ravel()
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise RuntimeError('SA-SF-TDA root has near-zero amplitude norm')
    return x / norm


def _amplitude_overlap(x_ref, tdobj, root):
    x = _normalized_x(tdobj, root)
    return abs(np.vdot(x_ref, x))


def _copy_scf_settings(mf_ref, mf):
    mf.verbose = 0
    for key in (
        'max_memory', 'conv_tol', 'conv_tol_grad', 'max_cycle', 'diis_space',
        'diis_start_cycle', 'level_shift', 'damp', 'direct_scf',
    ):
        if hasattr(mf_ref, key) and hasattr(mf, key):
            setattr(mf, key, getattr(mf_ref, key))

    if isinstance(mf_ref, dft.KohnShamDFT):
        mf.grids.level = mf_ref.grids.level
        mf.grids.prune = mf_ref.grids.prune
        mf.grids.radi_method = mf_ref.grids.radi_method
        mf.grids.becke_scheme = mf_ref.grids.becke_scheme
        mf.small_rho_cutoff = mf_ref.small_rho_cutoff
    return mf


def _make_displaced_mf(mf_ref, mol):
    if isinstance(mf_ref, (dft.roks.ROKS, dft.rks_symm.SymAdaptedROKS)):
        mf = mol.ROKS(xc=mf_ref.xc)
    elif isinstance(mf_ref, (scf.rohf.ROHF, scf.hf_symm.SymAdaptedROHF)):
        mf = mol.ROHF()
    else:
        raise NotImplementedError(
            'SATDA finite-difference gradients currently support ROKS/ROHF '
            'references only'
        )
    return _copy_scf_settings(mf_ref, mf)


class Gradients(rhf_grad.GradientsBase):
    '''Finite-difference gradients for :class:`pyscf.sftda.satda.SATDA`.

    The returned gradient is for the total excited-state energy

        E_ref + omega_SATDA

    matching the convention of PySCF TD gradient objects.  Root tracking is
    performed by overlap of the flattened SATDA X amplitudes with the reference
    root, which supports both ``deltaS=-1`` and ``deltaS=0`` amplitude layouts.
    '''

    _keys = rhf_grad.GradientsBase._keys | {
        'state', 'step', 'nstates', 'root_overlap_tol', 'method',
    }

    def __init__(self, td):
        rhf_grad.GradientsBase.__init__(self, td)
        self.state = 1
        self.step = 1e-3
        self.nstates = td.nstates
        self.root_overlap_tol = 0.7
        self.method = 'finite_diff'

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** finite-difference SATDA gradients for %s ********',
                 self.base.__class__)
        log.info('State ID = %d', self.state)
        log.info('step = %.6g Bohr', self.step)
        log.info('nstates = %d', self.nstates)
        log.info('root_overlap_tol = %.6g', self.root_overlap_tol)
        log.info('method = %s', self.method)
        log.info('unit = Eh/Bohr')
        if self.method == 'finite_diff':
            log.warn('SATDA analytical gradients are not implemented yet; '
                     'using central finite differences of E_ref + omega.')
        return self

    def _run_td_at(self, coords_bohr, x_ref=None):
        mol = self.mol.copy()
        mol.set_geom_(coords_bohr, unit='Bohr')

        mf = _make_displaced_mf(self.base._scf, mol)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Displaced ROKS/ROHF reference did not converge')

        td = self.base.__class__(mf)
        td.deltaS = self.base.deltaS
        td.nstates = self.nstates
        td.conv_tol = self.base.conv_tol
        td.lindep = self.base.lindep
        td.max_cycle = self.base.max_cycle
        td.max_memory = self.base.max_memory
        td.verbose = 0
        td.kernel()

        if x_ref is None:
            root = self.state - 1
            overlap = 1.0
        else:
            overlaps = np.array([_amplitude_overlap(x_ref, td, i)
                                 for i in range(len(td.e))])
            root = int(np.argmax(overlaps))
            overlap = overlaps[root]
            if overlap < self.root_overlap_tol:
                raise RuntimeError(
                    'State tracking failed: best overlap %.6f below threshold '
                    '%.6f' % (overlap, self.root_overlap_tol)
                )

        if not np.asarray(td.converged)[root]:
            raise RuntimeError(
                'Displaced SATDA tracked root %d did not converge' % (root + 1)
            )

        return mf.e_tot + td.e[root], root, overlap

    def _kernel_finite_diff(self, atmlst):
        coords0 = self.mol.atom_coords()
        x_ref = _normalized_x(self.base, self.state - 1)
        atmlst = tuple(atmlst)
        de = np.zeros((len(atmlst), 3))

        for k, ia in enumerate(atmlst):
            for xyz in range(3):
                coords_p = coords0.copy()
                coords_m = coords0.copy()
                coords_p[ia, xyz] += self.step
                coords_m[ia, xyz] -= self.step

                e_p, root_p, ovlp_p = self._run_td_at(coords_p, x_ref)
                e_m, root_m, ovlp_m = self._run_td_at(coords_m, x_ref)
                de[k, xyz] = (e_p - e_m) / (2 * self.step)

                logger.debug(
                    self,
                    'atom %d xyz %d roots %+d/%+d overlaps %.6f/%.6f',
                    ia, xyz, root_p + 1, root_m + 1, ovlp_p, ovlp_m,
                )
        return de

    def _kernel_analytic(self, xy, atmlst):
        raise NotImplementedError(
            'SATDA analytical nuclear gradients are not implemented yet.  Use '
            'method="finite_diff" for central finite differences of '
            'E_ref + omega.'
        )

    def kernel(self, state=None, atmlst=None, step=None, method=None):
        if state is not None:
            self.state = state
        if step is not None:
            self.step = step
        if method is not None:
            self.method = method
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)

        if self.state == 0:
            logger.warn(self, 'state=0 requested; returning ground-state gradient')
            return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

        if self.base.xy is None:
            self.base.run()
        if self.state < 1 or self.state > len(self.base.xy):
            raise ValueError('state must be in [1, %d]' % len(self.base.xy))

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        xy = self.base.xy[self.state - 1]
        if self.method == 'finite_diff':
            de = self._kernel_finite_diff(atmlst)
        elif self.method == 'analytic':
            de = self._kernel_analytic(xy, atmlst)
        else:
            raise ValueError('Unknown SATDA gradient method %s' % self.method)

        self.de = de
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------- finite-difference SATDA gradients '
                        'for state %d ----------', self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '--------------------------------------------')


Grad = Gradients
