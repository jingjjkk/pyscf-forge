"""PySCF-style SATDA gradient interface for the new block/delta package."""

import numpy as np

from pyscf import dft
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.sftda.satda import SATDA

from ._blocks import _amplitude_overlap, _normalized_x
from ._delta_grad import satda_delta_gradient_zvec
from ._fd import _make_displaced_mf
from ._sfbase_grad import satda_sfbase_gradient_zvec


def _copy_td_settings(src, dst):
    dst.nstates = src.nstates
    dst.conv_tol = src.conv_tol
    dst.lindep = src.lindep
    dst.max_cycle = src.max_cycle
    dst.max_memory = src.max_memory
    dst.verbose = 0
    return dst


class Gradients(rhf_grad.GradientsBase):
    """Gradient driver for :class:`pyscf.sftda.satda.SATDA`.

    The old monolithic ``pyscf.grad.tdsatda`` implementation was removed
    because its analytical assembly was not reliable.  This replacement keeps
    the SATDA public interface usable while the new block-resolved analytical
    path is assembled in ``tdsatda_delta``.

    ``method='analytic'`` is the trusted HF/ROKS ``deltaS=-1`` total gradient
    for ``E_ref + omega_SATDA``.  ``method='finite_diff'`` remains available
    as an expensive reference check.
    """

    _keys = rhf_grad.GradientsBase._keys | {
        'state', 'step', 'nstates', 'root_overlap_tol', 'method',
        'cphf_max_cycle', 'cphf_conv_tol', 'hessian_solver',
    }

    def __init__(self, tdobj):
        rhf_grad.GradientsBase.__init__(self, tdobj)
        self.state = 1
        self.step = 1e-3
        self.nstates = tdobj.nstates
        self.root_overlap_tol = 0.7
        self.method = 'analytic'
        self.cphf_max_cycle = 50
        self.cphf_conv_tol = 1e-8
        self.hessian_solver = 'dense'

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** SATDA gradients for %s ********', self.base.__class__)
        log.info('State ID = %d', self.state)
        log.info('step = %.6g Bohr', self.step)
        log.info('nstates = %d', self.nstates)
        log.info('root_overlap_tol = %.6g', self.root_overlap_tol)
        log.info('method = %s', self.method)
        log.info('hessian_solver = %s', self.hessian_solver)
        log.info('unit = Eh/Bohr')
        if self.method == 'finite_diff':
            log.warn('Using central finite differences of E_ref + omega_SATDA.')
        return self

    def _run_td_at(self, coords_bohr, x_ref=None):
        mol = self.mol.copy()
        mol.set_geom_(coords_bohr, unit='Bohr')

        mf = _make_displaced_mf(self.base._scf, mol)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Displaced ROKS/ROHF reference did not converge')

        tdobj = SATDA(mf).set(deltaS=self.base.deltaS)
        _copy_td_settings(self.base, tdobj)
        tdobj.kernel()

        if x_ref is None:
            root = self.state - 1
            overlap = 1.0
        else:
            overlaps = np.array([
                _amplitude_overlap(x_ref, tdobj, i)
                for i in range(len(tdobj.e))
            ])
            root = int(np.argmax(overlaps))
            overlap = overlaps[root]
            if overlap < self.root_overlap_tol:
                raise RuntimeError(
                    'State tracking failed: best overlap %.6f below threshold '
                    '%.6f' % (overlap, self.root_overlap_tol)
                )

        if not np.asarray(tdobj.converged)[root]:
            raise RuntimeError(
                'Displaced SATDA tracked root %d did not converge' % (root + 1)
            )
        return mf.e_tot + tdobj.e[root], root, overlap

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
        mf = self.base._scf
        xctype = mf._numint._xc_type(mf.xc) if isinstance(mf, dft.KohnShamDFT) else 'HF'
        if getattr(self.base, 'deltaS', None) != -1:
            raise NotImplementedError(
                'SATDA analytic gradients currently support only deltaS=-1'
            )
        if xctype != 'HF':
            raise NotImplementedError(
                'SATDA analytic gradients currently support only HF '
                'references. Current reference type: %s.' % xctype
            )

        gs_grad = self.base._scf.nuc_grad_method().set(
            verbose=0,
        ).kernel(atmlst=atmlst)
        sf_grad, sf_details = satda_sfbase_gradient_zvec(
            self, self.base, xy, atmlst=atmlst,
            hessian_solver=self.hessian_solver,
        )
        delta_grad, delta_details = satda_delta_gradient_zvec(
            self, self.base, xy, atmlst=atmlst,
            hessian_solver=self.hessian_solver,
        )
        self.satda_details = {
            'ground_state': gs_grad,
            'sfbase': sf_details,
            'delta': delta_details,
            'omega_gradient': sf_grad + delta_grad,
        }
        return gs_grad + sf_grad + delta_grad

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
        elif self.method == 'analytic_experimental':
            raise ValueError(
                'method="analytic_experimental" was the old unsafe SATDA '
                'gradient interface and has been removed.  Use '
                'method="analytic" for the verified HF/ROKS deltaS=-1 '
                'implementation.'
            )
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
            logger.note(
                self, '--------- SATDA gradients for state %d ----------',
                self.state,
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '---------------------------------------------')


Grad = Gradients
