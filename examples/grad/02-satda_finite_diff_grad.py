#!/usr/bin/env python

'''
Finite-difference nuclear gradient for SA-SF-TDA.

Run from the repository root with:

    PYTHONPATH=$PWD python examples/grad/02-satda_finite_diff_grad.py

The script computes central finite differences of both

    E_exc(R) = omega_SATDA(R)
    E_tot(R) = E_ROKS(R) + omega_SATDA(R)

for one selected SA-SF-TDA state.  State tracking is based on the overlap
between the reference SA-SF-TDA amplitude and displaced-geometry amplitudes.
The final interface gradient should match the independent finite difference of
E_tot.
'''

import numpy as np

from pyscf import gto, lib
from pyscf.data import nist
from pyscf.sftda.satda import SATDA


ATOM_SYMBOLS = ('O', 'O')
ATOM_COORDS_ANG = np.array([
    [0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 1.207500],
])

BASIS = 'sto-3g'
XC = 'B3LYP'
SPIN = 2
CHARGE = 0

DELTAS = -1           # -1 for Sf=Si-1, 0 for Sf=Si
TARGET_STATE = 4      # 1-based SATDA root index
NSTATES = 4
DELTA = 1e-3         # Bohr
GRID_LEVEL = 1


def build_mol(coords_bohr):
    atom = [(sym, coords_bohr[i]) for i, sym in enumerate(ATOM_SYMBOLS)]
    return gto.M(
        atom=atom,
        unit='Bohr',
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        symmetry=False,
        verbose=0,
    )


def normalized_x(td, root):
    x = np.asarray(td.xy[root][0]).ravel()
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise RuntimeError('SA-SF-TDA root has near-zero amplitude norm')
    return x / norm


def amplitude_overlap(x_ref, td, root):
    x = normalized_x(td, root)
    return abs(np.vdot(x_ref, x))


def run_satda(coords_bohr, x_ref=None):
    mol = build_mol(coords_bohr)

    mf = mol.ROKS(xc=XC)
    mf.conv_tol = 1e-10
    mf.grids.level = GRID_LEVEL
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('ROKS reference did not converge')

    td = SATDA(mf).set(deltaS=DELTAS, nstates=NSTATES,
                       verbose=0, conv_tol=1e-8)
    td.kernel()
    if not np.all(td.converged):
        raise RuntimeError('SA-SF-TDA solver did not converge for all roots')

    if x_ref is None:
        root = TARGET_STATE - 1
        overlap = 1.0
    else:
        overlaps = np.array([amplitude_overlap(x_ref, td, i)
                             for i in range(len(td.e))])
        root = int(np.argmax(overlaps))
        overlap = overlaps[root]

    return {
        'mf': mf,
        'td': td,
        'root': root,
        'overlap': overlap,
        'e_ground': mf.e_tot,
        'e_exc': td.e[root],
        'e_total': mf.e_tot + td.e[root],
        'x': normalized_x(td, root),
    }


def finite_difference_gradient(coords_bohr, x_ref):
    grad_exc = np.zeros_like(coords_bohr)
    grad_total = np.zeros_like(coords_bohr)

    for ia in range(coords_bohr.shape[0]):
        for xyz in range(3):
            coords_p = coords_bohr.copy()
            coords_m = coords_bohr.copy()
            coords_p[ia, xyz] += DELTA
            coords_m[ia, xyz] -= DELTA

            plus = run_satda(coords_p, x_ref)
            minus = run_satda(coords_m, x_ref)

            grad_exc[ia, xyz] = (plus['e_exc'] - minus['e_exc']) / (2 * DELTA)
            grad_total[ia, xyz] = (
                plus['e_total'] - minus['e_total']) / (2 * DELTA)

            print(
                'atom %d coord %d: roots %+d/%+d, overlaps %.6f %.6f, '
                'Eexc %+ .8f/%+ .8f eV' % (
                    ia + 1, xyz,
                    plus['root'] + 1, minus['root'] + 1,
                    plus['overlap'], minus['overlap'],
                    plus['e_exc'] * nist.HARTREE2EV,
                    minus['e_exc'] * nist.HARTREE2EV,
                )
            )

    return grad_exc, grad_total


def print_gradient(title, grad):
    print('\n' + title)
    print('atom             d/dx              d/dy              d/dz')
    for ia, row in enumerate(grad, start=1):
        print('%4d  %16.10f %16.10f %16.10f' %
              (ia, row[0], row[1], row[2]))


def main():
    coords_bohr = ATOM_COORDS_ANG / lib.param.BOHR

    ref = run_satda(coords_bohr)
    print('Reference root: SATDA state %d' % TARGET_STATE)
    print('deltaS: %d' % DELTAS)
    print('Ground-state ROKS energy: %.12f Hartree' % ref['e_ground'])
    print('SA-SF-TDA excitation energy: %.12f Hartree  %.8f eV' %
          (ref['e_exc'], ref['e_exc'] * nist.HARTREE2EV))
    print('Total excited-state energy: %.12f Hartree' % ref['e_total'])
    print('Finite-difference step: %.3e Bohr' % DELTA)

    grad_exc, grad_total = finite_difference_gradient(coords_bohr, ref['x'])
    grad_interface = ref['td'].Gradients().set(
        verbose=0, nstates=NSTATES, root_overlap_tol=0.2
    ).kernel(state=TARGET_STATE, method='finite_diff', step=DELTA)

    print_gradient('Gradient of SA-SF-TDA excitation energy, Hartree/Bohr',
                   grad_exc)
    print_gradient('Gradient of E_ROKS + E_SATDA, Hartree/Bohr',
                   grad_total)
    print_gradient('SATDA Gradients() interface, Hartree/Bohr',
                   grad_interface)
    print('\nMax |interface - independent total FD| = %.6e' %
          abs(grad_interface - grad_total).max())


if __name__ == '__main__':
    lib.num_threads(1)
    main()
