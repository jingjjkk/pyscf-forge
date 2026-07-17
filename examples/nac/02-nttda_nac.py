#!/usr/bin/env python
"""Complete and ETF-only NTTDA ``deltaS=-1`` NAC example."""

from pyscf import dft, gto
from pyscf.sftda.nttda import NTTDA


mol = gto.M(
    atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
    basis="sto-3g",
    spin=2,
    unit="Bohr",
    verbose=0,
)
mf = dft.ROKS(mol).set(xc="PBE", conv_tol=1e-12, verbose=0)
mf.grids.level = 0
mf.run()

td = NTTDA(mf).set(
    deltaS=-1,
    nstates=3,
    conv_tol=1e-9,
    verbose=0,
).run()
nac = td.NAC().set(state_I=1, state_J=2, ediff=True, verbose=0)

complete = nac.kernel(use_etfs=False)
etf_only = nac.kernel(use_etfs=True)

print("Complete NTTDA NAC (1/Bohr):")
print(complete)
print("\nETF/Hellmann-Feynman-only NAC (1/Bohr):")
print(etf_only)

# Expensive independent reference for a selected atom:
# finite_difference = nac.finite_difference(atmlst=[0], step=2e-4)
# print(finite_difference)
