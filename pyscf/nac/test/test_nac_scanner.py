
from pyscf import gto, dft
from pyscf.sftda import uks_sf
from pyscf.nac import tduks_sf
import numpy as np
from pyscf.sftda.tools_td import transition_analyze 
# ========== 1. Build initial molecule (H2O) ==========
mol = gto.Mole()
mol.atom = '''
O    0.000000    0.000000    0.000000
H    0.757000    0.586000    0.000000
H   -0.757000    0.586000    0.000000
'''
mol.basis = '6-31g'
mol.spin = 2  # Triplet reference for SF
mol.build()

# ========== 2. Run UKS + SF-TDA ==========
mf = dft.UKS(mol)
mf.xc = 'svwn'
mf.kernel()

td = uks_sf.TDA_SF(mf)
td.nstates = 4
td.extype = 1
td.max_space = 4000
td.kernel()
print(transition_analyze(mf, td, td.e[0], td.xy[0], tdtype='TDA'))  #spin analysis
print(transition_analyze(mf, td, td.e[2], td.xy[2], tdtype='TDA'))
nac_scanner = tduks_sf.NAC(td)
nac_scanner.state_I = 1  # 1st excited state
nac_scanner.state_J = 3  # 2nd excited state
nac_scanner.use_etfs = True
nac_scanner.ediff = True






nac_initial = nac_scanner(mol)  # First call: computes everything
print("NAC vector (au) at initial geom:")
for i, (symbol, _) in enumerate(mol._atom):
    print(f"  {symbol:>2s}: [{nac_initial[i,0]:9.5f}, {nac_initial[i,1]:9.5f}, {nac_initial[i,2]:9.5f}]")


coords = mol.atom_coords().copy()
coords[1, 0] += 0.01  # Move first H atom along x-axis
mol_perturbed = mol.copy()
mol_perturbed.set_geom_(coords, unit='Angstrom')



nac_perturbed = nac_scanner(mol_perturbed)  
print("NAC vector (au) at perturbed geom:")
for i, (symbol, _) in enumerate(mol_perturbed._atom):
    print(f"  {symbol:>2s}: [{nac_perturbed[i,0]:9.5f}, {nac_perturbed[i,1]:9.5f}, {nac_perturbed[i,2]:9.5f}]")


