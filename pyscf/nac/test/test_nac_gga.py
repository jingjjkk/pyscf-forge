from pyscf import gto, scf, tdscf
from pyscf import dft
from pyscf import sftda, grad
from pyscf.grad import tduks_sf  # this import is necessary.
from pyscf.nac import tduks_sf  # this import is necessary.
try:
    import mcfun  # mcfun>=0.2.5 must be used.
except ImportError:
    mcfun = None
from pyscf.sftda.tools_td import transition_analyze 

mol = gto.Mole()
mol.atom = '''
C    0.000000    0.000000    0.000000
O    1.215000    0.000000    0.000000
H    0.582000    0.940000    0.000000
H    0.582000   -0.940000    0.000000
'''
mol.basis = '6-31g'
mol.spin = 2  
mol.build()

mf = dft.UKS(mol)
mf.xc = 'b3lyp' 
mf.kernel()

# TDA_SF object
mftd1 = sftda.uks_sf.TDA_SF(mf)

mftd1.max_space = 4000 #necessary
mftd1.nstates = 4  # the number of excited states
mftd1.extype = 1  
mftd1.collinear_samples = 200

mftd1.kernel()


print(transition_analyze(mf, mftd1, mftd1.e[0], mftd1.xy[0], tdtype='TDA'))  #spin analysis
print(transition_analyze(mf, mftd1, mftd1.e[2], mftd1.xy[2], tdtype='TDA'))

# nac object
nac_grad = tduks_sf.NAC(mftd1)
nac_grad.state_I = 1  # S0 
nac_grad.state_J = 3  # S1

nac_grad.use_etfs = True  # whether to use ETF correction
nac_grad.ediff = True     # whether divide by energy difference


nac = nac_grad.kernel()


print("\nNon-Adiabatic Coupling (NAC)with ETF between S0 and S1:")
print("NAC shape:", nac.shape)  # should be (n_atoms, 3)
for i, atom in enumerate(mol._atom):
    print(f"Atom {i + 1} ({atom[0]}): {nac[i]}")

nac_grad.use_etfs = False 
nac = nac_grad.kernel()
print("\nNon-Adiabatic Coupling (NAC)without ETF between S0 and S1:")
for i, atom in enumerate(mol._atom):
    print(f"Atom {i + 1} ({atom[0]}): {nac[i]}")