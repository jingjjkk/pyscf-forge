# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#
# Author: Yu Jing <2501110377@stu.pku.edu.cn>
# Description: Non-adiabatic coupling for spin-flip TDDFT (part of manuscript in preparation)
from pyscf import gto, scf, tdscf
from pyscf import dft
from pyscf import sftda, grad
from pyscf.grad import tduks_sf  # this import is necessary.
from pyscf.tdnac import tduks_sf  # this import is necessary.
try:
    import mcfun  # mcfun>=0.2.5 must be used.
except ImportError:
    mcfun = None
from pyscf.sftda.tools_td import transition_analyze 

mol = gto.Mole()
mol.atom = '''
C -0.0162234867 0.0769559254 -0.0522234638
    C -0.0135260315 0.0045248978 1.3509656717
    H 0.8723284795 0.1408256249 -0.7059889228
    H -0.9404373818 -0.0233644513 -0.6359326024
    H 0.7801312555 0.6678088741 1.7600787531
    H 0.7413165352 -0.8668496245 1.2934242485
'''
mol.basis = '6-31g'
mol.spin = 2  
mol.build()

mf = dft.UKS(mol)
mf.xc = 'svwn' 
mf.kernel()

# TDA_SF object
mftd1 = sftda.uks_sf.TDDFT_SF(mf)

mftd1.max_space = 4000 #necessary
mftd1.nstates = 4  # the number of excited states
mftd1.extype = 1  
mftd1.collinear_samples = 50

mftd1.kernel()


print(transition_analyze(mf, mftd1, mftd1.e[0], mftd1.xy[0], tdtype='TDDFT'))  #spin analysis
print(transition_analyze(mf, mftd1, mftd1.e[2], mftd1.xy[2], tdtype='TDDFT'))
e1=mf.e_tot + mftd1.e[0]
e2=mf.e_tot + mftd1.e[2]
print(f"S1 energy: {e1}")
print(f"S3 energy: {e2}")
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