import numpy as np
import pyscf
from pyscf import dft, sftda


from fssh_sf import FSSH_SF 


def generate_velocities(masses, temperature=300.0):
    k_b = 1.380649e-23 # J/K
    AMU = 1.66053906660e-27 # kg
    FS = 1e-15 # s

    N = len(masses)
    masses_kg = masses * AMU
    
    # Maxwell-Boltzmann distribution for velocities
    std_dev = np.sqrt(k_b * temperature / masses_kg)[:, np.newaxis] # in m/s
    velocities = np.random.normal(0, std_dev, (N, 3)) # in m/s
    
    # Remove total momentum
    total_momentum = np.sum(velocities * masses_kg[:, np.newaxis], axis=0)
    velocities -= total_momentum / np.sum(masses_kg)
    
    # Scale to target temperature
    kinetic_energy = 0.5 * np.sum(masses_kg * np.sum(velocities**2, axis=1)) 
    target_energy = 1.5 * N * k_b * temperature  
    scaling_factor = np.sqrt(target_energy / kinetic_energy)
    velocities *= scaling_factor
    
    # Convert from m/s to Angstrom/fs
    velocities = velocities * 0.01)
    return velocities



atom = '''
C       -1.333526     0.151557     0.002045
H       -0.936384    -0.804118     0.494306
H       -2.333962    -0.014210    -0.531314
C       -0.597312     1.438604    -0.002665
H       -0.796505     2.277230     0.698398
H        0.245907     1.677226    -0.653921
'''

mol = pyscf.M(atom=atom, basis='6-31g', unit='A', spin=2)
mf = dft.UKS(mol, xc='svwn')
mf.kernel()

# 进行 SF-TDDFT 计算
sftd = sftda.uks_sf.TDA_SF(mf)
sftd.extype = 1
sftd.collinear_samples = 50
sftd.max_space = 4000
sftd.nstates = 5
sftd.kernel()

# (可选但推荐) 预先检查一下态的性质
from pyscf.sftda.tools_td import transition_analyze
print("--- Initial State Analysis ---")
for i in range(sftd.nstates):
    print(transition_analyze(mf, sftd, sftd.e[i], sftd.xy[i], tdtype='TDA'))
print("----------------------------")

# 准备动力学初始条件
fssh_sim = FSSH_SF(sftd, states=[1, 2], nsteps=200, dt=0.3)
fssh_sim.cur_state = 2  # 从 S1 开始

# 生成初始速度
masses = mol.atom_mass_list(isotope_avg=True)
initial_velocities = generate_velocities(masses, temperature=300.0)

# 准备初始量子系数 (100% 在 S1 态)
initial_coeffs = np.array([0, 1], dtype=complex)

# 4. 运行 FSSH_SF 模拟
fssh_sim.kernel(
    position=mol.atom_coords(), 
    velocity=initial_velocities, 
    coefficient=initial_coeffs
)
