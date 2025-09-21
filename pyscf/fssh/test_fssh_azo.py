import numpy as np
from pyscf import gto, dft, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
# =======================================================
# === 像使用真正的库一样，从您的模块中导入 FSSH 引擎 ===
# =======================================================
from fssh_sf import FSSH_SF_NACV
from tools import extract_state
from pyscf.sftda.tools_td import transition_analyze
# 1. 设置 SF-TDDFT 计算
mol = gto.Mole()
mol.atom = '''
N   -0.618676    0.046313    0.000000
N    0.618676   -0.046313    0.000000
C   -1.341103    1.222340    0.000000
H   -0.816103    2.177340    0.000000
H   -2.016103    1.192340    0.870000
H   -2.016103    1.192340   -0.870000
C    1.341103   -1.222340    0.000000
H    0.816103   -2.177340    0.000000
H    2.016103   -1.192340    0.870000
H    2.016103   -1.192340   -0.870000
'''
mol.basis = 'sto-3g'
mol.spin = 2
mol.build()

mf = dft.UKS(mol)
mf.xc = 'svwn'
mf.kernel()
a, b = get_ab_sf(mf, collinear_samples=50)
A_baba, A_abab = a
B_baab, B_abba = b

# 获取轨道维度信息
mo_occ = mf.mo_occ
occ_a = np.where(mo_occ[0] > 0)[0]
occ_b = np.where(mo_occ[1] > 0)[0]
virt_a = np.where(mo_occ[0] == 0)[0]
virt_b = np.where(mo_occ[1] == 0)[0]

n_occ_a, n_virt_b = len(occ_a), len(virt_b)
n_occ_b, n_virt_a = len(occ_b), len(virt_a)

# 将矩阵reshape为2D
A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b))
B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a))
B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b))
A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a))

# 构建完整的Casida矩阵
Casida_matrix = np.block([
    [ A_abab_2d, B_abba_2d],
    [-B_baab_2d,-A_baba_2d]  # 注意，PySCF的B矩阵定义可能与某些文献相反，这里用-B
])

# 对非对称矩阵使用np.linalg.eig进行全对角化
eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)

# 4. 筛选并归一化物理上有意义的解 (这部分是关键！)
# 按照能量大小排序
idxt = eigenvals.real.argsort()
eigenvals = eigenvals[idxt].real
eigenvecs = eigenvecs[:, idxt]

# 计算范数 ||X||^2 - ||Y||^2 来筛选物理的自旋翻转解
norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2

# 保留范数 > 0 的解
sfd_eigenvals = eigenvals[norms > 0]
sfd_eigenvecs = eigenvecs[:, norms > 0].T

# 定义归一化函数 (与您提供的tdgrad中的函数一致)
def norm_xy(z):
    x_flat = z[:n_occ_a*n_virt_b]
    y_flat = z[n_occ_a*n_virt_b:]
    norm = np.linalg.norm(x_flat)**2 - np.linalg.norm(y_flat)**2
    norm = np.sqrt(1./norm)
    
    x = x_flat.reshape(n_occ_a, n_virt_b) * norm
    y = y_flat.reshape(n_occ_b, n_virt_a) * norm
    
    # 按照PySCF的格式返回 ((y_aa, x_ab), (y_ba, x_bb))
    # 对于alpha->beta的翻转 (extype=1)，激发部分是x_ab，退激发是y_ba
    return ((0, x), (y, 0))

# 5. 创建TDDFT_SF对象并手动填充结果
mftd1 = sftda.uks_sf.TDDFT_SF(mf)
mftd1.max_space = 2000
mftd1.extype = 1
mftd1.collinear_samples =50
mftd1.e = sfd_eigenvals
mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]
mftd1.nstates = len(mftd1.e)
S = extract_state(mf, mftd1, Smin=0.0, Smax=0.7)
target1 = S[0]
target2 = S[1]
E1 =mf.e_tot+ mftd1.e[target1]
E2 =mf.e_tot+ mftd1.e[target2]
print(transition_analyze(mf, mftd1, mftd1.e[target1], mftd1.xy[target1], tdtype='TDDFT'))  #spin analysis
print(transition_analyze(mf, mftd1, mftd1.e[target2], mftd1.xy[target2], tdtype='TDDFT'))
# 2. 定义 FSSH 模拟参数
active_states = [target1+1, target2+1]

# 3. 实例化 FSSH 引擎
fssh_simulation = FSSH_SF_NACV(
    sftda_obj=mftd1,
    states=active_states,
    nac_options={'use_etfs': False, 'ediff': False}, # 传递给 nac 模块的选项
    cphf_options={'max_cycle': 100, 'conv_tol': 1e-6},
    dt=0.2,
    nsteps=40,
    output_dir='azomethane_traj'
)

# 4. 设置初始条件
initial_pos = mol.atom_coords()
np.random.seed(123)
initial_vel = (np.random.rand(*initial_pos.shape) - 0.5) * 0.01
initial_coeffs = np.array([0.0, 1.0], dtype=complex)

# 5. 运行 FSSH kernel
try:
    final_pos, final_vel, final_coeffs = fssh_simulation.kernel(
        position=initial_pos,
        velocity=initial_vel,
        coefficient=initial_coeffs
    )
    print("\nFSSH Simulation Finished Successfully.")
    print("Final positions (Angstrom):\n", final_pos)
    print("Final populations:", np.abs(final_coeffs)**2)
except Exception as e:
    print(f"\nAn error occurred during the FSSH simulation: {e}")
    import traceback
    traceback.print_exc()