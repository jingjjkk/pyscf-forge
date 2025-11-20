# %%
import numpy as np
from pyscf import gto, scf, tdscf,lib
from pyscf.geomopt import geometric_solver
from pyscf import dft
from pyscf import sftda, grad
from pyscf.grad import tduks_sf  # this import is necessary.
from pyscf.nac import tduks_sf as nac
from pyscf.lib import logger
# ================================================================
try:
    import mcfun  # mcfun>=0.2.5 must be used.
except ImportError:
    mcfun = None
from pyscf.sftda.tools_td import transition_analyze
from tools import extract_state
from pyscf.geomopt import berny_solver, as_pyscf_method
from pyscf.sftda.uhf_sf import get_ab_sf

def calculate_properties(mol, state_indices=(0, 1)):
    """
    在一个给定的几何构型下，计算两个目标态的能量、梯度和NAC向量。

    Args:
        mol (gto.Mole): PySCF molecule object.
        state_indices (tuple): 追踪的两个态的索引 (e.g., (0, 1) for the two lowest singlet states).
    
    Returns:
        tuple: (E1, E2, g1, g2, x2) or (None, ... ) if failed.
    """
    print("--- Running Quantum Chemistry Calculation ---")
    # ... 在这里填充你的SF-TDA计算、梯度计算、NAC计算的代码 ...
 
    mf = dft.UKS(mol)
    mf.xc = '0.5*HF+0.5*B88,LYP'
    mf.kernel()
    #mf.xc = '0.5*HF'
    a, b = get_ab_sf(mf, collinear_samples=50)
    A_baba, A_abab = a
    B_baab, B_abba = b

    mo_occ = mf.mo_occ
    n_occ_a = (mo_occ[0] > 0).sum()
    n_virt_a = (mo_occ[0] == 0).sum()
    n_occ_b = (mo_occ[1] > 0).sum()
    n_virt_b = (mo_occ[1] == 0).sum()

    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b))
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a))
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b))
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a))
    
    Casida_matrix = np.block([
        [ A_abab_2d, B_abba_2d],
        [-B_baab_2d,-A_baba_2d]
    ])

    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    
    idxt = eigenvals.real.argsort()
    eigenvals = eigenvals[idxt].real
    eigenvecs = eigenvecs[:, idxt]

    # c. 筛选并归一化物理上有意义的解
    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2 - \
            np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2

    sfd_eigenvals = eigenvals[norms > 0]
    sfd_eigenvecs = eigenvecs[:, norms > 0].T

    def norm_xy(z):
        x_flat = z[:n_occ_a*n_virt_b]
        y_flat = z[n_occ_a*n_virt_b:]
        norm_val = np.linalg.norm(x_flat)**2 - np.linalg.norm(y_flat)**2
        norm_val = np.sqrt(1./norm_val)
        
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return ((0, x), (y, 0))

        # d. 创建TDDFT_SF对象并手动填充结果
    mftd1 = sftda.uks_sf.TDA_SF(mf, extype=1)
    mftd1.collinear_samples = 50
    mftd1.e = sfd_eigenvals
    mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]
    mftd1.nstates = len(mftd1.e)
    #mftd1 = sftda.uks_sf.TDA_SF(mf, extype=1)
    #mftd1.nstates = 10
    #mftd1.max_space = 500     
    #mftd1.collinear_samples = 50

    

    S = extract_state(mf, mftd1, Smin=0.0, Smax=0.5)
    target1 = S[0]
    target2 = S[1]
    E1 =mf.e_tot+ mftd1.e[target1]
    E2 =mf.e_tot+ mftd1.e[target2]

    g1 = tduks_sf.Gradients(mftd1).kernel(state=target1 + 1)
    g2 = tduks_sf.Gradients(mftd1).kernel(state=target2 + 1)
    try:
        nac_grad = nac.NAC(mftd1)
        nac_grad.state_I = target1 + 1
        nac_grad.state_J = target2 + 1
        nac_grad.use_etfs = False  # 是否使用ETF校正
        nac_grad.ediff = False  # 是否除以能量差，False下会直接输出h_IJ,True则会输出d_IJ
        x2 = nac_grad.kernel()
    except ImportError:
        print("Warning: NAC calculator not found. Setting NAC vector (x2) to zero.")
        x2 = np.zeros_like(g1)
    print(f"State {state_indices[0]} (index {target1}): Energy = {E1} ")
    print(f"State {state_indices[1]} (index {target2}): Energy = {E2} ")
   
    print("--- Calculation Finished ---")   


    
    
    return E1, E2, g1, g2, x2

def project_on_plane_lstsq(x3, x1, x2):
    """
    使用最小二乘法将向量x3投影到由向量x1和x2定义的平面上。
    """
    x3 = x3.ravel()
    x1 = x1.ravel()
    x2 = x2.ravel()
    A = np.column_stack([x1, x2])
    coeffs, _, _, _ = np.linalg.lstsq(A, x3, rcond=None)
    projection = A @ coeffs
    return projection.reshape(-1, 3)



class MECPScanner:
    """
    一个与 pyscf.geomopt.geometric_solver 兼容的扫描器类，用于寻找MECP。
    """
    def __init__(self, mol, states=(0, 1)):
        self.mol = mol
        self.states = states
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.log = logger.Logger(self.stdout, self.verbose)
        self.converged = False
        self.base = self

    def __call__(self, mol_or_geom):
        if isinstance(mol_or_geom, gto.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.log.info("\n--- MECP Optimizer Step (using geometric_solver) ---")

        E1, E2, g1, g2, x2 = calculate_properties(mol, self.states)
        
        if E1 is None:
            self.log.warn("Quantum chemistry calculation failed. Returning high energy.")
            self.converged = False
            return 1e10, np.zeros_like(mol.atom_coords())

        self.converged = True

        self.log.info(f"  Total Energies: E1={E1:.6f}, E2={E2:.6f}")
        self.log.info(f"  Energy Gap (E2-E1): {E2-E1:.6f} Ha")

        x1 = g1 - g2
        x1_norm_val = np.linalg.norm(x1)
        x1_norm_vec = x1 / x1_norm_val if x1_norm_val > 1e-9 else np.zeros_like(x1)

        x2_norm_val = np.linalg.norm(x2)
        x2_norm_vec = x2 / x2_norm_val if x2_norm_val > 1e-9 else np.zeros_like(x2)
        
        g_aver=(g1+g2)/2.0
        g_on_plane = project_on_plane_lstsq(g_aver, x1_norm_vec, x2_norm_vec)
        g_proj = g_aver - g_on_plane
    
        f = (E1 - E2) * x1_norm_vec
        g_bar = g_proj + f*10 # MECP有效力

        self.log.info(f"  ||Seam Grad (g_proj)||: {np.linalg.norm(g_proj):.6f}")
        self.log.info(f"  ||Degeneracy Grad (f)||: {np.linalg.norm(f):.6f}")
        self.log.info(f"  ||Total Effective Grad||: {np.linalg.norm(g_bar):.6f}")
        self.log.info("----------------------------------------------------------------")

        energy_for_optimizer = (E1 + E2) / 2.0
        return energy_for_optimizer, g_bar
    
    def as_scanner(self):
        return self

    def nuc_grad_method(self):
        return self


# === ConicalIntersectionOptimizer 使用全局 MECPScanner ===
class ConicalIntersectionOptimizer:
    """
    实现了用于定位势能面交叉点上最低能量点的直接方法。
    这个版本使用 pyscf.geomopt.geometric_solver 作为核心优化器。
    """
    def __init__(self, mol, states=(0, 1)):
        if len(states) != 2:
            raise ValueError("`states`必须是两个态索引的元组。")
        self.mol = mol
        self.states = tuple(sorted(states))
        self.log = logger.new_logger(self, self.mol.verbose)

    def kernel(self, geom=None, **kwargs):
        return self.optimize(geom, **kwargs)

    def optimize(self, geom=None, **kwargs):
        if geom is not None:
            self.mol.atom = geom  # 注意：更推荐用 set_geom_ 或 copy + set
            
        mecp_scanner = MECPScanner(self.mol, states=self.states)  # ← 正确：使用全局类
        optimized_mol = geometric_solver.optimize(
    mecp_scanner,
    algorithm='RFO',
    # --- 收敛标准 (使用正确的 geomeTRIC 关键字) ---
    convergence_energy=5e-4,  # 替代 energyTol
    convergence_gmax=5e-4,    # 替代 gradientTol (放宽最大梯度)
    # convergence_grms=...   # 您也可以选择性地设置 RMS 梯度
    convergence_dmax=5e-4,    # 替代 stepTol (⚠️ 关键！放宽最大位移)
    convergence_drms=5e-4 ,    # 您也可以选择性地设置 RMS 位移
    
    # --- 信任半径设置 ---
    tmin=1e-4,                # 替代 minTrust
    tmax=0.2,                 # 替代 maxTrust
    trust=5e-4 ,         # 您可能还想设置初始信任半径 'trust'
    
    maxiter=100,
    **kwargs
)
        self.mol = optimized_mol
        return self.mol
    
if __name__ == '__main__':
    # 1. 创建一个分子对象
    # 使用一个稍微扭曲的构型作为初始点，以避免对称性问题
    mol = gto.Mole()
    
    mol.atom = '''
 C   0.044079  -0.236208   0.352258   
 C   0.009153   0.681657   1.461499   
 H   0.929962  -0.151761  -0.282805   
 H  -0.833261  -0.199295  -0.298995  
 H  -0.007400   0.945740   2.491184   
 H   0.067386  -1.248318   0.785123   
'''

 
    BOHR = 0.529177210927
    mol.unit = 'A'  # 使用 A 作为单位
    mol.basis = '6-31g**'
    mol.spin = 2  # 双重态
    mol.verbose = 3
    mol.build()

    # 2. 实例化 ConicalIntersectionOptimizer
    # 假设我们寻找最低的两个态（索引0和1）之间的交叉点
    optimizer = ConicalIntersectionOptimizer(mol, states=(0, 1))

    # 3. 运行优化
    print("Starting MECP optimization...")
    # 增加优化循环次数

    optimized_mol = optimizer.optimize(max_cycles=50)

    # 4. 打印结果
    print("\n==================== MECP Optimization Finished ====================")
    
    print("\nOptimized Geometry (Bohr):")
    for atom in optimized_mol.atom:
        print(f"  {atom[0]:<2} {atom[1][0]:>12.8f} {atom[1][1]:>12.8f} {atom[1][2]:>12.8f}")
    print("====================================================================")
    
    print("\nOptimized Geometry (A):")
    for atom in optimized_mol.atom:
        print(f"  {atom[0]:<2} {atom[1][0]*BOHR:>12.8f} {atom[1][1]*BOHR:>12.8f} {atom[1][2]*BOHR:>12.8f}")
    print("====================================================================")



