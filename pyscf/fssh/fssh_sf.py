# 文件路径: pyscf/fssh/fssh_sf.py
import copy
import numpy as np
import time
import logging
from typing import Tuple, Optional, List, Dict
from pathlib import Path
from pyscf import lib, dft, gto, sftda
from pyscf.sftda import uks_sf
from pyscf.tdnac import tduks_sf as nac_sf
from pyscf.tdgrad import tduks_sf 
from pyscf.lib import logger
# 导入您提供的标准FSSH基类
from fssh import FSSH # 假设标准FSSH类在 fssh.py 中
from tools import extract_state
logger = logging.getLogger(__name__)
FS2AUTIME = 41.34137        # Conversion factor: femtoseconds to atomic time units
A2BOHR = 1.889726           # Conversion factor: Angstrom to Bohr radius
AMU2AU = 1822.8884858012984  # Conversion factor: amu to atomic mass units
def _vector_dot(vec1, vec2):
    '''
    Compute symplectic inner product between two SF-TDDFT excitation vectors:
        <vec1 | vec2> = X1^T * X2 - Y1^T * Y2

    Args:
        vec1, vec2 (tuple): Each is ((X_alpha_beta, X_beta_alpha), (Y_alpha_beta, Y_beta_alpha)),
    Returns:
        float: Inner product value.
    '''
    x1, y1 = vec1
    x2, y2 = vec2
    
    dot_x = 0.0
    if isinstance(x1[0], np.ndarray) and isinstance(x2[0], np.ndarray):
        dot_x += np.dot(x1[0].ravel(), x2[0].ravel())
    if isinstance(x1[1], np.ndarray) and isinstance(x2[1], np.ndarray):
        dot_x += np.dot(x1[1].ravel(), x2[1].ravel())
        
    dot_y = 0.0
    if isinstance(y1[0], np.ndarray) and isinstance(y2[0], np.ndarray):
        dot_y += np.dot(y1[0].ravel(), y2[0].ravel())
    if isinstance(y1[1], np.ndarray) and isinstance(y2[1], np.ndarray):
        dot_y += np.dot(y1[1].ravel(), y2[1].ravel())
        
    return dot_x - dot_y

def _scale_vector(vec, factor=-1.0):
    '''
    Multiply an excitation vector by a scalar factor.

    Args:
        vec (tuple): ((x_a, x_b), (y_a, y_b))
        factor (float): Scaling factor.

    Returns:
        tuple: Scaled vector in same format.
    '''
    x, y = vec
    
    x_a_new = x[0] * factor if isinstance(x[0], np.ndarray) else 0
    x_b_new = x[1] * factor if isinstance(x[1], np.ndarray) else 0
    
    y_a_new = y[0] * factor if isinstance(y[0], np.ndarray) else 0
    y_b_new = y[1] * factor if isinstance(y[1], np.ndarray) else 0
    
    return ((x_a_new, x_b_new), (y_a_new, y_b_new))
class FSSH_SF(FSSH):
    """
    一个用于Spin-Flip TDDFT的FSSH实现。
    它继承自通用的FSSH类，并重写了calc_electronic方法以正确处理
    SF-TDDFT的计算。
    """
    def __init__(self, tddft, states:List[int], **kwargs):
        """
        初始化FSSH_SF模拟。
        
        Args:
            tddft: 一个已经“标准化”的sftda对象。
            states (List[int]): 1-based的激发态索引。
            **kwargs: 其他FSSH参数。
        """
        if any(s <= 0 for s in states):
            raise ValueError("SF-TDDFT states must be 1-based positive integers.")
        
        self.tddft = tddft  # 现在 tddft 是一个 sftda.uks_sf.TDDFT_SF 对象
        self.tdgrad = tduks_sf.Gradients(self.tddft)
        self.tdnac = nac_sf.NAC(self.tddft)
        self.tdnac.etfs= True
        self.tdnac.ediff= True
        self.tddft.mol.unit = 'Bohr'
        self.tddft._scf.mol.unit = 'Bohr'
        
            # 2. 检查并设置CPHF选项
        if 'cphf_options' in kwargs:
            cphf_opts = kwargs['cphf_options']
            max_cycle = cphf_opts.get('max_cycle', 100)
            conv_tol = cphf_opts.get('conv_tol', 1e-7)
            
            if hasattr(self.tdgrad, 'cphf_max_cycle'):
                self.tdgrad.cphf_max_cycle = max_cycle
                self.tdgrad.cphf_conv_tol = conv_tol
            if hasattr(self.tdnac, 'cphf_max_cycle'):
                self.tdnac.cphf_max_cycle = max_cycle
                self.tdnac.cphf_conv_tol = conv_tol
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("At least two electronic states must be specified")
        if any(not isinstance(s, int) or s <= 0 for s in states):
            raise ValueError("All state indices for FSSH-SF must be 1-based positive integers (e.g., [1, 2])")
        

        #self.prev_wavefunctions_flat = None  # 用于态追踪的历史波函数
        self.prev_ci_vectors = None  # 用于相位校正的历史CI向量
        # 3. 基本属性 (Basic Attributes)
        self.states = list(states)
        self.Nstates = len(states)      
        self.cur_state = states[0]  # Start from the first specified state
        
        
        
        
        
        # Set default simulation parameters
        self.dt = 0.5 * FS2AUTIME  # Default: 0.5 fs in atomic units
        
        self.output_dir = Path('.')
        # 4. 物理属性 (Physical Properties)
        self.mass = self.tddft.mol.atom_mass_list(True).reshape(-1, 1) * AMU2AU

        # 5. 算法设置 (Algorithm Setup)
        self.nac_idx = [(i, j) for i in range(self.Nstates-1) 
                        for j in range(i+1, self.Nstates)]

        # 6. 动力学参数默认值 (Dynamics Param Defaults)
        self.dt = 0.5 * FS2AUTIME
        self.nsteps = 1
        self.output_dir = Path('.')

        # 7. 动力学参数用户覆盖 (Dynamics Param Override)
        for key, value in kwargs.items():
            if key == 'dt' and isinstance(value, (int, float)):
                if value <= 0:
                    raise ValueError("Time step must be positive")
                self.dt = value * FS2AUTIME
            elif key == 'nsteps' and isinstance(value, int):
                if value <= 0:
                    raise ValueError("Number of steps must be positive")
                self.nsteps = value
            elif key == 'output_dir':
                self.output_dir = Path(value)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            else:
                setattr(self, key, value)

        # 8. 日志 (Logging)
        logger.info(f"FSSH-SF simulation initialized with {self.Nstates} states, "
                   f"dt={self.dt/FS2AUTIME:.3f} fs, {self.nsteps} steps")

    def calc_electronic(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算给定核位置下的电子能量、力和非绝热耦合向量 (NACV)。
        Args:
            position (np.ndarray): 核位置 (Natoms × 3)，单位为Bohr。
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - energy (np.ndarray): 电子能量 (Nstates,)
                - force (np.ndarray): 力 (Natoms × 3)
                - Nacv (np.ndarray): 非绝热耦合向量 (Nstates × Nstates × Natoms × 3)
        """
        current_mol = self.tddft.mol.copy()
        current_mol.set_geom_(position.reshape(current_mol.natm, 3), unit='Bohr')
        
        # 1. *** 创建一个全新的、纯净的UKS对象 ***
        #    从self.tddft._scf中获取必要的参数，如泛函和自旋
        xc_fun = self.tddft._scf.xc
        mf = dft.UKS(current_mol, xc=xc_fun)
        mf.spin = 2
        
        # 2. 运行SCF。
        mf.kernel()
        ref_energy = mf.e_tot

        # 3. 基于这个全新的mf，创建一个全新的TDA_SF对象
        mftd = sftda.uks_sf.TDA_SF(mf)
        mftd.extype = self.tddft.extype
        mftd.collinear_samples = self.tddft.collinear_samples
        mftd.max_space = self.tddft.max_space
        mftd.nstates = self.tddft.nstates
        mftd.kernel()
        
    # 4. 识别单重态 (S1, S2, ...)
        singlet_indices = extract_state(mf, mftd, Smin=0.0, Smax=0.7)
        if len(singlet_indices) < len(self.states):
            raise RuntimeError(f"Not enough singlet states found to track {len(self.states)} states.")

        sorted_singlets_pyscf_indices = sorted(singlet_indices, key=lambda i: mftd.e[i])

        # 5. 建立态标签到pyscf索引的映射
        state_map = {s: sorted_singlets_pyscf_indices[s-1] for s in self.states}
        
        # 6. 提取能量
        energy = np.array([ref_energy + mftd.e[state_map[s]] for s in self.states])
        
        # 7. 获取当前活动态的pyscf索引
        active_pyscf_idx = state_map[self.cur_state]
       
        # 8. 创建全新的Gradients对象并计算梯度
        mftdg = tduks_sf.Gradients(mftd)
        mftdg.cphf_max_cycle = self.tdgrad.cphf_max_cycle
        mftdg.cphf_conv_tol = self.tdgrad.cphf_conv_tol
        grad = mftdg.kernel(state=active_pyscf_idx + 1)
        force = -grad
        natoms = current_mol.natm
        
        current_ci_vectors = {s: mftd.xy[state_map[s]] for s in self.states}

        # 2. 如果不是第一步，进行相位校正
        if self.prev_ci_vectors is not None:
            for s in self.states:
                # 使用 _vector_dot 计算重叠积分
                overlap = _vector_dot(self.prev_ci_vectors[s], current_ci_vectors[s])
                
                # 如果重叠为负，说明相位需要翻转
                if overlap < 0:
                    # 使用 _scale_vector 进行相位校正
                    current_ci_vectors[s] = _scale_vector(current_ci_vectors[s], -1.0)

        # 3. 将校正后的CI向量放回mftd对象中，以供NAC计算器使用
        for s in self.states:
            pyscf_idx = state_map[s]
            mftd.xy[pyscf_idx] = current_ci_vectors[s]

        # 4. 创建NAC计算器并计算NACV
        Nacv = np.zeros((self.Nstates, self.Nstates, current_mol.natm, 3))
        
        nac = nac_sf.NAC(mftd)
        # 遍历所有需要计算的态对，例如 (0,1), (0,2), (1,2)...
        for i, j in self.nac_idx:
            # i, j 是 self.states 列表中的索引
            # 找到它们对应的 pyscf 索引
            
            pyscf_idx_I = state_map[self.states[i]]
            pyscf_idx_J = state_map[self.states[j]]
            nac.cphf_max_cycle = self.tdnac.cphf_max_cycle
            nac.cphf_conv_tol = self.tdnac.cphf_conv_tol
            # 设置并计算 NAC
            nac.state_I = pyscf_idx_I + 1
            nac.state_J = pyscf_idx_J + 1
            nac.etfs= True
            nac.ediff= True
           
            nac_vector = nac.kernel()
            # 打印nac
            print(f"NAC vector (a.u.) between states {nac.state_I} and {nac.state_J}:\n{nac_vector}\n")
                   
            Nacv[i, j] = nac_vector
            Nacv[j, i] = -nac_vector
        self.prev_ci_vectors = current_ci_vectors    

        return energy, force, Nacv
    def exp_propagator(self, c: np.ndarray, Veff: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate quantum coefficients using matrix exponential.
        dc/dt = -i V_eff(R,P) c

        The matrix exponential is computed efficiently using eigenvalue decomposition:
        exp(-i * V_eff * dt) = U * diag(exp(-i * λ_k * dt)) * U†

        Args:
            c (np.ndarray): Current quantum coefficients (Nstates,)
            Veff (np.ndarray): Effective Hamiltonian matrix (Nstates * Nstates)
            dt (float): Time step in atomic units
        
        Returns:
            np.ndarray: Updated quantum coefficients (Nstates,)
        
        Note:
            The effective Hamiltonian includes both diagonal energies and
            off-diagonal nonadiabatic coupling terms.        
        """
        # Diagonalize the effective Hamiltonian
        diags, coeff = np.linalg.eigh(Veff)

        # Compute the matrix exponential
        U = coeff @ np.diag(np.exp(-1j * diags * dt)) @ coeff.T.conj()

        # Apply propagator to coefficients
        c_new = np.dot(U, c)

        # Normalize coefficients
        c_new = c_new / np.linalg.norm(c_new)

        return c_new

    def update_coefficient(self, coeffs: np.ndarray, 
                           energy: np.ndarray, 
                           nact: np.ndarray) -> np.ndarray:
        """
        Update quantum coefficients using the effective Hamiltonian.
        
        The effective Hamiltonian in the FSSH method combines:
        1. Diagonal electronic energies: E_ii(R)
        2. Off-diagonal nonadiabatic coupling: -i * κ_ij
        
        V_eff = E(R) - i * d(R) * P/m = diag(E) - i * κTDC
        
        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            energy (np.ndarray): Electronic energies (Nstates,)
            nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)
        
        Returns:
            np.ndarray: Updated quantum coefficients (Nstates,)
        """
        # Construct effective Hamiltonian
        Veff = np.diag(energy) - 1j * nact

        # Propagate coefficients
        c_new = self.exp_propagator(coeffs, Veff, self.dt)

        return c_new
    
    def compute_hopping_probability(self, 
                                    coeffs: np.ndarray, 
                                    nact: np.ndarray) -> np.ndarray:
        """
        Calculate surface hopping probabilities using Tully's formula.
        
        The hopping probability from the current state i to state j is:
        g_ij = (2 * Re(κ_ij * c_i* * c_j) - 2 / ħ * Im(V_ij * c_i* * c_j)) * dt / |c_i|²
        p_ij = max(0, g_ij)
        p_ij = min(1, p_ij)

        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)
        
        Returns:
            np.ndarray: Hopping probabilities from current state (Nstates,)
        """

        # Get index of current state in the states list
        state_idx = self.states.index(self.cur_state)

        # Current state coefficient
        c_i = coeffs[state_idx]

        # Calculate hopping probabilities
        g_ij = 2 * (nact[state_idx] * c_i.conj() * coeffs).real * self.dt / (np.abs(c_i)**2)

        # Adjust hopping probabilities
        p_ij = np.where(g_ij < 0, 0, g_ij)
        p_ij = np.where(p_ij > 1, 1, p_ij)
        
        return p_ij
    
    def check_hop(self, r: float, p_ij: np.ndarray) -> int:
        """
        Determine if a surface hop occurs.

        The hopping decision is made by comparing a random number r ∈ [0,1)
        with cumulative probabilities. A hop to state k occurs if:
        Σ_{j=0}^{k-1} p_j < r ≤ Σ_{j=0}^{k} p_j
        
        Args:
            r (float): Random number between 0 and 1
            p_ij (np.ndarray): Hopping probabilities (Nstates,)
        
        Returns:
            int: Index of target state (-1 if no hop occurs)
        
        Note:
            Returns -1 if no hop occurs (r falls in the "stay" probability region)
        """
        # Calculate cumulative probabilities
        cumu_p_ij = np.cumsum(p_ij)

        # Check each state for hopping condition
        for k, u_bound in enumerate(cumu_p_ij):
            l_bound = 0.0 if k == 0 else cumu_p_ij[k-1]

            if l_bound < r <= u_bound:
                return k
            
        return -1
    
    def rescale_velocity(self, 
                         hop_index: int,
                         energy: np.ndarray,
                         velocity: np.ndarray,
                         d_vec: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Rescale nuclear velocities to conserve total energy after surface hopping.
        
        When a surface hop occurs, the nuclear kinetic energy must be adjusted to
        compensate for the change in electronic energy. This is achieved by solving
        the energy conservation equation:
        
        1/2m(v')² + E_new = 1/2mv² + E_old

        The new velocity is:
        v' = v - gamma * d_vec / mass

        if delta > 0:
            gamma = (b +- sqrt(b^2 - 4ac)) / 2a
            a = sum_i (d_i^2 / 2m_i)
            b = sum_i (v_i * d_i)
            c = E_new - E_old
        else:
            gamma = b / a

        Args:
            hop_index (int): Index of target state in states list
            energy (np.ndarray): Electronic energies for all states
            velocity (np.ndarray): Current nuclear velocities (Natoms × 3)
            d_vec (np.ndarray): Difference vector for velocity adjustment
        
        Returns:
            Tuple[bool, np.ndarray]: 
                - hop_allowed: Whether the hop is energetically allowed
                - velocity: Updated nuclear velocities
        """

        # To conserve energy, the new velocity v' = v - gamma * d_vec / mass must satisfy 
        # the energy conservation equation, which leads to a quadratic equation for the 
        # scaling factor gamma:
        #     a*gamma^2 - b*gamma + c = 0
        # where:
        #     a = sum_i (d_i^2 / 2m_i)
        #     b = sum_i (v_i * d_i)
        #     c = E_new - E_old

        # Get index of current state in the states list
        state_idx = self.states.index(self.cur_state)

        # Coefficients for the quadratic equation
        a = np.sum(d_vec**2 / (2 * self.mass))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[state_idx]

        # Discriminant of the quadratic equation
        delta = b**2 - 4 * a * c

        if delta >= 0:
            gamma = (b + np.sqrt(delta)) / (2 * a) if b < 0 else (b - np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass
            return False, velocity
    
    # NOT TESTED YET!!!!
    # def decoherence(self,
    #                 coeffs: np.ndarray,
    #                 velocity: np.ndarray,
    #                 energy: np.ndarray) -> np.ndarray:
    #     """
    #     Decoherence.

    #     c_j = c_j * exp(-dt / tau_ji)
    #     c_i = c_i * sqrt((1 - sum_j(j!=i) |c_j|**2) / |c_i|**2)
    #     tau_ji = ħ / |E_jj - E_ii| * (1 + a / E_kin)
    #     """

    #     E_kin = 0.5 * self.mass * np.sum(velocity ** 2)
    #     cumu_sum = 0
        
    #     for i in range(len(coeffs)):
    #         if i != self.cur_state:
    #             tau_ji = 1 / np.abs(energy[i] - energy[self.cur_state]) * (1 + self.alpha / E_kin)
    #             coeffs[i] = coeffs[i] * np.exp(-self.dt / tau_ji)
    #             cumu_sum += np.abs(coeffs[i]) ** 2
        
    #     coeffs[self.cur_state] = np.sqrt((1 - cumu_sum) / np.abs(coeffs[self.cur_state]) ** 2) * coeffs[self.cur_state]
    #     return coeffs

    def write_trajectory(self, 
                         step: int, 
                         position: np.ndarray, 
                         velocity: np.ndarray,
                         energy: np.ndarray, 
                         coeffs: np.ndarray, 
                         filename: str = 'trajectory.xyz') -> None:
        """
        Write current trajectory frame to XYZ file with comprehensive metadata.
        
        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            energy (np.ndarray): Electronic energies in Hartree
            coeffs (np.ndarray): Quantum coefficients
            filename (str): Output filename
        """
        filepath = self.output_dir / filename
        mode = 'w' if step == 0 else 'a'
        
        with open(filepath, mode) as f:
            # Write number of atoms
            f.write(f'{self.tddft.mol.natm}\n')
            
            # Write comment line with simulation data
            time_fs = step * self.dt / FS2AUTIME
            current_energy = energy[self.states.index(self.cur_state)]
            
            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {self.cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')
            
            # Write atomic coordinates
            for i, coord in enumerate(position):
                symbol = self.tddft.mol.atom_pure_symbol(i)
                x, y, z = coord / A2BOHR  # Convert to Angstrom
                f.write(f'{symbol:4.2s} {x:12.6f} {y:12.6f} {z:12.6f}\n')
    
    def print_step_info(self, 
                        step: int, 
                        total_time: float, 
                        energy: np.ndarray,
                        coeffs: np.ndarray, 
                        ) -> None:
        """
        Print detailed information about the current simulation step.
        
        Args:
            step (int): Current step number
            total_time (float): Total simulation time in fs
            energy (np.ndarray): Electronic energies
            coeffs (np.ndarray): Quantum coefficients
            nact (np.ndarray): κTDC coupling matrix
            hop_occurred (bool): Whether a hop occurred in this step
        """
        
        current_idx = self.states.index(self.cur_state)
        current_energy = energy[current_idx]
        populations = np.abs(coeffs)**2
            
        # Format output
        logger.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, "
              f"Energy {current_energy:12.8f} Ha, Populations: {populations}")

    def kernel(self, 
               position: Optional[np.ndarray] = None, 
               velocity: Optional[np.ndarray] = None, 
               coefficient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the main FSSH trajectory simulation.
        
        This method implements the complete FSSH algorithm using the velocity Verlet
        integration scheme.

        Integration Frame Ref:
            Nonadiabatic Field on Quantum Phase Space: A Century after Ehrenfest
            Baihua Wu, Xin He, and Jian Liu
            The Journal of Physical Chemistry Letters 2024 15 (2), 644-658
            DOI: 10.1021/acs.jpclett.3c03385

        Args:
            position (Optional[np.ndarray]): Initial nuclear coordinates in Angstrom
                If None, uses equilibrium geometry from TDDFT object
            velocity (Optional[np.ndarray]): Initial nuclear velocities in Angstrom/fs
                Must be provided for dynamics simulation
            coefficient (Optional[np.ndarray]): Initial quantum coefficients
                If None, starts in the first specified state
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Final nuclear positions in Angstrom
                - Final nuclear velocities in Angstrom/fs
                - Final quantum coefficients
        """
    
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"Starting FSSH trajectory simulation at {now_str}")
        start_time = time.time()
        
        # Initialize or validate input parameters
        if position is None:
            position = self.tddft.mol.atom_coords(unit='Bohr')  # (Na,3) Angstrom
            print(f"[DEBUG_else] Before conversion: first atom (Bohr) = {position[0]}")
        else:
            position = position 
            print(f"[DEBUG_else] After conversion: first atom (Bohr) = {position[0]}")
            

        velocity = velocity*A2BOHR/ (FS2AUTIME * 1e3) # (Na,D) Bohr/a.u.Time

        norm = np.linalg.norm(coefficient)
        coefficient /= norm
        
        # Calculate initial electronic structure
        energy, force, nacv = self.calc_electronic(position)
        
        # Write initial trajectory frame
        self.write_trajectory(0, position, velocity, energy, coefficient)
        
        total_time = 0.0
        
        # Main simulation loop
        logger.info(f"Starting main simulation loop for {self.nsteps} steps")
        
        for i in range(self.nsteps):
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity
            
            # 3. calculte new energy, force, and nacv
            energy, force, nacv = self.calc_electronic(position)
            
            # 4. update the electronic amplitude within a full-time step
            nact = np.einsum('ijnd,nd->ij', nacv, velocity)
            coefficient = self.update_coefficient(coefficient, energy, nact)
            
            # 5. evaluate the switching probability
            p_ij = self.compute_hopping_probability(coefficient, nact)
            r = np.random.rand()
            hop_index = self.check_hop(r, p_ij)

            logger.debug(f"Switching probability: {p_ij}, Random number: {r}")
            
            # 6. adjust nuclear velocity
            cur_idx = self.states.index(self.cur_state)
            if hop_index != -1 and hop_index != cur_idx:
          
                # Attempt velocity rescaling
                d_vec = nacv[cur_idx, hop_index]
                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)
                
                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]
                    
                    logger.info(f"Hop: {old_state} → {self.cur_state} at step {i + 1}")

                else:
                    logger.debug(f"Hop to state {self.states[hop_index]} rejected "
                                 f"due to insufficient kinetic energy")
            
            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 8. update total time
            total_time += self.dt / FS2AUTIME

            # 9. decoherence
            # coefficient = self.decoherence(coefficient, velocity, energy)
            
            self.write_trajectory(i + 1, position, velocity, energy, coefficient)   
            self.print_step_info(i + 1, total_time, energy, coefficient)
        
        # Simulation completed successfully
        elapsed_time = time.time() - start_time
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"FSSH simulation completed successfully at {now_str}")
        logger.info(f"Total simulation time: {elapsed_time:.2f} s")
            
        # Convert results back to user units
        final_position = position / A2BOHR  # Angstrom
        final_velocity = velocity * (FS2AUTIME * 1e3) / A2BOHR  # Angstrom/fs
        
        return final_position, final_velocity, coefficient