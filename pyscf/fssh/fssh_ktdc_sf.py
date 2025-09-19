import numpy as np
import time
import logging
from typing import Tuple, Optional, List
from pathlib import Path
import copy

 

# ==============================================================================
# 假设您的FSSH类在一个名为 fssh_api.py 的文件中
# from pyscf.fssh.fssh_api import FSSH 
# 为了方便演示，我将FSSH类直接粘贴在这里
# ==============================================================================

# Physical constants for unit conversions
FS2AUTIME = 41.34137        # Conversion factor: femtoseconds to atomic time units
A2BOHR = 1.889726           # Conversion factor: Angstrom to Bohr radius
AMU2AU = 1822.8884858012984 # Conversion factor: atomic mass units to atomic units

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FSSH:
    """
    This class implements the FSSH algorithm with κTDC (kappa Time-Derivative Coupling)
    for efficient nonadiabatic molecular dynamics simulations.

    The FSSH method treats nuclear motion classically while quantum mechanically
    describing electronic transitions between different potential energy surfaces.
    The κTDC approach provides a computationally efficient alternative to traditional
    nonadiabatic coupling vector calculations.    
    
    Attributes:
        tddft: Time-dependent density functional theory object
        tdgrad: Nuclear gradient scanner for force calculations
        states (List[int]): List of electronic states to include in simulation
        cur_state (int): Current active electronic state
        mass (np.ndarray): Nuclear masses in atomic units
        dt (float): Time step in atomic units
        nsteps (int): Number of simulation steps
    """
    def __init__(self, 
                 tddft, 
                 states:list[int], 
                 **kwargs):
        """
        Initialize the FSSH simulation with comprehensive parameter validation.
        
        Args:
            tddft: Time-dependent DFT object providing electronic structure
            states (List[int]): Electronic states to include in simulation. Note: 1-based indexing for excited states.
            **kwargs: Additional simulation parameters including:
                - dt (float): Time step in femtoseconds (default: 0.5)
                - nsteps (int): Number of simulation steps (default: 1)
                - output_dir (str): Directory for output files (default: current)
                - verbose (bool): Enable verbose output (default: True)
        """

        # Validate input parameters
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("At least two electronic states must be specified")
        
        if any(not isinstance(s, int) or s <= 0 for s in states):
            raise ValueError("All state indices must be positive integers (1-based)")
        
        # Initialize core simulation objects
        self.tddft = tddft
        # This line will be overridden in our child class
        if hasattr(self.tddft, 'nuc_grad_method'):
            self.tdgrad = self.tddft.nuc_grad_method().as_scanner()

        # Set up electronic state configuration
        self.states = list(states)
        self.Nstates = len(states)
        self.cur_state = states[0]  # Start from the first specified state

        # Calculate nuclear masses and convert to atomic units
        self.mass = self.tddft.mol.atom_mass_list(True).reshape(-1, 1) * AMU2AU  # (Na,1)  Unit: a.u.

        # Generate indices for nonadiabatic coupling calculations
        # Only consider unique pairs (i,j) where i < j to avoid redundancy
        self.nac_idx = [(i, j) for i in range(self.Nstates-1) 
                        for j in range(i+1, self.Nstates)]
        
        # Set default simulation parameters
        self.dt = 0.5 * FS2AUTIME  # Default: 0.5 fs in atomic units
        self.nsteps = 1
        self.output_dir = Path('.')
        
        # Override defaults with user-provided parameters
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
        
        logger.info(f"FSSH simulation initialized with {self.Nstates} states, "
                   f"dt={self.dt/FS2AUTIME:.3f} fs, {self.nsteps} steps")

    # ... (kTDC, exp_propagator, update_coefficient, compute_hopping_probability, check_hop, rescale_velocity, write_trajectory, print_step_info remain unchanged) ...
    def kTDC(self, 
             energy_t: np.ndarray, 
             energy_p: np.ndarray, 
             energy_pp: np.ndarray) -> np.ndarray:
        nact = np.zeros((self.Nstates, self.Nstates))
        for idx in self.nac_idx:
            i, j = idx
            dVt = energy_t[j] - energy_t[i]
            dVp = energy_p[j] - energy_p[i]
            dVpp = energy_pp[j] - energy_pp[i]
            d2Vdt2 = (dVt - 2 * dVp + dVpp) / (self.dt**2)
            sqrt_part = d2Vdt2 / dVt
            if sqrt_part > 0:
                kappa_value = np.sqrt(sqrt_part)
            else:
                kappa_value = 0.0
            nact[j, i] = kappa_value
            nact[i, j] = -kappa_value
        return nact

    def calc_electronic(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # This method WILL BE OVERRIDDEN for SF-TDDFT
        mol_temp = self.tddft.mol.copy().set_geom_(position, unit='Bohr')
        self.tdgrad.state = self.cur_state
        energy, grad = self.tdgrad(mol_temp)
        force = -grad
        # This energy logic is for standard TDDFT, will be replaced
        all_energies = np.concatenate(([self.tdgrad.base._scf.e_tot], self.tdgrad.e_tot))
        energy = all_energies[self.states]
        return energy, force

    def exp_propagator(self, c: np.ndarray, Veff: np.ndarray, dt: float) -> np.ndarray:
        diags, coeff = np.linalg.eigh(Veff)
        U = coeff @ np.diag(np.exp(-1j * diags * dt)) @ coeff.T.conj()
        c_new = np.dot(U, c)
        c_new = c_new / np.linalg.norm(c_new)
        return c_new

    def update_coefficient(self, coeffs: np.ndarray, 
                           energy: np.ndarray, 
                           nact: np.ndarray) -> np.ndarray:
        Veff = np.diag(energy) - 1j * nact
        c_new = self.exp_propagator(coeffs, Veff, self.dt)
        return c_new

    def compute_hopping_probability(self, 
                                    coeffs: np.ndarray, 
                                    nact: np.ndarray) -> np.ndarray:
        state_idx = self.states.index(self.cur_state)
        c_i = coeffs[state_idx]
        if np.abs(c_i)**2 < 1e-8: # Avoid division by zero
            return np.zeros(self.Nstates)
        g_ij = 2 * (nact[state_idx] * c_i.conj() * coeffs).real * self.dt / (np.abs(c_i)**2)
        p_ij = np.where(g_ij < 0, 0, g_ij)
        p_ij = np.where(p_ij > 1, 1, p_ij)
        return p_ij

    def check_hop(self, r: float, p_ij: np.ndarray) -> int:
        cumu_p_ij = np.cumsum(p_ij)
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
        state_idx = self.states.index(self.cur_state)
        a = np.sum(d_vec**2 / (2 * self.mass))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[state_idx]
        delta = b**2 - 4 * a * c
        if delta >= 0:
            gamma = (b + np.sqrt(delta)) / (2 * a) if b < 0 else (b - np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass
            return False, velocity
            
    def write_trajectory(self, 
                         step: int, 
                         position: np.ndarray, 
                         velocity: np.ndarray,
                         energy: np.ndarray, 
                         coeffs: np.ndarray, 
                         filename: str = 'trajectory.xyz') -> None:
        filepath = self.output_dir / filename
        mode = 'w' if step == 0 else 'a'
        with open(filepath, mode) as f:
            f.write(f'{self.tddft.mol.natm}\n')
            time_fs = step * self.dt / FS2AUTIME
            current_energy = energy[self.states.index(self.cur_state)]
            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {self.cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')
            for i, coord in enumerate(position):
                symbol = self.tddft.mol.atom_pure_symbol(i)
                x, y, z = coord / A2BOHR
                f.write(f'{symbol:4.2s} {x:12.6f} {y:12.6f} {z:12.6f}\n')
    
    def print_step_info(self, 
                        step: int, 
                        total_time: float, 
                        energy: np.ndarray,
                        coeffs: np.ndarray, 
                        ) -> None:
        current_idx = self.states.index(self.cur_state)
        current_energy = energy[current_idx]
        populations = np.abs(coeffs)**2
        logger.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, "
              f"Energy {current_energy:12.8f} Ha, Populations: {populations}")

    def kernel(self, 
               position: Optional[np.ndarray] = None, 
               velocity: Optional[np.ndarray] = None, 
               coefficient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # This method WILL BE OVERRIDDEN to fix gradient calls
        # ... (original kernel logic) ...
        pass


# ==============================================================================
# The NEW Adapter Class for SF-TDDFT
# ==============================================================================
from pyscf import gto, scf, tdscf, lib, grad
from pyscf import dft
from pyscf import sftda
from pyscf.grad import tduks_sf # Important import for SF gradients

class FSSH_SF(FSSH):
    """
    An adapter class to run FSSH dynamics using SF-TDDFT electronic structure.

    This class inherits from the general FSSH implementation and overrides
    methods related to electronic structure calculations to make them compatible
    with PySCF's spin-flip TDDFT (sftda) module.
    """
    def __init__(self, 
                 sftda_obj,      # Takes an sftda object instead of tddft
                 states:list[int], 
                 **kwargs):
        """
        Initializes the FSSH simulation for a SF-TDDFT system.
        
        Args:
            sftda_obj: A converged PySCF TDA_SF or TDDFT_SF object.
            states (List[int]): Electronic states to include (1-based indices).
            **kwargs: Additional simulation parameters (dt, nsteps, etc.).
        """
        # Store the sftda object as self.tddft to match parent class attributes
        self.tddft = sftda_obj

        # Crucially, instantiate the correct gradient method for SF-TDDFT
        self.tdgrad = tduks_sf.Gradients(self.tddft).as_scanner()

        # Call the parent's __init__ method to set up the rest
        super().__init__(self.tddft, states, **kwargs)

    def calc_electronic(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate electronic energies and nuclear forces for all states using SF-TDDFT.
        
        This method overrides the parent implementation to correctly handle
        energy and gradient calculations for spin-flip excited states.
        
        Args:
            position (np.ndarray): Nuclear coordinates in Bohr (Natoms * 3)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - energy: Electronic energies for all states (Nstates,) in Hartree
                - force: Nuclear forces for current state (Natoms * 3) in Ha/Bohr 
        """
        # Create a temporary molecule object with the new coordinates for the scanner
        mol_temp = self.tddft.mol.copy().set_geom_(position, unit='Bohr')

        # Set the current state for the gradient calculation (1-based index)
        self.tdgrad.state = self.cur_state

        # The SF-TDDFT gradient scanner returns only the gradient
        grad = self.tdgrad(mol_temp) 
        force = -grad  # (Na,D)  Unit: Ha/bohr

        # Calculate energies for ALL excited states from the sftda object
        # Energy = E_reference_SCF + E_excitation
        all_energies = self.tddft._scf.e_tot + self.tddft.e

        # Select the energies for the states included in the FSSH simulation
        # Note: self.states are 1-based, so we subtract 1 for 0-based numpy indexing
        energy_indices = [s - 1 for s in self.states]
        energy = all_energies[energy_indices]  # (Nstates,)  Unit: Ha

        return energy, force
    
    def kernel(self, 
               position: Optional[np.ndarray] = None, 
               velocity: Optional[np.ndarray] = None, 
               coefficient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the main FSSH trajectory simulation, adapted for SF-TDDFT.
        
        This method overrides the parent kernel to ensure correct calls to the
        PySCF standard gradient scanner when checking for velocity rescaling.
        """
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"Starting FSSH-SF trajectory simulation at {now_str}")
        start_time = time.time()

        if position is None:
            position = self.tddft.mol.atom_coords() * A2BOHR # Internally use Bohr
        else:
            position = position * A2BOHR

        if velocity is None:
            raise ValueError("Initial velocities must be provided for a dynamics simulation.")
        # Convert Angstrom/fs to Bohr/a.u. time
        velocity = velocity * (A2BOHR / FS2AUTIME)

        if coefficient is None:
            coefficient = np.zeros(self.Nstates, dtype=complex)
            coefficient[0] = 1.0 # Start in the first state of the list
        norm = np.linalg.norm(coefficient)
        coefficient /= norm

        energy_list = [None, None, None]
        energy, force = self.calc_electronic(position)
        energy_list[0] = energy

        logger.info("Performing initial steps to establish energy history for kTDC")
        velocity = velocity + 0.5 * self.dt * force / self.mass
        position = position + self.dt * velocity
        energy, force = self.calc_electronic(position)
        energy_list[1] = energy
        velocity = velocity + 0.5 * self.dt * force / self.mass
        
        self.write_trajectory(0, position, velocity, energy, coefficient)   
        total_time = 0.0

        logger.info(f"Starting main simulation loop for {self.nsteps} steps")
        
        for i in range(self.nsteps):
            velocity = velocity + 0.5 * self.dt * force / self.mass
            position = position + self.dt * velocity
            energy, force = self.calc_electronic(position) 

            energy_list[(i+2)%3] = energy
            nact = self.kTDC(energy_list[(i+2)%3], energy_list[(i+1)%3], energy_list[i%3])
            coefficient = self.update_coefficient(coefficient, energy, nact)

            p_ij = self.compute_hopping_probability(coefficient, nact)
            r = np.random.rand()
            hop_index = self.check_hop(r, p_ij) # This is an index into self.states list (0 to Nstates-1)

            if hop_index != -1 and hop_index != self.states.index(self.cur_state):
                # This section is modified to correctly call the gradient scanner
                
                # Create a temporary molecule with the current geometry
                mol_temp = self.tddft.mol.copy().set_geom_(position, unit='Bohr')
                
                # Set the target state for the gradient calculation
                target_state_abs_idx = self.states[hop_index]
                self.tdgrad.state = target_state_abs_idx
                
                # Calculate the gradient of the target state
                grad_target_state = self.tdgrad(mol_temp)
                
                # Gradient of current state is simply -force
                grad_current_state = -force
                
                d_vec = grad_target_state - grad_current_state

                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)
                
                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]
                    logger.info(f"Hop: {old_state} -> {self.cur_state} at step {i + 1}")
                    # After a hop, the force acting on the nuclei is from the new state
                    force = -grad_target_state
                else:
                    logger.debug(f"Hop to state {self.states[hop_index]} rejected "
                                 f"due to insufficient kinetic energy")

            velocity = velocity + 0.5 * self.dt * force / self.mass
            total_time += self.dt / FS2AUTIME

            self.write_trajectory(i + 1, position, velocity, energy, coefficient)   
            self.print_step_info(i + 1, total_time, energy, coefficient)
        
        elapsed_time = time.time() - start_time
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"FSSH simulation completed successfully at {now_str}")
        logger.info(f"Total simulation time: {elapsed_time:.2f} s")
            
        final_position = position / A2BOHR
        final_velocity = velocity * (FS2AUTIME / A2BOHR)
        
        return final_position, final_velocity, coefficient


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == '__main__':
    # 1. Setup the SF-TDDFT calculation (as provided in your example)
    mol = gto.Mole()
    mol.atom = '''
    C    0.000000    0.000000    0.000000
    O    1.215000    0.000000    0.000000
    H    0.582000    0.940000    0.000000
    H    0.582000   -0.940000    0.000000
    '''
    mol.basis = '6-31g'
    mol.spin = 2  # High-spin triplet reference for SF
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'svwn' 
    mf.kernel()

    # TDA_SF object
    mftd1 = sftda.uks_sf.TDA_SF(mf)
    mftd1.max_space =2000
    mftd1.nstates = 4  # Calculate 4 target states
    mftd1.kernel()

    # 2. Define FSSH simulation parameters
    # Let's simulate hopping between the first (S1) and third (S3) excited states
    # Note: These are 1-based indices corresponding to the output of mftd1.e
    active_states = [1, 3] 
    initial_state = 1 # Start on the S1 state

    # 3. Instantiate our new FSSH_SF adapter class
    fssh_simulation = FSSH_SF(
        sftda_obj=mftd1,
        states=active_states,
        dt=0.1,                # Time step in fs
        nsteps=50,             # Number of steps
        output_dir='fssh_sf_traj'
    )
    
    # 4. Set initial conditions for the dynamics
    initial_pos = mol.atom_coords() # in Angstrom

    # Create some small random initial velocities (in Angstrom/fs)
    # In a real simulation, these would come from a distribution (e.g., Wigner)
    np.random.seed(42)
    initial_vel = (np.random.rand(*initial_pos.shape) - 0.5) * 0.1 

    # Initial quantum coefficients: start purely in the first state of our active space list
    # The active space is [S1, S3]. Starting on S1 means the first coefficient is 1.
    initial_coeffs = np.array([1.0, 0.0], dtype=complex)

    # 5. Run the FSSH kernel
    final_pos, final_vel, final_coeffs = fssh_simulation.kernel(
        position=initial_pos,
        velocity=initial_vel,
        coefficient=initial_coeffs
    )

    print("\nFSSH Simulation Finished.")
    print("Final positions (Angstrom):")
    print(final_pos)
    print("\nFinal quantum coefficients:")
    print(final_coeffs)
    print("Final populations:", np.abs(final_coeffs)**2)