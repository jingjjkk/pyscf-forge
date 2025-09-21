import numpy as np
import time
import logging
from typing import Tuple, Optional, List, Dict
from pathlib import Path

from pyscf.grad import tduks_sf
from pyscf.nac.tduks_sf import NAC as SFTD_NAC_Calculator

FS2AUTIME = 41.34137
A2BOHR = 1.889726
AMU2AU = 1822.8884858012984

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FSSH_SF_NACV:
    def __init__(self, sftda_obj, states:List[int], nac_options:Optional[Dict] = None, cphf_options: Optional[Dict] = None, **kwargs):
        if any(s <= 0 for s in states):
            raise ValueError("SF-TDDFT states must be positive integers (1-based)")

        self.tddft = sftda_obj
        self.states = list(states)
        self.cur_state = states[0]
        self.Nstates = len(states)
        self.mass = self.tddft.mol.atom_mass_list(True).reshape(-1, 1) * AMU2AU
        self.nac_idx = [(i, j) for i in range(self.Nstates - 1) for j in range(i + 1, self.Nstates)]

        self.dt = 0.5 * FS2AUTIME
        self.nsteps = 1
        self.output_dir = Path('.')
        for key, value in kwargs.items():
            if key == 'dt': self.dt = value * FS2AUTIME
            elif key == 'nsteps': self.nsteps = value
            elif key == 'output_dir': self.output_dir = Path(value)
            else: setattr(self, key, value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        grad_calculator = tduks_sf.Gradients(self.tddft)
        self.sftd_nac_calc = SFTD_NAC_Calculator(self.tddft)

        if cphf_options:
            max_cycle = cphf_options.get('max_cycle', 50)
            conv_tol = cphf_options.get('conv_tol', 1e-7)
            logger.info(f"Applying custom CPHF settings: max_cycle={max_cycle}, conv_tol={conv_tol}")
            grad_calculator.cphf_max_cycle = max_cycle
            grad_calculator.cphf_conv_tol = conv_tol
            if hasattr(self.sftd_nac_calc, 'cphf_max_cycle'):
                self.sftd_nac_calc.cphf_max_cycle = max_cycle
                self.sftd_nac_calc.cphf_conv_tol = conv_tol
        
        self.tdgrad = grad_calculator.as_scanner()
        
        if nac_options:
            for key, value in nac_options.items():
                setattr(self.sftd_nac_calc, key, value)
        
        logger.info(f"FSSH-SF-NACV simulation initialized with {self.Nstates} states, dt={self.dt/FS2AUTIME:.3f} fs, {self.nsteps} steps")

    def calc_electronic(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mol_temp = self.tddft.mol.copy().set_geom_(position, unit='Bohr')
        self.tdgrad.state = self.cur_state
        _, grad = self.tdgrad(mol_temp)
        force = -grad
        all_exc_energies = self.tddft.e
        ref_energy = self.tddft._scf.e_tot
        energy_indices = [s - 1 for s in self.states]
        energy = ref_energy + all_exc_energies[energy_indices]
        natm = mol_temp.natm
        Nacv = np.zeros((self.Nstates, self.Nstates, natm, 3))
        for i, j in self.nac_idx:
            state_I_abs, state_J_abs = self.states[i], self.states[j]
            self.sftd_nac_calc.state_I, self.sftd_nac_calc.state_J = state_I_abs, state_J_abs
            nk = self.sftd_nac_calc(mol_temp)
            Nacv[i, j], Nacv[j, i] = nk, -nk
        return energy, force, Nacv

    def exp_propagator(self, c: np.ndarray, Veff: np.ndarray, dt: float) -> np.ndarray:
        diags, coeff = np.linalg.eigh(Veff)
        U = coeff @ np.diag(np.exp(-1j * diags * dt)) @ coeff.T.conj()
        c_new = np.dot(U, c)
        return c_new / np.linalg.norm(c_new)

    def update_coefficient(self, coeffs: np.ndarray, energy: np.ndarray, nact: np.ndarray) -> np.ndarray:
        Veff = np.diag(energy) - 1j * nact
        return self.exp_propagator(coeffs, Veff, self.dt)

    def compute_hopping_probability(self, coeffs: np.ndarray, nact: np.ndarray) -> np.ndarray:
        state_idx = self.states.index(self.cur_state)
        c_i = coeffs[state_idx]
        if np.abs(c_i)**2 < 1e-8: return np.zeros(self.Nstates)
        g_ij = 2 * (nact[state_idx] * c_i.conj() * coeffs).real * self.dt / (np.abs(c_i)**2)
        return np.maximum(0, g_ij)

    def check_hop(self, r: float, p_ij: np.ndarray) -> int:
        cumu_p_ij = np.cumsum(p_ij)
        for k, u_bound in enumerate(cumu_p_ij):
            if r <= u_bound: return k
        return -1

    def rescale_velocity(self, hop_index: int, energy: np.ndarray, velocity: np.ndarray, d_vec: np.ndarray) -> Tuple[bool, np.ndarray]:
        state_idx = self.states.index(self.cur_state)
        a = np.sum(d_vec**2 / (2 * self.mass))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[state_idx]
        delta = b**2 - 4 * a * c
        if a < 1e-9: return c <= 0, velocity
        if delta >= 0:
            gamma = (b - np.sign(b) * np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass
            return False, velocity

    def write_trajectory(self, step: int, position: np.ndarray, energy: np.ndarray, coeffs: np.ndarray) -> None:
        filepath = self.output_dir / 'trajectory.xyz'
        mode = 'w' if step == 0 else 'a'
        with open(filepath, mode) as f:
            f.write(f'{self.tddft.mol.natm}\n')
            time_fs = step * self.dt / FS2AUTIME
            current_relative_idx = self.states.index(self.cur_state)
            current_energy = energy[current_relative_idx]
            pop_str = " ".join([f"{abs(c)**2:.4f}" for c in coeffs])
            comment = (f'Step={step} Time={time_fs:.3f} State={self.cur_state} Energy={current_energy:.8f} Pop=[{pop_str}]')
            f.write(comment + '\n')
            for i, coord in enumerate(position):
                symbol = self.tddft.mol.atom_pure_symbol(i)
                x, y, z = coord / A2BOHR
                f.write(f'{symbol:4s} {x:12.6f} {y:12.6f} {z:12.6f}\n')
    
    def print_step_info(self, step: int, total_time: float, energy: np.ndarray, coeffs: np.ndarray) -> None:
        current_relative_idx = self.states.index(self.cur_state)
        current_energy = energy[current_relative_idx]
        populations = np.abs(coeffs)**2
        logger.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, Energy {current_energy:12.8f} Ha, Populations: {populations}")

    def kernel(self, position: np.ndarray, velocity: np.ndarray, coefficient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Starting FSSH trajectory simulation at {time.ctime()}")
        start_time = time.time()
        position, velocity = position * A2BOHR, velocity * (A2BOHR / FS2AUTIME)
        coefficient /= np.linalg.norm(coefficient)
        energy, force, nacv = self.calc_electronic(position)
        self.write_trajectory(0, position, energy, coefficient)
        self.print_step_info(0, 0.0, energy, coefficient)
        total_time = 0.0
        logger.info(f"Starting main simulation loop for {self.nsteps} steps")
        for i in range(self.nsteps):
            step = i + 1
            velocity += 0.5 * self.dt * force / self.mass
            position += self.dt * velocity
            energy, force, nacv = self.calc_electronic(position)
            nact = np.einsum('ijad,ad->ij', nacv, velocity)
            coefficient = self.update_coefficient(coefficient, energy, nact)
            p_ij = self.compute_hopping_probability(coefficient, nact)
            r = np.random.rand()
            hop_index = self.check_hop(r, p_ij)
            cur_idx = self.states.index(self.cur_state)
            if hop_index != -1 and hop_index != cur_idx:
                d_vec = nacv[cur_idx, hop_index]
                hop_allowed, new_velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)
                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]
                    velocity = new_velocity
                    logger.info(f"Hop: {old_state} -> {self.cur_state} at step {step}")
                    self.tdgrad.state = self.cur_state
                    mol_temp = self.tddft.mol.copy().set_geom_(position, unit='Bohr')
                    _, new_grad = self.tdgrad(mol_temp)
                    force = -new_grad
                else:
                    velocity = new_velocity
                    logger.debug(f"Hop to state {self.states[hop_index]} rejected (frustrated hop)")
            velocity += 0.5 * self.dt * force / self.mass
            total_time += self.dt / FS2AUTIME
            self.write_trajectory(step, position, energy, coefficient)   
            self.print_step_info(step, total_time, energy, coefficient)
        logger.info(f"FSSH simulation completed at {time.ctime()} in {time.time() - start_time:.2f} s")
        final_position, final_velocity = position / A2BOHR, velocity * (FS2AUTIME / A2BOHR)
        return final_position, final_velocity, coefficient