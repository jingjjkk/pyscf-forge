import numpy as np
from pyscf.sftda.tools_td import spin_square

def extract_state(mf, mftd, tdtype='TDDFT', Smin=0.0, Smax=0.7):
    oa = mf.mol.nelec[0]
    nvir = mf.mol.nao_nr() - mf.mol.nelec[1]
    S = []
    j = 0

    for i in range(min(20, len(mftd.xy))):
        xy = mftd.xy[i]
        s2 = spin_square(mf, mftd.xy[i], extype=1, tdtype=tdtype)
        x = xy[0][1].flatten()
        norm = x.conj() * x
        idx_mo = np.argsort(norm)
        idx_u = idx_mo[-1]
        idx_u2 = idx_mo[-2]

        a_i_mo_idx = (idx_u//nvir+1, idx_u%nvir+1)
        a_i_mo_idx2 =(idx_u2//nvir+1, idx_u2%nvir+1)

        if Smin <= s2 <= Smax:
            j += 1
            print(f'{i}-th state with excitation energy = {mftd.e[i]*27.2114:.4f},', end=' ')
            print(f'S^2 = {s2:.4f}', end=' ')
            print(f'Norms: {norm[idx_u]:.3f}@{a_i_mo_idx}, {norm[idx_u2]:.3f}@{a_i_mo_idx2}')
            S.append(i)
    return S


def extract_S(mf, mftd, tdtype='TDDFT', tolerance=0.1):
    '''
    find S0, S1, and S2 states from spin-flip-down TDDFT/TDA calculations
    '''
    oa = mf.mol.nelec[0]
    nvir = mf.mol.nao_nr() - mf.mol.nelec[1]
    S = []
    j = -1

    for i in range(len(mftd.xy)):
        xy = mftd.xy[i]
        s2 = spin_square(mf, mftd.xy[i], extype=1, tdtype=tdtype)
        x = xy[0][1].flatten()
        norm = x.conj() * x
        idx_mo = np.argsort(norm)
        idx_u = idx_mo[-1]
        idx_u2 = idx_mo[-2]

        a_i_mo_idx = (idx_u//nvir+1, idx_u%nvir+1)
        a_i_mo_idx2 =(idx_u2//nvir+1, idx_u2%nvir+1)

        if s2 < tolerance:
            j += 1
            print(f'S{j} state with S^2 = {s2:.3f}', end=' ')
            print(f'Norms: {norm[idx_u]:.2f}@{a_i_mo_idx}, {norm[idx_u2]:.2f}@{a_i_mo_idx2}')
            S.append(i)
        if j == 2:
            return S
    if j < 2:
        print('No enough singlet states!')
        return S

def extract_T(mf, mftd, tdtype='TDDFT', tolerance=0.1):
    '''
    find T1 state from spin-flip-down TDDFT/TDA calculations
    '''
    oa = mf.mol.nelec[0]
    nvir = mf.mol.nao_nr() - mf.mol.nelec[1]
    S = []
    j = 0

    for i in range(len(mftd.xy)):
        xy = mftd.xy[i]
        s2 = spin_square(mf, mftd.xy[i], extype=1, tdtype=tdtype)
        x = xy[0][1].flatten()
        norm = x.conj() * x
        idx_mo = np.argsort(norm)
        idx_u = idx_mo[-1]
        idx_u2 = idx_mo[-2]

        a_i_mo_idx = (idx_u//nvir+1, idx_u%nvir+1)
        a_i_mo_idx2 =(idx_u2//nvir+1, idx_u2%nvir+1)

        if 2 - tolerance < s2 < 2 + tolerance:
            j += 1
            print(f'T{j} state with S^2 = {s2:.3f}', end=' ')
            print(f'Norms: {norm[idx_u]:.2f}@{a_i_mo_idx}, {norm[idx_u2]:.2f}@{a_i_mo_idx2}')
            S.append(i)
        if j == 1:
            return S
    if j < 1:
        print('No enough triplet states!')
        return S