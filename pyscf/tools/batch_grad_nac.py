"""
批量梯度和NAC联合计算接口

此模块提供了高效的批量计算接口,用于同时计算多个电子态的梯度和它们之间的NAC。
通过智能缓存管理,避免重复计算,特别适用于:
- FSSH动力学模拟(需要多个态的梯度和NAC)
- MECP搜索(需要两个态的梯度和NAC)
- 势能面扫描

主要优化:
1. 所有态共享XC核计算
2. 梯度计算的中间量被缓存供NAC使用
3. 自动管理缓存生命周期

Example FSSH usage:
    >>> from batch_grad_nac import compute_gradients_and_nacs
    >>> states = [1, 2, 3]  # S0, S1, S2
    >>> grads, nacs = compute_gradients_and_nacs(mftd, states)
    >>> # grads[i] 是 states[i] 的梯度
    >>> # nacs[(i,j)] 是 states[i] 和 states[j] 之间的NAC

Example MECP usage:
    >>> from batch_grad_nac import compute_mecp_data
    >>> grad1, grad2, nac = compute_mecp_data(mftd, state_I=1, state_J=2)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pyscf.lib import logger
from .gradient_nac_cache import get_cache_manager, clear_all_caches
from pyscf.tdgrad import tduks_sf as tduks_sf_grad
from pyscf.tdnac import tduks_sf as tduks_sf_nac


def compute_gradients_and_nacs(
    tddft_obj,
    states: List[int],
    nac_pairs: Optional[List[Tuple[int, int]]] = None,
    atmlst: Optional[List[int]] = None,
    ediff: bool = True,
    use_etfs: bool = True,
    max_memory: int = 6000,
    verbose: int = logger.INFO
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    批量计算多个态的梯度和NAC,利用缓存避免冗余计算
    
    计算策略:
    1. 首次调用时,XC核被计算一次并缓存
    2. 按顺序计算每个态的梯度,中间量被缓存
    3. 计算NAC时,直接使用已缓存的梯度中间量
    
    Args:
        tddft_obj: SF-TDDFT对象 (TDA_SF或TDDFT_SF)
        states: 需要计算的态列表(1-based索引),例如[1,2,3]表示S0,S1,S2
        nac_pairs: NAC态对列表,例如[(1,2),(1,3)]. 如果为None,计算所有相邻态对
        atmlst: 原子索引列表,如果为None则计算所有原子
        ediff: 是否用能量差归一化NAC
        use_etfs: 是否只使用ETF修正的Hellmann-Feynman项
        max_memory: 最大内存(MB)
        verbose: 日志级别
    
    Returns:
        (gradients, nacs):
            - gradients: 字典 {state_id: gradient_array}
            - nacs: 字典 {(state_i, state_j): nac_vector}
    
    Example:
        >>> grads, nacs = compute_gradients_and_nacs(mftd, [1, 2, 3])
        >>> grad_s1 = grads[2]  # S1的梯度
        >>> nac_s0_s1 = nacs[(1, 2)]  # S0-S1的NAC
    """
    log = logger.new_logger(tddft_obj, verbose)
    cache_mgr = get_cache_manager()
    
    # 验证输入
    if not states:
        raise ValueError("states列表不能为空")
    if min(states) < 1:
        raise ValueError("态索引必须从1开始(1-based)")
    
    # 如果未指定NAC态对,计算所有相邻态对
    if nac_pairs is None:
        nac_pairs = [(states[i], states[i+1]) for i in range(len(states)-1)]
    
    log.info(f"批量计算 {len(states)} 个态的梯度和 {len(nac_pairs)} 对NAC")
    log.info(f"态列表: {states}")
    log.info(f"NAC对: {nac_pairs}")
    
    # ========== 第一阶段: 计算所有梯度并缓存中间量 ==========
    gradients = {}
    grad_obj = tduks_sf_grad.Gradients(tddft_obj)
    grad_obj.max_memory = max_memory
    grad_obj.verbose = verbose
    
    log.info("阶段1: 计算梯度并缓存中间量")
    for state_id in states:
        log.info(f"  计算态 {state_id} 的梯度...")
        xy = tddft_obj.xy[state_id - 1]
        
        # use_cache=True 会自动缓存中间量
        grad = grad_obj.grad_elec(xy, atmlst=atmlst, state_id=state_id, use_cache=True)
        gradients[state_id] = grad
        
        log.debug(f"  态 {state_id} 梯度完成,中间量已缓存")
    
    # ========== 第二阶段: 利用缓存计算NAC ==========
    nacs = {}
    nac_obj = tduks_sf_nac.NonAdiabaticCouplings(tddft_obj)
    nac_obj.ediff = ediff
    nac_obj.use_etfs = use_etfs
    nac_obj.max_memory = max_memory
    nac_obj.verbose = verbose
    
    log.info("阶段2: 利用缓存计算NAC")
    for state_i, state_j in nac_pairs:
        if state_i not in states or state_j not in states:
            log.warn(f"跳过NAC({state_i},{state_j}): 态不在states列表中")
            continue
        
        log.info(f"  计算NAC({state_i}, {state_j})...")
        
        # use_cache=True 会尝试使用梯度计算时缓存的中间量
        nac = nac_obj.compute_nac(
            state_I=state_i,
            state_J=state_j,
            atmlst=atmlst,
            ediff=ediff,
            use_etfs=use_etfs,
            use_cache=True
        )
        
        nacs[(state_i, state_j)] = nac
        log.debug(f"  NAC({state_i}, {state_j}) 完成")
    
    # 打印缓存统计
    if verbose >= logger.INFO:
        cache_mgr.print_stats()
    
    return gradients, nacs


def compute_mecp_data(
    tddft_obj,
    state_I: int,
    state_J: int,
    atmlst: Optional[List[int]] = None,
    ediff: bool = False,
    use_etfs: bool = True,
    max_memory: int = 6000,
    verbose: int = logger.INFO
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为MECP搜索计算所需数据:两个态的梯度和它们之间的NAC
    
    这是compute_gradients_and_nacs的简化接口,专门用于MECP场景
    
    Args:
        tddft_obj: SF-TDDFT对象
        state_I, state_J: 两个态的索引(1-based)
        atmlst: 原子列表
        ediff: 是否用能量差归一化NAC
        use_etfs: 是否只用ETF修正项
        max_memory: 最大内存(MB)
        verbose: 日志级别
    
    Returns:
        (grad_I, grad_J, nac_IJ):
            - grad_I: 态I的梯度
            - grad_J: 态J的梯度
            - nac_IJ: I和J之间的NAC向量
    
    Example:
        >>> grad1, grad2, nac = compute_mecp_data(mftd, 1, 2)
        >>> # 用于MECP优化的梯度差和垂直梯度
        >>> g_parallel = grad1 - grad2
        >>> g_perpendicular = nac
    """
    grads, nacs = compute_gradients_and_nacs(
        tddft_obj,
        states=[state_I, state_J],
        nac_pairs=[(state_I, state_J)],
        atmlst=atmlst,
        ediff=ediff,
        use_etfs=use_etfs,
        max_memory=max_memory,
        verbose=verbose
    )
    
    return grads[state_I], grads[state_J], nacs[(state_I, state_J)]


def compute_fssh_data(
    tddft_obj,
    states: List[int],
    active_state: int,
    atmlst: Optional[List[int]] = None,
    ediff: bool = True,
    use_etfs: bool = True,
    max_memory: int = 6000,
    verbose: int = logger.INFO
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    为FSSH单步计算所需数据:当前态梯度和所有NAC
    
    Args:
        tddft_obj: SF-TDDFT对象
        states: 参与动力学的所有态
        active_state: 当前活动态
        atmlst: 原子列表
        ediff: 是否用能量差归一化NAC
        use_etfs: 是否只用ETF修正项
        max_memory: 最大内存(MB)
        verbose: 日志级别
    
    Returns:
        (force, nacs):
            - force: 当前活动态的力(负梯度)
            - nacs: 字典 {(i, j): nac_vector},包含所有需要的NAC对
    """
    # 生成所有需要的NAC对(只计算上三角,因为NAC反对称)
    nac_pairs = [(states[i], states[j]) 
                 for i in range(len(states)-1) 
                 for j in range(i+1, len(states))]
    
    grads, nacs = compute_gradients_and_nacs(
        tddft_obj,
        states=states,
        nac_pairs=nac_pairs,
        atmlst=atmlst,
        ediff=ediff,
        use_etfs=use_etfs,
        max_memory=max_memory,
        verbose=verbose
    )
    
    # 返回力(负梯度)和NAC
    force = -grads[active_state]
    
    return force, nacs


def compute_single_gradient_cached(
    tddft_obj,
    state_id: int,
    atmlst: Optional[List[int]] = None,
    max_memory: int = 6000,
    verbose: int = logger.INFO
) -> np.ndarray:
    """
    计算单个态的梯度并缓存中间量
    
    适用于只需要计算一个态梯度的场景,但未来可能需要相关NAC
    
    Args:
        tddft_obj: SF-TDDFT对象
        state_id: 态索引(1-based)
        atmlst: 原子列表
        max_memory: 最大内存(MB)
        verbose: 日志级别
    
    Returns:
        梯度向量 (natm, 3)
    """
    grad_obj = tduks_sf_grad.Gradients(tddft_obj)
    grad_obj.max_memory = max_memory
    grad_obj.verbose = verbose
    
    xy = tddft_obj.xy[state_id - 1]
    grad = grad_obj.grad_elec(xy, atmlst=atmlst, state_id=state_id, use_cache=True)
    
    return grad


def clear_cache_for_new_geometry():
    """
    清除所有缓存,准备新几何构型的计算
    
    在轨迹优化或动力学模拟的每一步开始时调用
    """
    clear_all_caches()
    print("已清除所有缓存,准备新几何构型计算")


# ========== 便捷函数 ==========

def precompute_gradients(tddft_obj, states: List[int], **kwargs) -> Dict[int, np.ndarray]:
    """
    预计算并缓存多个态的梯度
    
    这在NAC密集计算前很有用,可以一次性缓存所有需要的中间量
    """
    grads, _ = compute_gradients_and_nacs(
        tddft_obj, states, nac_pairs=[], **kwargs
    )
    return grads


def compute_nac_only(
    tddft_obj,
    state_I: int,
    state_J: int,
    require_gradients: bool = True,
    **kwargs
) -> np.ndarray:
    """
    只计算NAC,可选择是否自动计算所需梯度
    
    Args:
        require_gradients: 如果True且缓存未命中,自动计算梯度
    """
    cache_mgr = get_cache_manager()
    mol = tddft_obj.mol
    mf = tddft_obj._scf
    
    # 检查是否有缓存的梯度数据
    has_I = cache_mgr.has_gradient_data(mf, mol, state_I)
    has_J = cache_mgr.has_gradient_data(mf, mol, state_J)
    
    if require_gradients and (not has_I or not has_J):
        # 缺少梯度数据,先计算梯度
        print(f"NAC所需梯度数据不在缓存中,先计算梯度...")
        states_to_compute = []
        if not has_I:
            states_to_compute.append(state_I)
        if not has_J:
            states_to_compute.append(state_J)
        
        precompute_gradients(tddft_obj, states_to_compute, **kwargs)
    
    # 计算NAC
    nac_obj = tduks_sf_nac.NonAdiabaticCouplings(tddft_obj)
    nac_obj.state_I = state_I
    nac_obj.state_J = state_J
    
    for key, val in kwargs.items():
        if hasattr(nac_obj, key):
            setattr(nac_obj, key, val)
    
    return nac_obj.kernel(use_cache=True)