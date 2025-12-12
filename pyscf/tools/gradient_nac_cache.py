"""
缓存管理系统用于梯度和NAC联合计算

此模块提供了一个全局缓存系统,用于存储和共享梯度与NAC计算的中间结果,
从而消除冗余计算,显著提升FSSH和MECP等需要同时计算多个态梯度和NAC的任务效率。

主要功能:
1. 缓存XC核 (fxc_sf, kxc_sf)
2. 缓存每个态的密度矩阵
3. 缓存响应项 (f1vo, f1oo, vxc1, k1ao等)
4. 自动管理缓存生命周期和几何构型变化

Example:
    >>> from gradient_nac_cache import get_cache_manager
    >>> cache_mgr = get_cache_manager()
    >>> # 梯度计算时存储中间量
    >>> cache_mgr.store_gradient_data(state_id, data)
    >>> # NAC计算时读取
    >>> data = cache_mgr.get_gradient_data(state_id)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class GradientNACCacheManager:
    """
    梯度和NAC计算的缓存管理器
    
    使用复合键缓存:(mf_id, geometry_bytes, state_id) 
    可以自动处理几何构型变化和不同态的数据分离
    """
    
    def __init__(self):
        # XC核缓存 - 只依赖于几何构型,不依赖于态
        self._xc_kernel_cache: Dict[Tuple, Tuple] = {}
        
        # 梯度数据缓存 - 每个态一份
        self._gradient_cache: Dict[Tuple, Dict[str, Any]] = {}
        
        # 统计信息
        self._stats = {
            'xc_kernel_hits': 0,
            'xc_kernel_misses': 0,
            'gradient_hits': 0,
            'gradient_misses': 0
        }
    
    def _make_geometry_key(self, mol) -> bytes:
        """创建几何构型的唯一标识"""
        return mol.atom_coords().tobytes()
    
    def _make_xc_cache_key(self, mf, mol) -> Tuple:
        """创建XC核缓存键: (mf对象ID, 几何构型)"""
        return (id(mf), self._make_geometry_key(mol))
    
    def _make_state_cache_key(self, mf, mol, state_id: int) -> Tuple:
        """创建态相关缓存键: (mf对象ID, 几何构型, 态ID)"""
        return (id(mf), self._make_geometry_key(mol), state_id)
    
    # ========== XC核缓存管理 ==========
    
    def get_xc_kernel(self, mf, mol) -> Optional[Tuple]:
        """
        获取缓存的XC核
        
        Returns:
            (fxc_sf, kxc_sf) 如果缓存命中, 否则返回None
        """
        key = self._make_xc_cache_key(mf, mol)
        if key in self._xc_kernel_cache:
            self._stats['xc_kernel_hits'] += 1
            logger.debug("XC kernel cache hit")
            return self._xc_kernel_cache[key]
        else:
            self._stats['xc_kernel_misses'] += 1
            logger.debug("XC kernel cache miss")
            return None
    
    def store_xc_kernel(self, mf, mol, fxc_sf, kxc_sf):
        """存储XC核到缓存"""
        key = self._make_xc_cache_key(mf, mol)
        self._xc_kernel_cache[key] = (fxc_sf, kxc_sf)
        logger.debug(f"Stored XC kernel in cache (total: {len(self._xc_kernel_cache)})")
    
    # ========== 梯度数据缓存管理 ==========
    
    def get_gradient_data(self, mf, mol, state_id: int) -> Optional[Dict]:
        """
        获取指定态的梯度计算中间数据
        
        Args:
            mf: Mean-field对象
            mol: 分子对象
            state_id: 态索引(1-based)
        
        Returns:
            包含中间量的字典,如果缓存未命中则返回None
        """
        key = self._make_state_cache_key(mf, mol, state_id)
        if key in self._gradient_cache:
            self._stats['gradient_hits'] += 1
            logger.debug(f"Gradient data cache hit for state {state_id}")
            return self._gradient_cache[key]
        else:
            self._stats['gradient_misses'] += 1
            logger.debug(f"Gradient data cache miss for state {state_id}")
            return None
    
    def store_gradient_data(self, mf, mol, state_id: int,
                           dmvo: Tuple,
                           dmoo: Optional[Tuple] = None,
                           f1vo: Optional[np.ndarray] = None,
                           f1oo: Optional[np.ndarray] = None,
                           vxc1: Optional[np.ndarray] = None,
                           veff_terms: Optional[Dict] = None):
        """
        存储梯度计算的中间数据
        
        注意: k1ao不被缓存,因为NAC中的k1ao依赖于两个态的交叉项,
              与单态梯度中的k1ao不同
        
        Args:
            mf: Mean-field对象
            mol: 分子对象
            state_id: 态索引(1-based)
            dmvo: 密度矩阵 ((dmxpy_ab, dmxpy_ba), (dmxmy_ab, dmxmy_ba))
            dmoo: 占据-占据密度矩阵 (dmzoo_a, dmzoo_b)
            f1vo: 虚-占据响应(只依赖单态,可缓存)
            f1oo: 占据-占据响应(只依赖单态,可缓存)
            vxc1: 泛函梯度(只依赖单态,可缓存)
            veff_terms: 其他有效势项
        """
        key = self._make_state_cache_key(mf, mol, state_id)
        
        cache_data = {
            'dmvo': dmvo,
            'dmoo': dmoo,
            'f1vo': f1vo,
            'f1oo': f1oo,
            'vxc1': vxc1,
            'veff_terms': veff_terms
            # 注意: k1ao被故意排除,因为它在NAC中的定义不同
        }
        
        self._gradient_cache[key] = cache_data
        logger.debug(f"Stored gradient data for state {state_id} "
                    f"(total states cached: {len(self._gradient_cache)})")
    
    def has_gradient_data(self, mf, mol, state_id: int) -> bool:
        """检查是否有指定态的缓存数据"""
        key = self._make_state_cache_key(mf, mol, state_id)
        return key in self._gradient_cache
    
    # ========== 缓存管理 ==========
    
    def clear(self):
        """清空所有缓存"""
        self._xc_kernel_cache.clear()
        self._gradient_cache.clear()
        logger.info("Cleared all caches")
    
    def clear_geometry(self, mf, mol):
        """清除特定几何构型的所有缓存"""
        geom_bytes = self._make_geometry_key(mol)
        
        # 清除XC核缓存
        xc_key = (id(mf), geom_bytes)
        if xc_key in self._xc_kernel_cache:
            del self._xc_kernel_cache[xc_key]
        
        # 清除所有相关的梯度缓存
        keys_to_remove = [k for k in self._gradient_cache.keys() 
                         if k[0] == id(mf) and k[1] == geom_bytes]
        for k in keys_to_remove:
            del self._gradient_cache[k]
        
        logger.debug(f"Cleared {len(keys_to_remove)} gradient cache entries")
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_xc = self._stats['xc_kernel_hits'] + self._stats['xc_kernel_misses']
        total_grad = self._stats['gradient_hits'] + self._stats['gradient_misses']
        
        stats = self._stats.copy()
        if total_xc > 0:
            stats['xc_kernel_hit_rate'] = self._stats['xc_kernel_hits'] / total_xc
        if total_grad > 0:
            stats['gradient_hit_rate'] = self._stats['gradient_hits'] / total_grad
        
        stats['xc_cache_size'] = len(self._xc_kernel_cache)
        stats['gradient_cache_size'] = len(self._gradient_cache)
        
        return stats
    
    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("Gradient-NAC Cache Statistics")
        print("="*60)
        print(f"XC Kernel Cache:")
        print(f"  Hits: {stats['xc_kernel_hits']}, Misses: {stats['xc_kernel_misses']}")
        if 'xc_kernel_hit_rate' in stats:
            print(f"  Hit Rate: {stats['xc_kernel_hit_rate']*100:.1f}%")
        print(f"  Cache Size: {stats['xc_cache_size']}")
        print(f"\nGradient Data Cache:")
        print(f"  Hits: {stats['gradient_hits']}, Misses: {stats['gradient_misses']}")
        if 'gradient_hit_rate' in stats:
            print(f"  Hit Rate: {stats['gradient_hit_rate']*100:.1f}%")
        print(f"  Cache Size: {stats['gradient_cache_size']}")
        print("="*60 + "\n")


# 全局缓存管理器单例
_CACHE_MANAGER: Optional[GradientNACCacheManager] = None


def get_cache_manager() -> GradientNACCacheManager:
    """获取全局缓存管理器单例"""
    global _CACHE_MANAGER
    if _CACHE_MANAGER is None:
        _CACHE_MANAGER = GradientNACCacheManager()
    return _CACHE_MANAGER


def clear_all_caches():
    """清空所有缓存(便捷函数)"""
    get_cache_manager().clear()


def print_cache_stats():
    """打印缓存统计(便捷函数)"""
    get_cache_manager().print_stats()