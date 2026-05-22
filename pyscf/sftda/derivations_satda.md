# SASF TDA 公式推导 — 2026-05-22

**目标：** 将 SASFTDA_Rev(2).pdf 中的理论与 `satda.py` 代码逐行对应，覆盖 $\Delta S = -1$ (spin-flip, $S_f = S_i - 1$) 和 $\Delta S = 0$ (spin-conserving, $S_f = S_i$) 两种情形的完整推导。

**代码对应：** `pyscf-forge/pyscf/sftda/satda.py` — `SATDA` 类及其 `gen_rohf_response_sf/sc`、`gen_vind_sf/sc` 函数。

---

## 1. ROKS 基态与轨道划分

### 1.1 自旋量子数

分子总自旋 $S$ 由 $\alpha$ 和 $\beta$ 电子数之差确定：

$$
S = \frac{N_{\alpha} - N_{\beta}}{2}
$$

**代码对应：** `satda.py:161`, `satda.py:296`, `satda.py:391`, `satda.py:510`
```python
s = (mol.nelec[0] - mol.nelec[1]) * 0.5
```

### 1.2 轨道空间划分 (ROKS 基态)

基于 ROKS 占据数 $n_p^{\text{ROKS}} \in \{2, 1, 0\}$ 将分子轨道分为三类：

| 轨道类型 | 占据数 | 指标 | 数目 | 代码变量 |
|---|---|---|---|---|
| 闭壳 (core) | $n_p = 2$ | $c,d \in \text{cs}$ | $N_{cs}$ | `csidx`, `orbcs` |
| 开壳 (open) | $n_p = 1$ | $u,v,w \in \text{os}$ | $N_{os}$ | `osidx`, `orbos` |
| 虚轨道 (virtual) | $n_p = 0$ | $a,b \in \text{vs}$ | $N_{vs}$ | `vsidx`, `orbvs` |

基态为 $S_z = S$ 的高自旋态，所有开壳轨道被 $\alpha$ 电子占据：

$$
|\text{ROKS}\rangle = |\underbrace{c_1\bar{c}_1 \dots c_{N_{cs}}\bar{c}_{N_{cs}}}_{\text{闭壳}} \underbrace{u_1 \dots u_{N_{os}}}_{\text{开壳(全 $\alpha$)}}\rangle
$$

其中 $N_{os} = 2S$.

**代码对应：** `satda.py:394-402`, `satda.py:513-521`
```python
csidx = np.where(mo_occ == 2)[0]   # 闭壳
osidx = np.where(mo_occ == 1)[0]   # 开壳
vsidx = np.where(mo_occ == 0)[0]   # 虚轨道
orbcs = mo_coeff[:, csidx]
orbos = mo_coeff[:, osidx]
orbvs = mo_coeff[:, vsidx]
```

### 1.3 ROKS Fock 矩阵

ROKS 的有效单粒子算符为（PDF §1）：

$$
\mathbf{F}^{\text{ROKS}} = \mathbf{h} + \mathbf{J}[\mathbf{D}_{\text{闭壳}} + \mathbf{D}_{\text{开壳}}] + \mathbf{V}_{xc}[\mathbf{D}_{\text{总}}]
$$

其中 $\mathbf{F}^{\alpha}$ 和 $\mathbf{F}^{\beta}$ 由 ROKS 耦合系数决定。代码中用 `mf.get_fock()` 直接获取:

**代码对应：** `satda.py:410-412`, `satda.py:530-531`
```python
fock = mf.get_fock()
focka = fock.focka   # F_α: α 自旋 Fock 矩阵
fockb = fock.fockb   # F_β: β 自旋 Fock 矩阵
```

---

## 2. $\Delta S = -1$: 自旋翻转 (Spin-Flip, $S_f = S_i - 1$)

### 2.1 激发算符与激发空间 (PDF §2.1)

对于 $\Delta S = -1$，自旋匹配的激发算符为（PDF §2.1, Eq. 1-2）：

$$
\mathbb{G}_{c \to u}^{\dagger} |\text{ROKS}\rangle = \frac{1}{\sqrt{2S}} \left(\hat{a}_{u\alpha}^{\dagger} \hat{a}_{c\beta} + \sum_{v=1}^{2S} \hat{a}_{u\beta}^{\dagger} \hat{a}_{v\alpha} \hat{a}_{v\beta}^{\dagger} \hat{a}_{c\beta}\right) |\text{ROKS}\rangle
$$

$$
\mathbb{G}_{u \to a}^{\dagger} |\text{ROKS}\rangle = \frac{1}{\sqrt{2S}} \left(\hat{a}_{a\alpha}^{\dagger} \hat{a}_{u\beta} + \sum_{v=1}^{2S} \hat{a}_{a\beta}^{\dagger} \hat{a}_{v\alpha} \hat{a}_{v\beta}^{\dagger} \hat{a}_{u\beta}\right) |\text{ROKS}\rangle
$$

这些算符产生 $S_f = S-1$ 的自旋纯态。

在 Tamm-Dancoff 近似下，激发空间由四种子块组成：

$$
\mathbf{X} = \begin{pmatrix}
\mathbf{X}_{co} & \mathbf{X}_{cv} \\
\mathbf{X}_{oo} & \mathbf{X}_{ov}
\end{pmatrix}
$$

其中:
- $\mathbf{X}_{co}$: $c \to u$ (闭壳 $\to$ 开壳) — shape $(N_{cs}, N_{os})$
- $\mathbf{X}_{cv}$: $c \to a$ (闭壳 $\to$ 虚轨道) — shape $(N_{cs}, N_{vs})$
- $\mathbf{X}_{oo}$: $u \to v$ (开壳 $\to$ 开壳) — shape $(N_{os}, N_{os})$
- $\mathbf{X}_{ov}$: $u \to a$ (开壳 $\to$ 虚轨道) — shape $(N_{os}, N_{vs})$

**代码对应：** `satda.py:522-523,564-569`
```python
nocc = ncs + nos
nvir = nos + nvs
zs = np.asarray(zs).reshape(-1, nocc, nvir)
zs_co = zs[:, :ncs, :nos]      # c→u
zs_cv = zs[:, :ncs, nos:]       # c→a
zs_oo = zs[:, ncs:, :nos]       # u→v
zs_ov = zs[:, ncs:, nos:]       # u→a
```

### 2.2 TDA 本征方程 (PDF §2.2)

自旋翻转 TDA 的本征值问题为：

$$
\mathbf{A} \cdot \mathbf{X} = \omega \mathbf{X}
$$

其中 $\mathbf{A}$ 矩阵具有块结构：

$$
\mathbf{A} = \begin{pmatrix}
\mathbf{A}_{co,co} & \mathbf{A}_{co,cv} & \mathbf{A}_{co,oo} & \mathbf{A}_{co,ov} \\
\mathbf{A}_{cv,co} & \mathbf{A}_{cv,cv} & \mathbf{A}_{cv,oo} & \mathbf{A}_{cv,ov} \\
\mathbf{A}_{oo,co} & \mathbf{A}_{oo,cv} & \mathbf{A}_{oo,oo} & \mathbf{A}_{oo,ov} \\
\mathbf{A}_{ov,co} & \mathbf{A}_{ov,cv} & \mathbf{A}_{ov,oo} & \mathbf{A}_{ov,ov}
\end{pmatrix}
$$

### 2.3 $\mathbf{A}$ 矩阵的 Fock + 响应函数分解 (PDF §2.3)

$\mathbf{A}$ 矩阵可以分解为纯 Fock 部分和响应函数部分：

$$
\mathbf{A} = \mathbf{A}^{\text{Fock}} + \mathbf{A}^{\text{Resp}}
$$

其中 $\mathbf{A}^{\text{Resp}}$ 进一步分解（对于 DFT）：

$$
\mathbf{A}^{\text{Resp}} = \mathbf{A}^{\text{Coulomb}} + \mathbf{A}^{\text{HFex}} + \mathbf{A}^{\text{XC}}
$$

$\mathbf{A}^{\text{Fock}}$ 由 Fock 矩阵元之差给出；$\mathbf{A}^{\text{XC}}$ 由交换相关核的积分给出：

$$
A_{ia,jb}^{\text{XC}} = \int d\mathbf{r} d\mathbf{r}' \, \phi_i(\mathbf{r}) \phi_a(\mathbf{r}) \, f_{xc}(\mathbf{r},\mathbf{r}') \, \phi_j(\mathbf{r}') \phi_b(\mathbf{r}')
$$

代码通过 **sigma 向量** 方式计算 $\boldsymbol{\sigma} = \mathbf{A} \cdot \mathbf{X}$，避免显式构建 $\mathbf{A}$ 矩阵。

### 2.4 响应函数 `gen_rohf_response_sf` (PDF §2.3, §2.4)

**代码对应：** `satda.py:284-382`

#### 2.4.1 XC 核 `fxc_ref` 的构造

ROKS 的密度响应 XC 核为（PDF Eq. 11-12）：

$$
f_{xc}^{\text{ref}}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) - f_{xc}^{\alpha\beta}(r) - f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r) \Big]
$$

这是在 collinear 表示下 `fxc_d0[0, :, 0]` 和 `fxc_d0[:, 0, :]` 等的组合。

**代码对应：** `satda.py:305-307` (deltaS=-1 版本)、`satda.py:170-171` (deltaS=0 版本)
```python
fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])
```

对于 `gen_rohf_response_sf`（deltaS=-1），仅需 $f_{xc}^{\text{ref}}$，而对 `gen_rohf_response_sc`（deltaS=0）还需要额外的 UKS 核分量（见 §3.3）。

#### 2.4.2 密度矩阵构造

对于每个 sigma 向量计算，先将激发振幅转为 AO 基密度矩阵：

$$
D_{\mu\nu}^{co} = \sum_{c,u} X_{c u} \, C_{\nu c}^{*} C_{\mu u}
$$

$$
D_{\mu\nu}^{cv} = \sum_{c,a} X_{c a} \, C_{\nu c}^{*} C_{\mu a}
$$

$$
D_{\mu\nu}^{oo} = \sum_{u,v} X_{u v} \, C_{\nu u}^{*} C_{\mu v} \quad (= \text{开壳} \to \text{开壳})
$$

$$
D_{\mu\nu}^{ov} = \sum_{u,a} X_{u a} \, C_{\nu u}^{*} C_{\mu a}
$$

**代码对应：** `satda.py:570-573`
```python
dms_co = lib.einsum('xov,pv,qo->xpq', zs_co, orbos, orbcs.conj())  # X_{c,u}: col v=os, row o=cs
dms_cv = lib.einsum('xov,pv,qo->xpq', zs_cv, orbvs, orbcs.conj())
dms_oo = lib.einsum('xov,pv,qo->xpq', zs_oo, orbos, orbos.conj())
dms_ov = lib.einsum('xov,pv,qo->xpq', zs_ov, orbvs, orbos.conj())
```

理解 einsum 的指标映射：`xov` = 态指标×行指标×列指标, `pv` = AO×列指标(MO), `qo` = AO×行指标(MO) → `xpq` = 态指标×AO×AO.

对于 $D_{co}$: $p \leftrightarrow$ virtual index (`orbos`, $u$), $q \leftrightarrow$ occupied index (`orbcs` conjugate, $c$):
$$
D_{\mu\nu}^{co} = \sum_{cu} X_{cu} \, C_{\mu u} \, C_{\nu c}^{*}
$$

#### 2.4.3 XC 响应计算

$\mathbf{A}^{\text{XC}}$ 的作用通过 numerical integration 计算:

对于完整 4-block 密度 $D^0 = [D^{co}, D^{cv}, D^{oo}, D^{ov}]$:
$$
V_{\mu\nu}^{\text{ref0}} = \int d\mathbf{r} \, \phi_{\mu}(\mathbf{r}) \phi_{\nu}(\mathbf{r}) \, f_{xc}^{\text{ref}}(\mathbf{r}) \, \sum_{pq} D_{pq} \phi_p(\mathbf{r}) \phi_q(\mathbf{r})
$$

对于简化 2-block 密度 $D^1 = [D^{co}, D^{ov}]$:
$$
V_{\mu\nu}^{\text{ref1}} = \int d\mathbf{r} \, \phi_{\mu}(\mathbf{r}) \phi_{\nu}(\mathbf{r}) \, f_{xc}^{\text{ref}}(\mathbf{r}) \, \sum_{pq} D_{pq} \phi_p(\mathbf{r}) \phi_q(\mathbf{r})
$$

**代码对应：** `satda.py:330-331,334-340`
```python
dms0 = np.concatenate((dms_co, dms_cv, dms_oo, dms_ov), axis=0)
dms1 = np.concatenate((dms_co, dms_ov), axis=0)
vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, ...)
vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, ...)  # LDA
# 或 nr_rks_fxc1_gga(...) / nr_rks_fxc1_mgga(...) for GGA/MGGA
```

#### 2.4.4 HF 交换响应

对于 hybrid 泛函，需要加上 HF 交换项减去 XC 核中的交换:

$$
V_{\mu\nu}^{\text{HF}} = -c_{hyb} \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) D_{\lambda\sigma}
$$

即 `mf.get_k(mol, dms0, hermi) * hyb` (带有 range-separated 修正).

**代码对应：** `satda.py:346-352`
```python
if hybrid:
    vk = mf.get_k(mol, dms0, hermi) * hyb
    vj = mf.get_j(mol, dms1, hermi) * hyb
    if omega != 0:
        vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
        vj += mf.get_j(mol, dms1, hermi, omega=omega) * (alpha - hyb)
    vref0 -= vk
    vref1 -= vj
```

注意：此处对 $D^0$（4-block）用 exchange $K$，对 $D^1$（2-block）用 Coulomb $J$。这是因为在 UKS 的 spin-flip TDA 中，自旋翻转激发只有 exchange 型耦合（α→β 方向），无 Coulomb 耦合。而 $D^1$ 的 $J$ 来自基态 Fock 矩阵的自旋适配修正。

#### 2.4.5 自旋适配系数组装 (PDF Eq. 20-24)

将 AO 基的 $V^{\text{ref0,1}}$ 按自旋适配系数组合到各激发子块。对于 $\Delta S = -1$，系数矩阵为：

$$
\begin{pmatrix}
\mathbf{V}_{co} \\
\mathbf{V}_{cv} \\
\mathbf{V}_{oo} \\
\mathbf{V}_{ov}
\end{pmatrix}
=
\mathbf{C}_{\Delta S=-1}
\begin{pmatrix}
\mathbf{V}_{co}^{\text{ref0}} \\
\mathbf{V}_{cv}^{\text{ref0}} \\
\mathbf{V}_{oo}^{\text{ref0}} \\
\mathbf{V}_{ov}^{\text{ref0}} \\
\mathbf{V}_{co}^{\text{ref1}} \\
\mathbf{V}_{ov}^{\text{ref1}}
\end{pmatrix}
$$

其中系数矩阵 $\mathbf{C}_{\Delta S=-1}$ 的元素（PDF Eq. 20-24）为：

| | $V_{co}^{\text{ref0}}$ | $V_{cv}^{\text{ref0}}$ | $V_{oo}^{\text{ref0}}$ | $V_{ov}^{\text{ref0}}$ | $V_{co}^{\text{ref1}}$ | $V_{ov}^{\text{ref1}}$ |
|---|---|---|---|---|---|---|
| $V_{co}$ | $1$ | $\sqrt{\frac{2S+1}{2S}}$ | $\sqrt{\frac{2S}{2S-1}}$ | $\frac{2S}{2S-1}$ | $\frac{1}{2S-1}$ | $-\frac{1}{2S-1}$ |
| $V_{cv}$ | $\sqrt{\frac{2S+1}{2S}}$ | $1$ | $\sqrt{\frac{2S+1}{2S-1}}$ | $\sqrt{\frac{2S+1}{2S}}$ | $0$ | $0$ |
| $V_{oo}$ | $\sqrt{\frac{2S}{2S-1}}$ | $\sqrt{\frac{2S+1}{2S-1}}$ | $1$ | $\sqrt{\frac{2S}{2S-1}}$ | $0$ | $0$ |
| $V_{ov}$ | $\frac{2S}{2S-1}$ | $\sqrt{\frac{2S+1}{2S}}$ | $\sqrt{\frac{2S}{2S-1}}$ | $1$ | $-\frac{1}{2S-1}$ | $\frac{1}{2S-1}$ |

**代码对应：** `satda.py:361-368`

逐个验证：

**$V_{co}$ (line 361-362):**
```python
v1ao_co += vref0_co + vref1_co/(2*s - 1) + np.sqrt((2*s + 1)/2/s)*vref0_cv
v1ao_co += np.sqrt(2*s/(2*s - 1))*vref0_oo + 2*s/(2*s-1)*vref0_ov - vref1_ov/(2*s-1)
```
→ 系数: $(1, \sqrt{\frac{2S+1}{2S}}, \sqrt{\frac{2S}{2S-1}}, \frac{2S}{2S-1}, \frac{1}{2S-1}, -\frac{1}{2S-1})$ ✓

**$V_{cv}$ (line 363-364):**
```python
v1ao_cv += vref0_co*np.sqrt((2*s+1)/2/s) + vref0_cv + np.sqrt((2*s+1)/(2*s-1))*vref0_oo
v1ao_cv += np.sqrt((2*s+1)/2/s)*vref0_ov
```
→ 系数: $(\sqrt{\frac{2S+1}{2S}}, 1, \sqrt{\frac{2S+1}{2S-1}}, \sqrt{\frac{2S+1}{2S}}, 0, 0)$ ✓

**$V_{oo}$ (line 365-366):**
```python
v1ao_oo += np.sqrt(2*s/(2*s-1))*vref0_co + np.sqrt((2*s+1)/(2*s-1))*vref0_cv
v1ao_oo += vref0_oo + np.sqrt(2*s/(2*s-1))*vref0_ov
```
→ 系数: $(\sqrt{\frac{2S}{2S-1}}, \sqrt{\frac{2S+1}{2S-1}}, 1, \sqrt{\frac{2S}{2S-1}}, 0, 0)$ ✓

**$V_{ov}$ (line 367-368):**
```python
v1ao_ov += 2*s/(2*s-1)*vref0_co - vref1_co/(2*s-1) + np.sqrt((2*s+1)/2/s)*vref0_cv
v1ao_ov += np.sqrt(2*s/(2*s-1))*vref0_oo + vref0_ov + vref1_ov/(2*s-1)
```
→ 系数: $(\frac{2S}{2S-1}, \sqrt{\frac{2S+1}{2S}}, \sqrt{\frac{2S}{2S-1}}, 1, -\frac{1}{2S-1}, \frac{1}{2S-1})$ ✓

#### 2.4.6 Fockz: 开壳密度响应 (PDF §2.5)

`fockz = 0.5 * delta` 是开壳 Fock 矩阵的响应修正:

$$
\mathbf{F}^z = \frac{1}{2} \left[ \mathbf{K}^{\text{ref}}[\mathbf{D}_{oo}] - c_{hyb} \mathbf{K}^{\text{HF}}[\mathbf{D}_{oo}] \right]
$$

其中 $\mathbf{D}_{oo} = \sum_{u \in os} \phi_u \phi_u^{\dagger}$ 为开壳轨道密度。

**代码对应：** `satda.py:372-382`
```python
orbos = mo_coeff[:, np.where(mo_occ == 1)[0]]
dmoo = orbos @ orbos.T
if xctype != 'HF':
    delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, ...)
else:
    delta = np.zeros_like(dmoo)
if hybrid:
    delta -= mf.get_k(mol, dmoo, 1) * hyb
    if omega != 0:
        delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
return vind, 0.5 * delta
```

### 2.5 `gen_vind_sf`: 完整 Sigma 向量 (PDF §2.6)

**代码对应：** `satda.py:503-617`

#### 2.5.1 Fock 矩阵的分块定义

在得到 `vresp` (XC+HF 响应) 和 `fockz` 后，定义有效 Fock 矩阵:

$$
\mathbf{F}^0 = \mathbf{F}^{\alpha} - \mathbf{F}^z
$$

**代码对应：** `satda.py:532`
```python
fock0 = focka - fockz
```

然后在各轨道子空间投影:

**代码对应：** `satda.py:534-552`

| 名称 | 定义 | MO 基投影 |
|---|---|---|
| $F_{co}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{os}$ |
| $F_{co}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{cs}^{\dagger} (F^0 + F^z) \mathbf{C}_{cs}$ |
| $F_{co}^{2}$ | $F^z$ | $\mathbf{C}_{cs}^{\dagger} F^z \mathbf{C}_{cs}$ |
| $F_{cv}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ |
| $F_{cooo}^{0}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{cs}$ |
| $F_{cooo}^{1}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{cs}$ |
| $F_{cv}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{vs}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ |
| $F_{cv}^{2}$ | $F^z$ | $\mathbf{C}_{vs}^{\dagger} F^z \mathbf{C}_{vs}$ |
| $F_{cv}^{3}$ | $F^z$ | $\mathbf{C}_{cs}^{\dagger} F^z \mathbf{C}_{cs}$ |
| $F_{cvoo}$ | $F^z$ | $\mathbf{C}_{vs}^{\dagger} F^z \mathbf{C}_{cs}$ |
| $F_{c\tilde{v}}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{cs}$ (同 $F_{cooo}^{0}$) |
| $F_{ooo}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{os}$ (同 $F_{co}^{0}$) |
| $F_{ooo}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{os}$ |
| $F_{ooov}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ (同 $F_{cv}$) |
| $F_{ooov}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{vs}$ |
| $F_{ov}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{vs}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ (同 $F_{cv}^{0}$) |
| $F_{ov}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{os}$ (同 $F_{ooo}^{1}$) |
| $F_{ov}^{2}$ | $F^z$ | $\mathbf{C}_{vs}^{\dagger} F^z \mathbf{C}_{vs}$ (同 $F_{cv}^{2}$) |

**代码对应：** `satda.py:534-552`
```python
fock_coco0 = orbos.T @ (fock0 - fockz) @ orbos
fock_coco1 = orbcs.T @ (fock0 + fockz) @ orbcs
fock_coco2 = orbcs.T @ fockz @ orbcs
fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
fock_cooo0 = orbos.T @ (fock0 + fockz) @ orbcs
fock_cooo1 = orbos.T @ (fock0 - fockz) @ orbcs
fock_cvcv0 = orbvs.T @ (fock0 - fockz) @ orbvs
fock_cvcv1 = fock_coco1
fock_cvcv2 = orbvs.T @ fockz @ orbvs
fock_cvcv3 = fock_coco2
fock_cvoo = orbvs.T @ fockz @ orbcs
fock_cvov = fock_cooo0
fock_oooo0 = fock_coco0
fock_oooo1 = orbos.T @ (fock0 + fockz) @ orbos
fock_ooov0 = fock_cocv
fock_ooov1 = orbos.T @ (fock0 + fockz) @ orbvs
fock_ovov0 = fock_cvcv0
fock_ovov1 = fock_oooo1
fock_ovov2 = fock_cvcv2
```

#### 2.5.2 对角元 (preconditioner diagonal, PDF §2.6)

对于 TDA 迭代对角化的预处理器，需要 $\mathbf{A}$ 的对角近似:

**co 块 (line 555-556):**

$$
h_{\text{diag}}^{co} = \varepsilon_{u}^{F^0-F^z} \otimes \mathbf{1}_{c} - \mathbf{1}_{u} \otimes \varepsilon_{c}^{F^0+F^z} - \frac{2}{2S-1} \cdot \mathbf{1}_{u} \otimes \varepsilon_{c}^{F^z}
$$

即:
$$
h_{\text{diag}}^{co}[i, u] = \varepsilon_u^{(0)} - \varepsilon_c^{(1)} - \frac{2}{2S-1} \varepsilon_c^{(2)}
$$

其中 $\varepsilon^{(0)} = \text{diag}(F_{co}^{0})$, $\varepsilon^{(1)} = \text{diag}(F_{co}^{1})$, $\varepsilon^{(2)} = \text{diag}(F_{co}^{2})$.

**cv 块 (line 557-558):**

$$
h_{\text{diag}}^{cv} = \varepsilon_{a}^{F^0-F^z} \otimes \mathbf{1}_{c} - \mathbf{1}_{a} \otimes \varepsilon_{c}^{F^0+F^z} - \frac{1}{S} \varepsilon_{a}^{F^z} \otimes \mathbf{1}_{c} + \frac{1}{S} \cdot \mathbf{1}_{a} \otimes \varepsilon_{c}^{F^z}
$$

**oo 块 (line 559):**

$$
h_{\text{diag}}^{oo} = \varepsilon_{v}^{F^0-F^z} \otimes \mathbf{1}_{u} - \mathbf{1}_{v} \otimes \varepsilon_{u}^{F^0+F^z}
$$

**ov 块 (line 560-561):**

$$
h_{\text{diag}}^{ov} = \varepsilon_{a}^{F^0-F^z} \otimes \mathbf{1}_{u} - \mathbf{1}_{a} \otimes \varepsilon_{u}^{F^0+F^z} - \frac{2}{2S-1} \varepsilon_{a}^{F^z} \otimes \mathbf{1}_{u}
$$

**代码对应：** `satda.py:555-561`
```python
hdiag_co = fock_coco0.diagonal()[None, :] - fock_coco1.diagonal()[:, None]
hdiag_co -= fock_coco2.diagonal()[:, None] * 2 / (2 * s - 1)
hdiag_cv = fock_cvcv0.diagonal()[None, :] - fock_cvcv1.diagonal()[:, None]
hdiag_cv -= fock_cvcv2.diagonal()[None, :] / s + fock_cvcv3.diagonal()[:, None] / s
hdiag_oo = fock_oooo0.diagonal()[None, :] - fock_oooo1.diagonal()[:, None]
hdiag_ov = fock_ovov0.diagonal()[None, :] - fock_ovov1.diagonal()[:, None]
hdiag_ov -= fock_ovov2.diagonal()[None, :] * 2 / (2 * s - 1)
```

最终 `hdiag = np.block([[hdiag_co, hdiag_cv], [hdiag_oo, hdiag_ov]]).ravel()` (line 562).

#### 2.5.3 Vind 函数: Fock 项叠加 (PDF §2.6)

在计算 AO 基的 XC 响应 `v1ao_co/cv/oo/ov` 后，转为 MO 基并叠加 Fock 矩阵项。

**AO → MO 变换:**

$$
V_{cu}^{\text{MO}} = \sum_{\mu\nu} C_{\mu c}^{*} \, V_{\mu\nu}^{\text{AO}} \, C_{\nu u}
$$

**代码对应：** `satda.py:575-578`
```python
v1mo_co = lib.einsum('xpq,qo,pv->xov', v1ao_co, orbcs, orbos.conj())
v1mo_cv = lib.einsum('xpq,qo,pv->xov', v1ao_cv, orbcs, orbvs.conj())
v1mo_oo = lib.einsum('xpq,qo,pv->xov', v1ao_oo, orbos, orbos.conj())
v1mo_ov = lib.einsum('xpq,qo,pv->xov', v1ao_ov, orbos, orbvs.conj())
```

##### co 块 Fock 项 (PDF Eq. 25)

从自旋适配单粒子能差的推广:

$$
A_{cu, dv}^{\text{Fock}} = \delta_{cd} \, F_{vu}^{F^0-F^z} - \delta_{uv} \, F_{cd}^{F^0+F^z} - \frac{2}{2S-1} \delta_{uv} \, F_{cd}^{F^z}
$$

加上非对角耦合项（激发算符之间的单粒子耦合）:

$$
A_{cu, da}^{\text{Fock,coupling}} = \sqrt{\frac{2S+1}{2S}} \, \delta_{cd} \, F_{ua}^{F^0-F^z} \times X_{da}
$$

$$
A_{cu, vw}^{\text{Fock,coupling}} = -\sqrt{\frac{2S}{2S-1}} \, \delta_{uv} \, F_{wc}^{F^0+F^z} \times X_{vw} \;+\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{vw} \, F_{uc}^{F^0-F^z} \times X_{vc}
$$

**代码对应 (co 块):** `satda.py:580-585`
```python
v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco0, zs_co)          # δ_cd F^0-F^z_vu X_dv
v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco1, zs_co)          # -δ_uv F^0+F^z_cd X_dv
v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco2, zs_co) * 2/(2s-1) # -2/(2S-1) δ_uv F^z_cd X_dv
v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv, zs_cv) * sqrt((2s+1)/2/s)  # √(...)·δ_cd F^0-F^z_va X_da
v1mo_co -= lib.einsum('uv,wi,xwv->xiu', np.eye(nos), fock_cooo0, zs_oo) * sqrt(2s/(2s-1)) # -√(...)·δ_uv F^0+F^z_wc X_vw
v1mo_co += lib.einsum('vw,ui,xwv->xiu', np.eye(nos), fock_cooo1, zs_oo) / sqrt(2*s*(2*s-1))  # 1/√(...)·δ_vw F^0-F^z_uc X_vc
```

einsum 指标解析:
- `lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco0, zs_co)`:
  - `ij` = δ_cd (闭壳指标)
  - `uv` = F^0-F^z 矩阵在开壳空间中, indices (v,u) → (u,v) 转置
  - `xjv` = zs_co 在 (d→v) 指标
  - `->xiu` = 输出 (c→u)
  - 等价于: $V_{cu} = \sum_{d,v} \delta_{cd} \, F_{vu}^{F^0-F^z} \, X_{dv}$

##### cv 块 Fock 项 (PDF Eq. 26)

$$
A_{ca, db}^{\text{Fock}} = \delta_{cd} \, F_{ab}^{F^0-F^z} - \delta_{ab} \, F_{cd}^{F^0+F^z} - \frac{1}{S} \delta_{cd} \, F_{ab}^{F^z} - \frac{1}{S} \delta_{ab} \, F_{cd}^{F^z}
$$

加上耦合项:

$$
A_{ca, du}^{\text{coupling}} = \sqrt{\frac{2S+1}{2S}} \, \delta_{cd} \, (F_{ua}^{F^0-F^z})^{\dagger} \times X_{du}
$$

$$
A_{ca, vw}^{\text{coupling}} = -\frac{1}{S} \sqrt{\frac{2S+1}{2S-1}} \, \delta_{vw} \, F_{ac}^{F^z} \times X_{vw}
$$

$$
A_{ca, vb}^{\text{coupling}} = -\sqrt{\frac{2S+1}{2S}} \, \delta_{ab} \, F_{cv}^{F^0+F^z} \times X_{vb}
$$

**代码对应 (cv 块):** `satda.py:587-593`
```python
v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv.T, zs_co) * sqrt((2s+1)/2/s)
v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv0, zs_cv)
v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv1, zs_cv)
v1mo_cv -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv2, zs_cv) / s
v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv3, zs_cv) / s
v1mo_cv -= lib.einsum('vw,ai,xwv->xia', np.eye(nos), fock_cvoo, zs_oo) / s * sqrt((2s+1)/(2s-1))
v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_cvov, zs_ov) * sqrt((2s+1)/2/s)
```

##### oo 块 Fock 项 (PDF Eq. 27)

$$
A_{uv, wt}^{\text{Fock}} = \delta_{uw} \, F_{vt}^{F^0-F^z} - \delta_{vt} \, F_{uw}^{F^0+F^z}
$$

耦合项:

$$
A_{uv, cd}^{\text{coupling}} = -\sqrt{\frac{2S}{2S-1}} \, \delta_{vt} \, (F_{cu}^{F^0+F^z})^{\dagger} \times X_{cd}
\;+\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{ut} \, (F_{cv}^{F^0-F^z})^{\dagger} \times X_{cv}
$$

$$
A_{uv, ca}^{\text{coupling}} = -\frac{1}{S} \sqrt{\frac{2S+1}{2S-1}} \, \delta_{ut} \, (F_{cv}^{F^z})^{\dagger} \times X_{ca}
$$

$$
A_{uv, wa}^{\text{coupling}} = \sqrt{\frac{2S}{2S-1}} \, \delta_{uv} \, F_{ta}^{F^0-F^z} \times X_{wt}
\;-\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{ut} \, F_{va}^{F^0+F^z} \times X_{tv}
$$

**代码对应 (oo 块):** `satda.py:595-601`
```python
v1mo_oo -= lib.einsum('vt,ju,xjv->xut', np.eye(nos), fock_cooo0.T, zs_co) * sqrt(2*s/(2*s-1))
v1mo_oo += lib.einsum('ut,jv,xjv->xut', np.eye(nos), fock_cooo1.T, zs_co) / sqrt(2*s*(2*s-1))
v1mo_oo -= lib.einsum('ut,jb,xjb->xut', np.eye(nos), fock_cvoo.T, zs_cv) / s * sqrt((2*s+1)/(2*s-1))
v1mo_oo += lib.einsum('wu,tv,xwv->xut', np.eye(nos), fock_oooo0, zs_oo)
v1mo_oo -= lib.einsum('tv,wu,xwv->xut', np.eye(nos), fock_oooo1, zs_oo)
v1mo_oo += lib.einsum('uv,tb,xvb->xut', np.eye(nos), fock_ooov0, zs_ov) * sqrt(2*s/(2*s-1))
v1mo_oo -= lib.einsum('tu,vb,xvb->xut', np.eye(nos), fock_ooov1, zs_ov) / sqrt(2*s*(2*s-1))
```

##### ov 块 Fock 项 (PDF Eq. 28)

$$
A_{ua, vb}^{\text{Fock}} = \delta_{uv} \, F_{ab}^{F^0-F^z} - \delta_{ab} \, F_{vu}^{F^0+F^z} - \frac{2}{2S-1} \delta_{uv} \, F_{ab}^{F^z}
$$

耦合项:

$$
A_{ua, cb}^{\text{coupling}} = -\sqrt{\frac{2S+1}{2S}} \, \delta_{ab} \, (F_{cu}^{F^0+F^z})^{\dagger} \times X_{cb}
$$

$$
A_{ua, wv}^{\text{coupling}} = \sqrt{\frac{2S}{2S-1}} \, \delta_{uw} \, (F_{va}^{F^0-F^z})^{\dagger} \times X_{wv}
\;-\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{vw} \, (F_{ua}^{F^0+F^z})^{\dagger} \times X_{vw}
$$

**代码对应 (ov 块):** `satda.py:603-608`
```python
v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_cvov.T, zs_cv) * sqrt((2s+1)/2/s)
v1mo_ov += lib.einsum('uw,av,xwv->xua', np.eye(nos), fock_ooov0.T, zs_oo) * sqrt(2*s/(2*s-1))
v1mo_ov -= lib.einsum('vw,au,xwv->xua', np.eye(nos), fock_ooov1.T, zs_oo) / sqrt(2*s*(2*s-1))
v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov0, zs_ov)
v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), fock_ovov1, zs_ov)
v1mo_ov -= lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov2, zs_ov) * 2/(2*s-1)
```

最后，将四个子块组装为总复矢量 `v1mo[:, :ncs, :nos] = v1mo_co` 等 (lines 610-614)。

---

## 3. $\Delta S = 0$: 自旋守恒 (Spin-Conserving, $S_f = S_i$)

### 3.1 激发算符 (PDF §3.1)

对于自旋守恒激发 ($S_f = S_i = S$)，需要三类自旋匹配算符除去零激发态（纯 $D_{oo}$）：

**G_co (闭壳→开壳):**

$$
\mathbb{G}_{c \to u}^{\dagger} |\text{ROKS}\rangle = \frac{1}{\sqrt{2S+2}} \left(\hat{a}_{u\alpha}^{\dagger} \hat{a}_{c\alpha} + \hat{a}_{u\beta}^{\dagger} \hat{a}_{c\beta}\right) |\text{ROKS}\rangle
$$

**G_cv (闭壳→虚轨道 + 开壳→虚轨道 的叠加，非 cv0 道):**

两个正交的正交道用于激发:
- "cv" 道: $\mathbb{G}_{c \to a}^{(1)}$ — 对称形式的叠加
- "cv0" 道: $\mathbb{G}_{c \to a}^{(2)}$ — 反对称形式的叠加

**G_ov (开壳→虚轨道, 与 G_co 配对的 partner):**

与前两者配对的组合算符。

此外还有一个特殊的单参数算符 $\mathbb{G}_{oo}$ (开壳→开壳)，只有一个自由度。

### 3.2 激发空间 (PDF §3.2)

激发空间由 5 个矢量组成 (包括 `cv0`):

$$
\mathbf{X} = \begin{pmatrix}
\mathbf{X}_{co} & \mathbf{X}_{cv} \\
\mathbf{X}_{oo} & \mathbf{X}_{ov} \\
& \mathbf{X}_{cv0}
\end{pmatrix}
$$

**代码对应：** `satda.py:403-406`
```python
idx1 = ncs * nos          # co: ncs × nos
idx2 = idx1 + ncs * nvs   # cv: ncs × nvs
idx3 = idx2 + 1           # oo: 1 element (scalar!)
idx4 = idx3 + nos * nvs   # ov: nos × nvs
# cv0: rest, ncs × nvs
```

**关键点：** $oo$ 块仅 1 个元素，因为自旋守恒时只有一个独立的 $D_{oo}$ 算符（对角线）可以激发而不改变自旋。这与 $\Delta S = -1$ 的 $oo$ 块（shape $(N_{os}, N_{os})$）完全不同的物理根源。

### 3.3 响应函数 `gen_rohf_response_sc` (PDF §3.3-3.5)

**代码对应：** `satda.py:149-282`

#### 3.3.1 多个 XC 核的构造

对于 $\Delta S = 0$，仅 $f_{xc}^{\text{ref}}$ 不够，还需要 5 个额外的核分量（PDF Eq. 30-35）:

从 ROKS 密度核:
$$
f_{xc}^{\text{ref}}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) - f_{xc}^{\alpha\beta}(r) - f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r) \Big]
$$

**代码对应：** `satda.py:170-171`
```python
fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])
```

从 UKS 密度核（转为 UKS 以获取非零自旋密度的核）:
$$
f_{xc}^{s}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\alpha\beta}(r) + f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r) \Big]
$$

$$
f_{xc}^{cv0,cv}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\alpha\beta}(r) - f_{xc}^{\beta\alpha}(r) - f_{xc}^{\beta\beta}(r) \Big]
$$

$$
f_{xc}^{cv,cv0}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) - f_{xc}^{\alpha\beta}(r) + f_{xc}^{\beta\alpha}(r) - f_{xc}^{\beta\beta}(r) \Big]
$$

$$
f_{xc}^{cv0,co}(r) = f_{xc}^{\alpha\beta}(r) + f_{xc}^{\beta\beta}(r)
$$

$$
f_{xc}^{cv0,ov}(r) = f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\beta\alpha}(r)
$$

$$
f_{xc}^{co,cv0}(r) = f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r)
$$

$$
f_{xc}^{ov,cv0}(r) = f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\alpha\beta}(r)
$$

**代码对应：** `satda.py:172-181`
```python
umf = mf.to_uks()
uni = umf._numint
_, _, fxc = uni.cache_xc_kernel(mol, mf.grids, mf.xc, umf.mo_coeff, umf.mo_occ, 1)
fxc_s = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1] + fxc[1, :, 0] + fxc[1, :, 1])
fxc_cv0cv = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1] - fxc[1, :, 0] - fxc[1, :, 1])
fxc_cvcv0 = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1] + fxc[1, :, 0] - fxc[1, :, 1])
fxc_cv0co = fxc[0, :, 1] + fxc[1, :, 1]
fxc_cv0ov = fxc[0, :, 0] + fxc[1, :, 0]
fxc_cocv0 = fxc[1, :, 0] + fxc[1, :, 1]
fxc_ovcv0 = fxc[0, :, 0] + fxc[0, :, 1]
```

#### 3.3.2 Vind 函数: 四项贡献 (PDF §3.4)

响应函数由四部分构成：Coulomb 部分、HF 交换部分、K^Ref 部分、K^CV0 部分。

##### Coulomb 部分 (PDF Eq. 36-38)

$\Delta S = 0$ 时存在 Coulomb 耦合（两个激发算符通过 Coulomb 算符连接产生 $cv0$ 道）:

$$
\mathbf{V}_{co}^{\text{Coul}} = \sqrt{2} \, \mathbf{J}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{ov}^{\text{Coul}} = -\sqrt{2} \, \mathbf{J}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{cv0}^{\text{Coul}} = \sqrt{2} \, \mathbf{J}[\mathbf{D}_{co}] - \sqrt{2} \, \mathbf{J}[\mathbf{D}_{ov}] + 2 \, \mathbf{J}[\mathbf{D}_{cv0}]
$$

**代码对应：** `satda.py:206-214`
```python
dms_j = np.concatenate((dms_co, dms_ov, dms_cv0), axis=0)
vcoul = mf.get_j(mol, dms_j, hermi)
vcoul_co = vcoul[:idx1]
vcoul_ov = vcoul[idx1:idx1+n_ov]
vcoul_cv0 = vcoul[idx1+n_ov:]
v1ao_co += np.sqrt(2) * vcoul_cv0
v1ao_ov -= np.sqrt(2) * vcoul_cv0
v1ao_cv0 += np.sqrt(2) * vcoul_co - np.sqrt(2) * vcoul_ov + 2 * vcoul_cv0
```

##### HF 交换部分 (PDF Eq. 39-42)

对于 hybrid 泛函，HF 交换贡献带有自旋适配系数:

$$
\mathbf{V}_{co}^{\text{HF}} = -\mathbf{K}[\mathbf{D}_{co}] + \mathbf{J}[\mathbf{D}_{co}] - \sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{cv}] - \mathbf{J}[\mathbf{D}_{ov}] - \frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{cv}^{\text{HF}} = -\sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{co}] - \mathbf{K}[\mathbf{D}_{cv}] - \sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{ov}]
$$

$$
\mathbf{V}_{ov}^{\text{HF}} = -\mathbf{J}[\mathbf{D}_{co}] - \sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{cv}] + \mathbf{J}[\mathbf{D}_{ov}] - \mathbf{K}[\mathbf{D}_{ov}] + \frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{cv0}^{\text{HF}} = -\frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{co}] + \frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{ov}] - \mathbf{K}[\mathbf{D}_{cv0}]
$$

**代码对应：** `satda.py:216-233`
```python
dms = np.concatenate((dms_co, dms_cv, dms_ov, dms_cv0), axis=0)
vk = mf.get_k(mol, dms, hermi) * hyb
vj = vcoul[:idx1+n_ov] * hyb
# range-separated correction...
v1ao_co += - vk_co + vj_co - np.sqrt((s+1)/2/s) * vk_cv - vj_ov - np.sqrt(0.5) * vk_cv0
v1ao_cv += - np.sqrt((s+1)/2/s) * vk_co - vk_cv - np.sqrt((s+1)/2/s) * vk_ov
v1ao_ov += - vj_co - np.sqrt((s+1)/2/s) * vk_cv + vj_ov - vk_ov + np.sqrt(0.5) * vk_cv0
v1ao_cv0 += - np.sqrt(0.5) * vk_co + np.sqrt(0.5) * vk_ov - vk_cv0
```

##### K^Ref 部分 (PDF Eq. 43-46)

与 $\Delta S = -1$ 结构相似但系数用 $S$ 而非 $2S$:

$$
\mathbf{V}_{co}^{\text{Ref}} = \mathbf{V}_{co}^{\text{ref0}} - \mathbf{V}_{co}^{\text{ref1}} + \sqrt{\frac{S+1}{2S}} \mathbf{V}_{cv}^{\text{ref0}} + \mathbf{V}_{ov}^{\text{ref1}}
$$

$$
\mathbf{V}_{cv}^{\text{Ref}} = \sqrt{\frac{S+1}{2S}} \mathbf{V}_{co}^{\text{ref0}} + \mathbf{V}_{cv}^{\text{ref0}} + \sqrt{\frac{S+1}{2S}} \mathbf{V}_{ov}^{\text{ref0}}
$$

$$
\mathbf{V}_{ov}^{\text{Ref}} = \mathbf{V}_{co}^{\text{ref1}} + \sqrt{\frac{S+1}{2S}} \mathbf{V}_{cv}^{\text{ref0}} - \mathbf{V}_{ov}^{\text{ref1}} + \mathbf{V}_{ov}^{\text{ref0}}
$$

**代码对应：** `satda.py:235-255`
```python
# vref0 = fxc_ref applied to dms0=(dms_co, dms_cv, dms_ov)
# vref1 = fxc_ref applied to dms1=(dms_co, dms_ov)
v1ao_co += vref0_co - vref1_co + np.sqrt((s+1)/2/s) * vref0_cv + vref1_ov
v1ao_cv += np.sqrt((s+1)/2/s) * vref0_co + vref0_cv + np.sqrt((s+1)/2/s) * vref0_ov
v1ao_ov += vref1_co + np.sqrt((s+1)/2/s) * vref0_cv - vref1_ov + vref0_ov
```

##### K^CV0 部分 (PDF Eq. 47-50)

cv0 道需要额外的 XC 核分量:

$$
\mathbf{V}_{co}^{\text{CV0}} = \frac{1}{\sqrt{2}} \, f_{xc}^{co,cv0} \cdot \mathbf{D}_{cv0}
$$

$$
\mathbf{V}_{cv}^{\text{CV0}} = -\sqrt{\frac{S+1}{S}} \, f_{xc}^{cv,cv0} \cdot \mathbf{D}_{cv0}
$$

$$
\mathbf{V}_{ov}^{\text{CV0}} = -\frac{1}{\sqrt{2}} \, f_{xc}^{ov,cv0} \cdot \mathbf{D}_{cv0}
$$

$$
\mathbf{V}_{cv0}^{\text{CV0}} = \frac{1}{\sqrt{2}} \, f_{xc}^{cv0,co} \cdot \mathbf{D}_{co} - \sqrt{\frac{S+1}{S}} \, f_{xc}^{cv0,cv} \cdot \mathbf{D}_{cv}
- \frac{1}{\sqrt{2}} \, f_{xc}^{cv0,ov} \cdot \mathbf{D}_{ov} + f_{xc}^{s} \cdot \mathbf{D}_{cv0}
$$

**代码对应：** `satda.py:257-269`
```python
v_cocv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_cocv0, ...)
v_cvcv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_cvcv0, ...)
v_ovcv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_ovcv0, ...)
v_cv0co = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_co, 0, hermi, None, None, fxc_cv0co, ...)
v_cv0cv = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv, 0, hermi, None, None, fxc_cv0cv, ...)
v_cv0ov = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_ov, 0, hermi, None, None, fxc_cv0ov, ...)
v_cv0cv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_s, ...)
v1ao_co += np.sqrt(0.5) * v_cocv0
v1ao_cv -= np.sqrt((s+1)/s) * v_cvcv0
v1ao_ov -= np.sqrt(0.5) * v_ovcv0
v1ao_cv0 += np.sqrt(0.5)*v_cv0co - np.sqrt((s+1)/s)*v_cv0cv - np.sqrt(0.5)*v_cv0ov + v_cv0cv0
```

### 3.4 `gen_vind_sc`: 完整 Sigma 向量 (PDF §3.6)

**代码对应：** `satda.py:384-501`

#### 3.4.1 Fock 矩阵分块

基于 $\mathbf{F}^0 = \mathbf{F}^{\alpha} - \mathbf{F}^z$，投影到各轨道子空间:

**代码对应：** `satda.py:413-432`
```python
fock0 = focka - fockz
fock_coco1 = orbos.T @ (fock0 - fockz) @ orbos      # o→o with F^0-F^z
fock_coco2 = orbcs.T @ (fock0 - fockz) @ orbcs       # c→c with F^0-F^z
fock_cocv  = orbos.T @ (fock0 - fockz) @ orbvs       # o→v with F^0-F^z
fock_cvcv1 = orbvs.T @ (fock0 - fockz/s) @ orbvs     # v→v with F^0-F^z/s
fock_cvcv2 = orbcs.T @ (fock0 + fockz/s) @ orbcs     # c→c with F^0+F^z/s
fock_cocv0 = orbos.T @ fockb @ orbvs                 # o→v with F^β (用于 cv0 道)
fock_cvov  = orbos.T @ (fock0 + fockz) @ orbcs       # o→c with F^0+F^z
fock_cvcv01 = 0.5 * orbvs.T @ (focka - fockb) @ orbvs  # (F^α - F^β)/2 在 v 空间
fock_cvcv02 = 0.5 * orbcs.T @ (focka - fockb) @ orbcs  # (F^α - F^β)/2 在 c 空间
fock_ovov1 = orbvs.T @ (fock0 + fockz) @ orbvs       # v→v with F^0+F^z
fock_ovov2 = orbos.T @ (fock0 + fockz) @ orbos       # o→o with F^0+F^z
fock_ovcv0 = orbcs.T @ focka @ orbos                 # c→o with F^α
fock_cv0cv01 = 0.5 * orbvs.T @ (focka + fockb) @ orbvs  # (F^α+F^β)/2 在 v 空间 (cv0 道对角)
fock_cv0cv02 = 0.5 * orbcs.T @ (focka + fockb) @ orbcs  # (F^α+F^β)/2 在 c 空间 (cv0 道对角)
fock_cooo   = orbos.T @ (fock0 - fockz) @ orbcs      # o→c with F^0-F^z
fock_cvoo   = orbvs.T @ fockz @ orbcs                # v→c with F^z
fock_ovoo   = orbvs.T @ (fock0 + fockz) @ orbos      # v→o with F^0+F^z
fock_cv0oo  = 0.5 * orbvs.T @ (focka + fockb) @ orbcs  # v→c with (F^α+F^β)/2
```

#### 3.4.2 对角元 (PDF §3.6)

对于 5 个激发道:

**代码对应：** `satda.py:435-440`
```python
hdiag_co = (fock_coco1.diagonal()[None,:] - fock_coco2.diagonal()[:,None]).ravel()
hdiag_cv = (fock_cvcv1.diagonal()[None,:] - fock_cvcv2.diagonal()[:,None]).ravel()
hdiag_oo = np.array([0.0])                             # oo: 标量, 对角元为 0
hdiag_ov = (fock_ovov1.diagonal()[None,:] - fock_ovov2.diagonal()[:,None]).ravel()
hdiag_cv0 = (fock_cv0cv01.diagonal()[None,:] - fock_cv0cv02.diagonal()[:,None]).ravel()
```

注意 $oo$ 块对角元为 0，因为该块没有 Fock 矩阵之差，仅靠 XC 响应。

#### 3.4.3 Vind 函数: Fock 项叠加 (PDF §3.6)

与 $\Delta S = -1$ 类似，将 AO 基的响应转为 MO 基后叠加 Fock 项。在此仅给出关键的系数差异。

##### co 块 Fock 项 (line 460-464)

$$
A_{cu, dv}^{\text{Fock}} = \delta_{cd} \, F_{vu}^{F^0-F^z} - \delta_{uv} \, F_{cd}^{F^0-F^z}
$$

耦合项 (含从 $cv$、$oo$、$cv0$ 道的贡献):

$$
A_{cu, da}^{\text{coupling}} = \sqrt{\frac{S+1}{2S}} \, \delta_{cd} \, F_{ua}^{F^0-F^z} \times X_{da}
$$

$$
A_{cu, oo}^{\text{coupling}} = -F_{uc}^{F^0-F^z} \times X_{oo}
$$

$$
A_{cu, da}^{(cv0)\text{coupling}} = \frac{1}{\sqrt{2}} \, \delta_{cd} \, F_{ua}^{F^{\beta}} \times X_{da}
$$

**代码对应：** `satda.py:460-464`
```python
v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco1, zs_co)
v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco2, zs_co)
v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv, zs_cv) * np.sqrt((s+1)/2/s)
v1mo_co -= np.einsum('ui,xv->xiu', fock_cooo, zs_oo)
v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv0, zs_cv0) * np.sqrt(0.5)
```

##### cv 块 Fock 项 (line 466-472)

$$
A_{ca, db}^{\text{Fock}} = \delta_{cd} \, F_{ab}^{F^0-F^z/s} - \delta_{ab} \, F_{cd}^{F^0+F^z/s}
$$

耦合项 (来自 co, oo, ov, cv0):

$$
A_{ca, du}^{\text{coupling}} = \sqrt{\frac{S+1}{2S}} \, \delta_{cd} \, (F_{ua}^{F^0-F^z})^{\dagger} \times X_{du}
$$

$$
A_{ca, oo}^{\text{coupling}} = \sqrt{\frac{2(S+1)}{S}} \, F_{ac}^{F^z} \times X_{oo}
$$

$$
A_{ca, vb}^{\text{coupling}} = -\sqrt{\frac{S+1}{2S}} \, \delta_{ab} \, F_{cv}^{F^0+F^z} \times X_{vb}
$$

$$
A_{ca, db}^{(cv0)\text{coupling}} = -\sqrt{\frac{S+1}{S}} \, \delta_{cd} \, F_{ab}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
+ \sqrt{\frac{S+1}{S}} \, \delta_{ab} \, F_{cd}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
$$

**代码对应：** `satda.py:466-472`
```python
v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv.T, zs_co) * np.sqrt((s+1)/2/s)
v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv1, zs_cv)
v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv2, zs_cv)
v1mo_cv += np.einsum('ai,xv->xia', fock_cvoo, zs_oo) * np.sqrt(2*(s+1)/s)
v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_cvov, zs_ov) * np.sqrt((s+1)/2/s)
v1mo_cv -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv01, zs_cv0) * np.sqrt((s+1)/s)
v1mo_cv += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv02, zs_cv0) * np.sqrt((s+1)/s)
```

##### oo 块 Fock 项 (line 488-492)

$oo$ 仅有与 co、cv、ov、cv0 道的耦合项:

$$
A_{oo, cu} = -F_{cu}^{F^0-F^z} \times X_{cu}
$$

$$
A_{oo, ca} = \sqrt{\frac{2(S+1)}{S}} \, F_{ca}^{F^z} \times X_{ca}
$$

$$
A_{oo, ua} = F_{ua}^{F^0+F^z} \times X_{ua}
$$

$$
A_{oo, ca}^{(cv0)} = -\sqrt{2} \, F_{ca}^{(F^{\alpha}+F^{\beta})/2} \times X_{ca}
$$

**代码对应：** `satda.py:488-492`
```python
v1mo_oo = np.zeros((len(zs),))
v1mo_oo -= lib.einsum('jv,xjv->x', fock_cooo.T, zs_co)
v1mo_oo += lib.einsum('jb,xjb->x', fock_cvoo.T, zs_cv) * np.sqrt(2*(s+1)/s)
v1mo_oo += lib.einsum('vb,xvb->x', fock_ovoo.T, zs_ov)
v1mo_oo -= lib.einsum('jb,xjb->x', fock_cv0oo.T, zs_cv0) * np.sqrt(2)
```

##### ov 块 Fock 项 (line 474-478)

$$
A_{ua, vb}^{\text{Fock}} = \delta_{uv} \, F_{ab}^{F^0+F^z} - \delta_{ab} \, F_{vu}^{F^0+F^z}
$$

耦合项:

$$
A_{ua, cb} = -\sqrt{\frac{S+1}{2S}} \, \delta_{ab} \, (F_{cu}^{F^0+F^z})^{\dagger} \times X_{cb}
$$

$$
A_{ua, oo} = F_{ua}^{F^0+F^z} \times X_{oo}
$$

$$
A_{ua, cb}^{(cv0)} = \frac{1}{\sqrt{2}} \, \delta_{ab} \, F_{cu}^{F^{\alpha}} \times X_{cb}
$$

**代码对应：** `satda.py:474-478`
```python
v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_cvov.T, zs_cv) * np.sqrt((s+1)/2/s)
v1mo_ov += np.einsum('au,xv->xua', fock_ovoo, zs_oo)
v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov1, zs_ov)
v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), fock_ovov2, zs_ov)
v1mo_ov += lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_ovcv0, zs_cv0) * np.sqrt(0.5)
```

##### cv0 块 Fock 项 (line 480-486)

cv0 道没有纯 Fock 对角项（已在 $h_{\text{diag}}$ 和 XC 响应中处理），仅有与其他道的耦合:

$$
A_{ca, du}^{(cv0)} = \frac{1}{\sqrt{2}} \, \delta_{cd} \, (F_{ua}^{F^{\beta}})^{\dagger} \times X_{du}
$$

$$
A_{ca, db}^{(cv0)} = -\sqrt{\frac{S+1}{S}} \, \delta_{cd} \, F_{ab}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
$$

$$
A_{ca, db}^{(cv0)} = \sqrt{\frac{S+1}{S}} \, \delta_{ab} \, F_{cd}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
$$

$$
A_{ca, oo}^{(cv0)} = -\sqrt{2} \, F_{ca}^{(F^{\alpha}+F^{\beta})/2} \times X_{oo}
$$

$$
A_{ca, vb}^{(cv0)} = \frac{1}{\sqrt{2}} \, \delta_{ab} \, (F_{vc}^{F^{\alpha}})^{\dagger} \times X_{vb}
$$

$$
A_{ca, db}^{\text{Fock}} = \delta_{cd} \, F_{ab}^{(F^{\alpha}+F^{\beta})/2} - \delta_{ab} \, F_{cd}^{(F^{\alpha}+F^{\beta})/2}
$$

**代码对应：** `satda.py:480-486`
```python
v1mo_cv0 += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv0.T, zs_co) * np.sqrt(0.5)
v1mo_cv0 -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv01, zs_cv) * np.sqrt((s+1)/s)
v1mo_cv0 += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv02, zs_cv) * np.sqrt((s+1)/s)
v1mo_cv0 -= np.einsum('ai,xv->xia', fock_cv0oo, zs_oo) * np.sqrt(2)
v1mo_cv0 += lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_ovcv0.T, zs_ov) * np.sqrt(0.5)
v1mo_cv0 += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cv0cv01, zs_cv0)
v1mo_cv0 -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cv0cv02, zs_cv0)
```

---

## 4. Hybrid 泛函与 Range-Separated 修正

对于 hybrid 泛函，HF 交换系数 $c_{\text{hyb}}$ 和 range-separation 参数 ($\omega$, $\alpha$) 分别处理:

$$
\mathbf{K}_{\text{total}} = c_{\text{hyb}} \, \mathbf{K}_{\text{Coulomb}} + (\alpha - c_{\text{hyb}}) \, \mathbf{K}_{\omega}^{\text{SR}}
$$

其中 $\mathbf{K}_{\omega}^{\text{SR}}$ 是短程 exchange (erfc(ωr)/r 型).

**代码对应：** `satda.py:219-223`, `satda.py:347-351`
```python
vk = mf.get_k(mol, dms, hermi) * hyb
if omega != 0:
    vk += mf.get_k(mol, dms, hermi, omega=omega) * (alpha - hyb)
```

---

## 5. GGA 和 Meta-GGA 响应函数

### 5.1 nr_rks_fxc1_gga — GGA 梯度校正 XC 核响应 (PDF Appendix A)

**代码对应：** `satda.py:33-85`

对于 GGA，XC 核 $f_{xc}(r, r')$ 不仅依赖于密度，还依赖于密度梯度。响应函数需要处理:

$$
V_{\mu\nu} = \int \frac{\delta^2 E_{xc}}{\delta \rho(r) \delta \rho(r')} \, \phi_{\mu}(r) \phi_{\nu}(r) \, \delta\rho(r') \, dr \, dr'
$$

作为 $\delta\rho$ 的函数，$V_{\mu\nu}$ 由局部项和梯度项组成:

$$
V_{\mu\nu} = U_{00} \phi_{\mu} \phi_{\nu} \rho_0 + \sum_{i=1}^{3} U_{i0} \phi_{\mu} \phi_{\nu} \nabla_i \rho_0 + \sum_{i=1}^{3} U_{0i} \rho_0 (\nabla_i \phi_{\mu} \phi_{\nu}) + \sum_{i,j=1}^{3} U_{ij} \nabla_i(\phi_{\mu} \phi_{\nu}) \nabla_j \rho_0
$$

其中 $U_{ab}$ 矩阵 (4×4) 组合了 $f_{xc}$ 的各分量和密度梯度:

**代码对应：** `satda.py:69-78` (GGA U matrix)
```python
U = np.zeros((4, 4, weight.size))
U[0, 0] = _fxc[0, 0] * rho0
U[0, 0] += lib.einsum('ig,ig->g', _fxc[1:4, 0], L_grad)    # Σ_i f_{i0} L_i
U[0, 0] += lib.einsum('jg,jg->g', _fxc[0, 1:4], R_grad)    # Σ_j f_{0j} R_j
U[0, 0] += lib.einsum('ijg,ijg->g', _fxc[1:4, 1:4], tau)   # Σ_ij f_{ij} τ_ij
U[1:4, 0] = _fxc[1:4, 0] * rho0
U[1:4, 0] += lib.einsum('ijg,jg->ig', _fxc[1:4, 1:4], R_grad)
U[0, 1:4] = _fxc[0, 1:4] * rho0
U[0, 1:4] += lib.einsum('ijg,ig->jg', _fxc[1:4, 1:4], L_grad)
U[1:4, 1:4] = _fxc[1:4, 1:4] * rho0
```

其中:
- $\rho_0$ = `rho0` — 扰动密度的 0 阶值
- $L_{\text{grad}}$ = 左侧梯度 (将 DM 的梯度作用到 AO 上)
- $R_{\text{grad}}$ = 右侧梯度 (将 AO 的梯度作用到 DM 上)
- $\tau$ = 密度 Hessian 矩阵 (3×3 梯度积)

最终 AO 基的响应矩阵通过稀疏矩阵乘法计算 (line 80-84):

$$
V_{\mu\nu}^{(k)} = \sum_{i=0}^{3} \phi_{\mu}^{(i)} \, U_{i,*} \, \phi_{\nu}^{(*)}
$$

其中 $\phi^{(0)} = \phi$, $\phi^{(1,2,3)} = \nabla \phi$.

### 5.2 nr_rks_fxc1_mgga — Meta-GGA 动能密度响应 (PDF Appendix B)

**代码对应：** `satda.py:88-147`

Meta-GGA 额外依赖动能密度 $\tau(r) = \frac{1}{2} \sum_i |\nabla \phi_i|^2$。$f_{xc}$ 核扩展为 5×5 矩阵（指标 0-3 为密度和梯度, 指标 4 为 $\tau$）.

U 矩阵扩展了含 $\tau$ 的项 (line 128-139):

$$
U_{i0} += \frac{1}{2} f_{i,4} \, L_{\text{grad}}
$$

$$
U_{0i} += \frac{1}{2} f_{4,i} \, R_{\text{grad}}
$$

$$
U_{ij} += \frac{1}{2} f_{i,4} \, R_{\text{grad}} + \frac{1}{2} f_{4,i} \, L_{\text{grad}} + \frac{1}{4} f_{4,4} \, \tau
$$

**代码对应：** `satda.py:128-139`
```python
U[1:4, 0] += 0.5 * _fxc[4, 0].reshape(1, -1) * L_grad
U[1:4, 0] += 0.5 * lib.einsum('jg,ijg->ig', _fxc[4, 1:4], tau)
U[0, 1:4] += 0.5 * _fxc[0, 4].reshape(1, -1) * R_grad
U[0, 1:4] += 0.5 * lib.einsum('ig,ijg->jg', _fxc[1:4, 4], tau)
U[1:4, 1:4] += 0.5 * lib.einsum('ig,jg->ijg', _fxc[1:4, 4], R_grad)
U[1:4, 1:4] += 0.5 * lib.einsum('jg,ig->ijg', _fxc[4, 1:4], L_grad)
U[1:4, 1:4] += 0.25 * _fxc[4, 4].reshape(1, 1, -1) * tau
```

---

## 6. TDA 对角化与迭代算法

### 6.1 Davidson 迭代

**代码对应：** `satda.py:668-680`

$$
\mathbf{X}^{(k+1)} = \text{lr\_eigh}(\text{vind}, \mathbf{X}^{(k)}, \text{precond}, \dots)
$$

使用 PySCF 的 `lr_eigh` (Davidson/Lanczos 型) 对 sigma 向量函数 `vind` 进行迭代对角化。预处理矩阵用 `hdiag`（即 A 矩阵的对角近似）。

### 6.2 初值猜测 (SATDA.init_guess)

**代码对应：** `satda.py:629-636`

取 `hdiag` 最小的几个对角元对应的单位向量作为初值：

```python
n_init = min(nstates + 3, hdiag.size)
idx = np.argsort(hdiag)[:n_init]
x0 = np.zeros((n_init, hdiag.size))
x0[np.arange(n_init), idx] = 1.0
```

### 6.3 本征值分类与 XY 向量

**代码对应：** `satda.py:682-686`

- $\Delta S = 0$: 直接保留为向量 `xy = [(xi, 0) for xi in x1]`
- $\Delta S = -1$: 解为 `xy = [(xi.reshape(nocca, nvirb), 0) for xi in x1]`

reshape 为 $(N_{occ}^{\alpha}, N_{vir}^{\beta})$ ($N_{occ}^{\alpha} = N_{cs} + N_{os}$, $N_{vir}^{\beta} = N_{os} + N_{vs}$) 的矩阵形式，对应激发算符 $\hat{a}_a^{\dagger} \hat{a}_i$ 的系数矩阵。

---

## 7. 公式 → 代码 关键对照总结

| PDF 公式 / 定义 | 代码位置 | 说明 |
|---|---|---|
| $S$ 定义 | `s = (mol.nelec[0] - mol.nelec[1]) * 0.5` | 总自旋量子数 |
| 轨道划分 (cs, os, vs) | `csidx/osidx/vsidx` (line 394-399) | 按 mo_occ 划分 |
| $f_{xc}^{\text{ref}}$ | `fxc_ref` (line 170-171) | ROKS 密度 XC 核 |
| $f_{xc}^{s}$, $f_{xc}^{cv0,cv}$, ... | line 175-181 | UKS 核分量 (仅 ΔS=0) |
| $\mathbf{A}^{\text{Resp}}$ (ΔS=-1) | `gen_rohf_response_sf` (line 284-382) | XC + HF 响应函数 |
| $\mathbf{A}^{\text{Resp}}$ (ΔS=0) | `gen_rohf_response_sc` (line 149-282) | + Coulomb + 多核 XC |
| $\mathbf{F}^z$ | `fockz = 0.5*delta` (line 282, 382) | 开壳密度响应 |
| $\mathbf{A}^{\text{Fock}}$ (ΔS=-1) | `gen_vind_sf` (line 503-617) | 自旋适配 Fock 项 |
| $\mathbf{A}^{\text{Fock}}$ (ΔS=0) | `gen_vind_sc` (line 384-501) | 含 cv0 道 |
| GGA U 矩阵 | `nr_rks_fxc1_gga` (line 33-85) | 梯度校正 XC 响应 |
| MGGA U 矩阵 | `nr_rks_fxc1_mgga` (line 88-147) | 动能密度 XC 响应 |
| TDA 对角化 | `SATDA.kernel()` (line 638-693) | lr_eigh Davidson 对角化 |

---

## 附录: 自旋适配系数的物理来源

自旋适配系数来自 Wigner-Eckart 定理。两个自旋态之间的矩阵元可通过约化矩阵元和一个几何因子因式分解:

$$
\langle S', M' | \hat{O} | S, M \rangle = C_{SM, 1q}^{S'M'} \, \langle S' || \hat{O} || S \rangle
$$

其中 $C_{SM, 1q}^{S'M'}$ 是 Clebsch-Gordan 系数。

对于自旋翻转激发 $S_f = S_i - 1$:

| 跃迁类型 | 物理过程 | CG 系数 | 代码中的体现 |
|---|---|---|---|
| co (c→u) | 翻转一个闭壳电子的自旋 | $\frac{1}{\sqrt{2S}}$ | $\sqrt{\frac{2S+1}{2S}}$ 的配对系数 |
| cv (c→a) | 闭壳→虚轨道的自旋翻转 | $\frac{1}{\sqrt{2S(S+1)}}$ | $1/S$ 的 Fock 能差修正 |
| oo (u→v) | 开壳→开壳的保持自旋 | $\sqrt{\frac{2S}{2S-1}}$ | 自旋保持项 |
| ov (u→a) | 开壳→虚轨道的自旋翻转 | $\sqrt{\frac{2S}{2S+1}}$ | $\frac{2S}{2S-1}$ 的主系数 |

对于自旋守恒激发 $S_f = S_i = S$:

| 跃迁类型 | 物理过程 | CG 系数 | 代码中的体现 |
|---|---|---|---|
| co (c→u) | 保持自旋的激发 | $\frac{1}{\sqrt{2S+2}}$ | $\sqrt{\frac{S+1}{2S}}$ |
| cv0 (特殊道) | 纯 Coulomb 耦合 | $\sqrt{2}$ | co↔cv0, ov↔cv0 的 $\sqrt{2}$ 系数 |

这些系数确保激发态是自旋算符 $\hat{S}^2$ 和 $\hat{S}_z$ 的本征态。

---

*推导完成。代码对应: `pyscf-forge/pyscf/sftda/satda.py` (SATDA 类及其响应函数)。*
