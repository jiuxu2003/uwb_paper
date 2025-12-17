# Research: TH-UWB Communication System Simulation

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)
**Phase**: Phase 0 - Technical Research
**Date**: 2025-12-17

## Overview

本文档记录了实现 TH-UWB 通信系统仿真所需的关键技术研究结果，包括信号生成、信道建模、接收解调、可视化等核心模块的实现方案。

## 1. 二阶高斯导数脉冲生成（Gaussian Monocycle）

### 1.1 数学公式

二阶高斯导数脉冲（也称为 Gaussian Doublet 或 Mexican Hat Wavelet）是 UWB 系统中最常用的脉冲波形之一。其数学表达式为：

```
g''(t) = A · (1 - 4π(t/τ)²) · exp(-2π(t/τ)²)
```

其中：
- `A`: 脉冲幅度（amplitude）
- `τ`: 脉冲宽度参数（pulse width parameter），通常在纳秒级别（如 0.5ns）
- `t`: 时间变量

**物理特性**：
- 中心时刻（t=0）：脉冲达到最大正值
- 零交叉点：约在 t ≈ ±0.4τ 处
- 时域有限支撑：能量主要集中在 [-2τ, 2τ] 区间
- 频域特性：超宽带特性，带宽可达数 GHz

### 1.2 NumPy 实现方案

```python
import numpy as np

def generate_gaussian_doublet(t: np.ndarray,
                             tau: float = 0.5e-9,
                             amplitude: float = 1.0) -> np.ndarray:
    """
    生成二阶高斯导数脉冲

    参数:
        t: 时间数组（单位：秒）
        tau: 脉冲宽度参数（单位：秒），默认 0.5ns
        amplitude: 脉冲幅度，默认 1.0

    返回:
        脉冲波形数组
    """
    normalized_t = t / tau
    pulse = amplitude * (1 - 4 * np.pi * normalized_t**2) * \
            np.exp(-2 * np.pi * normalized_t**2)
    return pulse
```

**采样要求**：
- 采样频率：≥50 GHz（根据 Nyquist 定理，需远高于脉冲最高频率）
- 时间分辨率：≤20 ps（1/(50GHz) = 20ps）
- 脉冲持续时间：约 5τ（如 τ=0.5ns，则持续时间约 2.5ns）

**数值验证标准**：
1. 能量归一化：∫|g''(t)|²dt 应为有限值
2. 零均值：∫g''(t)dt ≈ 0（积分为零，满足无直流分量要求）
3. 峰值位置：max(g''(t)) 应在 t=0 附近

**工具链选择**：
- NumPy：向量化运算，避免显式循环
- SciPy.signal：可选用 `scipy.signal.gausspulse` 作为替代方案（但需手动求导）

### 1.3 参考来源

- Web Search 结果提供的二阶高斯导数脉冲实现模式
- SciPy 信号处理文档：[scipy.signal 模块](https://github.com/scipy/scipy/blob/main/doc/source/tutorial/signal.rst)

---

## 2. TH-PPM 调制原理

### 2.1 跳时序列（Time-Hopping Code）

**定义**：每个用户分配一个伪随机跳时序列 `{c_j}`，其中 `c_j ∈ [0, Nh-1]` 决定第 j 帧中脉冲在哪个时隙发射。

**生成算法**：
```python
def generate_th_code(user_id: int,
                    num_frames: int,
                    num_hops: int,
                    seed: int = None) -> np.ndarray:
    """
    为用户生成跳时序列

    参数:
        user_id: 用户 ID（用于种子多样化）
        num_frames: 帧数量
        num_hops: 每帧可用时隙数量（Nh）
        seed: 随机种子基础值

    返回:
        跳时序列数组，长度为 num_frames
    """
    if seed is None:
        seed = 0
    rng = np.random.default_rng([user_id, seed])  # 使用 SeedSequence 确保独立性
    th_code = rng.integers(0, num_hops, size=num_frames)
    return th_code
```

**重要性**：
- 多用户区分：不同用户的跳时序列尽量正交，减少碰撞概率
- MUI 分析基础：当多个用户的脉冲在同一时隙发射时产生干扰

### 2.2 PPM 调制（Pulse Position Modulation）

**二进制 PPM 规则**（基于 Clarification Q3）：
- 比特 0：脉冲位于时隙起始位置（无偏移）
- 比特 1：脉冲偏移 δ = Tc/2（时隙宽度的一半）

**TH-PPM 信号数学表达式**：
```
s(t) = Σ[j=0 to Nf-1] g''(t - jTf - c_j·Tc - d_j·δ)
```

其中：
- `Nf`: 总帧数
- `Tf`: 帧周期
- `Tc`: 时隙宽度
- `c_j`: 第 j 帧的跳时码
- `d_j ∈ {0, 1}`: 第 j 帧携带的数据比特
- `δ = Tc/2`: PPM 调制时延

**约束条件**：
- `Tf` 必须足够大以容纳 `Nh` 个时隙：`Tf ≥ Nh · Tc`
- `Tc` 必须足够大以容纳脉冲宽度：`Tc ≥ 5τ`（确保脉冲不重叠）
- `δ` 必须足够大以在接收端区分：`δ ≥ τ`（通常选择 Tc/2）

### 2.3 参考来源

- [System Simulations of DSTRD and TH-PPM for Ultra Wide ...](https://digitalcommons.unf.edu/cgi/viewcontent.cgi?article=1069&context=ojii_volumes)
- [An enhanced pulse position modulation (PPM) in ultra ...](https://scholarworks.uni.edu/cgi/viewcontent.cgi?article=1041&context=etd)
- NumPy 随机数生成最佳实践：[SeedSequence for parallel RNG](https://github.com/numpy/numpy/blob/main/doc/source/reference/random/parallel.rst)

---

## 3. 信道建模

### 3.1 高斯白噪声（AWGN）

**模型**：
```
r(t) = s(t) + n(t)
```

其中 `n(t) ~ N(0, σ²)` 是白噪声，功率谱密度为 `N0/2`。

**SNR 定义**：
```
SNR (dB) = 10·log10(Eb/N0)
```

其中：
- `Eb`: 每比特能量
- `N0`: 单边噪声功率谱密度

**NumPy 实现**：
```python
def add_awgn(signal: np.ndarray,
             snr_db: float,
             signal_power: float = None) -> np.ndarray:
    """
    添加高斯白噪声

    参数:
        signal: 原始信号
        snr_db: 信噪比（dB）
        signal_power: 信号功率（若为 None 则自动计算）

    返回:
        加噪后的信号
    """
    if signal_power is None:
        signal_power = np.mean(signal**2)

    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    return signal + noise
```

### 3.2 多用户干扰（MUI）

**模型**：当 K 个用户同时发射时：
```
r(t) = s_1(t) + Σ[k=2 to K] s_k(t) + n(t)
```

其中 `s_k(t)` 是第 k 个用户的 TH-PPM 信号（使用不同跳时序列）。

**干扰分析**：
- 时隙碰撞概率：`P_collision ≈ (K-1)/Nh`（假设跳时码均匀分布）
- 干扰功率随用户数线性增长：`P_MUI ∝ (K-1)`
- 预期性能：BER 随用户数单调上升（验证 SC-002）

### 3.3 窄带干扰（NBI）

**模型**（基于 Clarification Q2）：
```
i(t) = A_i · sin(2π·f_i·t + φ)
```

其中：
- `f_i = 2.4 GHz`（WiFi 频段）
- `A_i`: 干扰幅度（由 SIR 确定）
- `φ`: 随机初相位

**SIR 定义**：
```
SIR (dB) = 10·log10(P_signal / P_interference)
```

**NumPy 实现**：
```python
def generate_nbi(t: np.ndarray,
                 freq_hz: float = 2.4e9,
                 sir_db: float = 10.0,
                 signal_power: float = 1.0) -> np.ndarray:
    """
    生成窄带干扰（单频正弦波）

    参数:
        t: 时间数组
        freq_hz: 干扰频率（Hz），默认 2.4 GHz
        sir_db: 信干比（dB）
        signal_power: 信号功率

    返回:
        干扰信号
    """
    sir_linear = 10**(sir_db / 10)
    interference_power = signal_power / sir_linear
    amplitude = np.sqrt(2 * interference_power)  # RMS 转峰值

    phase = np.random.uniform(0, 2*np.pi)  # 随机初相位
    interference = amplitude * np.sin(2 * np.pi * freq_hz * t + phase)

    return interference
```

**抗干扰机制**：
- UWB 固有优势：信号能量分散在极宽频带上，单频干扰只影响很小一部分
- 理论预期：即使 SIR=-10dB，BER 仍应 <0.4（验证 SC-003）

### 3.4 参考来源

- [Simulation of Interference Effects of UWB Pulse Signal to the GPS Receiver](https://onlinelibrary.wiley.com/doi/10.1155/2021/9935543)
- NumPy 随机数生成：[default_rng seeding guide](https://github.com/numpy/numpy/blob/main/doc/source/reference/random/index.rst)

---

## 4. 相关接收机（Correlator Receiver）

### 4.1 工作原理

**假设**：接收端完美同步，已知跳时序列和精确时序（FR-008）。

**接收过程**：
1. **模板信号生成**：构造两个参考模板
   - 模板 0：对应比特 0 的理想信号（无 PPM 偏移）
   - 模板 1：对应比特 1 的理想信号（有 PPM 偏移 δ）

2. **相关运算**：
   ```
   Λ_0 = ∫[0 to Tf] r(t) · template_0(t) dt
   Λ_1 = ∫[0 to Tf] r(t) · template_1(t) dt
   ```

3. **判决准则**：
   ```
   决策比特 = argmax(Λ_0, Λ_1)
   ```

### 4.2 NumPy 实现

```python
def correlator_receiver(received_signal: np.ndarray,
                       template_0: np.ndarray,
                       template_1: np.ndarray,
                       fs: float,
                       frame_period: float) -> int:
    """
    相关接收机（单帧判决）

    参数:
        received_signal: 接收信号（单帧）
        template_0: 比特 0 的模板信号
        template_1: 比特 1 的模板信号
        fs: 采样频率
        frame_period: 帧周期

    返回:
        判决比特（0 或 1）
    """
    # 相关运算（离散形式）
    correlation_0 = np.sum(received_signal * template_0) / fs
    correlation_1 = np.sum(received_signal * template_1) / fs

    # 最大似然判决
    decision = 1 if correlation_1 > correlation_0 else 0

    return decision
```

**优化要点**：
- 向量化运算：避免逐帧循环，使用 NumPy 广播机制批量处理
- 内存管理：对于长仿真（10^4 比特），分块处理避免内存溢出

### 4.3 参考来源

- [Multiuser detection of time-hopping PPM UWB system](https://www.sciencedirect.com/science/article/abs/pii/S1434841106000203)
- [A performance analysis of the high-capacity TH multiple access UWB system using PPM](https://ieeexplore.ieee.org/document/5449760/)

---

## 5. 误码率（BER）计算

### 5.1 统计方法

**定义**：
```
BER = 错误比特数 / 总发送比特数
```

**统计稳定性要求**（SC-004）：
- 最小比特数：10^4 个
- 相对误差目标：<20%
- 置信区间估计：使用二项分布理论

**NumPy 实现**：
```python
def calculate_ber(transmitted_bits: np.ndarray,
                 received_bits: np.ndarray) -> float:
    """
    计算误码率

    参数:
        transmitted_bits: 发送比特序列
        received_bits: 接收比特序列

    返回:
        误码率（0到1之间）
    """
    errors = np.sum(transmitted_bits != received_bits)
    ber = errors / len(transmitted_bits)
    return ber
```

### 5.2 性能指标

**多用户干扰分析**（SC-002）：
- 固定 SNR = 10 dB
- 用户数：1, 2, 3, 5, 7, 10
- 预期趋势：BER 从 <10^-3（1 用户）增长至 ~10^-1（10 用户）

**窄带干扰分析**（SC-003）：
- 固定单用户场景
- SIR 范围：30 dB → 0 dB → -10 dB
- 预期趋势：SIR 降低时 BER 上升，但增长幅度不超过 2 个数量级

---

## 6. 学术论文级可视化

### 6.1 Matplotlib 配置标准

**全局 rcParams 配置**（适合学术论文）：

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局参数
mpl.rcParams['figure.figsize'] = [8.0, 6.0]  # 图表尺寸（英寸）
mpl.rcParams['figure.dpi'] = 100  # 屏幕显示 DPI
mpl.rcParams['savefig.dpi'] = 300  # 保存 DPI（满足 SC-005 的 ≥300 要求）
mpl.rcParams['savefig.transparent'] = False  # 白色背景
mpl.rcParams['font.size'] = 12  # 基础字体大小
mpl.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体
mpl.rcParams['axes.titlesize'] = 14  # 标题字体
mpl.rcParams['xtick.labelsize'] = 10  # X 轴刻度字体
mpl.rcParams['ytick.labelsize'] = 10  # Y 轴刻度字体
mpl.rcParams['legend.fontsize'] = 10  # 图例字体
mpl.rcParams['axes.grid'] = True  # 启用网格
mpl.rcParams['grid.alpha'] = 0.3  # 网格透明度
mpl.rcParams['grid.linestyle'] = '--'  # 网格线样式
mpl.rcParams['lines.linewidth'] = 1.5  # 线宽
```

**推荐使用 SciencePlots 样式**（如果可用）：
```python
# 需要安装: pip install SciencePlots
import scienceplots
plt.style.use(['science', 'ieee'])  # IEEE 期刊风格
```

### 6.2 三种核心图表

#### 图表 1：时域波形图（FR-010）

```python
def plot_waveform(t: np.ndarray, signal: np.ndarray,
                 title: str = "TH-PPM信号时域波形"):
    """
    生成时域波形图

    要求：
    - 清晰展示至少 3 帧信号（SC-001）
    - 标注时间轴（ns）和幅度轴
    - 包含网格、图例、标题
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1e9, signal, linewidth=1.0, label='TH-PPM 信号')
    ax.set_xlabel('时间 (ns)', fontsize=12)
    ax.set_ylabel('幅度', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('waveform.png', dpi=300, bbox_inches='tight')
    plt.show()
```

#### 图表 2：BER vs 用户数量（FR-011）

```python
def plot_ber_vs_users(user_counts: np.ndarray, ber_values: np.ndarray):
    """
    生成 BER vs 用户数量性能曲线

    要求：
    - 横轴：用户数量（线性）
    - 纵轴：BER（对数坐标）
    - 包含网格、轴标签、图例
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(user_counts, ber_values, 'o-', linewidth=2,
                markersize=8, label='SNR=10dB')
    ax.set_xlabel('用户数量', fontsize=12)
    ax.set_ylabel('误码率 (BER)', fontsize=12)
    ax.set_title('多用户干扰性能分析', fontsize=14)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('ber_vs_users.png', dpi=300, bbox_inches='tight')
    plt.show()
```

#### 图表 3：BER vs 信干比（FR-012）

```python
def plot_ber_vs_sir(sir_db: np.ndarray, ber_values: np.ndarray):
    """
    生成 BER vs SIR 性能曲线

    要求：
    - 横轴：SIR (dB)
    - 纵轴：BER（对数坐标）
    - 包含网格、轴标签、图例
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(sir_db, ber_values, 's-', linewidth=2,
                markersize=8, label='单用户', color='red')
    ax.set_xlabel('信干比 SIR (dB)', fontsize=12)
    ax.set_ylabel('误码率 (BER)', fontsize=12)
    ax.set_title('窄带干扰抑制性能分析', fontsize=14)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('ber_vs_sir.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 6.3 参考来源

- Matplotlib 官方文档：[savefig DPI 配置](https://github.com/matplotlib/matplotlib/blob/main/doc/release/prev_whats_new/whats_new_1.5.rst)
- Matplotlib rcParams 指南：[全局样式配置](https://github.com/matplotlib/matplotlib/blob/main/doc/release/prev_whats_new/dflt_style_changes.rst)
- [SciencePlots](https://github.com/garrettj403/scienceplots)（可选，用于 IEEE/Nature 期刊风格）

---

## 7. 可重复性保证（FR-014）

### 7.1 随机数种子管理

**NumPy 最佳实践**（基于 SeedSequence）：

```python
import numpy as np

# 全局种子设置
GLOBAL_SEED = 42

def initialize_simulation(seed: int = GLOBAL_SEED):
    """
    初始化仿真环境的随机数生成器

    参数:
        seed: 全局种子值

    返回:
        rng: NumPy 随机数生成器实例
    """
    rng = np.random.default_rng(seed)
    print(f"仿真使用随机种子: {seed}")
    return rng
```

**多用户场景的种子管理**：
```python
def spawn_user_rngs(base_seed: int, num_users: int):
    """
    为多个用户生成独立的随机数生成器

    使用 SeedSequence.spawn() 确保每个用户的跳时序列独立且可重复
    """
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(num_users)
    user_rngs = [np.random.default_rng(s) for s in child_seeds]
    return user_rngs
```

### 7.2 结果验证

**验证清单**：
1. 使用相同种子运行两次仿真，所有输出（BER、图表）完全一致
2. 改变种子值，结果应有统计差异但趋势保持一致
3. 记录种子值于每次仿真的日志或图表标题中

### 7.3 参考来源

- NumPy 随机数生成：[SeedSequence best practices](https://github.com/numpy/numpy/blob/main/doc/source/reference/random/bit_generators/index.rst)
- 并行仿真指南：[Parallel RNG with SeedSequence](https://github.com/numpy/numpy/blob/main/doc/source/reference/random/parallel.rst)

---

## 8. 性能优化建议

### 8.1 向量化运算

**避免显式循环**，利用 NumPy 广播机制：

```python
# 差的实现（显式循环）
signal = np.zeros(len(t))
for j in range(num_frames):
    pulse_time = j * frame_period + th_code[j] * slot_width + data_bits[j] * delta
    pulse_index = int(pulse_time * fs)
    signal[pulse_index:pulse_index+len(pulse_template)] += pulse_template

# 好的实现（向量化）
pulse_times = (np.arange(num_frames) * frame_period +
               th_code * slot_width +
               data_bits * delta)
# 使用 np.add.at() 或预分配数组批量处理
```

### 8.2 内存管理

**分块处理**（适用于 10^4 比特的长仿真）：
- 每次处理 1000 个帧，避免一次性生成超大数组
- 使用生成器模式逐帧处理并累积 BER 统计

### 8.3 性能目标验证（SC-006）

- 单点仿真（10^4 比特）：目标 <1 分钟
  - 瓶颈预测：脉冲生成（O(N·M)，N 为帧数，M 为采样点）
  - 优化方向：预计算脉冲模板，重复使用
- 完整曲线（10 个性能点）：目标 <15 分钟
  - 并行化潜力：多个 SNR/SIR 点可并行计算

---

## 9. 依赖项版本锁定

基于 Technical Context 中的要求，最终 `requirements.txt` 应包含：

```txt
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
pytest>=7.0.0
```

**可选依赖**（如需学术风格）：
```txt
scienceplots>=2.0.0  # IEEE/Nature 期刊样式
```

---

## 10. 研究结论

### 10.1 技术可行性

✅ 所有核心功能均有成熟的 NumPy/SciPy 实现方案
✅ 性能目标（SC-006）在合理范围内可达成
✅ 可视化需求可通过 Matplotlib 标准配置满足
✅ 可重复性通过 SeedSequence 机制保证

### 10.2 潜在风险

⚠️ **采样频率选择**：需平衡精度和计算时间（建议 50 GHz 起步）
⚠️ **大规模仿真内存**：10^4 比特 × 50 GHz × 5ns = 2.5M 采样点/帧，需优化
⚠️ **跳时码设计**：随机码可能导致高碰撞率，可考虑 Gold 序列（超出当前范围）

### 10.3 下一步行动

进入 **Phase 1: Design**，生成以下文档：
1. `data-model.md`：定义核心实体类（Signal, User, Channel, Receiver, Metrics）
2. `contracts/`：定义模块间接口契约
3. `quickstart.md`：提供快速运行示例

---

**研究完成日期**: 2025-12-17
**研究者**: Claude Sonnet 4.5
**审查状态**: 待 Phase 1 验证
