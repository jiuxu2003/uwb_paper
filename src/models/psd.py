"""
PSD (功率谱密度) 计算模块

本模块提供高斯导数脉冲生成和功率谱密度计算功能。

功能：
1. 生成 2 阶和 5 阶高斯导数脉冲
2. 计算脉冲的功率谱密度（PSD）
3. 频谱对比分析
"""

import numpy as np
from typing import Tuple, Optional


def generate_gaussian_derivative_pulse(
    n: int,
    tau: float,
    fc: float,
    sampling_rate: float,
    duration_factor: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 n 阶高斯导数脉冲

    参数:
        n: 阶数，取值 {2, 5}
        tau: 脉冲宽度（秒）
        fc: 中心频率（Hz），0 表示无调制
        sampling_rate: 采样率（Hz）
        duration_factor: 脉冲持续时间倍数，默认 10

    返回:
        waveform: 脉冲波形，能量归一化
        time_axis: 时间轴

    前置条件:
        - n ∈ {2, 5}
        - tau ∈ [0.1e-9, 2e-9]
        - sampling_rate ≥ 24e9

    后置条件:
        - 能量归一化：0.99 <= ∫|waveform|² dt <= 1.01

    数学公式（参见 research.md）:
        - 2阶: p_2(t) = A_2 · (1 - t²/τ²) · exp(-t²/(2τ²))
        - 5阶: p_5(t) = A_5 · [15t/τ² - 10t³/τ⁴ + t⁵/τ⁶] · exp(-t²/(2τ²))
        - 归一化系数: A_2 = sqrt(2/(√π·τ)), A_5 = sqrt(32/(√π·τ))
        - 载波调制（fc > 0）: p(t) = p_n(t) · cos(2πfct)
    """
    # 参数验证
    assert n in {2, 5}, f"阶数 n 必须是 2 或 5，当前值: {n}"
    assert 0.1e-9 <= tau <= 2e-9, f"脉冲宽度 tau 必须在 [0.1ns, 2ns] 范围内，当前值: {tau*1e9:.2f} ns"
    assert sampling_rate >= 24e9, f"采样率必须 ≥24 GHz，当前值: {sampling_rate/1e9:.2f} GHz"

    # 生成时间轴（脉冲中心在 t=0）
    duration = duration_factor * tau  # 脉冲总持续时间
    num_samples = int(duration * sampling_rate)
    time_axis = np.linspace(-duration/2, duration/2, num_samples)

    # 计算理论归一化系数（连续时间）
    sqrt_pi = np.sqrt(np.pi)
    if n == 2:
        # A_2 = sqrt(2/(√π · τ))
        A_n_theory = np.sqrt(2 / (sqrt_pi * tau))
    elif n == 5:
        # A_5 = sqrt(32/(√π · τ))
        A_n_theory = np.sqrt(32 / (sqrt_pi * tau))

    # 生成基带脉冲（无调制）
    t_normalized = time_axis / tau  # 归一化时间 t/τ
    gaussian = np.exp(-t_normalized**2 / 2)  # exp(-t²/(2τ²))

    if n == 2:
        # p_2(t) = A_2 · (1 - t²/τ²) · exp(-t²/(2τ²))
        waveform = A_n_theory * (1 - t_normalized**2) * gaussian
    elif n == 5:
        # p_5(t) = A_5 · [15t/τ² - 10t³/τ⁴ + t⁵/τ⁶] · exp(-t²/(2τ²))
        # 展开：= A_5 · [(15/τ²)t - (10/τ⁴)t³ + (1/τ⁶)t⁵] · exp(-t²/(2τ²))
        polynomial = 15*time_axis/tau**2 - 10*time_axis**3/tau**4 + time_axis**5/tau**6
        waveform = A_n_theory * polynomial * gaussian

    # 离散能量归一化校正
    dt = time_axis[1] - time_axis[0]
    energy_before = np.trapz(waveform**2, dx=dt)
    waveform = waveform / np.sqrt(energy_before)  # 归一化到单位能量

    # 载波调制（如果 fc > 0）
    if fc > 0:
        carrier = np.cos(2 * np.pi * fc * time_axis)
        waveform = waveform * carrier
        # 调制后重新归一化（cos² 的平均值是 1/2，导致能量减半）
        energy_after_mod = np.trapz(waveform**2, dx=dt)
        waveform = waveform / np.sqrt(energy_after_mod)

    # 能量归一化验证（后置条件）
    energy = np.trapz(waveform**2, dx=dt)
    assert 0.99 <= energy <= 1.01, \
        f"能量归一化失败：energy = {energy:.4f} 不在 [0.99, 1.01] 范围内"

    return waveform, time_axis


def compute_psd(
    waveform: np.ndarray,
    sampling_rate: float,
    freq_resolution: float = 10e6,
    freq_range: Tuple[float, float] = (0, 12e9),
    power_scale_db: float = -108.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算脉冲的功率谱密度（PSD）

    参数:
        waveform: 脉冲波形，1D 数组（能量归一化）
        sampling_rate: 采样率（Hz）
        freq_resolution: 频率分辨率（Hz），默认 10 MHz
        freq_range: 频率范围（Hz），默认 (0, 12 GHz)
        power_scale_db: 功率缩放因子（dB），默认 -108 dB
                       对应真实系统的 PRF × 脉冲能量组合效应
                       典型值：-108 dB（PRF=10MHz, E=50pJ）

    返回:
        frequencies: 频率轴（Hz）
        psd_dbm_per_mhz: 功率谱密度（dBm/MHz）

    前置条件:
        - waveform 为 1D 数组（能量归一化：∫|waveform|²dt = 1）
        - freq_resolution ≥ 10 MHz

    后置条件:
        - len(frequencies) ≥ 1200（0-12 GHz @ 10 MHz 分辨率）

    算法（参见 research.md Section 1.3 & 2.2）:
        1. FFT 点数：n_fft = 2^ceil(log2(sampling_rate / freq_resolution))
        2. 实数 FFT：np.fft.rfft(waveform, n=n_fft)
        3. 功率谱密度：S(f) = |P(f)|² / (N · f_s)
        4. 应用功率缩放：S(f) *= 10^(power_scale_db/10)
        5. 转换为 dBm/MHz：10·log10(S(f) * 1e9)

    注：归一化脉冲需要乘以真实系统的 PRF×E_real 才能得到物理功率谱密度
    """
    # 参数验证
    assert waveform.ndim == 1, "waveform 必须是 1D 数组"
    assert freq_resolution >= 10e6, f"频率分辨率必须 ≥10 MHz，当前值: {freq_resolution/1e6:.2f} MHz"

    # 计算 FFT 点数（确保是 2 的幂次）
    n_fft = int(2 ** np.ceil(np.log2(sampling_rate / freq_resolution)))

    # 实数 FFT（仅计算正频率部分）
    fft_result = np.fft.rfft(waveform, n=n_fft)
    freq_axis_full = np.fft.rfftfreq(n_fft, d=1/sampling_rate)

    # 计算功率谱密度（Welch 周期图法）
    # PSD定义：S(f) = |FFT(x)|² / (N · fs)
    # 其中 N 是 FFT 点数，fs 是采样率
    #
    # 推导过程：
    # 1. 离散傅里叶变换：X[k] = Σ x[n]·exp(-j2πkn/N)
    # 2. 功率谱密度（双边）：S[k] = |X[k]|² / (N · fs)
    # 3. Parseval定理：Σ|x[n]|² = (1/N)·Σ|X[k]|²
    #
    # 参考文献：
    # - Oppenheim & Schafer, "Discrete-Time Signal Processing", Section 10.5
    # - IEEE Std 686-2017, "IEEE Standard for Radar Definitions"

    N = n_fft  # FFT 点数
    psd = np.abs(fft_result)**2 / (N * sampling_rate)  # Power Spectral Density (Watts/Hz)

    # 对于实数信号的单边谱，正频率部分需要翻倍（负频率能量折叠到正频率）
    # 但DC（k=0）和奈奎斯特（k=N/2）分量不翻倍
    psd[1:-1] *= 2

    # 应用功率缩放因子（将归一化PSD缩放到真实系统功率）
    # power_scale_db 对应 PRF × E_real 的组合效应
    power_scale_linear = 10 ** (power_scale_db / 10)
    psd = psd * power_scale_linear

    # 转换为 dBm/MHz
    # PSD 当前单位是 Watts/Hz（功率谱密度）
    # 需要转换为 dBm/MHz：
    #
    # 1. Watts/Hz → Watts/MHz：乘以 1e6（因为 1 MHz = 1e6 Hz）
    # 2. Watts → mW：乘以 1e3
    # 3. mW → dBm：10·log10(mW)
    #
    # 综合：dBm/MHz = 10·log10(PSD_Watts/Hz * 1e6 * 1e3)
    #              = 10·log10(PSD_Watts/Hz) + 10·log10(1e9)
    #              = 10·log10(PSD_Watts/Hz) + 90

    psd_watts_per_mhz = psd * 1e6  # Watts/Hz → Watts/MHz
    psd_mw_per_mhz = psd_watts_per_mhz * 1e3  # Watts → mW

    # 避免 log(0)，设置最小值
    psd_mw_per_mhz = np.maximum(psd_mw_per_mhz, 1e-30)
    psd_dbm_per_mhz = 10 * np.log10(psd_mw_per_mhz)

    # 裁剪到指定频率范围
    freq_mask = (freq_axis_full >= freq_range[0]) & (freq_axis_full <= freq_range[1])
    frequencies = freq_axis_full[freq_mask]
    psd_dbm_per_mhz = psd_dbm_per_mhz[freq_mask]

    # 后置条件验证：频率分辨率
    if freq_range == (0, 12e9):
        assert len(frequencies) >= 1200, \
            f"频率采样点不足：len(frequencies) = {len(frequencies)} < 1200"

    return frequencies, psd_dbm_per_mhz


def analyze_spectrum_shift(
    psd_2nd: np.ndarray,
    psd_5th: np.ndarray,
    frequencies: np.ndarray
) -> dict:
    """
    分析 2 阶和 5 阶脉冲的频谱对比

    参数:
        psd_2nd: 2 阶脉冲 PSD（dBm/MHz）
        psd_5th: 5 阶脉冲 PSD（dBm/MHz）
        frequencies: 频率轴（Hz）

    返回:
        analysis: 分析报告字典，包含：
            - f_peak_2nd: 2 阶脉冲峰值频率（Hz）
            - f_peak_5th: 5 阶脉冲峰值频率（Hz）
            - frequency_ratio: 频率比（f_peak_5th / f_peak_2nd）
            - report: 文本报告
    """
    # TODO: 实现频谱对比分析（将在 T018 中完成）
    raise NotImplementedError("analyze_spectrum_shift 函数尚未实现")
