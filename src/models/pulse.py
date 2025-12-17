"""
脉冲波形生成模块

实现二阶高斯导数脉冲（Gaussian Monocycle / Mexican Hat Wavelet）的生成与管理。
"""

from dataclasses import dataclass
import numpy as np
from src.config import SystemConfig


def generate_gaussian_doublet(
    t: np.ndarray, tau: float = 0.5e-9, amplitude: float = 1.0
) -> np.ndarray:
    """
    生成二阶高斯导数脉冲（Mexican Hat Wavelet）

    数学公式:
        g''(t) = A · (1 - 4π(t/τ)²) · exp(-2π(t/τ)²)

    参数:
        t: 时间轴数组（秒），shape (N,)
        tau: 脉冲宽度参数 τ（秒），控制脉冲持续时间，默认 0.5ns
        amplitude: 脉冲幅度 A，默认 1.0

    返回:
        pulse: 脉冲波形数组，shape (N,)

    前置条件:
        - t 必须是 1D NumPy 数组
        - tau > 0，范围推荐 [0.1ns, 2ns]
        - amplitude > 0

    后置条件:
        - 返回数组 shape 与 t 一致
        - 零均值: |mean(pulse)| < 1e-6（由于对称性）
        - 峰值在 t=0 附近: pulse(0) = A

    示例:
        >>> t = np.linspace(-2.5e-9, 2.5e-9, 250)
        >>> pulse = generate_gaussian_doublet(t, tau=0.5e-9, amplitude=1.0)
        >>> np.abs(np.mean(pulse)) < 1e-6  # 验证零均值
        True
        >>> np.abs(pulse[125] - 1.0) < 0.01  # 验证峰值（中心点）
        True

    参考:
        - research.md Section 1.2: 二阶高斯导数脉冲公式
        - contracts/pulse.md: generate_gaussian_doublet 契约
    """
    # 验证输入
    if not isinstance(t, np.ndarray):
        raise TypeError(f"t 必须是 NumPy 数组，当前类型: {type(t)}")
    if t.ndim != 1:
        raise ValueError(f"t 必须是 1D 数组，当前维度: {t.ndim}")
    if tau <= 0:
        raise ValueError(f"tau 必须为正数，当前值: {tau}")
    if amplitude <= 0:
        raise ValueError(f"amplitude 必须为正数，当前值: {amplitude}")

    # 归一化时间
    normalized_t = t / tau

    # 计算二阶高斯导数
    pulse = amplitude * (1 - 4 * np.pi * normalized_t**2) * np.exp(
        -2 * np.pi * normalized_t**2
    )

    return pulse


@dataclass
class Pulse:
    """
    二阶高斯导数脉冲模板

    缓存预计算的脉冲波形，供所有用户复用，避免重复计算。

    属性:
        config: 系统配置参数
        waveform: 脉冲波形数组，shape (M,)，M 由采样率和脉冲持续时间决定
        time_axis: 对应的时间轴，shape (M,)，单位秒

    示例:
        >>> from src.config import SystemConfig
        >>> config = SystemConfig(
        ...     frame_period=100e-9, slot_width=10e-9, num_slots=8,
        ...     ppm_delay=5e-9, pulse_tau=0.5e-9, pulse_amplitude=1.0,
        ...     sampling_rate=50e9, num_bits=1000, random_seed=42
        ... )
        >>> pulse = Pulse.generate(config)
        >>> pulse.waveform.shape[0] == pulse.time_axis.shape[0]
        True
        >>> pulse.duration > 0
        True
        >>> pulse.energy > 0
        True
    """

    config: SystemConfig
    waveform: np.ndarray
    time_axis: np.ndarray

    @classmethod
    def generate(cls, config: SystemConfig) -> "Pulse":
        """
        生成脉冲波形（工厂方法）

        脉冲持续时间设定为 5τ（覆盖 99.7% 能量），时间轴居中于零点。

        参数:
            config: 系统配置参数

        返回:
            Pulse 实例，waveform 和 time_axis 已预计算

        后置条件:
            - len(waveform) == len(time_axis)
            - duration ≈ 5τ
            - 峰值位于 time_axis 中心
            - 零均值: |mean(waveform)| < 1e-6

        性能:
            - 时间复杂度: O(M)，M = duration × sampling_rate
            - 对于 τ=0.5ns, fs=50GHz: M ≈ 125 采样点

        参考:
            - data-model.md Section 1.2: Pulse 实体定义
            - contracts/pulse.md: Pulse.generate() 契约
        """
        # 计算脉冲持续时间：5τ（覆盖 99.7% 能量）
        duration = 5 * config.pulse_tau

        # 计算采样点数
        num_samples = int(duration * config.sampling_rate)

        # 确保至少有 3 个采样点
        if num_samples < 3:
            num_samples = 3

        # 生成时间轴：[-2.5τ, 2.5τ]，居中于零点
        t = np.linspace(-duration / 2, duration / 2, num_samples)

        # 生成二阶高斯导数脉冲
        waveform = generate_gaussian_doublet(t, config.pulse_tau, config.pulse_amplitude)

        return cls(config=config, waveform=waveform, time_axis=t)

    @property
    def duration(self) -> float:
        """
        脉冲持续时间（秒）

        返回:
            脉冲从起始到结束的时间长度

        示例:
            >>> pulse.duration
            2.5e-09  # 2.5 ns (for tau=0.5ns)
        """
        return self.time_axis[-1] - self.time_axis[0]

    @property
    def energy(self) -> float:
        """
        脉冲能量

        计算公式:
            E = ∫ |g''(t)|² dt ≈ Σ |g''[n]|² · Δt

        返回:
            脉冲能量（焦耳），用于归一化和功率计算

        后置条件:
            - energy > 0（脉冲非零）

        示例:
            >>> pulse.energy > 0
            True

        参考:
            - contracts/pulse.md: Pulse.energy 契约
        """
        # 计算采样间隔
        dt = 1 / self.config.sampling_rate

        # 数值积分：梯形法则
        energy = np.sum(self.waveform**2) * dt

        return energy
