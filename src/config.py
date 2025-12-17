"""
系统配置模块

提供 TH-UWB 仿真的配置参数管理，作为配置的单一来源（Single Source of Truth）。
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    """
    TH-UWB 系统配置参数

    集中管理所有仿真参数，确保全局一致性。所有时间单位为秒（s），所有频率单位为赫兹（Hz）。

    属性:
        frame_period: 帧周期 Tf（秒），例如 100ns
        slot_width: 时隙宽度 Tc（秒），例如 10ns
        num_slots: 每帧时隙数量 Nh，例如 8
        ppm_delay: PPM 调制时延 δ（秒），必须 = Tc/2
        pulse_tau: 脉冲宽度参数 τ（秒），例如 0.5ns
        pulse_amplitude: 脉冲幅度 A，默认 1.0
        sampling_rate: 采样频率 fs（Hz），例如 50 GHz
        num_bits: 仿真比特数，默认 10000（满足统计稳定性）
        random_seed: 随机种子，默认 42（确保可重复性）

    约束条件:
        - Tf >= Nh · Tc（帧周期必须容纳所有时隙）
        - Tc >= 5τ（避免脉冲重叠）
        - δ = Tc/2（标准 PPM 调制）
        - fs >= 50 GHz（Nyquist 采样定理）

    示例:
        >>> config = SystemConfig(
        ...     frame_period=100e-9,
        ...     slot_width=10e-9,
        ...     num_slots=8,
        ...     ppm_delay=5e-9,
        ...     pulse_tau=0.5e-9,
        ...     pulse_amplitude=1.0,
        ...     sampling_rate=50e9,
        ...     num_bits=10000,
        ...     random_seed=42
        ... )
    """

    # 时间参数（Time Parameters）
    frame_period: float  # Tf: 帧周期
    slot_width: float  # Tc: 时隙宽度
    num_slots: int  # Nh: 每帧时隙数量
    ppm_delay: float  # δ: PPM 调制时延

    # 脉冲参数（Pulse Parameters）
    pulse_tau: float  # τ: 脉冲宽度参数
    pulse_amplitude: float  # A: 脉冲幅度

    # 采样参数（Sampling Parameters）
    sampling_rate: float  # fs: 采样频率

    # 仿真参数（Simulation Parameters）
    num_bits: int  # 仿真比特数
    random_seed: int  # 随机种子

    def __post_init__(self):
        """
        验证配置参数的有效性

        异常:
            AssertionError: 当参数违反约束条件时
        """
        # 验证帧周期约束
        assert (
            self.frame_period >= self.num_slots * self.slot_width
        ), f"Tf={self.frame_period:.2e}s 必须 >= Nh·Tc={self.num_slots * self.slot_width:.2e}s"

        # 验证时隙宽度约束（避免脉冲重叠）
        assert (
            self.slot_width >= 5 * self.pulse_tau
        ), f"Tc={self.slot_width:.2e}s 必须 >= 5τ={5 * self.pulse_tau:.2e}s（避免脉冲重叠）"

        # 验证 PPM 调制时延约束
        expected_delay = self.slot_width / 2
        assert (
            abs(self.ppm_delay - expected_delay) < 1e-12
        ), f"δ={self.ppm_delay:.2e}s 必须 = Tc/2={expected_delay:.2e}s（标准 PPM）"

        # 验证采样频率约束（Nyquist 定理）
        assert (
            self.sampling_rate >= 50e9
        ), f"fs={self.sampling_rate:.2e}Hz 必须 >= 50 GHz（Nyquist 采样定理要求）"

        # 验证仿真比特数为正整数
        assert self.num_bits > 0, f"num_bits={self.num_bits} 必须为正整数"

        # 验证其他参数为正数
        assert self.frame_period > 0, "frame_period 必须为正数"
        assert self.slot_width > 0, "slot_width 必须为正数"
        assert self.num_slots > 0, "num_slots 必须为正整数"
        assert self.ppm_delay > 0, "ppm_delay 必须为正数"
        assert self.pulse_tau > 0, "pulse_tau 必须为正数"
        assert self.pulse_amplitude > 0, "pulse_amplitude 必须为正数"
