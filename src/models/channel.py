"""
信道模块

实现 TH-UWB 通信系统的信道模型，包括加性高斯白噪声（AWGN）、窄带干扰（NBI）和多用户信号传输。
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from src.config import SystemConfig
from src.models.pulse import Pulse


@dataclass
class Channel:
    """
    TH-UWB 通信信道

    模拟信号通过无线信道的传输过程，包括：
    - 多用户干扰（MUI）：多个用户信号叠加
    - 加性高斯白噪声（AWGN）：模拟环境噪声
    - 窄带干扰（NBI）：模拟强窄带干扰源（如 WiFi）

    属性:
        config: 系统配置参数
        snr_db: 信噪比（dB），范围 [-5, 30]，np.inf 表示无噪声
        sir_db: 信干比（dB），范围 [-10, 40]，np.inf 表示无干扰
        nbi_frequency: 窄带干扰中心频率（Hz），通常为 2.4 GHz

    示例:
        >>> config = SystemConfig(...)
        >>> channel = Channel(config=config, snr_db=10.0, sir_db=20.0, nbi_frequency=2.4e9)
        >>> users = [User.create(user_id=k, config=config) for k in range(3)]
        >>> pulse = Pulse.generate(config)
        >>> received_signal, time_axis = channel.transmit(users, pulse)

    参考:
        - data-model.md Section 1.5: Channel 实体定义
        - contracts/channel.md: Channel 契约
    """

    config: SystemConfig
    snr_db: float
    sir_db: float = np.inf  # 默认无干扰
    nbi_frequency: float = 2.4e9  # 默认 2.4 GHz

    def __post_init__(self):
        """验证信道参数"""
        if self.snr_db < -10 and self.snr_db != np.inf:
            raise ValueError(f"snr_db 必须 >= -10 或 np.inf，当前值: {self.snr_db}")
        if self.sir_db < -20 and self.sir_db != np.inf:
            raise ValueError(f"sir_db 必须 >= -20 或 np.inf，当前值: {self.sir_db}")
        if self.nbi_frequency <= 0:
            raise ValueError(f"nbi_frequency 必须 > 0，当前值: {self.nbi_frequency}")

    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        """
        添加高斯白噪声（AWGN）

        根据指定的 SNR 向信号添加加性高斯白噪声。

        参数:
            signal: 原始信号，shape (N,)

        返回:
            加噪后的信号，shape (N,)

        前置条件:
            - signal 是 1D NumPy 数组
            - self.snr_db 已设置（-10 到 30 dB 或 np.inf）

        后置条件:
            - 返回数组 shape 与输入相同
            - 噪声功率满足 SNR 定义：10·log10(P_signal / P_noise) = snr_db（误差 < 0.1 dB）

        异常:
            ValueError: 如果 signal 不是 1D 数组

        示例:
            >>> channel = Channel(config=config, snr_db=10.0)
            >>> signal = np.random.randn(10000)
            >>> noisy_signal = channel.add_awgn(signal)

        参考:
            - contracts/channel.md Section 1: add_awgn() 契约
            - research.md Section 3.1: AWGN 信道模型
        """
        # 验证输入
        if signal.ndim != 1:
            raise ValueError(f"signal 必须是 1D 数组，当前维度: {signal.ndim}")

        # 如果 SNR 为无穷大，不添加噪声
        if self.snr_db == np.inf:
            return signal.copy()

        # 计算信号功率
        signal_power = np.mean(signal**2)

        # 计算噪声功率：P_noise = P_signal / 10^(SNR_dB/10)
        noise_power = signal_power / (10 ** (self.snr_db / 10))

        # 生成高斯白噪声（标准差 = sqrt(noise_power)）
        noise = np.random.randn(len(signal)) * np.sqrt(noise_power)

        return signal + noise

    def add_nbi(self, signal: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
        """
        添加窄带干扰（NBI）

        在指定频率上添加单频正弦波干扰，模拟强窄带干扰源（如 WiFi、蓝牙）。

        参数:
            signal: 原始信号，shape (N,)
            time_axis: 时间轴，shape (N,)，单位秒

        返回:
            加干扰后的信号，shape (N,)

        前置条件:
            - len(signal) == len(time_axis)
            - self.sir_db 已设置（若为 np.inf 则无干扰）
            - self.nbi_frequency 已设置（通常为 2.4 GHz）

        后置条件:
            - 如果 sir_db == np.inf，返回原信号（无修改）
            - 否则，干扰功率满足 SIR 定义：10·log10(P_signal / P_interference) = sir_db（误差 < 0.1 dB）

        异常:
            ValueError: 如果 signal 和 time_axis 长度不匹配

        示例:
            >>> channel = Channel(config=config, sir_db=10.0, nbi_frequency=2.4e9)
            >>> signal = np.random.randn(10000)
            >>> time_axis = np.arange(10000) / config.sampling_rate
            >>> signal_with_nbi = channel.add_nbi(signal, time_axis)

        参考:
            - contracts/channel.md Section 2: add_nbi() 契约
            - research.md Section 3.2: 窄带干扰模型
        """
        # 验证输入
        if len(signal) != len(time_axis):
            raise ValueError(
                f"signal 和 time_axis 长度必须一致，"
                f"len(signal)={len(signal)}, len(time_axis)={len(time_axis)}"
            )

        # 如果 SIR 为无穷大，不添加干扰
        if self.sir_db == np.inf:
            return signal.copy()

        # 计算信号功率
        signal_power = np.mean(signal**2)

        # 计算干扰功率：P_interference = P_signal / 10^(SIR_dB/10)
        interference_power = signal_power / (10 ** (self.sir_db / 10))

        # 生成单频正弦波干扰：A·sin(2πft)
        # 正弦波的功率 = A²/2，因此 A = sqrt(2·P_interference)
        amplitude = np.sqrt(2 * interference_power)
        interference = amplitude * np.sin(2 * np.pi * self.nbi_frequency * time_axis)

        return signal + interference

    def transmit(
        self, users: List["User"], pulse: Pulse
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟多用户信号通过信道传输

        聚合多个用户的信号，模拟多用户干扰（MUI），然后依次添加窄带干扰（NBI）和高斯白噪声（AWGN）。

        信道传输流程：
        1. 聚合所有用户信号（MUI）：Σ s_k(t)
        2. 添加窄带干扰：s(t) + I(t)
        3. 添加高斯白噪声：s(t) + I(t) + n(t)

        参数:
            users: 用户列表，长度 [1, 20]
            pulse: 脉冲模板

        返回:
            (received_signal, time_axis): 接收信号和时间轴
                - received_signal: shape (N,)，包含 MUI + NBI + AWGN
                - time_axis: shape (N,)，范围 [0, num_bits * frame_period]

        前置条件:
            - len(users) >= 1 and <= 20
            - 所有用户的 config 与 self.config 一致
            - pulse.config 与 self.config 一致

        后置条件:
            - len(received_signal) == len(time_axis)
            - 接收信号包含所有用户的信号叠加（MUI）+ NBI + AWGN
            - 时间轴范围 [0, num_bits * frame_period]

        异常:
            ValueError: 如果 users 为空或超过 20 个
            ValueError: 如果用户配置不一致

        示例:
            >>> config = SystemConfig(..., num_bits=1000)
            >>> channel = Channel(config=config, snr_db=10.0, sir_db=20.0)
            >>> users = [User.create(user_id=k, config=config) for k in range(3)]
            >>> pulse = Pulse.generate(config)
            >>> received_signal, time_axis = channel.transmit(users, pulse)

        性能:
            - 时间复杂度：O(K·N·M)，K = num_users，N = num_bits，M = len(pulse)
            - 对于 K=10 用户，N=10,000 比特，执行时间 < 30 秒

        参考:
            - contracts/channel.md Section 3: transmit() 契约
            - research.md Section 3.3: 多用户干扰（MUI）建模
        """
        # 验证输入
        if len(users) == 0:
            raise ValueError("users 列表不能为空")
        if len(users) > 20:
            raise ValueError(f"users 列表最多支持 20 个用户，当前: {len(users)}")

        # 验证所有用户配置一致
        for user in users:
            if user.config is not self.config:
                raise ValueError(
                    f"用户 {user.user_id} 的 config 与信道 config 不一致"
                )

        # 验证 pulse 配置一致
        if pulse.config is not self.config:
            raise ValueError("pulse.config 必须与 channel.config 一致")

        # 生成时间轴
        total_duration = self.config.num_bits * self.config.frame_period
        num_samples = int(total_duration * self.config.sampling_rate)
        time_axis = np.arange(num_samples) / self.config.sampling_rate

        # 1. 聚合所有用户信号（MUI）
        aggregated_signal = np.zeros(num_samples)
        for user in users:
            user_signal = user.generate_signal(pulse)
            aggregated_signal += user_signal

        # 2. 添加窄带干扰（NBI）
        signal_with_nbi = self.add_nbi(aggregated_signal, time_axis)

        # 3. 添加高斯白噪声（AWGN）
        received_signal = self.add_awgn(signal_with_nbi)

        return received_signal, time_axis
