"""
接收机模块

实现 TH-UWB 通信系统的相关接收机，用于解调 TH-PPM 信号。
"""

from dataclasses import dataclass
import numpy as np
from src.config import SystemConfig
from src.models.pulse import Pulse


@dataclass
class Receiver:
    """
    相关接收机（Correlator Receiver）

    使用最大似然检测（Maximum Likelihood Detection）解调 TH-PPM 信号。

    工作原理：
    1. 生成两个参考模板（template_0 和 template_1）
    2. 对接收信号每一帧计算与两个模板的相关值
    3. 选择相关值更大的比特作为判决结果

    属性:
        config: 系统配置参数
        target_user: 目标用户 ID（接收机锁定的用户）
        pulse: 脉冲模板
        template_0: 比特 0 的参考模板，shape (samples_per_frame,)
        template_1: 比特 1 的参考模板，shape (samples_per_frame,)

    示例:
        >>> config = SystemConfig(...)
        >>> pulse = Pulse.generate(config)
        >>> receiver = Receiver(config=config, target_user=0, pulse=pulse)
        >>> # 假设已有接收信号和目标用户
        >>> user = User.create(user_id=0, config=config)
        >>> received_signal, time_axis = channel.transmit([user], pulse)
        >>> decoded_bits = receiver.demodulate(received_signal, user.th_code)

    参考:
        - research.md Section 4.1: 相关接收机工作原理
    """

    config: SystemConfig
    target_user: int
    pulse: Pulse
    template_0: np.ndarray = None
    template_1: np.ndarray = None

    def __post_init__(self):
        """初始化后自动生成模板"""
        self.generate_templates()

    def generate_templates(self) -> None:
        """
        生成相关接收机的参考模板

        为目标用户生成两个参考模板：
        - template_0: 对应比特 0 的理想信号（无 PPM 偏移）
        - template_1: 对应比特 1 的理想信号（有 PPM 偏移 δ）

        这两个模板用于后续的相关运算，覆盖单个帧周期。

        前置条件:
            - self.pulse 已初始化
            - self.config 已设置

        后置条件:
            - self.template_0 和 self.template_1 已生成
            - 模板长度 = samples_per_frame = int(frame_period * sampling_rate)

        参考:
            - research.md Section 4.1: 模板信号生成
        """
        # 计算单帧的采样点数
        samples_per_frame = int(self.config.frame_period * self.config.sampling_rate)

        # 初始化两个模板（全零数组）
        self.template_0 = np.zeros(samples_per_frame)
        self.template_1 = np.zeros(samples_per_frame)

        # 脉冲放置位置：跳时时隙的起始位置
        # 注意：这里使用跳时时隙 0 作为参考（实际接收时会根据目标用户的跳时码调整）
        # template_0: 无 PPM 偏移
        pulse_start_idx_0 = 0
        pulse_end_idx_0 = pulse_start_idx_0 + len(self.pulse.waveform)

        # template_1: 有 PPM 偏移 δ
        pulse_start_idx_1 = int(self.config.ppm_delay * self.config.sampling_rate)
        pulse_end_idx_1 = pulse_start_idx_1 + len(self.pulse.waveform)

        # 将脉冲复制到模板中（边界检查）
        if pulse_end_idx_0 <= samples_per_frame:
            self.template_0[pulse_start_idx_0:pulse_end_idx_0] = self.pulse.waveform
        else:
            # 脉冲超出帧边界，截断
            remaining = samples_per_frame - pulse_start_idx_0
            if remaining > 0:
                self.template_0[pulse_start_idx_0:samples_per_frame] = self.pulse.waveform[:remaining]

        if pulse_end_idx_1 <= samples_per_frame:
            self.template_1[pulse_start_idx_1:pulse_end_idx_1] = self.pulse.waveform
        else:
            remaining = samples_per_frame - pulse_start_idx_1
            if remaining > 0:
                self.template_1[pulse_start_idx_1:samples_per_frame] = self.pulse.waveform[:remaining]

    def demodulate(
        self, received_signal: np.ndarray, th_code: "TimeHoppingCode"
    ) -> np.ndarray:
        """
        解调接收信号，恢复数据比特

        对每一帧接收信号执行相关检测，判决比特值。

        算法：
        1. 将接收信号按帧周期分割
        2. 对每一帧：
           a. 根据跳时码 c_j 调整帧的起始位置
           b. 提取该帧信号
           c. 计算与 template_0 和 template_1 的相关值
           d. 选择相关值更大的比特

        参数:
            received_signal: 接收信号，shape (N,)
            th_code: 目标用户的跳时序列

        返回:
            解调后的比特序列，shape (num_bits,)，元素为 0 或 1

        前置条件:
            - len(received_signal) >= num_bits * frame_period * sampling_rate
            - len(th_code) == config.num_bits
            - self.template_0 和 self.template_1 已生成

        后置条件:
            - 返回数组长度 = config.num_bits
            - 所有元素为 0 或 1

        示例:
            >>> receiver = Receiver(config=config, target_user=0, pulse=pulse)
            >>> user = User.create(user_id=0, config=config)
            >>> channel = Channel(config=config, snr_db=10.0)
            >>> received_signal, _ = channel.transmit([user], pulse)
            >>> decoded_bits = receiver.demodulate(received_signal, user.th_code)
            >>> # 比较原始比特和解调比特
            >>> errors = np.sum(decoded_bits != user.data_bits)

        性能:
            - 时间复杂度：O(N·M)，N = num_bits，M = samples_per_frame
            - 对于 N=10,000 比特，执行时间 < 5 秒

        参考:
            - research.md Section 4.2: 相关接收机 NumPy 实现
        """
        # 验证输入
        if len(th_code) != self.config.num_bits:
            raise ValueError(
                f"跳时码长度必须等于比特数，"
                f"len(th_code)={len(th_code)}, num_bits={self.config.num_bits}"
            )

        # 计算单帧的采样点数
        samples_per_frame = int(self.config.frame_period * self.config.sampling_rate)

        # 初始化解调比特数组
        decoded_bits = np.zeros(self.config.num_bits, dtype=int)

        # 逐帧解调
        for j in range(self.config.num_bits):
            # 计算第 j 帧的起始位置（考虑跳时码 c_j）
            # frame_start = j * T_f + c_j * T_c
            frame_start_time = (
                j * self.config.frame_period + th_code[j] * self.config.slot_width
            )
            frame_start_idx = int(frame_start_time * self.config.sampling_rate)
            frame_end_idx = frame_start_idx + samples_per_frame

            # 提取该帧信号（边界检查）
            if frame_end_idx <= len(received_signal):
                frame_signal = received_signal[frame_start_idx:frame_end_idx]
            else:
                # 帧超出接收信号边界，用零填充
                remaining = len(received_signal) - frame_start_idx
                if remaining > 0:
                    frame_signal = np.zeros(samples_per_frame)
                    frame_signal[:remaining] = received_signal[frame_start_idx:]
                else:
                    frame_signal = np.zeros(samples_per_frame)

            # 相关运算（离散形式）：Λ = Σ r(t) · template(t) / fs
            correlation_0 = np.sum(frame_signal * self.template_0) / self.config.sampling_rate
            correlation_1 = np.sum(frame_signal * self.template_1) / self.config.sampling_rate

            # 最大似然判决
            decoded_bits[j] = 1 if correlation_1 > correlation_0 else 0

        return decoded_bits
