"""
调制模块

实现 TH-PPM（Time-Hopping Pulse Position Modulation）调制方案，包括跳时序列生成和用户信号生成。
"""

from dataclasses import dataclass
import numpy as np
from src.config import SystemConfig
from src.models.pulse import Pulse


def generate_th_code(
    user_id: int, num_frames: int, num_hops: int, rng: np.random.Generator
) -> np.ndarray:
    """
    生成伪随机跳时序列

    每个用户使用独立的跳时序列来避免多用户干扰（MUI）。跳时码决定脉冲在每帧中的发射时隙。

    参数:
        user_id: 用户 ID（0-based），范围 [0, 100]
        num_frames: 帧数量，通常等于数据比特数，范围 [1, 10^6]
        num_hops: 每帧可用时隙数量 Nh，范围 [2, 64]
        rng: NumPy 随机数生成器（使用 SeedSequence 确保独立性）

    返回:
        跳时序列数组，shape (num_frames,)，值域 [0, num_hops-1]

    前置条件:
        - user_id >= 0
        - num_frames > 0
        - num_hops >= 2（至少 2 个时隙才能跳时）
        - rng 必须是 np.random.Generator 实例

    后置条件:
        - 返回数组长度 = num_frames
        - 所有元素满足 0 <= code[i] < num_hops
        - 数组元素分布近似均匀（长序列）

    异常:
        ValueError: 如果参数超出范围
        TypeError: 如果 rng 不是 Generator 类型

    示例:
        >>> rng = np.random.default_rng([user_id, seed])
        >>> code = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng)
        >>> len(code)
        100
        >>> np.all((code >= 0) & (code < 8))
        True

    参考:
        - research.md Section 2.1: 跳时序列生成算法
        - contracts/modulation.md: generate_th_code 契约
    """
    # 验证输入参数
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng 必须是 np.random.Generator 实例，当前类型: {type(rng)}")
    if user_id < 0:
        raise ValueError(f"user_id 必须 >= 0，当前值: {user_id}")
    if user_id > 100:
        raise ValueError(f"user_id 必须 <= 100（支持范围），当前值: {user_id}")
    if num_frames <= 0:
        raise ValueError(f"num_frames 必须 > 0，当前值: {num_frames}")
    if num_frames > 10**6:
        raise ValueError(f"num_frames 必须 <= 10^6（内存限制），当前值: {num_frames}")
    if num_hops < 2:
        raise ValueError(f"num_hops 必须 >= 2（至少 2 个时隙才能跳时），当前值: {num_hops}")
    if num_hops > 64:
        raise ValueError(f"num_hops 必须 <= 64（合理范围），当前值: {num_hops}")

    # 生成伪随机跳时序列
    code = rng.integers(0, num_hops, size=num_frames)

    return code


@dataclass
class TimeHoppingCode:
    """
    跳时序列

    为每个用户生成唯一的伪随机跳时序列，用于 TH-UWB 通信。跳时码决定脉冲在每帧中的时隙位置。

    属性:
        user_id: 用户 ID
        config: 系统配置参数
        code: 跳时序列数组，shape (num_frames,)，值域 [0, Nh-1]

    示例:
        >>> config = SystemConfig(...)
        >>> th_code = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)
        >>> len(th_code)
        100
        >>> th_code[0]  # 第一帧的跳时时隙索引
        5

    参考:
        - data-model.md Section 1.3: TimeHoppingCode 实体定义
    """

    user_id: int
    config: SystemConfig
    code: np.ndarray

    @classmethod
    def generate(
        cls, user_id: int, config: SystemConfig, num_frames: int, seed: int = None
    ) -> "TimeHoppingCode":
        """
        为用户生成跳时序列（工厂方法）

        使用 SeedSequence 确保不同用户的码序列独立且可重复。

        参数:
            user_id: 用户 ID（0-based）
            config: 系统配置
            num_frames: 帧数量，通常等于数据比特数
            seed: 基础随机种子（若为 None 则使用 config.random_seed）

        返回:
            TimeHoppingCode 实例

        示例:
            >>> config = SystemConfig(..., random_seed=42)
            >>> th_code = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)
            >>> # 相同 user_id 和 seed 生成相同序列
            >>> th_code2 = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)
            >>> np.array_equal(th_code.code, th_code2.code)
            True

        参考:
            - research.md Section 7: NumPy SeedSequence 确保独立性
        """
        if seed is None:
            seed = config.random_seed

        # 使用 SeedSequence 确保不同用户的码序列独立
        rng = np.random.default_rng([user_id, seed])
        code = generate_th_code(user_id, num_frames, config.num_slots, rng)

        return cls(user_id=user_id, config=config, code=code)

    def __len__(self) -> int:
        """返回跳时序列长度"""
        return len(self.code)

    def __getitem__(self, index: int) -> int:
        """获取指定帧的跳时时隙索引"""
        return self.code[index]


@dataclass
class User:
    """
    TH-UWB 通信用户

    表示一个通信用户，包含其跳时序列和数据比特流。每个用户通过唯一的跳时码区分。

    属性:
        user_id: 用户 ID（0-based）
        config: 系统配置参数
        th_code: 跳时序列对象
        data_bits: 数据比特序列（0/1），shape (num_bits,)

    示例:
        >>> config = SystemConfig(..., num_bits=100, random_seed=42)
        >>> user = User.create(user_id=0, config=config)
        >>> user.data_bits.shape
        (100,)
        >>> np.all((user.data_bits == 0) | (user.data_bits == 1))
        True

    参考:
        - data-model.md Section 1.4: User 实体定义
    """

    user_id: int
    config: SystemConfig
    th_code: TimeHoppingCode
    data_bits: np.ndarray

    @classmethod
    def create(cls, user_id: int, config: SystemConfig) -> "User":
        """
        创建用户实例（工厂方法）

        自动生成跳时序列和随机数据比特。使用 SeedSequence 确保可重复性。

        参数:
            user_id: 用户 ID（0-based）
            config: 系统配置

        返回:
            User 实例，包含跳时序列和随机数据比特

        示例:
            >>> config = SystemConfig(..., num_bits=100, random_seed=42)
            >>> user = User.create(user_id=0, config=config)
            >>> len(user.th_code)
            100
            >>> len(user.data_bits)
            100

        参考:
            - data-model.md Section 1.4: User.create() 工厂方法
        """
        # 生成跳时序列
        th_code = TimeHoppingCode.generate(
            user_id=user_id, config=config, num_frames=config.num_bits
        )

        # 生成随机数据比特（使用不同的种子分量确保独立性）
        rng = np.random.default_rng([user_id, config.random_seed, 999])
        data_bits = rng.integers(0, 2, size=config.num_bits)

        return cls(user_id=user_id, config=config, th_code=th_code, data_bits=data_bits)

    def generate_signal(self, pulse: Pulse) -> np.ndarray:
        """
        生成该用户的 TH-PPM 信号

        根据 TH-PPM 调制公式，逐帧叠加脉冲，脉冲位置由跳时码和数据比特共同决定。

        数学公式:
            s(t) = Σ[j=0 to Nf-1] g''(t - jTf - c_j·Tc - d_j·δ)

        其中:
            - c_j = self.th_code[j]（跳时码）
            - d_j = self.data_bits[j]（数据比特）
            - δ = self.config.ppm_delay（PPM 时延）
            - Tf = self.config.frame_period（帧周期）
            - Tc = self.config.slot_width（时隙宽度）

        参数:
            pulse: 脉冲模板对象

        返回:
            完整的时域信号数组，长度 = num_bits * frame_period * sampling_rate

        前置条件:
            - pulse.config 与 self.config 一致（同一个 SystemConfig 实例）

        后置条件:
            - 返回信号长度 = int(config.num_bits * config.frame_period * config.sampling_rate)
            - 信号功率有限: np.mean(signal**2) < inf
            - 信号包含 num_bits 个脉冲

        性能:
            - 时间复杂度: O(N·M)，N = num_bits，M = len(pulse.waveform)
            - 对于 N=10,000 比特: < 5 秒（单用户）

        异常:
            ValueError: 如果 pulse 与 config 不兼容

        示例:
            >>> config = SystemConfig(..., num_bits=100)
            >>> user = User.create(user_id=0, config=config)
            >>> pulse = Pulse.generate(config)
            >>> signal = user.generate_signal(pulse)
            >>> len(signal)
            500000  # 100 bits × 100 ns/bit × 50 GHz = 500,000 samples

        参考:
            - research.md Section 2.2: TH-PPM 调制公式
            - contracts/modulation.md: User.generate_signal() 契约
        """
        # 验证 pulse 配置一致性
        if pulse.config is not self.config:
            raise ValueError(
                "pulse.config 必须与 user.config 一致（同一个 SystemConfig 实例）"
            )

        # 计算总信号长度
        total_duration = self.config.num_bits * self.config.frame_period
        num_samples = int(total_duration * self.config.sampling_rate)
        signal = np.zeros(num_samples)

        # 逐帧叠加脉冲
        for j in range(self.config.num_bits):
            # 计算脉冲发射时刻: t = jTf + c_j·Tc + d_j·δ
            pulse_time = (
                j * self.config.frame_period  # 第 j 帧的起始时间
                + self.th_code[j] * self.config.slot_width  # 跳时时隙偏移
                + self.data_bits[j] * self.config.ppm_delay  # PPM 调制偏移
            )

            # 转换为采样点索引
            pulse_start_idx = int(pulse_time * self.config.sampling_rate)
            pulse_end_idx = pulse_start_idx + len(pulse.waveform)

            # 叠加脉冲（边界检查）
            if pulse_end_idx <= num_samples:
                signal[pulse_start_idx:pulse_end_idx] += pulse.waveform
            else:
                # 部分脉冲落在信号范围外，截断
                remaining_samples = num_samples - pulse_start_idx
                if remaining_samples > 0:
                    signal[pulse_start_idx:num_samples] += pulse.waveform[
                        :remaining_samples
                    ]

        return signal
