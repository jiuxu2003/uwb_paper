"""
性能指标模块

提供 BER 计算、统计分析和仿真结果封装功能。
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from src.config import SystemConfig


@dataclass
class PerformanceMetrics:
    """
    性能指标（Performance Metrics）

    计算并存储通信系统的误码率（BER）和相关统计指标。

    属性:
        transmitted_bits: 发送的原始比特序列，shape (num_bits,)，元素为 0 或 1
        received_bits: 接收并解调后的比特序列，shape (num_bits,)，元素为 0 或 1

    示例:
        >>> transmitted = np.array([0, 1, 1, 0, 1])
        >>> received = np.array([0, 1, 0, 0, 1])  # 第 3 位出错
        >>> metrics = PerformanceMetrics(transmitted, received)
        >>> metrics.ber
        0.2
        >>> metrics.num_errors
        1
        >>> metrics.num_bits
        5
        >>> metrics.ber_confidence_interval(confidence=0.95)
        (0.007, 0.716)  # Wilson 95% 置信区间

    参考:
        - data-model.md Section 1.7: PerformanceMetrics 实体定义
        - research.md Section 5.1: BER 计算统计方法
    """

    transmitted_bits: np.ndarray
    received_bits: np.ndarray

    def __post_init__(self):
        """验证输入数据"""
        # 验证输入数组维度
        if self.transmitted_bits.ndim != 1:
            raise ValueError(
                f"transmitted_bits 必须是 1D 数组，当前维度: {self.transmitted_bits.ndim}"
            )
        if self.received_bits.ndim != 1:
            raise ValueError(
                f"received_bits 必须是 1D 数组，当前维度: {self.received_bits.ndim}"
            )

        # 验证长度一致
        if len(self.transmitted_bits) != len(self.received_bits):
            raise ValueError(
                f"transmitted_bits 和 received_bits 长度必须一致，"
                f"len(transmitted)={len(self.transmitted_bits)}, "
                f"len(received)={len(self.received_bits)}"
            )

        # 验证比特值合法性
        if not np.all(np.isin(self.transmitted_bits, [0, 1])):
            raise ValueError("transmitted_bits 必须只包含 0 或 1")
        if not np.all(np.isin(self.received_bits, [0, 1])):
            raise ValueError("received_bits 必须只包含 0 或 1")

    @property
    def ber(self) -> float:
        """
        误码率（Bit Error Rate）

        定义：BER = 错误比特数 / 总发送比特数

        返回:
            误码率，范围 [0, 1]

        示例:
            >>> metrics.ber
            0.001  # 千分之一的误码率
        """
        errors = np.sum(self.transmitted_bits != self.received_bits)
        return float(errors / len(self.transmitted_bits))

    @property
    def num_errors(self) -> int:
        """
        错误比特数

        返回:
            发送与接收不一致的比特数量

        示例:
            >>> metrics.num_errors
            10  # 10 个比特出错
        """
        return int(np.sum(self.transmitted_bits != self.received_bits))

    @property
    def num_bits(self) -> int:
        """
        总比特数

        返回:
            发送的总比特数量

        示例:
            >>> metrics.num_bits
            10000  # 发送了 10000 个比特
        """
        return len(self.transmitted_bits)

    def ber_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算 BER 的置信区间（基于二项分布理论）

        使用 Wilson 置信区间方法，相比简单的正态近似更精确，特别是在样本量较小或
        成功率接近 0 或 1 时。

        算法：
        对于二项分布（n 次试验，p 成功率），Wilson 置信区间为：
        ```
        center = (p_hat + z²/(2n)) / (1 + z²/n)
        margin = z·sqrt(p_hat(1-p_hat)/n + z²/(4n²)) / (1 + z²/n)
        CI = [center - margin, center + margin]
        ```
        其中：
        - p_hat = observed_errors / n（观测误码率）
        - z = 1.96 (95% 置信水平) 或 2.576 (99% 置信水平)

        参数:
            confidence: 置信水平，范围 (0, 1)，默认 0.95
                       支持 0.95（95%）和 0.99（99%）

        返回:
            (lower_bound, upper_bound): BER 的置信区间下界和上界

        前置条件:
            - confidence 必须为 0.95 或 0.99

        后置条件:
            - 返回区间 [lower, upper] 满足 0 ≤ lower ≤ upper ≤ 1
            - 真实 BER 以 confidence 概率落在区间内

        异常:
            ValueError: 如果 confidence 不是 0.95 或 0.99

        示例:
            >>> metrics = PerformanceMetrics(transmitted=..., received=...)
            >>> metrics.ber
            0.001
            >>> metrics.ber_confidence_interval(confidence=0.95)
            (0.0008, 0.0012)  # 95% 置信区间
            >>> metrics.ber_confidence_interval(confidence=0.99)
            (0.0007, 0.0013)  # 99% 置信区间（更宽）

        参考:
            - data-model.md Section 1.7: Wilson 置信区间公式
            - research.md Section 5.1: 统计稳定性要求
            - Wilson, E.B. (1927). "Probable inference, the law of succession,
              and statistical inference". Journal of the American Statistical
              Association. 22 (158): 209–212.
        """
        # 验证置信水平
        if confidence not in [0.95, 0.99]:
            raise ValueError(f"confidence 必须为 0.95 或 0.99，当前值: {confidence}")

        # 获取统计量
        errors = self.num_errors
        n = self.num_bits
        p_hat = self.ber

        # 确定 z-score（标准正态分布分位数）
        z = 1.96 if confidence == 0.95 else 2.576

        # 计算 Wilson 置信区间
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = (
            z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
        )

        # 确保区间在 [0, 1] 范围内
        lower_bound = max(0.0, center - margin)
        upper_bound = min(1.0, center + margin)

        return (lower_bound, upper_bound)

    def __repr__(self) -> str:
        """
        字符串表示

        示例:
            >>> print(metrics)
            PerformanceMetrics(BER=1.0000e-03, Errors=10/10000)
        """
        return (
            f"PerformanceMetrics(BER={self.ber:.4e}, "
            f"Errors={self.num_errors}/{self.num_bits})"
        )


@dataclass
class SimulationResult:
    """
    单次仿真实验结果（Simulation Result）

    封装一次完整仿真实验的所有输入参数和输出结果，便于批量分析和结果序列化。

    属性:
        config: 系统配置参数
        num_users: 用户数量（1-20）
        snr_db: 信噪比（dB），范围 [-10, 30]，np.inf 表示无噪声
        sir_db: 信干比（dB），范围 [-20, 40]，np.inf 表示无干扰
        metrics: 性能指标对象（包含 BER 等）
        execution_time: 仿真执行时间（秒）
        signal_samples: （可选）保存的信号样本，用于可视化，shape (N,)
        time_axis: （可选）信号的时间轴，shape (N,)，单位秒

    示例:
        >>> config = SystemConfig(...)
        >>> metrics = PerformanceMetrics(transmitted=..., received=...)
        >>> result = SimulationResult(
        ...     config=config,
        ...     num_users=3,
        ...     snr_db=10.0,
        ...     sir_db=np.inf,
        ...     metrics=metrics,
        ...     execution_time=12.34
        ... )
        >>> result.to_dict()
        {
            'num_users': 3,
            'snr_db': 10.0,
            'sir_db': inf,
            'ber': 0.001,
            'num_errors': 10,
            'num_bits': 10000,
            'execution_time': 12.34
        }

    参考:
        - data-model.md Section 1.8: SimulationResult 实体定义
    """

    config: SystemConfig
    num_users: int
    snr_db: float
    sir_db: float
    metrics: PerformanceMetrics
    execution_time: float
    signal_samples: Optional[np.ndarray] = None
    time_axis: Optional[np.ndarray] = None

    def __post_init__(self):
        """验证输入参数"""
        # 验证用户数量
        if self.num_users < 1 or self.num_users > 20:
            raise ValueError(f"num_users 必须在 [1, 20]，当前值: {self.num_users}")

        # 验证 SNR
        if self.snr_db < -10 and self.snr_db != np.inf:
            raise ValueError(f"snr_db 必须 >= -10 或 np.inf，当前值: {self.snr_db}")

        # 验证 SIR（扩展到-50dB以支持极强干扰测试）
        if self.sir_db < -50 and self.sir_db != np.inf:
            raise ValueError(f"sir_db 必须 >= -50 或 np.inf，当前值: {self.sir_db}")

        # 验证执行时间
        if self.execution_time < 0:
            raise ValueError(f"execution_time 必须 >= 0，当前值: {self.execution_time}")

        # 验证信号样本和时间轴的一致性
        if self.signal_samples is not None and self.time_axis is not None:
            if len(self.signal_samples) != len(self.time_axis):
                raise ValueError(
                    f"signal_samples 和 time_axis 长度必须一致，"
                    f"len(signal)={len(self.signal_samples)}, "
                    f"len(time)={len(self.time_axis)}"
                )

    def to_dict(self) -> dict:
        """
        转换为字典（用于序列化）

        将仿真结果转换为 Python 字典，便于：
        - 保存到 JSON/YAML 文件
        - 存储到数据库
        - 生成报告
        - 批量分析

        返回:
            字典，包含主要仿真参数和结果

        后置条件:
            - 返回的字典只包含可序列化的基本类型（int, float, str）
            - 不包含 NumPy 数组和复杂对象

        示例:
            >>> result_dict = result.to_dict()
            >>> import json
            >>> json.dumps(result_dict)
            '{"num_users": 3, "snr_db": 10.0, ...}'

        参考:
            - data-model.md Section 1.8: to_dict() 方法定义
        """
        return {
            "num_users": self.num_users,
            "snr_db": self.snr_db,
            "sir_db": self.sir_db,
            "ber": self.metrics.ber,
            "num_errors": self.metrics.num_errors,
            "num_bits": self.metrics.num_bits,
            "execution_time": self.execution_time,
        }

    def __repr__(self) -> str:
        """
        字符串表示

        示例:
            >>> print(result)
            SimulationResult(Users=3, SNR=10.0dB, SIR=inf dB, BER=1.0000e-03, Time=12.34s)
        """
        return (
            f"SimulationResult(Users={self.num_users}, "
            f"SNR={self.snr_db}dB, SIR={self.sir_db}dB, "
            f"BER={self.metrics.ber:.4e}, Time={self.execution_time:.2f}s)"
        )
