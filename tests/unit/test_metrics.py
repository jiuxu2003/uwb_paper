"""
Metrics 模块单元测试

测试 BER 计算、置信区间和仿真结果封装功能。
"""

import pytest
import numpy as np
from src.simulation.metrics import PerformanceMetrics, SimulationResult
from src.config import SystemConfig


@pytest.fixture
def config():
    """测试配置"""
    return SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=1000,
        random_seed=42,
    )


def test_performance_metrics_initialization():
    """测试 PerformanceMetrics 初始化"""
    transmitted = np.array([0, 1, 1, 0, 1, 0])
    received = np.array([0, 1, 0, 0, 1, 1])

    metrics = PerformanceMetrics(transmitted, received)

    assert isinstance(metrics.transmitted_bits, np.ndarray)
    assert isinstance(metrics.received_bits, np.ndarray)
    assert len(metrics.transmitted_bits) == 6
    assert len(metrics.received_bits) == 6


def test_performance_metrics_validation():
    """测试 PerformanceMetrics 输入验证"""
    # 长度不一致
    with pytest.raises(ValueError, match="长度必须一致"):
        PerformanceMetrics(np.array([0, 1]), np.array([0, 1, 1]))

    # 不是 1D 数组
    with pytest.raises(ValueError, match="必须是 1D 数组"):
        PerformanceMetrics(np.array([[0, 1]]), np.array([0, 1]))

    # 包含非法值
    with pytest.raises(ValueError, match="必须只包含 0 或 1"):
        PerformanceMetrics(np.array([0, 2, 1]), np.array([0, 1, 1]))


def test_metrics_ber_calculation():
    """
    测试 BER 计算正确性

    BER = 错误比特数 / 总比特数
    """
    # 场景 1：无错误
    transmitted = np.array([0, 1, 1, 0, 1])
    received = np.array([0, 1, 1, 0, 1])
    metrics = PerformanceMetrics(transmitted, received)
    assert metrics.ber == 0.0

    # 场景 2：1 个错误
    transmitted = np.array([0, 1, 1, 0, 1])
    received = np.array([0, 1, 0, 0, 1])  # 第 3 位出错
    metrics = PerformanceMetrics(transmitted, received)
    assert metrics.ber == 0.2  # 1/5 = 0.2

    # 场景 3：多个错误
    transmitted = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    received = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 1])  # 5 个错误
    metrics = PerformanceMetrics(transmitted, received)
    assert metrics.ber == 0.5  # 5/10 = 0.5

    # 场景 4：全部错误
    transmitted = np.array([0, 1, 0, 1])
    received = np.array([1, 0, 1, 0])
    metrics = PerformanceMetrics(transmitted, received)
    assert metrics.ber == 1.0


def test_metrics_num_errors():
    """测试错误比特数计算"""
    transmitted = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    received = np.array([0, 1, 0, 0, 0, 0, 1, 0])  # 第 3, 5, 8 位出错

    metrics = PerformanceMetrics(transmitted, received)
    assert metrics.num_errors == 3


def test_metrics_num_bits():
    """测试总比特数"""
    transmitted = np.zeros(10000)
    received = np.zeros(10000)

    metrics = PerformanceMetrics(transmitted, received)
    assert metrics.num_bits == 10000


def test_metrics_ber_confidence_interval_95():
    """
    测试 95% 置信区间计算

    使用已知 BER 验证置信区间合理性
    """
    # 场景：100 比特中 10 个错误，BER = 0.1
    transmitted = np.concatenate([np.ones(10), np.zeros(90)])
    received = np.concatenate([np.zeros(10), np.zeros(90)])  # 前 10 个错误

    metrics = PerformanceMetrics(transmitted, received)
    lower, upper = metrics.ber_confidence_interval(confidence=0.95)

    # 验证区间包含真实 BER
    assert lower <= metrics.ber <= upper

    # 验证区间在 [0, 1] 范围内
    assert 0 <= lower <= 1
    assert 0 <= upper <= 1

    # 验证 lower < upper
    assert lower < upper


def test_metrics_ber_confidence_interval_99():
    """测试 99% 置信区间（应该比 95% 更宽）"""
    transmitted = np.random.randint(0, 2, 1000)
    received = transmitted.copy()
    # 添加 5% 错误
    error_indices = np.random.choice(1000, 50, replace=False)
    received[error_indices] = 1 - received[error_indices]

    metrics = PerformanceMetrics(transmitted, received)

    lower_95, upper_95 = metrics.ber_confidence_interval(confidence=0.95)
    lower_99, upper_99 = metrics.ber_confidence_interval(confidence=0.99)

    # 99% 置信区间应该比 95% 更宽
    assert lower_99 <= lower_95
    assert upper_99 >= upper_95


def test_metrics_ber_confidence_interval_validation():
    """测试置信区间输入验证"""
    transmitted = np.array([0, 1, 1, 0])
    received = np.array([0, 1, 0, 0])

    metrics = PerformanceMetrics(transmitted, received)

    # 只支持 0.95 和 0.99
    with pytest.raises(ValueError, match="confidence 必须为 0.95 或 0.99"):
        metrics.ber_confidence_interval(confidence=0.90)


def test_metrics_ber_confidence_interval_edge_cases():
    """测试置信区间边界情况"""
    # 场景 1：BER = 0（无错误）
    transmitted = np.array([0, 1, 1, 0] * 25)  # 100 比特
    received = transmitted.copy()

    metrics = PerformanceMetrics(transmitted, received)
    lower, upper = metrics.ber_confidence_interval(confidence=0.95)

    assert lower == 0.0  # 下界应该是 0
    assert upper > 0.0  # 上界应该 > 0
    assert upper < 0.1  # 但不应该太大

    # 场景 2：BER = 1（全部错误）
    received_all_wrong = 1 - transmitted

    metrics = PerformanceMetrics(transmitted, received_all_wrong)
    lower, upper = metrics.ber_confidence_interval(confidence=0.95)

    assert lower < 1.0  # 下界应该 < 1
    assert np.isclose(upper, 1.0, atol=1e-10)  # 上界应该接近 1（浮点精度）


def test_metrics_repr():
    """测试字符串表示"""
    transmitted = np.array([0, 1, 1, 0, 1] * 20)  # 100 比特
    received = np.array([0, 1, 0, 0, 1] * 20)  # 20 个错误

    metrics = PerformanceMetrics(transmitted, received)
    repr_str = repr(metrics)

    # 验证包含关键信息
    assert "PerformanceMetrics" in repr_str
    assert "BER" in repr_str
    assert "Errors" in repr_str
    assert "20/100" in repr_str  # 20 个错误 / 100 比特


def test_simulation_result_initialization(config):
    """测试 SimulationResult 初始化"""
    transmitted = np.zeros(100)
    received = np.zeros(100)
    metrics = PerformanceMetrics(transmitted, received)

    result = SimulationResult(
        config=config,
        num_users=3,
        snr_db=10.0,
        sir_db=20.0,
        metrics=metrics,
        execution_time=12.34,
    )

    assert result.config == config
    assert result.num_users == 3
    assert result.snr_db == 10.0
    assert result.sir_db == 20.0
    assert result.metrics == metrics
    assert result.execution_time == 12.34
    assert result.signal_samples is None
    assert result.time_axis is None


def test_simulation_result_validation(config):
    """测试 SimulationResult 输入验证"""
    transmitted = np.zeros(100)
    received = np.zeros(100)
    metrics = PerformanceMetrics(transmitted, received)

    # 用户数量超出范围
    with pytest.raises(ValueError, match="num_users 必须在 \\[1, 20\\]"):
        SimulationResult(
            config=config,
            num_users=0,
            snr_db=10.0,
            sir_db=np.inf,
            metrics=metrics,
            execution_time=1.0,
        )

    with pytest.raises(ValueError, match="num_users 必须在 \\[1, 20\\]"):
        SimulationResult(
            config=config,
            num_users=25,
            snr_db=10.0,
            sir_db=np.inf,
            metrics=metrics,
            execution_time=1.0,
        )

    # SNR 过低
    with pytest.raises(ValueError, match="snr_db 必须 >= -10"):
        SimulationResult(
            config=config,
            num_users=3,
            snr_db=-15.0,
            sir_db=np.inf,
            metrics=metrics,
            execution_time=1.0,
        )

    # SIR 过低
    with pytest.raises(ValueError, match="sir_db 必须 >= -20"):
        SimulationResult(
            config=config,
            num_users=3,
            snr_db=10.0,
            sir_db=-25.0,
            metrics=metrics,
            execution_time=1.0,
        )

    # 执行时间为负
    with pytest.raises(ValueError, match="execution_time 必须 >= 0"):
        SimulationResult(
            config=config,
            num_users=3,
            snr_db=10.0,
            sir_db=np.inf,
            metrics=metrics,
            execution_time=-1.0,
        )


def test_simulation_result_signal_validation(config):
    """测试 SimulationResult 信号数组验证"""
    transmitted = np.zeros(100)
    received = np.zeros(100)
    metrics = PerformanceMetrics(transmitted, received)

    # 信号和时间轴长度不一致
    with pytest.raises(ValueError, match="signal_samples 和 time_axis 长度必须一致"):
        SimulationResult(
            config=config,
            num_users=3,
            snr_db=10.0,
            sir_db=np.inf,
            metrics=metrics,
            execution_time=1.0,
            signal_samples=np.zeros(100),
            time_axis=np.zeros(50),  # 长度不匹配
        )


def test_simulation_result_to_dict(config):
    """测试 to_dict 方法"""
    transmitted = np.array([0, 1, 1, 0, 1] * 20)  # 100 比特
    received = np.array([0, 1, 0, 0, 1] * 20)  # 20 个错误
    metrics = PerformanceMetrics(transmitted, received)

    result = SimulationResult(
        config=config,
        num_users=3,
        snr_db=10.0,
        sir_db=20.0,
        metrics=metrics,
        execution_time=12.34,
    )

    result_dict = result.to_dict()

    # 验证字典包含所有关键字段
    assert result_dict["num_users"] == 3
    assert result_dict["snr_db"] == 10.0
    assert result_dict["sir_db"] == 20.0
    assert result_dict["ber"] == 0.2  # 20/100
    assert result_dict["num_errors"] == 20
    assert result_dict["num_bits"] == 100
    assert result_dict["execution_time"] == 12.34

    # 验证字典可以序列化为 JSON
    import json

    json_str = json.dumps(result_dict)
    assert isinstance(json_str, str)


def test_simulation_result_repr(config):
    """测试字符串表示"""
    transmitted = np.zeros(100)
    received = np.zeros(100)
    metrics = PerformanceMetrics(transmitted, received)

    result = SimulationResult(
        config=config,
        num_users=3,
        snr_db=10.0,
        sir_db=20.0,
        metrics=metrics,
        execution_time=12.34,
    )

    repr_str = repr(result)

    # 验证包含关键信息
    assert "SimulationResult" in repr_str
    assert "Users=3" in repr_str
    assert "SNR=10.0dB" in repr_str
    assert "SIR=20.0dB" in repr_str
    assert "BER" in repr_str
    assert "Time=12.34s" in repr_str


def test_metrics_properties_are_consistent():
    """测试 PerformanceMetrics 属性一致性"""
    transmitted = np.random.randint(0, 2, 1000)
    received = transmitted.copy()

    # 添加 10% 错误
    error_indices = np.random.choice(1000, 100, replace=False)
    received[error_indices] = 1 - received[error_indices]

    metrics = PerformanceMetrics(transmitted, received)

    # 验证属性之间的一致性
    assert metrics.ber == metrics.num_errors / metrics.num_bits
    assert metrics.num_bits == len(transmitted)
    assert metrics.num_errors == np.sum(transmitted != received)
