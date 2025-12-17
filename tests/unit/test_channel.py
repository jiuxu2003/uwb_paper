"""
Channel 模块单元测试

测试 AWGN、NBI 和多用户信号传输功能。
"""

import pytest
import numpy as np
from src.models.channel import Channel
from src.models.pulse import Pulse
from src.models.modulation import User
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
        num_bits=100,  # 使用较小的比特数加快测试
        random_seed=42,
    )


@pytest.fixture
def pulse(config):
    """测试脉冲"""
    return Pulse.generate(config)


@pytest.fixture
def users(config):
    """创建多个测试用户"""
    return [User.create(user_id=k, config=config) for k in range(3)]


def test_channel_initialization(config):
    """测试 Channel 初始化"""
    # 正常初始化
    channel = Channel(config=config, snr_db=10.0, sir_db=20.0, nbi_frequency=2.4e9)
    assert channel.snr_db == 10.0
    assert channel.sir_db == 20.0
    assert channel.nbi_frequency == 2.4e9

    # 测试默认值
    channel2 = Channel(config=config, snr_db=10.0)
    assert channel2.sir_db == np.inf
    assert channel2.nbi_frequency == 2.4e9


def test_channel_validation(config):
    """测试 Channel 参数验证"""
    # SNR 过低
    with pytest.raises(ValueError, match="snr_db 必须 >= -10"):
        Channel(config=config, snr_db=-15.0)

    # SIR 过低
    with pytest.raises(ValueError, match="sir_db 必须 >= -20"):
        Channel(config=config, snr_db=10.0, sir_db=-25.0)

    # NBI 频率非法
    with pytest.raises(ValueError, match="nbi_frequency 必须 > 0"):
        Channel(config=config, snr_db=10.0, nbi_frequency=-1.0)


def test_add_awgn_snr(config):
    """
    测试 AWGN 噪声功率满足 SNR 定义

    验证：10·log10(P_signal / P_noise) = snr_db（误差 < 0.5 dB）
    """
    # 创建测试信号
    signal = np.random.randn(10000)
    signal_power = np.mean(signal**2)

    # 测试不同 SNR
    for snr_db in [0, 5, 10, 15, 20]:
        channel = Channel(config=config, snr_db=snr_db)
        noisy_signal = channel.add_awgn(signal)

        # 计算噪声
        noise = noisy_signal - signal
        noise_power = np.mean(noise**2)

        # 计算实际 SNR
        actual_snr_db = 10 * np.log10(signal_power / noise_power)

        # 验证 SNR 误差 < 0.5 dB
        assert abs(actual_snr_db - snr_db) < 0.5, (
            f"SNR mismatch: expected {snr_db} dB, got {actual_snr_db:.2f} dB"
        )


def test_add_awgn_no_noise(config):
    """测试 SNR = inf 时不添加噪声"""
    signal = np.random.randn(1000)
    channel = Channel(config=config, snr_db=np.inf)
    noisy_signal = channel.add_awgn(signal)

    # 信号应该完全相同（除了浮点误差）
    np.testing.assert_array_almost_equal(noisy_signal, signal, decimal=10)


def test_add_awgn_validation(config):
    """测试 add_awgn 输入验证"""
    channel = Channel(config=config, snr_db=10.0)

    # 输入必须是 1D 数组
    with pytest.raises(ValueError, match="signal 必须是 1D 数组"):
        channel.add_awgn(np.zeros((10, 10)))


def test_add_nbi_sir(config):
    """
    测试 NBI 干扰功率满足 SIR 定义

    验证：10·log10(P_signal / P_interference) = sir_db（误差 < 0.5 dB）
    """
    # 创建测试信号
    num_samples = 10000
    signal = np.random.randn(num_samples)
    time_axis = np.arange(num_samples) / config.sampling_rate
    signal_power = np.mean(signal**2)

    # 测试不同 SIR
    for sir_db in [10, 20, 30]:
        channel = Channel(config=config, snr_db=np.inf, sir_db=sir_db)
        signal_with_nbi = channel.add_nbi(signal, time_axis)

        # 计算干扰
        interference = signal_with_nbi - signal
        interference_power = np.mean(interference**2)

        # 计算实际 SIR
        actual_sir_db = 10 * np.log10(signal_power / interference_power)

        # 验证 SIR 误差 < 0.5 dB
        assert abs(actual_sir_db - sir_db) < 0.5, (
            f"SIR mismatch: expected {sir_db} dB, got {actual_sir_db:.2f} dB"
        )


def test_add_nbi_frequency(config):
    """
    测试 NBI 干扰信号基本特性

    验证：
    1. 干扰信号不全为零
    2. 干扰信号是振荡的（有正负值）
    3. 干扰功率符合 SIR 定义
    """
    # 创建测试信号（使用非零信号，因为 SIR 是相对于信号功率定义的）
    num_samples = 10000
    signal = np.random.randn(num_samples) * 0.1  # 小幅度随机信号
    signal_power = np.mean(signal**2)
    time_axis = np.arange(num_samples) / config.sampling_rate

    # 设置 2.4 GHz 干扰，SIR = 10 dB
    nbi_freq = 2.4e9
    sir_db = 10.0
    channel = Channel(config=config, snr_db=np.inf, sir_db=sir_db, nbi_frequency=nbi_freq)
    signal_with_nbi = channel.add_nbi(signal, time_axis)

    # 计算干扰
    interference = signal_with_nbi - signal

    # 验证 1：干扰信号不全为零
    assert np.any(interference != 0), "NBI should produce non-zero interference"

    # 验证 2：干扰信号是振荡的（有正负值）
    assert np.any(interference > 0) and np.any(interference < 0), (
        "NBI should be oscillatory (sinusoidal)"
    )

    # 验证 3：干扰功率符合 SIR 定义
    interference_power = np.mean(interference**2)
    actual_sir_db = 10 * np.log10(signal_power / interference_power)
    assert abs(actual_sir_db - sir_db) < 0.5, (
        f"SIR mismatch: expected {sir_db} dB, got {actual_sir_db:.2f} dB"
    )


def test_add_nbi_no_interference(config):
    """测试 SIR = inf 时不添加干扰"""
    num_samples = 1000
    signal = np.random.randn(num_samples)
    time_axis = np.arange(num_samples) / config.sampling_rate

    channel = Channel(config=config, snr_db=np.inf, sir_db=np.inf)
    signal_with_nbi = channel.add_nbi(signal, time_axis)

    # 信号应该完全相同（除了浮点误差）
    np.testing.assert_array_almost_equal(signal_with_nbi, signal, decimal=10)


def test_add_nbi_validation(config):
    """测试 add_nbi 输入验证"""
    channel = Channel(config=config, snr_db=10.0, sir_db=10.0)

    signal = np.zeros(100)
    time_axis = np.arange(50) / config.sampling_rate  # 长度不匹配

    # signal 和 time_axis 长度必须一致
    with pytest.raises(ValueError, match="signal 和 time_axis 长度必须一致"):
        channel.add_nbi(signal, time_axis)


def test_channel_transmit_single_user(config, pulse, users):
    """测试单用户信道传输"""
    channel = Channel(config=config, snr_db=10.0, sir_db=np.inf)
    single_user = [users[0]]

    received_signal, time_axis = channel.transmit(single_user, pulse)

    # 验证输出格式
    assert isinstance(received_signal, np.ndarray)
    assert isinstance(time_axis, np.ndarray)
    assert len(received_signal) == len(time_axis)

    # 验证时间轴范围
    expected_duration = config.num_bits * config.frame_period
    assert abs(time_axis[-1] - expected_duration) < 1e-10


def test_channel_transmit_multi_user(config, pulse, users):
    """
    测试多用户信道传输

    验证：
    1. 接收信号包含所有用户的信号叠加
    2. 多用户信号功率 > 单用户信号功率
    """
    channel = Channel(config=config, snr_db=np.inf, sir_db=np.inf)

    # 单用户传输
    single_user = [users[0]]
    signal_single, _ = channel.transmit(single_user, pulse)
    power_single = np.mean(signal_single**2)

    # 多用户传输（3 个用户）
    signal_multi, _ = channel.transmit(users, pulse)
    power_multi = np.mean(signal_multi**2)

    # 多用户信号功率应该更大（因为信号叠加）
    assert power_multi > power_single, (
        f"Multi-user power ({power_multi:.2e}) should be > "
        f"single-user power ({power_single:.2e})"
    )


def test_channel_transmit_with_nbi_and_awgn(config, pulse, users):
    """测试完整信道传输（MUI + NBI + AWGN）"""
    channel = Channel(config=config, snr_db=10.0, sir_db=20.0, nbi_frequency=2.4e9)
    received_signal, time_axis = channel.transmit(users, pulse)

    # 验证输出
    assert len(received_signal) == len(time_axis)

    # 验证信号不全为零（包含信号 + 噪声 + 干扰）
    assert np.any(received_signal != 0)


def test_channel_transmit_validation(config, pulse, users):
    """测试 transmit 输入验证"""
    channel = Channel(config=config, snr_db=10.0)

    # 用户列表不能为空
    with pytest.raises(ValueError, match="users 列表不能为空"):
        channel.transmit([], pulse)

    # 用户数量不能超过 20
    too_many_users = [User.create(user_id=k, config=config) for k in range(25)]
    with pytest.raises(ValueError, match="users 列表最多支持 20 个用户"):
        channel.transmit(too_many_users, pulse)


def test_channel_transmit_config_consistency(config, pulse, users):
    """测试配置一致性验证"""
    # 创建不同配置的用户
    other_config = SystemConfig(
        frame_period=200e-9,  # 不同的帧周期
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=100,
        random_seed=42,
    )
    user_other = User.create(user_id=0, config=other_config)

    channel = Channel(config=config, snr_db=10.0)

    # 用户配置必须一致
    with pytest.raises(ValueError, match="config 与信道 config 不一致"):
        channel.transmit([user_other], pulse)


def test_channel_transmit_reproducibility(config, pulse, users):
    """测试可重复性（相同配置产生相同结果）"""
    # 由于使用 random seed，两次传输应该产生相同结果
    # 但需要重新创建 channel 以重置随机状态
    np.random.seed(42)
    channel1 = Channel(config=config, snr_db=10.0, sir_db=20.0)
    signal1, _ = channel1.transmit(users, pulse)

    np.random.seed(42)
    channel2 = Channel(config=config, snr_db=10.0, sir_db=20.0)
    signal2, _ = channel2.transmit(users, pulse)

    # 两次传输结果应该相同（除了浮点误差）
    np.testing.assert_array_almost_equal(signal1, signal2, decimal=8)
