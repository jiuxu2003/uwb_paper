"""
Receiver 模块单元测试

测试相关接收机的模板生成和解调功能。
"""

import pytest
import numpy as np
from src.simulation.receiver import Receiver
from src.models.pulse import Pulse
from src.models.modulation import User
from src.models.channel import Channel
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
def user(config):
    """测试用户"""
    return User.create(user_id=0, config=config)


def test_receiver_initialization(config, pulse):
    """测试 Receiver 初始化"""
    receiver = Receiver(config=config, target_user=0, pulse=pulse)

    assert receiver.config == config
    assert receiver.target_user == 0
    assert receiver.pulse == pulse
    assert receiver.template_0 is not None
    assert receiver.template_1 is not None


def test_receiver_generate_templates(config, pulse):
    """测试模板生成"""
    receiver = Receiver(config=config, target_user=0, pulse=pulse)

    # 验证模板形状
    samples_per_frame = int(config.frame_period * config.sampling_rate)
    assert len(receiver.template_0) == samples_per_frame
    assert len(receiver.template_1) == samples_per_frame

    # 验证模板不全为零
    assert np.any(receiver.template_0 != 0), "template_0 should contain pulse"
    assert np.any(receiver.template_1 != 0), "template_1 should contain pulse"

    # 验证两个模板不同（PPM 偏移）
    assert not np.array_equal(
        receiver.template_0, receiver.template_1
    ), "template_0 and template_1 should differ by PPM delay"


def test_receiver_template_ppm_delay(config, pulse):
    """
    测试模板的 PPM 偏移正确性

    template_1 应该比 template_0 延迟 δ = ppm_delay
    """
    receiver = Receiver(config=config, target_user=0, pulse=pulse)

    # 计算 PPM 偏移的采样点数
    ppm_offset_samples = int(config.ppm_delay * config.sampling_rate)

    # 找到 template_0 中脉冲的起始位置
    nonzero_0 = np.where(receiver.template_0 != 0)[0]
    if len(nonzero_0) > 0:
        pulse_start_0 = nonzero_0[0]

        # 找到 template_1 中脉冲的起始位置
        nonzero_1 = np.where(receiver.template_1 != 0)[0]
        if len(nonzero_1) > 0:
            pulse_start_1 = nonzero_1[0]

            # 验证偏移量
            actual_offset = pulse_start_1 - pulse_start_0
            assert actual_offset == ppm_offset_samples, (
                f"PPM offset should be {ppm_offset_samples} samples, "
                f"got {actual_offset} samples"
            )


def test_receiver_demodulate_perfect_channel(config, pulse, user):
    """
    测试理想信道下的解调（无噪声、无干扰）

    在无噪声条件下，解调应该完全正确（BER = 0）
    """
    # 生成信号（无噪声）
    signal = user.generate_signal(pulse)

    # 接收解调
    receiver = Receiver(config=config, target_user=0, pulse=pulse)
    decoded_bits = receiver.demodulate(signal, user.th_code)

    # 验证解调正确性
    assert len(decoded_bits) == len(user.data_bits)
    errors = np.sum(decoded_bits != user.data_bits)
    ber = errors / len(user.data_bits)

    # 理想信道下应该完全正确
    assert ber == 0.0, f"Expected BER=0 in perfect channel, got BER={ber}"


def test_receiver_demodulate_with_awgn(config, pulse, user):
    """
    测试 AWGN 信道下的解调

    在中等 SNR（10 dB）下，BER 应该较低（< 0.1）
    """
    # 信道传输（SNR = 10 dB）
    channel = Channel(config=config, snr_db=10.0, sir_db=np.inf)
    received_signal, _ = channel.transmit([user], pulse)

    # 接收解调
    receiver = Receiver(config=config, target_user=0, pulse=pulse)
    decoded_bits = receiver.demodulate(received_signal, user.th_code)

    # 验证解调结果
    assert len(decoded_bits) == len(user.data_bits)
    errors = np.sum(decoded_bits != user.data_bits)
    ber = errors / len(user.data_bits)

    # 在 SNR=10dB 下，BER 应该较低
    assert ber < 0.1, f"Expected BER < 0.1 at SNR=10dB, got BER={ber}"


def test_receiver_demodulate_output_format(config, pulse, user):
    """测试解调输出格式"""
    signal = user.generate_signal(pulse)
    receiver = Receiver(config=config, target_user=0, pulse=pulse)
    decoded_bits = receiver.demodulate(signal, user.th_code)

    # 验证输出类型和形状
    assert isinstance(decoded_bits, np.ndarray)
    assert decoded_bits.shape == (config.num_bits,)
    assert decoded_bits.dtype == int

    # 验证比特值为 0 或 1
    assert np.all(np.isin(decoded_bits, [0, 1]))


def test_receiver_demodulate_validation(config, pulse, user):
    """测试解调输入验证"""
    receiver = Receiver(config=config, target_user=0, pulse=pulse)

    # 创建错误长度的跳时码
    from src.models.modulation import TimeHoppingCode

    short_config = SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=3,  # 不同的比特数
        random_seed=42,
    )
    wrong_th_code = TimeHoppingCode.generate(
        user_id=0, config=short_config, num_frames=3
    )  # 长度不匹配

    signal = user.generate_signal(pulse)

    # 跳时码长度必须等于 num_bits
    with pytest.raises(ValueError, match="跳时码长度必须等于比特数"):
        receiver.demodulate(signal, wrong_th_code)


def test_receiver_demodulate_multi_user_interference(config, pulse):
    """
    测试多用户干扰条件下的解调

    多用户场景下 BER 应该比单用户更高
    """
    # 创建多个用户
    users = [User.create(user_id=k, config=config) for k in range(3)]

    # 信道传输（3 个用户，SNR=10dB）
    channel = Channel(config=config, snr_db=10.0, sir_db=np.inf)
    received_signal, _ = channel.transmit(users, pulse)

    # 解调第一个用户的数据
    receiver = Receiver(config=config, target_user=0, pulse=pulse)
    decoded_bits = receiver.demodulate(received_signal, users[0].th_code)

    # 计算 BER
    errors = np.sum(decoded_bits != users[0].data_bits)
    ber_multi = errors / len(users[0].data_bits)

    # 对比单用户场景
    signal_single, _ = channel.transmit([users[0]], pulse)
    decoded_single = receiver.demodulate(signal_single, users[0].th_code)
    errors_single = np.sum(decoded_single != users[0].data_bits)
    ber_single = errors_single / len(users[0].data_bits)

    # 多用户 BER 应该 >= 单用户 BER（允许相等是因为样本量小）
    assert ber_multi >= ber_single, (
        f"Multi-user BER ({ber_multi}) should be >= single-user BER ({ber_single})"
    )


def test_receiver_demodulate_reproducibility(config, pulse, user):
    """测试解调可重复性"""
    # 生成相同的噪声信号
    np.random.seed(42)
    channel1 = Channel(config=config, snr_db=10.0)
    signal1, _ = channel1.transmit([user], pulse)

    receiver = Receiver(config=config, target_user=0, pulse=pulse)
    decoded1 = receiver.demodulate(signal1, user.th_code)

    # 重复相同过程
    np.random.seed(42)
    channel2 = Channel(config=config, snr_db=10.0)
    signal2, _ = channel2.transmit([user], pulse)
    decoded2 = receiver.demodulate(signal2, user.th_code)

    # 结果应该相同
    np.testing.assert_array_equal(decoded1, decoded2)


def test_receiver_correlation_decision(config, pulse):
    """
    测试相关检测判决逻辑

    手动构造信号，验证相关检测是否选择正确的比特
    """
    receiver = Receiver(config=config, target_user=0, pulse=pulse)

    samples_per_frame = int(config.frame_period * config.sampling_rate)

    # 构造一个帧信号，包含比特 0（无 PPM 偏移）
    frame_bit_0 = receiver.template_0.copy()

    # 计算相关值
    correlation_0 = np.sum(frame_bit_0 * receiver.template_0) / config.sampling_rate
    correlation_1 = np.sum(frame_bit_0 * receiver.template_1) / config.sampling_rate

    # 比特 0 应该与 template_0 相关性更高
    assert correlation_0 > correlation_1, (
        f"Bit 0 should correlate better with template_0: "
        f"corr_0={correlation_0}, corr_1={correlation_1}"
    )

    # 构造一个帧信号，包含比特 1（有 PPM 偏移）
    frame_bit_1 = receiver.template_1.copy()

    # 计算相关值
    correlation_0 = np.sum(frame_bit_1 * receiver.template_0) / config.sampling_rate
    correlation_1 = np.sum(frame_bit_1 * receiver.template_1) / config.sampling_rate

    # 比特 1 应该与 template_1 相关性更高
    assert correlation_1 > correlation_0, (
        f"Bit 1 should correlate better with template_1: "
        f"corr_0={correlation_0}, corr_1={correlation_1}"
    )


def test_receiver_edge_case_short_signal(config, pulse, user):
    """测试边界情况：接收信号长度不足"""
    # 创建较短的配置
    short_config = SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=10,
        random_seed=42,
    )

    user_short = User.create(user_id=0, config=short_config)
    pulse_short = Pulse.generate(short_config)
    signal_short = user_short.generate_signal(pulse_short)

    # 截断信号（模拟信号长度不足）
    truncated_signal = signal_short[: len(signal_short) // 2]

    # 解调应该能处理（用零填充）
    receiver = Receiver(config=short_config, target_user=0, pulse=pulse_short)
    decoded_bits = receiver.demodulate(truncated_signal, user_short.th_code)

    # 应该返回完整长度的解调结果
    assert len(decoded_bits) == short_config.num_bits
