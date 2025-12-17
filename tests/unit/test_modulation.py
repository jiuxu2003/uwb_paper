"""
Modulation 模块单元测试

测试 TH-PPM 调制方案的跳时序列生成和用户信号生成功能。
"""

import pytest
import numpy as np
from scipy.stats import chisquare
from src.config import SystemConfig
from src.models.pulse import Pulse
from src.models.modulation import generate_th_code, TimeHoppingCode, User


def get_default_config() -> SystemConfig:
    """获取默认测试配置"""
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


class TestGenerateThCode:
    """测试 generate_th_code() 函数"""

    def test_generate_th_code_length(self):
        """
        T022: 验证跳时码长度正确

        跳时码数组长度应等于帧数量。
        """
        rng = np.random.default_rng(42)
        num_frames = 100
        code = generate_th_code(user_id=0, num_frames=num_frames, num_hops=8, rng=rng)

        assert len(code) == num_frames, f"跳时码长度应为 {num_frames}，实际: {len(code)}"

    def test_generate_th_code_range(self):
        """验证跳时码值域范围"""
        rng = np.random.default_rng(42)
        num_hops = 8
        code = generate_th_code(user_id=0, num_frames=1000, num_hops=num_hops, rng=rng)

        assert np.all(code >= 0), "跳时码所有值应 >= 0"
        assert np.all(code < num_hops), f"跳时码所有值应 < {num_hops}"

    def test_generate_th_code_uniformity(self):
        """
        T023: 验证跳时码分布均匀性（卡方检验）

        长序列的跳时码应服从均匀分布（p-value > 0.05）。
        """
        rng = np.random.default_rng(42)
        num_hops = 8
        num_frames = 10000  # 大样本确保统计有效性
        code = generate_th_code(user_id=0, num_frames=num_frames, num_hops=num_hops, rng=rng)

        # 统计每个时隙出现次数
        observed = np.bincount(code, minlength=num_hops)
        expected = np.full(num_hops, num_frames / num_hops)

        # 卡方检验：H0 = 均匀分布
        _, p_value = chisquare(observed, expected)

        assert (
            p_value > 0.05
        ), f"跳时码应服从均匀分布（p-value > 0.05），实际 p-value={p_value:.4f}"

    def test_generate_th_code_independence(self):
        """
        验证跳时码随机性

        使用不同 RNG 实例应生成不同序列（除非种子完全相同）。
        """
        # 使用不同种子的 RNG
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)  # 不同种子

        code1 = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng1)
        code2 = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng2)

        # 不同种子应生成不同序列
        assert not np.array_equal(
            code1, code2
        ), "不同 RNG 种子应生成不同跳时码"

    def test_generate_th_code_reproducibility(self):
        """验证跳时码的可重复性"""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        code1 = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng1)
        code2 = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng2)

        # 相同 user_id 和种子应生成相同序列
        assert np.array_equal(code1, code2), "相同参数应生成相同跳时码（可重复性）"

    def test_generate_th_code_input_validation(self):
        """验证输入参数验证"""
        rng = np.random.default_rng(42)

        # 测试非 Generator 类型
        with pytest.raises(TypeError):
            generate_th_code(user_id=0, num_frames=100, num_hops=8, rng="not_a_generator")

        # 测试负 user_id
        with pytest.raises(ValueError):
            generate_th_code(user_id=-1, num_frames=100, num_hops=8, rng=rng)

        # 测试超大 user_id
        with pytest.raises(ValueError):
            generate_th_code(user_id=101, num_frames=100, num_hops=8, rng=rng)

        # 测试非正 num_frames
        with pytest.raises(ValueError):
            generate_th_code(user_id=0, num_frames=0, num_hops=8, rng=rng)

        # 测试超大 num_frames
        with pytest.raises(ValueError):
            generate_th_code(user_id=0, num_frames=10**7, num_hops=8, rng=rng)

        # 测试 num_hops < 2
        with pytest.raises(ValueError):
            generate_th_code(user_id=0, num_frames=100, num_hops=1, rng=rng)

        # 测试 num_hops > 64
        with pytest.raises(ValueError):
            generate_th_code(user_id=0, num_frames=100, num_hops=65, rng=rng)


class TestTimeHoppingCode:
    """测试 TimeHoppingCode 类"""

    def test_timehoppingcode_generate(self):
        """验证 TimeHoppingCode.generate() 工厂方法"""
        config = get_default_config()
        th_code = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)

        assert isinstance(th_code, TimeHoppingCode), "应返回 TimeHoppingCode 实例"
        assert th_code.user_id == 0, "用户 ID 应为 0"
        assert len(th_code) == 100, "跳时码长度应为 100"
        assert len(th_code.code) == 100, "code 数组长度应为 100"

    def test_timehoppingcode_indexing(self):
        """验证 TimeHoppingCode 索引访问"""
        config = get_default_config()
        th_code = TimeHoppingCode.generate(user_id=0, config=config, num_frames=10)

        # 测试索引访问
        for i in range(10):
            assert (
                th_code[i] == th_code.code[i]
            ), f"索引 {i} 访问应等于 code[{i}]"

    def test_timehoppingcode_seed_consistency(self):
        """验证相同种子生成相同跳时码"""
        config = get_default_config()

        th_code1 = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)
        th_code2 = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)

        assert np.array_equal(
            th_code1.code, th_code2.code
        ), "相同 user_id 和 seed 应生成相同跳时码"

    def test_timehoppingcode_user_independence(self):
        """验证不同用户的跳时码独立"""
        config = get_default_config()

        th_code0 = TimeHoppingCode.generate(user_id=0, config=config, num_frames=100)
        th_code1 = TimeHoppingCode.generate(user_id=1, config=config, num_frames=100)

        # 不同用户应有不同的跳时码（即使使用相同基础种子）
        assert not np.array_equal(
            th_code0.code, th_code1.code
        ), "不同用户的跳时码应不同（通过 SeedSequence 确保独立性）"


class TestUser:
    """测试 User 类"""

    def test_user_create(self):
        """验证 User.create() 工厂方法"""
        config = get_default_config()
        user = User.create(user_id=0, config=config)

        assert isinstance(user, User), "应返回 User 实例"
        assert user.user_id == 0, "用户 ID 应为 0"
        assert len(user.th_code) == config.num_bits, "跳时码长度应等于比特数"
        assert len(user.data_bits) == config.num_bits, "数据比特长度应等于比特数"

        # 验证数据比特为二进制
        assert np.all((user.data_bits == 0) | (user.data_bits == 1)), "数据比特应为 0 或 1"

    def test_user_generate_signal_length(self):
        """
        T024: 验证生成信号长度正确

        信号长度应等于 num_bits × frame_period × sampling_rate。
        """
        config = get_default_config()
        user = User.create(user_id=0, config=config)
        pulse = Pulse.generate(config)

        signal = user.generate_signal(pulse)

        expected_length = int(
            config.num_bits * config.frame_period * config.sampling_rate
        )
        assert (
            len(signal) == expected_length
        ), f"信号长度应为 {expected_length}，实际: {len(signal)}"

    def test_user_generate_signal_power(self):
        """验证生成信号功率有限"""
        config = get_default_config()
        user = User.create(user_id=0, config=config)
        pulse = Pulse.generate(config)

        signal = user.generate_signal(pulse)
        power = np.mean(signal**2)

        assert power > 0, "信号功率应为正"
        assert not np.isinf(power), "信号功率应有限"
        assert not np.isnan(power), "信号功率不应为 NaN"

    def test_user_generate_signal_reproducibility(self):
        """
        T025: 验证信号生成的可重复性

        相同配置和种子应生成相同信号。
        """
        config = get_default_config()
        pulse = Pulse.generate(config)

        # 第一次生成
        user1 = User.create(user_id=0, config=config)
        signal1 = user1.generate_signal(pulse)

        # 第二次生成（相同参数）
        user2 = User.create(user_id=0, config=config)
        signal2 = user2.generate_signal(pulse)

        # 验证完全一致
        assert np.array_equal(
            signal1, signal2
        ), "相同参数应生成相同信号（可重复性）"

    def test_user_generate_signal_pulse_config_validation(self):
        """验证 pulse 配置一致性检查"""
        config1 = get_default_config()
        config2 = SystemConfig(
            frame_period=200e-9,  # 不同配置
            slot_width=20e-9,
            num_slots=8,
            ppm_delay=10e-9,
            pulse_tau=1.0e-9,
            pulse_amplitude=1.0,
            sampling_rate=50e9,
            num_bits=1000,
            random_seed=42,
        )

        user = User.create(user_id=0, config=config1)
        pulse = Pulse.generate(config2)  # 不同配置的脉冲

        with pytest.raises(ValueError):
            user.generate_signal(pulse)  # 应抛出异常

    def test_user_different_users_different_signals(self):
        """验证不同用户生成不同信号"""
        config = get_default_config()
        pulse = Pulse.generate(config)

        user0 = User.create(user_id=0, config=config)
        user1 = User.create(user_id=1, config=config)

        signal0 = user0.generate_signal(pulse)
        signal1 = user1.generate_signal(pulse)

        # 不同用户的信号应不同（不同跳时码和数据比特）
        assert not np.array_equal(
            signal0, signal1
        ), "不同用户应生成不同信号"

    def test_user_signal_contains_pulses(self):
        """验证信号包含预期数量的脉冲"""
        config = SystemConfig(
            frame_period=100e-9,
            slot_width=10e-9,
            num_slots=8,
            ppm_delay=5e-9,
            pulse_tau=0.5e-9,
            pulse_amplitude=1.0,
            sampling_rate=50e9,
            num_bits=10,  # 少量比特便于验证
            random_seed=42,
        )

        user = User.create(user_id=0, config=config)
        pulse = Pulse.generate(config)
        signal = user.generate_signal(pulse)

        # 验证信号不全为零（包含脉冲）
        assert not np.allclose(signal, 0), "信号应包含非零脉冲"

        # 验证信号峰值与脉冲幅度相近
        peak_signal = np.max(np.abs(signal))
        peak_pulse = np.max(np.abs(pulse.waveform))
        assert (
            peak_signal >= peak_pulse * 0.9
        ), "信号峰值应接近或等于脉冲峰值"
