# Contract: Modulation Module

**Module**: `src/models/modulation.py`
**Entities**: TimeHoppingCode, User (from [data-model.md](../data-model.md))
**Version**: 1.0.0

## 1. generate_th_code()

**职责**: 为用户生成伪随机跳时序列

### 签名

```python
def generate_th_code(
    user_id: int,
    num_frames: int,
    num_hops: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    生成跳时序列

    参数:
        user_id: 用户 ID（0-based），范围 [0, 100]
        num_frames: 帧数量，范围 [1, 10^6]
        num_hops: 每帧可用时隙数量 Nh，范围 [2, 64]
        rng: NumPy 随机数生成器

    返回:
        跳时序列数组，shape (num_frames,)，值域 [0, num_hops-1]

    异常:
        ValueError: 如果参数超出范围
        TypeError: 如果 rng 不是 Generator 类型
    """
```

### 前置条件

1. `user_id >= 0`
2. `num_frames > 0`
3. `num_hops >= 2`（至少 2 个时隙才能跳时）
4. `rng` 必须是 `np.random.Generator` 实例

### 后置条件

1. 返回数组长度 = `num_frames`
2. 所有元素满足 `0 <= code[i] < num_hops`
3. 数组元素的分布近似均匀（卡方检验 p-value > 0.05）

### 测试用例

```python
def test_generate_th_code_length():
    rng = np.random.default_rng(42)
    code = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng)
    assert len(code) == 100

def test_generate_th_code_range():
    rng = np.random.default_rng(42)
    code = generate_th_code(user_id=0, num_frames=1000, num_hops=8, rng=rng)
    assert np.all(code >= 0) and np.all(code < 8)

def test_generate_th_code_uniformity():
    rng = np.random.default_rng(42)
    code = generate_th_code(user_id=0, num_frames=10000, num_hops=8, rng=rng)
    # 卡方检验：均匀性
    from scipy.stats import chisquare
    observed = np.bincount(code, minlength=8)
    expected = np.full(8, 10000/8)
    _, p_value = chisquare(observed, expected)
    assert p_value > 0.05  # 不拒绝均匀分布假设

def test_generate_th_code_independence():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    code1 = generate_th_code(user_id=0, num_frames=100, num_hops=8, rng=rng1)
    code2 = generate_th_code(user_id=1, num_frames=100, num_hops=8, rng=rng2)
    # 不同用户的码序列应不同
    assert not np.array_equal(code1, code2)
```

---

## 2. User.generate_signal()

**职责**: 生成用户的 TH-PPM 调制信号

### 签名

```python
def generate_signal(self, pulse: Pulse) -> np.ndarray:
    """
    生成该用户的 TH-PPM 信号

    参数:
        pulse: 脉冲模板对象

    返回:
        完整的时域信号数组，长度 = num_bits * frame_period * sampling_rate

    异常:
        ValueError: 如果 pulse 与 config 不兼容
    """
```

### 前置条件

1. `pulse.config` 与 `self.config` 一致（同一个 SystemConfig 实例）
2. `self.data_bits` 长度 = `self.config.num_bits`
3. `self.th_code` 长度 = `self.config.num_bits`

### 后置条件

1. 返回信号长度 = `int(config.num_bits * config.frame_period * config.sampling_rate)`
2. 信号功率有限：`np.mean(signal**2) < inf`
3. 信号包含 `num_bits` 个脉冲，每个脉冲位于对应帧的跳时时隙

### 数学公式

```
s(t) = Σ[j=0 to Nf-1] g''(t - jTf - c_j·Tc - d_j·δ)
```

其中：
- `c_j = self.th_code[j]`（跳时码）
- `d_j = self.data_bits[j]`（数据比特）
- `δ = self.config.ppm_delay`（PPM 时延）

### 测试用例

```python
def test_user_generate_signal_length():
    config = get_default_config()
    user = User.create(user_id=0, config=config)
    pulse = Pulse.generate(config)

    signal = user.generate_signal(pulse)
    expected_length = int(config.num_bits * config.frame_period * config.sampling_rate)
    assert len(signal) == expected_length

def test_user_generate_signal_power():
    config = get_default_config()
    user = User.create(user_id=0, config=config)
    pulse = Pulse.generate(config)

    signal = user.generate_signal(pulse)
    power = np.mean(signal**2)
    assert power > 0 and power < np.inf

def test_user_generate_signal_reproducibility():
    config = get_default_config()
    config.random_seed = 42

    user1 = User.create(user_id=0, config=config)
    pulse = Pulse.generate(config)
    signal1 = user1.generate_signal(pulse)

    # 使用相同种子重新生成
    user2 = User.create(user_id=0, config=config)
    signal2 = user2.generate_signal(pulse)

    assert np.array_equal(signal1, signal2)
```

### 性能契约

- 时间复杂度：O(N·M)，N = num_bits，M = len(pulse.waveform)
- 对于 N=10,000 比特，执行时间 < 5 秒（单用户）

---

## 3. 使用示例

```python
from src.models.modulation import generate_th_code, User
from src.models.pulse import Pulse
from src.config import SystemConfig
import numpy as np

# 初始化配置和脉冲
config = SystemConfig(
    frame_period=100e-9,
    slot_width=10e-9,
    num_slots=8,
    ppm_delay=5e-9,
    pulse_tau=0.5e-9,
    pulse_amplitude=1.0,
    sampling_rate=50e9,
    num_bits=100,
    random_seed=42
)

pulse = Pulse.generate(config)

# 创建用户
user = User.create(user_id=0, config=config)
print(f"用户 {user.user_id} 跳时序列前 10 个: {user.th_code.code[:10]}")
print(f"用户 {user.user_id} 数据比特前 10 个: {user.data_bits[:10]}")

# 生成信号
signal = user.generate_signal(pulse)
print(f"信号长度: {len(signal)} 采样点")
print(f"信号功率: {np.mean(signal**2):.6f}")

# 可视化前 3 帧
import matplotlib.pyplot as plt

frames_to_plot = 3
samples_per_frame = int(config.frame_period * config.sampling_rate)
signal_snippet = signal[:frames_to_plot * samples_per_frame]
t_snippet = np.arange(len(signal_snippet)) / config.sampling_rate

plt.figure(figsize=(12, 4))
plt.plot(t_snippet * 1e9, signal_snippet, linewidth=0.8)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title(f'TH-PPM Signal (User {user.user_id}, First {frames_to_plot} Frames)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

**契约版本**: 1.0.0
**参考研究**: [research.md](../research.md#2-th-ppm-调制原理)
**最后更新**: 2025-12-17
