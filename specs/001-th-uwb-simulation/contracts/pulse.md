# Contract: Pulse Generation Module

**Module**: `src/models/pulse.py`
**Entities**: Pulse (from [data-model.md](../data-model.md#12-pulse脉冲波形))
**Version**: 1.0.0

## 1. generate_gaussian_doublet()

**职责**: 生成二阶高斯导数脉冲波形

### 签名

```python
def generate_gaussian_doublet(
    t: np.ndarray,
    tau: float = 0.5e-9,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    生成二阶高斯导数脉冲

    参数:
        t: 时间数组（秒），shape (N,)
        tau: 脉冲宽度参数（秒），范围 [0.1ns, 2ns]
        amplitude: 脉冲幅度，范围 [0, 10]

    返回:
        脉冲波形数组，shape (N,)，与 t 同长度

    异常:
        ValueError: 如果 tau 或 amplitude 超出范围
        TypeError: 如果 t 不是 np.ndarray
    """
```

### 前置条件（Preconditions）

1. `t` 必须是 1D NumPy 数组，dtype = float64
2. `t` 必须单调递增，无重复值
3. `tau > 0`，建议范围 [0.1ns, 2ns]
4. `amplitude > 0`，通常为 1.0

### 后置条件（Postconditions）

1. 返回数组 shape 与 `t` 完全一致
2. 脉冲在 t=0 附近达到最大值（峰值位置误差 < 0.1τ）
3. 脉冲零均值：`np.abs(np.mean(pulse)) < 1e-6`（数值误差范围内）
4. 脉冲能量有限：`np.sum(pulse**2) < inf`

### 数学公式

```
g''(t) = A · (1 - 4π(t/τ)²) · exp(-2π(t/τ)²)
```

### 测试用例

```python
def test_gaussian_doublet_zero_mean():
    t = np.linspace(-5e-9, 5e-9, 10000)
    pulse = generate_gaussian_doublet(t, tau=0.5e-9)
    assert np.abs(np.mean(pulse)) < 1e-6

def test_gaussian_doublet_peak_at_zero():
    t = np.linspace(-5e-9, 5e-9, 10000)
    pulse = generate_gaussian_doublet(t, tau=0.5e-9)
    peak_idx = np.argmax(np.abs(pulse))
    assert np.abs(t[peak_idx]) < 0.1 * 0.5e-9

def test_gaussian_doublet_invalid_tau():
    t = np.linspace(-1e-9, 1e-9, 1000)
    with pytest.raises(ValueError):
        generate_gaussian_doublet(t, tau=-0.5e-9)  # 负值
```

### 性能契约

- 时间复杂度：O(N)，N = len(t)
- 对于 N=10,000，执行时间 < 10 ms（单线程）

---

## 2. Pulse.generate() (类方法)

**职责**: 工厂方法，生成 Pulse 实例

### 签名

```python
@classmethod
def generate(cls, config: SystemConfig) -> 'Pulse':
    """
    生成脉冲实例

    参数:
        config: 系统配置对象

    返回:
        Pulse 实例，包含预计算的 waveform 和 time_axis

    异常:
        ConfigurationError: 如果 config 参数不合法
    """
```

### 前置条件

1. `config` 必须是有效的 `SystemConfig` 实例
2. `config.pulse_tau` 必须 > 0
3. `config.sampling_rate` 必须 >= 50 GHz

### 后置条件

1. 返回的 `Pulse.waveform` 长度 = int(5 * tau * fs)
2. 返回的 `Pulse.time_axis` 范围为 [-2.5τ, 2.5τ]
3. `Pulse.energy` 为有限正值

### 测试用例

```python
def test_pulse_generate():
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

    assert len(pulse.waveform) == len(pulse.time_axis)
    assert pulse.time_axis[0] < 0 < pulse.time_axis[-1]
    assert pulse.energy > 0 and pulse.energy < np.inf
```

---

## 3. Pulse.energy (属性)

**职责**: 计算脉冲能量

### 签名

```python
@property
def energy(self) -> float:
    """
    脉冲能量 ∫|g(t)|²dt

    返回:
        能量值（无量纲）
    """
```

### 后置条件

1. 返回值 > 0
2. 返回值 < inf
3. 对相同脉冲，多次调用返回相同值（幂等性）

### 测试用例

```python
def test_pulse_energy_positive():
    config = get_default_config()
    pulse = Pulse.generate(config)
    assert pulse.energy > 0

def test_pulse_energy_finite():
    config = get_default_config()
    pulse = Pulse.generate(config)
    assert pulse.energy < np.inf

def test_pulse_energy_idempotent():
    config = get_default_config()
    pulse = Pulse.generate(config)
    energy1 = pulse.energy
    energy2 = pulse.energy
    assert energy1 == energy2
```

---

## 4. 使用示例

```python
from src.models.pulse import Pulse, generate_gaussian_doublet
from src.config import SystemConfig
import numpy as np
import matplotlib.pyplot as plt

# 方式 1：直接生成脉冲波形
t = np.linspace(-5e-9, 5e-9, 10000)  # -5ns 到 5ns
pulse_waveform = generate_gaussian_doublet(t, tau=0.5e-9, amplitude=1.0)

plt.plot(t*1e9, pulse_waveform)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title('Second-Order Gaussian Derivative Pulse')
plt.grid(True)
plt.show()

# 方式 2：通过配置生成 Pulse 对象（推荐）
config = SystemConfig(
    frame_period=100e-9,
    slot_width=10e-9,
    num_slots=8,
    ppm_delay=5e-9,
    pulse_tau=0.5e-9,
    pulse_amplitude=1.0,
    sampling_rate=50e9,
    num_bits=10000,
    random_seed=42
)

pulse = Pulse.generate(config)
print(f"脉冲持续时间: {pulse.duration*1e9:.2f} ns")
print(f"脉冲能量: {pulse.energy:.6f}")

plt.plot(pulse.time_axis*1e9, pulse.waveform)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title(f'UWB Pulse (τ={config.pulse_tau*1e9:.1f}ns)')
plt.grid(True)
plt.show()
```

---

**契约版本**: 1.0.0
**参考研究**: [research.md](../research.md#1-二阶高斯导数脉冲生成gaussian-monocycle)
**最后更新**: 2025-12-17
