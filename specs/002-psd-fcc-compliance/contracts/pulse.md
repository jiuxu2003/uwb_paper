# Contract: Gaussian Derivative Pulse Generation

**Feature**: 002-psd-fcc-compliance
**Module**: src/models/psd.py (新增模块)
**Date**: 2025-12-17

本契约定义了高斯导数脉冲生成的接口规范，包括前置条件、后置条件和性能要求。

---

## Function Signature

```python
def generate_gaussian_derivative_pulse(
    n: int,
    tau: float,
    fc: float,
    sampling_rate: float,
    duration_factor: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成 n 阶高斯导数脉冲的时域波形

    参数:
        n: 脉冲阶数，支持 {2, 5}
        tau: 脉冲宽度参数，单位秒 (s)，范围 [0.1ns, 2ns]
        fc: 中心频率，单位赫兹 (Hz)，范围 [0, 15e9]
        sampling_rate: 采样率，单位赫兹 (Hz)，≥24e9
        duration_factor: 脉冲持续时间倍数，默认 10（即 ±5τ）

    返回:
        waveform: np.ndarray, shape (N,), 时域波形（能量归一化）
        time_axis: np.ndarray, shape (N,), 时间轴（单位：秒）

    异常:
        ValueError: 如果参数不在有效范围内
        RuntimeError: 如果能量归一化失败

    示例:
        >>> waveform, t = generate_gaussian_derivative_pulse(
        ...     n=5, tau=0.5e-9, fc=6.85e9, sampling_rate=50e9
        ... )
        >>> len(waveform)
        1000  # 取决于 sampling_rate 和 duration_factor
        >>> np.trapz(waveform**2, t)
        0.9999  # 能量归一化（误差 < 1%）
    """
    pass
```

---

## Preconditions（前置条件）

### 输入参数验证

| 参数 | 约束 | 验证方式 | 错误处理 |
|------|------|----------|----------|
| `n` | n ∈ {2, 5} | `assert n in [2, 5]` | `ValueError("n must be 2 or 5")` |
| `tau` | 0.1e-9 ≤ tau ≤ 2e-9 | `assert 0.1e-9 <= tau <= 2e-9` | `ValueError("tau must be in [0.1ns, 2ns]")` |
| `fc` | fc ≥ 0 | `assert fc >= 0` | `ValueError("fc must be non-negative")` |
| `sampling_rate` | sampling_rate ≥ 2 * max_freq | `assert sampling_rate >= 24e9` | `ValueError("sampling_rate too low")` |
| `duration_factor` | duration_factor > 0 | `assert duration_factor > 0` | `ValueError("duration_factor must be positive")` |

### 物理约束验证

1. **奈奎斯特采样定理**：
   - `sampling_rate ≥ 2 * max_frequency`
   - 其中 `max_frequency` 取决于脉冲带宽：≈ 1/tau
   - 推荐 `sampling_rate ≥ 50 * (1/tau)`（确保精度）

2. **脉冲持续时间充分性**：
   - `duration = duration_factor * tau`
   - 推荐 `duration_factor ≥ 10`（确保脉冲尾部能量 < 1%）

---

## Postconditions（后置条件）

### 输出验证

| 验证项 | 期望值 | 验证方式 | 容忍误差 |
|--------|--------|----------|----------|
| 波形长度 | `N = int(duration * sampling_rate)` | `assert len(waveform) == N` | 严格相等 |
| 时间轴长度 | `len(time_axis) == len(waveform)` | `assert len(time_axis) == len(waveform)` | 严格相等 |
| 能量归一化 | `∫|waveform|² dt = 1` | `assert abs(energy - 1.0) < 0.01` | < 1% |
| 时域对称性（n=2） | `waveform(t) ≈ waveform(-t)` | 计算对称性误差 | < 1% |
| 峰值位置（无调制） | `argmax(|waveform|) ≈ N/2` | 计算峰值偏移 | < 5% |

### 能量守恒验证

```python
# 时域能量计算
energy_time_domain = np.trapz(waveform**2, time_axis)
assert 0.99 <= energy_time_domain <= 1.01, "能量归一化失败"
```

### 频域特性验证（可选）

```python
# FFT 验证
fft_result = np.fft.rfft(waveform)
energy_freq_domain = np.sum(np.abs(fft_result)**2) / len(waveform)
assert abs(energy_freq_domain - energy_time_domain) < 0.01, "Parseval 定理验证失败"
```

---

## Algorithm Specification

### 归一化系数计算

**2 阶脉冲**：
```python
A_2 = np.sqrt(2 / (np.sqrt(np.pi) * tau))
```

**5 阶脉冲**：
```python
A_5 = np.sqrt(32 / (np.sqrt(np.pi) * tau))
```

**证明**：
通过对时域公式进行能量积分，确保 ∫|p_n(t)|² dt = 1。

### 时域波形生成

**Step 1**: 生成时间轴
```python
duration = duration_factor * tau
num_samples = int(duration * sampling_rate)
time_axis = np.linspace(-duration/2, duration/2, num_samples)
```

**Step 2**: 计算基础高斯函数
```python
gaussian = np.exp(-time_axis**2 / (2 * tau**2))
```

**Step 3**: 应用导数公式

**n=2**:
```python
waveform = A_2 * (1 - time_axis**2 / tau**2) * gaussian
```

**n=5**:
```python
term1 = 15 * time_axis / tau**2
term2 = -10 * (time_axis**3) / (tau**4)
term3 = (time_axis**5) / (tau**6)
waveform = A_5 * (term1 + term2 + term3) * gaussian
```

**Step 4**: 载波调制（如果 fc > 0）
```python
if fc > 0:
    carrier = np.cos(2 * np.pi * fc * time_axis)
    waveform = waveform * carrier
```

**Step 5**: 能量归一化验证
```python
energy = np.trapz(waveform**2, time_axis)
if abs(energy - 1.0) > 0.01:
    # 修正归一化系数
    waveform = waveform / np.sqrt(energy)
```

---

## Performance Requirements

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| 执行时间 | < 10 ms | `timeit` 模块测量 |
| 内存占用 | < 1 MB | `sys.getsizeof()` 测量 |
| 数值精度 | 相对误差 < 1e-6 | 对比解析解 |

### Benchmark Test

```python
def test_performance():
    import time
    start = time.perf_counter()
    for _ in range(100):
        generate_gaussian_derivative_pulse(
            n=5, tau=0.5e-9, fc=6.85e9, sampling_rate=50e9
        )
    elapsed = time.perf_counter() - start
    assert elapsed / 100 < 0.01, f"平均执行时间 {elapsed/100*1000:.2f} ms 超过 10 ms"
```

---

## Error Handling

### 异常类型

1. **ValueError**:
   - 原因：输入参数不在有效范围内
   - 处理：抛出明确的错误消息，指出哪个参数违规

2. **RuntimeError**:
   - 原因：能量归一化失败（理论上不应发生）
   - 处理：记录警告日志，尝试修正归一化系数

3. **MemoryError**:
   - 原因：`num_samples` 过大导致内存不足
   - 处理：捕获异常，建议减小 `duration_factor` 或降低 `sampling_rate`

### 输入修正策略

对于边界外的参数，提供自动修正（可选）：

```python
def sanitize_parameters(n, tau, fc, sampling_rate):
    # 修正 n 到最近的有效值
    if n not in [2, 5]:
        n = 2 if abs(n - 2) < abs(n - 5) else 5
        warnings.warn(f"n 修正为 {n}")

    # 修正 tau 到有效范围
    tau = np.clip(tau, 0.1e-9, 2e-9)

    # 修正 fc 到非负
    fc = max(fc, 0)

    return n, tau, fc, sampling_rate
```

---

## Testing Strategy

### Unit Tests

1. **test_pulse_energy_normalization**:
   - 验证能量归一化（n=2 和 n=5）
   - 断言：`0.99 <= energy <= 1.01`

2. **test_pulse_symmetry**:
   - 验证 2 阶脉冲的对称性（无调制时）
   - 断言：`max(|waveform(t) - waveform(-t)|) < 0.01`

3. **test_carrier_modulation**:
   - 验证载波调制（fc > 0）
   - 断言：频域峰值频率 ≈ fc

4. **test_sampling_rate_effect**:
   - 验证不同采样率下的波形一致性
   - 断言：归一化后的波形形状相同

5. **test_boundary_conditions**:
   - 验证边界参数（tau=0.1ns, tau=2ns）
   - 断言：无异常抛出，能量守恒

### Integration Tests

1. **test_psd_calculation_from_pulse**:
   - 生成脉冲 → 计算 PSD → 验证 Parseval 定理

2. **test_fcc_compliance_check**:
   - 生成 5 阶脉冲（fc=6.85 GHz） → 应判定为"Compliant"
   - 生成 2 阶脉冲（fc=0） → 应判定为"Non-compliant"

---

## Reference Implementation

**模块路径**: `src/models/psd.py`

**依赖**:
- NumPy >= 1.24.0（FFT、数学函数）
- SciPy >= 1.10.0（信号处理，可选）

**相关契约**:
- [psd.md](psd.md) - PSD 计算契约
- [fcc_mask.md](fcc_mask.md) - FCC 掩蔽罩契约

**相关文档**:
- [research.md](../research.md) Section 1.1: 时域公式推导
- [data-model.md](../data-model.md) Entity 1: Gaussian Derivative Pulse

---

**Contract Status**: ✅ 完成
**Last Updated**: 2025-12-17
