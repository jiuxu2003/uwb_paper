# Contract: Power Spectral Density (PSD) Calculation

**Feature**: 002-psd-fcc-compliance
**Module**: src/models/psd.py
**Date**: 2025-12-17

本契约定义了从时域脉冲波形计算功率谱密度（PSD）的接口规范。

---

## Function Signature

```python
def compute_psd(
    waveform: np.ndarray,
    sampling_rate: float,
    freq_resolution: float = 10e6,
    freq_range: tuple[float, float] = (0, 12e9)
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算时域波形的功率谱密度（PSD）

    参数:
        waveform: np.ndarray, shape (N,), 时域波形
        sampling_rate: float, 采样率（Hz）
        freq_resolution: float, 频率分辨率（Hz），默认 10 MHz
        freq_range: tuple, 频率范围（Hz），默认 (0, 12 GHz)

    返回:
        frequencies: np.ndarray, shape (M,), 频率轴（Hz）
        psd_dbm_per_mhz: np.ndarray, shape (M,), PSD值（dBm/MHz）

    异常:
        ValueError: 如果 waveform 不是 1D 数组或为空
        RuntimeError: 如果 FFT 计算失败

    示例:
        >>> waveform, t = generate_gaussian_derivative_pulse(...)
        >>> freq, psd = compute_psd(waveform, sampling_rate=50e9)
        >>> len(freq)
        1200  # ≥1200 采样点（0-12 GHz, 10 MHz 分辨率）
        >>> freq[0], freq[-1]
        (0.0, 12e9)  # 频率范围
    """
    pass
```

---

## Preconditions

| 参数 | 约束 | 验证 |
|------|------|------|
| `waveform` | 1D 数组，长度 > 0 | `assert waveform.ndim == 1 and len(waveform) > 0` |
| `sampling_rate` | ≥ 24 GHz | `assert sampling_rate >= 24e9` |
| `freq_resolution` | ≥ 10 MHz（SC-002） | `assert freq_resolution >= 10e6` |
| `freq_range[1]` | ≤ sampling_rate / 2（奈奎斯特） | `assert freq_range[1] <= sampling_rate / 2` |

---

## Postconditions

| 验证项 | 期望值 | 容忍误差 |
|--------|--------|----------|
| 频率点数 | `len(frequencies) ≥ (freq_range[1] - freq_range[0]) / freq_resolution` | ±10% |
| 频率分辨率 | `frequencies[1] - frequencies[0] ≤ freq_resolution` | 严格 |
| Parseval 定理 | `∫PSD df ≈ ∫|waveform|² dt` | < 1% |
| PSD 范围 | `-100 ≤ psd_dbm_per_mhz ≤ +20` | - |

---

## Algorithm

### Step 1: FFT 点数计算

```python
# 确保 FFT 点数为 2 的幂次（高效计算）
n_fft = int(2 ** np.ceil(np.log2(sampling_rate / freq_resolution)))
n_fft = max(n_fft, len(waveform))  # 至少与波形长度相同
```

### Step 2: 执行实数 FFT

```python
fft_result = np.fft.rfft(waveform, n=n_fft)
frequencies = np.fft.rfftfreq(n_fft, d=1/sampling_rate)
```

### Step 3: 计算单边功率谱

```python
# 归一化功率谱（单边）
power_spectrum = np.abs(fft_result)**2 / (n_fft * sampling_rate)

# 除直流和奈奎斯特频率外，单边谱需乘以 2
power_spectrum[1:-1] *= 2
```

### Step 4: 转换为 dBm/MHz

```python
# 归一化到 1 MHz 带宽
psd_linear = power_spectrum * sampling_rate / 1e6  # 功率/MHz

# 转换为 dBm
psd_dbm_per_mhz = 10 * np.log10(psd_linear / 1e-3)
```

### Step 5: 截取频率范围

```python
mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
frequencies = frequencies[mask]
psd_dbm_per_mhz = psd_dbm_per_mhz[mask]
```

---

## Performance Requirements

| 指标 | 目标值 |
|------|--------|
| 执行时间 | < 1 秒（N=8192） |
| 内存占用 | < 10 MB |
| 频率分辨率 | ≤ 10 MHz |

---

## Testing Strategy

### Unit Tests

1. **test_psd_energy_conservation**:
   - 验证 Parseval 定理
   - `∫PSD df ≈ ∫|waveform|² dt`

2. **test_frequency_resolution**:
   - 验证频率分辨率满足 SC-002
   - `frequencies[1] - frequencies[0] ≤ 10e6`

3. **test_psd_range**:
   - 验证 PSD 值在合理范围内
   - `-100 ≤ psd_dbm_per_mhz ≤ +20`

---

## Reference

- [research.md](../research.md) Section 1.3: FFT 参数选择
- [data-model.md](../data-model.md) Entity 2: Power Spectral Density
- NumPy FFT 文档: https://numpy.org/doc/stable/reference/routines.fft.html

---

**Contract Status**: ✅ 完成
