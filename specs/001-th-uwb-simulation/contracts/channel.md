# Contract: Channel Module

**Module**: `src/models/channel.py`
**Entities**: Channel (from [data-model.md](../data-model.md#15-channel信道))
**Version**: 1.0.0

## 1. Channel.add_awgn()

**职责**: 向信号添加高斯白噪声

### 签名

```python
def add_awgn(self, signal: np.ndarray) -> np.ndarray:
    """
    添加高斯白噪声（AWGN）

    参数:
        signal: 原始信号，shape (N,)

    返回:
        加噪后的信号，shape (N,)

    异常:
        ValueError: 如果 signal 不是 1D 数组
    """
```

### 前置条件

1. `signal` 是 1D NumPy 数组
2. `self.snr_db` 已设置（通常为 -5 到 30 dB）

### 后置条件

1. 返回数组 shape 与输入相同
2. 噪声功率满足 SNR 定义：`10·log10(P_signal / P_noise) = snr_db`（误差 < 0.1 dB）

### 测试用例

```python
def test_add_awgn_shape():
    config = get_default_config()
    channel = Channel(config=config, snr_db=10.0)
    signal = np.random.randn(10000)

    noisy_signal = channel.add_awgn(signal)
    assert noisy_signal.shape == signal.shape

def test_add_awgn_snr():
    config = get_default_config()
    channel = Channel(config=config, snr_db=10.0)
    signal = np.random.randn(100000)  # 大样本确保统计稳定

    noisy_signal = channel.add_awgn(signal)
    noise = noisy_signal - signal

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    assert np.abs(measured_snr_db - 10.0) < 0.1  # 误差 < 0.1 dB
```

---

## 2. Channel.add_nbi()

**职责**: 向信号添加窄带干扰（单频正弦波）

### 签名

```python
def add_nbi(self, signal: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    """
    添加窄带干扰（NBI）

    参数:
        signal: 原始信号，shape (N,)
        time_axis: 时间轴，shape (N,)，单位秒

    返回:
        加干扰后的信号，shape (N,)

    异常:
        ValueError: 如果 signal 和 time_axis 长度不匹配
    """
```

### 前置条件

1. `len(signal) == len(time_axis)`
2. `self.sir_db` 已设置（若为 `np.inf`则无干扰）
3. `self.nbi_frequency` 已设置（通常为 2.4 GHz）

### 后置条件

1. 如果 `sir_db == np.inf`，返回原信号（无修改）
2. 否则，干扰功率满足 SIR 定义：`10·log10(P_signal / P_interference) = sir_db`（误差 < 0.1 dB）

### 测试用例

```python
def test_add_nbi_no_interference():
    config = get_default_config()
    channel = Channel(config=config, sir_db=np.inf)
    signal = np.random.randn(1000)
    time_axis = np.linspace(0, 1e-6, 1000)

    signal_with_nbi = channel.add_nbi(signal, time_axis)
    assert np.array_equal(signal_with_nbi, signal)

def test_add_nbi_sir():
    config = get_default_config()
    channel = Channel(config=config, sir_db=10.0, nbi_frequency=2.4e9)
    signal = np.random.randn(100000)
    time_axis = np.linspace(0, 1e-5, 100000)

    signal_with_nbi = channel.add_nbi(signal, time_axis)
    interference = signal_with_nbi - signal

    signal_power = np.mean(signal**2)
    interference_power = np.mean(interference**2)
    measured_sir_db = 10 * np.log10(signal_power / interference_power)

    assert np.abs(measured_sir_db - 10.0) < 0.1  # 误差 < 0.1 dB

def test_add_nbi_frequency():
    config = get_default_config()
    channel = Channel(config=config, sir_db=0.0, nbi_frequency=2.4e9)
    signal = np.zeros(100000)  # 纯干扰，无信号
    fs = 50e9
    time_axis = np.arange(100000) / fs

    interference_signal = channel.add_nbi(signal, time_axis)

    # FFT 验证频率
    from scipy.fft import fft, fftfreq
    freqs = fftfreq(len(interference_signal), 1/fs)
    spectrum = np.abs(fft(interference_signal))

    # 找峰值频率
    peak_idx = np.argmax(spectrum[:len(spectrum)//2])
    peak_freq = np.abs(freqs[peak_idx])

    assert np.abs(peak_freq - 2.4e9) / 2.4e9 < 0.01  # 相对误差 < 1%
```

---

## 3. Channel.transmit()

**职责**: 模拟多用户信号通过信道传输

### 签名

```python
def transmit(
    self,
    users: List[User],
    pulse: Pulse
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟多用户信号通过信道传输

    参数:
        users: 用户列表，长度 [1, 20]
        pulse: 脉冲模板

    返回:
        (received_signal, time_axis): 接收信号和时间轴

    异常:
        ValueError: 如果 users 为空或超过 20 个
        ConfigurationError: 如果用户配置不一致
    """
```

### 前置条件

1. `len(users) >= 1`
2. 所有用户的 `config` 与 `self.config` 一致
3. `pulse.config` 与 `self.config` 一致

### 后置条件

1. `len(received_signal) == len(time_axis)`
2. 接收信号包含所有用户的信号叠加（MUI）+ NBI + AWGN
3. 时间轴范围 [0, num_bits * frame_period]

### 测试用例

```python
def test_channel_transmit_single_user():
    config = get_default_config()
    channel = Channel(config=config, snr_db=10.0, sir_db=np.inf)
    user = User.create(user_id=0, config=config)
    pulse = Pulse.generate(config)

    received_signal, time_axis = channel.transmit([user], pulse)

    assert len(received_signal) == len(time_axis)
    assert time_axis[0] == 0
    assert time_axis[-1] <= config.num_bits * config.frame_period

def test_channel_transmit_multi_user():
    config = get_default_config()
    channel = Channel(config=config, snr_db=10.0, sir_db=np.inf)
    users = [User.create(user_id=k, config=config) for k in range(3)]
    pulse = Pulse.generate(config)

    received_signal, time_axis = channel.transmit(users, pulse)

    # 多用户信号功率应大于单用户
    single_user_signal = users[0].generate_signal(pulse)
    single_user_power = np.mean(single_user_signal**2)
    multi_user_power = np.mean(received_signal**2)

    # 近似检查：多用户功率 > 单用户功率（考虑噪声）
    assert multi_user_power > single_user_power * 0.8
```

### 性能契约

- 时间复杂度：O(K·N·M)，K = num_users，N = num_bits，M = len(pulse)
- 对于 K=10 用户，N=10,000 比特，执行时间 < 30 秒

---

**契约版本**: 1.0.0
**参考研究**: [research.md](../research.md#3-信道建模)
**最后更新**: 2025-12-17
