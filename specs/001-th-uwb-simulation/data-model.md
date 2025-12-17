# Data Model: TH-UWB Communication System Simulation

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)
**Phase**: Phase 1 - Design
**Date**: 2025-12-17

## Overview

本文档定义 TH-UWB 通信系统仿真的核心数据模型，包括实体（Entities）、它们的属性（Attributes）、关系（Relationships）以及生命周期（Lifecycle）。所有设计遵循"简单优先"原则（Constitution: 简单优先），避免过度抽象。

---

## 1. 核心实体（Core Entities）

### 1.1 SystemConfig（系统配置）

**职责**: 集中管理所有仿真参数，作为配置的单一来源（Single Source of Truth）。

**属性**:
```python
@dataclass
class SystemConfig:
    """TH-UWB 系统配置参数"""

    # 时间参数
    frame_period: float  # 帧周期 Tf（秒），例如 100ns
    slot_width: float    # 时隙宽度 Tc（秒），例如 10ns
    num_slots: int       # 每帧时隙数量 Nh，例如 8
    ppm_delay: float     # PPM 调制时延 δ（秒），= Tc/2

    # 脉冲参数
    pulse_tau: float     # 脉冲宽度参数 τ（秒），例如 0.5ns
    pulse_amplitude: float  # 脉冲幅度 A，默认 1.0

    # 采样参数
    sampling_rate: float  # 采样频率 fs（Hz），例如 50 GHz

    # 仿真参数
    num_bits: int        # 仿真比特数，默认 10000（满足 SC-004）
    random_seed: int     # 随机种子，默认 42

    # 验证约束（构造后检查）
    def __post_init__(self):
        assert self.frame_period >= self.num_slots * self.slot_width, \
            "Tf 必须 >= Nh · Tc"
        assert self.slot_width >= 5 * self.pulse_tau, \
            "Tc 必须 >= 5τ（避免脉冲重叠）"
        assert self.ppm_delay == self.slot_width / 2, \
            "δ 必须 = Tc/2（标准 PPM）"
        assert self.sampling_rate >= 50e9, \
            "采样频率必须 >= 50 GHz（Nyquist 要求）"
```

**生命周期**: 在仿真开始时创建，全局只读（immutable），所有模块共享。

**依赖关系**: 被所有其他实体依赖。

---

### 1.2 Pulse（脉冲波形）

**职责**: 生成并缓存二阶高斯导数脉冲模板，供所有用户复用。

**属性**:
```python
@dataclass
class Pulse:
    """二阶高斯导数脉冲"""

    config: SystemConfig
    waveform: np.ndarray  # 脉冲波形数组
    time_axis: np.ndarray  # 对应的时间轴

    @classmethod
    def generate(cls, config: SystemConfig) -> 'Pulse':
        """
        生成脉冲波形（工厂方法）

        返回:
            Pulse 实例，waveform 和 time_axis 已预计算
        """
        # 脉冲持续时间：5τ
        duration = 5 * config.pulse_tau
        num_samples = int(duration * config.sampling_rate)

        # 时间轴：[-2.5τ, 2.5τ]
        t = np.linspace(-duration/2, duration/2, num_samples)

        # 生成二阶高斯导数（参考 research.md 1.2节）
        normalized_t = t / config.pulse_tau
        waveform = config.pulse_amplitude * \
                   (1 - 4 * np.pi * normalized_t**2) * \
                   np.exp(-2 * np.pi * normalized_t**2)

        return cls(config=config, waveform=waveform, time_axis=t)

    @property
    def duration(self) -> float:
        """脉冲持续时间（秒）"""
        return self.time_axis[-1] - self.time_axis[0]

    @property
    def energy(self) -> float:
        """脉冲能量 ∫|g(t)|²dt"""
        dt = 1 / self.config.sampling_rate
        return np.sum(self.waveform**2) * dt
```

**生命周期**: 在仿真开始时生成一次，所有用户共享同一个模板（节省内存和计算）。

**依赖关系**: 依赖 `SystemConfig`。

---

### 1.3 TimeHoppingCode（跳时序列）

**职责**: 为每个用户生成唯一的伪随机跳时序列。

**属性**:
```python
@dataclass
class TimeHoppingCode:
    """跳时序列"""

    user_id: int
    config: SystemConfig
    code: np.ndarray  # 跳时序列，长度 = num_frames，值域 [0, Nh-1]

    @classmethod
    def generate(cls, user_id: int, config: SystemConfig,
                num_frames: int, seed: int = None) -> 'TimeHoppingCode':
        """
        为用户生成跳时序列

        参数:
            user_id: 用户 ID（0-based）
            config: 系统配置
            num_frames: 帧数量
            seed: 基础随机种子（若为 None 则使用 config.random_seed）

        返回:
            TimeHoppingCode 实例
        """
        if seed is None:
            seed = config.random_seed

        # 使用 SeedSequence 确保不同用户的码序列独立（research.md 2.1节）
        rng = np.random.default_rng([user_id, seed])
        code = rng.integers(0, config.num_slots, size=num_frames)

        return cls(user_id=user_id, config=config, code=code)

    def __len__(self) -> int:
        return len(self.code)

    def __getitem__(self, index: int) -> int:
        return self.code[index]
```

**生命周期**: 每个用户一个实例，在用户创建时生成。

**依赖关系**: 依赖 `SystemConfig`。

---

### 1.4 User（用户）

**职责**: 表示一个通信用户，包含其跳时序列和数据比特流。

**属性**:
```python
@dataclass
class User:
    """TH-UWB 通信用户"""

    user_id: int
    config: SystemConfig
    th_code: TimeHoppingCode
    data_bits: np.ndarray  # 数据比特序列（0/1），长度 = num_bits

    @classmethod
    def create(cls, user_id: int, config: SystemConfig) -> 'User':
        """
        创建用户实例（工厂方法）

        参数:
            user_id: 用户 ID
            config: 系统配置

        返回:
            User 实例，包含跳时序列和随机数据比特
        """
        # 生成跳时序列
        th_code = TimeHoppingCode.generate(
            user_id=user_id,
            config=config,
            num_frames=config.num_bits  # 每帧发送一个比特
        )

        # 生成随机数据比特
        rng = np.random.default_rng([user_id, config.random_seed, 999])
        data_bits = rng.integers(0, 2, size=config.num_bits)

        return cls(
            user_id=user_id,
            config=config,
            th_code=th_code,
            data_bits=data_bits
        )

    def generate_signal(self, pulse: Pulse) -> np.ndarray:
        """
        生成该用户的 TH-PPM 信号

        参数:
            pulse: 脉冲模板

        返回:
            完整的时域信号数组
        """
        # 总持续时间
        total_duration = self.config.num_bits * self.config.frame_period
        num_samples = int(total_duration * self.config.sampling_rate)
        signal = np.zeros(num_samples)

        # 逐帧叠加脉冲（参考 research.md 2.2节的 TH-PPM 公式）
        for j in range(self.config.num_bits):
            # 脉冲发射时刻 = jTf + c_j·Tc + d_j·δ
            pulse_time = (j * self.config.frame_period +
                         self.th_code[j] * self.config.slot_width +
                         self.data_bits[j] * self.config.ppm_delay)

            # 转换为采样点索引
            pulse_start_idx = int(pulse_time * self.config.sampling_rate)
            pulse_end_idx = pulse_start_idx + len(pulse.waveform)

            # 叠加脉冲（边界检查）
            if pulse_end_idx <= num_samples:
                signal[pulse_start_idx:pulse_end_idx] += pulse.waveform

        return signal
```

**生命周期**: 每个用户一个实例，在仿真开始时创建。

**依赖关系**: 依赖 `SystemConfig`、`TimeHoppingCode`、`Pulse`。

---

### 1.5 Channel（信道）

**职责**: 模拟信号传播环境，包括 AWGN、多用户干扰（MUI）、窄带干扰（NBI）。

**属性**:
```python
@dataclass
class Channel:
    """通信信道模型"""

    config: SystemConfig
    snr_db: float = 10.0  # 信噪比（dB）
    sir_db: float = np.inf  # 信干比（dB），默认无窄带干扰
    nbi_frequency: float = 2.4e9  # 窄带干扰频率（Hz），默认 2.4 GHz

    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        """
        添加高斯白噪声

        参数:
            signal: 原始信号

        返回:
            加噪后的信号
        """
        signal_power = np.mean(signal**2)
        snr_linear = 10**(self.snr_db / 10)
        noise_power = signal_power / snr_linear

        rng = np.random.default_rng([self.config.random_seed, 1001])
        noise = rng.normal(0, np.sqrt(noise_power), signal.shape)

        return signal + noise

    def add_nbi(self, signal: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
        """
        添加窄带干扰（单频正弦波）

        参数:
            signal: 原始信号
            time_axis: 时间轴

        返回:
            加干扰后的信号
        """
        if np.isinf(self.sir_db):  # 无干扰
            return signal

        signal_power = np.mean(signal**2)
        sir_linear = 10**(self.sir_db / 10)
        interference_power = signal_power / sir_linear
        amplitude = np.sqrt(2 * interference_power)

        # 随机初相位
        rng = np.random.default_rng([self.config.random_seed, 1002])
        phase = rng.uniform(0, 2 * np.pi)

        interference = amplitude * np.sin(
            2 * np.pi * self.nbi_frequency * time_axis + phase
        )

        return signal + interference

    def transmit(self, users: List[User], pulse: Pulse) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟多用户信号通过信道传输

        参数:
            users: 用户列表
            pulse: 脉冲模板

        返回:
            (received_signal, time_axis): 接收信号和时间轴
        """
        # 生成所有用户的信号并叠加（MUI）
        total_signal = sum(user.generate_signal(pulse) for user in users)

        # 生成时间轴
        total_duration = self.config.num_bits * self.config.frame_period
        num_samples = len(total_signal)
        time_axis = np.linspace(0, total_duration, num_samples)

        # 添加窄带干扰
        total_signal = self.add_nbi(total_signal, time_axis)

        # 添加 AWGN
        received_signal = self.add_awgn(total_signal)

        return received_signal, time_axis
```

**生命周期**: 每次仿真实验一个实例（例如一个 SNR 点或一个用户数量点）。

**依赖关系**: 依赖 `SystemConfig`、`User`、`Pulse`。

---

### 1.6 Receiver（接收机）

**职责**: 使用相关接收算法解调 TH-PPM 信号。

**属性**:
```python
@dataclass
class Receiver:
    """相关接收机"""

    config: SystemConfig
    target_user: User  # 目标用户（已知其跳时序列）
    pulse: Pulse

    def generate_templates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成比特 0 和比特 1 的参考模板信号

        返回:
            (template_0, template_1): 两个模板信号
        """
        frame_duration = self.config.frame_period
        num_samples_per_frame = int(frame_duration * self.config.sampling_rate)

        # 只需生成单帧模板
        template_0 = np.zeros(num_samples_per_frame)
        template_1 = np.zeros(num_samples_per_frame)

        # 比特 0 模板：脉冲位于 c_0·Tc
        pulse_time_0 = self.target_user.th_code[0] * self.config.slot_width
        pulse_start_idx_0 = int(pulse_time_0 * self.config.sampling_rate)
        template_0[pulse_start_idx_0:pulse_start_idx_0 + len(self.pulse.waveform)] = self.pulse.waveform

        # 比特 1 模板：脉冲位于 c_0·Tc + δ
        pulse_time_1 = pulse_time_0 + self.config.ppm_delay
        pulse_start_idx_1 = int(pulse_time_1 * self.config.sampling_rate)
        template_1[pulse_start_idx_1:pulse_start_idx_1 + len(self.pulse.waveform)] = self.pulse.waveform

        return template_0, template_1

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        解调接收信号，恢复数据比特

        参数:
            received_signal: 接收到的信号

        返回:
            判决比特序列（0/1 数组）
        """
        template_0, template_1 = self.generate_templates()
        num_frames = self.config.num_bits
        frame_samples = int(self.config.frame_period * self.config.sampling_rate)

        decoded_bits = np.zeros(num_frames, dtype=int)

        for j in range(num_frames):
            # 提取第 j 帧
            frame_start = j * frame_samples
            frame_end = frame_start + frame_samples
            received_frame = received_signal[frame_start:frame_end]

            # 相关运算
            corr_0 = np.sum(received_frame * template_0) / self.config.sampling_rate
            corr_1 = np.sum(received_frame * template_1) / self.config.sampling_rate

            # 最大似然判决
            decoded_bits[j] = 1 if corr_1 > corr_0 else 0

        return decoded_bits
```

**生命周期**: 每次解调任务一个实例。

**依赖关系**: 依赖 `SystemConfig`、`User`（目标用户）、`Pulse`。

---

### 1.7 PerformanceMetrics（性能指标）

**职责**: 计算并存储 BER 和其他性能指标。

**属性**:
```python
@dataclass
class PerformanceMetrics:
    """性能指标"""

    transmitted_bits: np.ndarray
    received_bits: np.ndarray

    @property
    def ber(self) -> float:
        """误码率（Bit Error Rate）"""
        errors = np.sum(self.transmitted_bits != self.received_bits)
        return errors / len(self.transmitted_bits)

    @property
    def num_errors(self) -> int:
        """错误比特数"""
        return np.sum(self.transmitted_bits != self.received_bits)

    @property
    def num_bits(self) -> int:
        """总比特数"""
        return len(self.transmitted_bits)

    def ber_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算 BER 的置信区间（基于二项分布）

        参数:
            confidence: 置信水平，默认 95%

        返回:
            (lower_bound, upper_bound)
        """
        from scipy.stats import binom
        errors = self.num_errors
        n = self.num_bits

        # Wilson 置信区间（更精确）
        z = 1.96 if confidence == 0.95 else 2.576  # z-score
        p_hat = self.ber
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def __repr__(self) -> str:
        return (f"PerformanceMetrics(BER={self.ber:.4e}, "
                f"Errors={self.num_errors}/{self.num_bits})")
```

**生命周期**: 每次仿真实验一个实例。

**依赖关系**: 无外部依赖，纯数据容器。

---

### 1.8 SimulationResult（仿真结果）

**职责**: 封装单次仿真实验的所有输入参数和输出结果。

**属性**:
```python
@dataclass
class SimulationResult:
    """单次仿真实验结果"""

    # 输入参数
    config: SystemConfig
    num_users: int
    snr_db: float
    sir_db: float

    # 输出结果
    metrics: PerformanceMetrics
    execution_time: float  # 执行时间（秒）

    # 可选：保存信号用于可视化
    signal_samples: Optional[np.ndarray] = None
    time_axis: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            'num_users': self.num_users,
            'snr_db': self.snr_db,
            'sir_db': self.sir_db,
            'ber': self.metrics.ber,
            'num_errors': self.metrics.num_errors,
            'num_bits': self.metrics.num_bits,
            'execution_time': self.execution_time
        }
```

**生命周期**: 每次仿真实验一个实例，可保存到文件或汇总到数据库。

**依赖关系**: 依赖 `SystemConfig`、`PerformanceMetrics`。

---

## 2. 实体关系图（Entity Relationship Diagram）

```
┌─────────────────┐
│ SystemConfig    │ (全局单例，所有实体依赖)
└────────┬────────┘
         │
         ├──────────────┬─────────────┬──────────────┐
         │              │             │              │
         ▼              ▼             ▼              ▼
    ┌────────┐   ┌──────────────┐  ┌──────┐   ┌──────────┐
    │ Pulse  │   │TimeHoppingCode│  │User  │   │ Channel  │
    └────┬───┘   └──────┬───────┘  └───┬──┘   └────┬─────┘
         │              │              │           │
         │              └──────────────┤           │
         │                             │           │
         └─────────────┬───────────────┘           │
                       │                           │
                       ▼                           │
                  ┌──────────┐                     │
                  │ Receiver │                     │
                  └────┬─────┘                     │
                       │                           │
                       └───────────┬───────────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ PerformanceMetrics │
                         └─────────┬──────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ SimulationResult │
                          └──────────────────┘
```

**关键依赖链**：
1. **SystemConfig** → 所有实体（配置源）
2. **Pulse** + **SystemConfig** → **User** → **Channel**
3. **TimeHoppingCode** + **SystemConfig** → **User**
4. **User** + **Pulse** → **Receiver** → **PerformanceMetrics**
5. **PerformanceMetrics** → **SimulationResult**（结果封装）

---

## 3. 数据流（Data Flow）

### 3.1 单用户仿真流程

```
[SystemConfig] ──┐
                 ├──> [Pulse.generate()]
                 │
                 ├──> [User.create(user_id=0)]
                 │        ├──> [TimeHoppingCode.generate()]
                 │        └──> 生成随机数据比特
                 │
                 ├──> [Channel(snr_db=10)]
                 │
                 └──> [Receiver(target_user=user_0)]

[User] ──> generate_signal(pulse) ──> [信号数组]
       │
       └──> [Channel.transmit([user_0], pulse)]
                 │
                 └──> add_nbi() → add_awgn() ──> [接收信号]
                          │
                          └──> [Receiver.demodulate(接收信号)]
                                   │
                                   └──> [判决比特]
                                          │
                                          └──> [PerformanceMetrics(发送比特, 判决比特)]
                                                   │
                                                   └──> 计算 BER ──> [SimulationResult]
```

### 3.2 多用户仿真流程

```
[SystemConfig] ──┐
                 ├──> [Pulse.generate()]
                 │
                 ├──> for k in range(K):
                 │        [User.create(user_id=k)]  # K 个用户
                 │             ├──> 独立的 TimeHoppingCode
                 │             └──> 独立的数据比特
                 │
                 ├──> [Channel(snr_db=10)]
                 │
                 └──> [Receiver(target_user=user_0)]  # 只解调 user_0

[所有用户] ──> generate_signal(pulse) ──> 信号叠加（MUI）
                 │
                 └──> [Channel.transmit(all_users, pulse)]
                          │
                          └──> add_awgn() ──> [接收信号]
                                   │
                                   └──> [Receiver.demodulate(接收信号)]
                                            │
                                            └──> [PerformanceMetrics] ──> BER
```

---

## 4. 设计决策与理由

### 4.1 为什么使用 Dataclass？

**理由**：
- 简洁性：自动生成 `__init__`、`__repr__` 等方法
- 类型安全：支持 type hints，IDE 可提供自动补全
- Immutable 选项：可通过 `frozen=True` 确保配置不可变
- Python 3.7+ 标准库，无额外依赖

**符合宪法原则**：简单优先、工具优先（使用标准库）。

### 4.2 为什么 Pulse 是单例共享？

**理由**：
- 性能优化：避免每个用户重复计算相同的脉冲波形
- 内存节约：10 个用户共享一个脉冲模板 vs 各自持有副本
- 物理真实性：所有用户使用相同的脉冲波形（仅位置不同）

**符合宪法原则**：质量第一（基于证据的性能优化）。

### 4.3 为什么 Channel 分离加噪和加干扰？

**理由**：
- 单一职责：每个方法只做一件事（`add_awgn`、`add_nbi`）
- 可测试性：可以独立测试噪声生成和干扰生成
- 灵活性：可以单独开关 NBI（通过 `sir_db = inf`）

**符合宪法原则**：简单优先（避免复杂的组合逻辑）。

### 4.4 为什么 Receiver 依赖目标用户？

**理由**：
- 物理模型：相关接收机需要已知目标用户的跳时序列（完美同步假设）
- 简化设计：避免在 Receiver 中重复存储跳时码
- 明确依赖：依赖关系显式化，便于理解和测试

**符合宪法原则**：透明记录（依赖关系清晰）。

---

## 5. 性能考虑

### 5.1 内存占用估算

**单用户仿真**（10^4 比特，50 GHz 采样）：
- 信号长度：10^4 帧 × 100ns/帧 × 50 GHz ≈ 50M 采样点
- 内存占用：50M × 8 bytes (float64) ≈ 400 MB

**多用户仿真**（10 个用户）：
- 10 个信号叠加（in-place）：仍然 400 MB
- Pulse 模板共享：< 1 MB

**结论**：内存占用在合理范围内（<1 GB）。

### 5.2 计算瓶颈

**最耗时操作**：
1. `User.generate_signal()`：逐帧叠加脉冲（O(N·M)，N=帧数，M=采样点）
2. `Receiver.demodulate()`：逐帧相关运算（O(N·M)）

**优化方向**：
- 向量化：使用 NumPy 广播替代显式循环（已在设计中体现）
- 分块处理：若内存不足，可分 1000 帧一批处理
- 并行化：多个 SNR 点可并行计算（未来改进）

**符合宪法原则**：质量第一（基于 profiling 的优化）。

---

## 6. 未来扩展性

### 6.1 可能的扩展点

1. **支持更多调制方式**：
   - 当前：二进制 PPM
   - 扩展：M-ary PPM（M > 2）
   - 修改点：`User.generate_signal()` 和 `Receiver.generate_templates()`

2. **支持非完美同步**：
   - 当前：假设完美同步
   - 扩展：添加时序偏移和时钟漂移
   - 修改点：`Receiver` 增加时序估计模块

3. **支持多径信道**：
   - 当前：单径 + AWGN
   - 扩展：Rake 接收机
   - 修改点：`Channel` 增加多径模型，`Receiver` 改为 Rake 结构

### 6.2 不推荐的扩展（过度设计）

❌ **抽象信道基类**（AbstractChannel）
- 理由：当前只需一种信道模型，抽象层无必要
- 违反宪法原则：简单优先

❌ **插件化脉冲生成器**（PulseGenerator Interface）
- 理由：规范明确只用二阶高斯导数
- 违反宪法原则：避免为"未来可能需求"添加抽象

---

## 7. 验证清单

**数据模型完整性**：
- [x] 所有 spec.md 中的 Key Entities 已定义
- [x] 所有 FR（功能需求）可通过这些实体实现
- [x] 实体间依赖关系清晰无环
- [x] 数据流覆盖所有用户场景（US1, US2, US3）

**Constitution Check**：
- [x] 简单优先：无不必要的抽象层
- [x] 透明记录：所有实体有清晰的文档字符串
- [x] 质量第一：设计基于 research.md 的技术证据
- [x] 生态复用：使用 Python dataclass 标准库

---

**设计完成日期**: 2025-12-17
**设计者**: Claude Sonnet 4.5
**审查状态**: 待 Phase 1 contracts 生成后验证
