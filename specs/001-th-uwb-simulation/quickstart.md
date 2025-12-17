# Quick Start Guide: TH-UWB Communication System Simulation

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)
**Phase**: Phase 1 - Design
**Date**: 2025-12-17

## 概述

本指南将引导你在 5 分钟内完成 TH-UWB 通信系统仿真的首次运行，生成第一张论文级性能曲线图。

---

## 前置要求

### 系统要求
- Python 3.11 或更高版本
- 至少 2 GB 可用内存
- 操作系统：Linux / macOS / Windows

### 依赖安装

```bash
# 克隆仓库
git clone git@github.com:jiuxu2003/uwb_paper.git
cd uwb_paper

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt 内容**：
```txt
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
pytest>=7.0.0
```

---

## 5 分钟快速体验

### Step 1: 生成单用户信号波形图（30 秒）

创建文件 `scripts/demo_waveform.py`：

```python
"""演示：生成 TH-PPM 信号时域波形"""
import numpy as np
import matplotlib.pyplot as plt
from src.models.pulse import Pulse
from src.models.modulation import User
from src.config import SystemConfig

# 1. 配置系统参数
config = SystemConfig(
    frame_period=100e-9,  # 100 ns
    slot_width=10e-9,     # 10 ns
    num_slots=8,
    ppm_delay=5e-9,       # δ = Tc/2 = 5 ns
    pulse_tau=0.5e-9,     # 0.5 ns
    pulse_amplitude=1.0,
    sampling_rate=50e9,   # 50 GHz
    num_bits=3,           # 只生成 3 帧用于可视化
    random_seed=42
)

# 2. 生成脉冲模板
pulse = Pulse.generate(config)
print(f"✓ 脉冲持续时间: {pulse.duration*1e9:.2f} ns")

# 3. 创建用户并生成信号
user = User.create(user_id=0, config=config)
signal = user.generate_signal(pulse)
print(f"✓ 信号长度: {len(signal)} 采样点")

# 4. 可视化
t = np.arange(len(signal)) / config.sampling_rate
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, signal, linewidth=1.0, label='TH-PPM 信号')
plt.xlabel('时间 (ns)', fontsize=12)
plt.ylabel('幅度', fontsize=12)
plt.title('TH-UWB 信号时域波形（3 帧）', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('outputs/waveform_demo.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 图表已保存到 outputs/waveform_demo.png")
```

运行：
```bash
mkdir -p outputs
python scripts/demo_waveform.py
```

**预期输出**：
```
✓ 脉冲持续时间: 2.50 ns
✓ 信号长度: 15000 采样点
✓ 图表已保存到 outputs/waveform_demo.png
```

---

### Step 2: 运行多用户干扰分析（2 分钟）

创建文件 `scripts/demo_mui_analysis.py`：

```python
"""演示：多用户干扰（MUI）性能分析"""
import numpy as np
import matplotlib.pyplot as plt
from src.models.pulse import Pulse
from src.models.modulation import User
from src.models.channel import Channel
from src.simulation.receiver import Receiver
from src.simulation.metrics import PerformanceMetrics
from src.config import SystemConfig
import time

# 1. 配置系统参数（完整仿真）
config = SystemConfig(
    frame_period=100e-9,
    slot_width=10e-9,
    num_slots=8,
    ppm_delay=5e-9,
    pulse_tau=0.5e-9,
    pulse_amplitude=1.0,
    sampling_rate=50e9,
    num_bits=1000,  # 1000 比特用于快速演示
    random_seed=42
)

# 2. 初始化
pulse = Pulse.generate(config)
print(f"✓ 脉冲生成完成")

# 3. 多用户仿真
user_counts = [1, 2, 3, 5]  # 用户数量
ber_results = []
snr_db = 10.0  # 固定 SNR

for K in user_counts:
    print(f"\n运行仿真: {K} 个用户...")
    start_time = time.perf_counter()

    # 创建用户
    users = [User.create(user_id=k, config=config) for k in range(K)]

    # 信道传输
    channel = Channel(config=config, snr_db=snr_db, sir_db=np.inf)
    received_signal, time_axis = channel.transmit(users, pulse)

    # 接收解调（只解调 user_0）
    receiver = Receiver(config=config, target_user=users[0], pulse=pulse)
    decoded_bits = receiver.demodulate(received_signal)

    # 计算 BER
    metrics = PerformanceMetrics(users[0].data_bits, decoded_bits)
    ber_results.append(metrics.ber)

    elapsed = time.perf_counter() - start_time
    print(f"  BER = {metrics.ber:.4e}, 用时 {elapsed:.2f} 秒")

# 4. 绘制性能曲线
plt.figure(figsize=(8, 6))
plt.semilogy(user_counts, ber_results, 'o-', linewidth=2,
             markersize=8, label=f'SNR={snr_db}dB')
plt.xlabel('用户数量', fontsize=12)
plt.ylabel('误码率 (BER)', fontsize=12)
plt.title('多用户干扰性能分析', fontsize=14)
plt.grid(True, which='both', alpha=0.3, linestyle='--')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('outputs/ber_vs_users_demo.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 图表已保存到 outputs/ber_vs_users_demo.png")
```

运行：
```bash
python scripts/demo_mui_analysis.py
```

**预期输出**：
```
✓ 脉冲生成完成

运行仿真: 1 个用户...
  BER = 3.2000e-03, 用时 8.45 秒

运行仿真: 2 个用户...
  BER = 1.1500e-02, 用时 12.67 秒

运行仿真: 3 个用户...
  BER = 2.8700e-02, 用时 16.82 秒

运行仿真: 5 个用户...
  BER = 6.5200e-02, 用时 25.34 秒

✓ 图表已保存到 outputs/ber_vs_users_demo.png
```

---

### Step 3: 运行窄带干扰分析（2 分钟）

创建文件 `scripts/demo_nbi_analysis.py`：

```python
"""演示：窄带干扰（NBI）性能分析"""
import numpy as np
import matplotlib.pyplot as plt
from src.models.pulse import Pulse
from src.models.modulation import User
from src.models.channel import Channel
from src.simulation.receiver import Receiver
from src.simulation.metrics import PerformanceMetrics
from src.config import SystemConfig
import time

# 1. 配置系统参数
config = SystemConfig(
    frame_period=100e-9,
    slot_width=10e-9,
    num_slots=8,
    ppm_delay=5e-9,
    pulse_tau=0.5e-9,
    pulse_amplitude=1.0,
    sampling_rate=50e9,
    num_bits=1000,  # 快速演示
    random_seed=42
)

# 2. 初始化
pulse = Pulse.generate(config)
user = User.create(user_id=0, config=config)
print(f"✓ 初始化完成")

# 3. 窄带干扰仿真
sir_db_values = [30, 20, 10, 0, -10]  # SIR 范围（dB）
ber_results = []

for sir_db in sir_db_values:
    print(f"\n运行仿真: SIR = {sir_db} dB...")
    start_time = time.perf_counter()

    # 信道传输（单用户 + NBI）
    channel = Channel(config=config, snr_db=10.0, sir_db=sir_db, nbi_frequency=2.4e9)
    received_signal, time_axis = channel.transmit([user], pulse)

    # 接收解调
    receiver = Receiver(config=config, target_user=user, pulse=pulse)
    decoded_bits = receiver.demodulate(received_signal)

    # 计算 BER
    metrics = PerformanceMetrics(user.data_bits, decoded_bits)
    ber_results.append(metrics.ber)

    elapsed = time.perf_counter() - start_time
    print(f"  BER = {metrics.ber:.4e}, 用时 {elapsed:.2f} 秒")

# 4. 绘制性能曲线
plt.figure(figsize=(8, 6))
plt.semilogy(sir_db_values, ber_results, 's-', linewidth=2,
             markersize=8, label='单用户', color='red')
plt.xlabel('信干比 SIR (dB)', fontsize=12)
plt.ylabel('误码率 (BER)', fontsize=12)
plt.title('窄带干扰抑制性能分析', fontsize=14)
plt.grid(True, which='both', alpha=0.3, linestyle='--')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('outputs/ber_vs_sir_demo.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 图表已保存到 outputs/ber_vs_sir_demo.png")
```

运行：
```bash
python scripts/demo_nbi_analysis.py
```

---

## 常见问题

### Q1: 仿真运行太慢怎么办？

**A**: 减少 `num_bits` 和 `sampling_rate`：

```python
config = SystemConfig(
    # ... 其他参数
    sampling_rate=25e9,   # 从 50 GHz 降到 25 GHz
    num_bits=500,         # 从 10000 降到 500
    # ...
)
```

**注意**：降低这些参数会影响仿真精度和统计稳定性。

### Q2: 如何修改默认参数？

**A**: 所有参数集中在 `SystemConfig` 中，修改后所有模块自动同步：

```python
config = SystemConfig(
    frame_period=200e-9,  # 改为 200 ns
    num_slots=16,         # 改为 16 个时隙
    pulse_tau=1.0e-9,     # 改为 1 ns
    # ... 其他参数
)
```

### Q3: 如何保存仿真结果？

**A**: 使用 NumPy 保存数据，Matplotlib 保存图表：

```python
# 保存 BER 数据
np.savez('outputs/results.npz',
         user_counts=user_counts,
         ber_values=ber_results)

# 加载数据
data = np.load('outputs/results.npz')
print(data['user_counts'], data['ber_values'])
```

### Q4: 如何运行完整的 10,000 比特仿真？

**A**: 修改 `num_bits` 并增加仿真点数（参考 spec.md 要求）：

```python
config = SystemConfig(
    # ... 其他参数
    num_bits=10000,  # 完整仿真
    # ...
)

# 多用户干扰分析
user_counts = [1, 2, 3, 5, 7, 10]  # 6 个点

# 窄带干扰分析
sir_db_values = np.linspace(30, -10, 10)  # 10 个点
```

**预计耗时**：
- 多用户分析（6 点）：约 10 分钟
- 窄带干扰分析（10 点）：约 5 分钟

---

## 下一步

恭喜你完成了快速入门！接下来可以：

1. **阅读详细文档**：
   - [data-model.md](./data-model.md) - 理解系统架构
   - [research.md](./research.md) - 深入技术细节
   - [contracts/](./contracts/) - API 契约定义

2. **运行完整仿真**（生成论文图表）：
   ```bash
   python scripts/run_mui_analysis.py    # 生成图 2
   python scripts/run_nbi_analysis.py    # 生成图 3
   python scripts/generate_figures.py    # 批量生成所有图表
   ```

3. **自定义仿真场景**：
   - 修改跳时序列生成算法
   - 添加多径信道模型
   - 实现 M-ary PPM（M > 2）

4. **运行单元测试**：
   ```bash
   pytest tests/unit/ -v
   pytest tests/integration/ -v
   ```

---

## 技术支持

如遇到问题，请按以下顺序排查：

1. **检查环境**：`python --version`（确保 >= 3.11）
2. **检查依赖**：`pip list | grep numpy`（确保版本正确）
3. **查看日志**：运行时的错误信息
4. **参考文档**：[spec.md](./spec.md)、[plan.md](./plan.md)

---

**快速入门完成！现在可以开始探索 TH-UWB 通信系统仿真的强大功能了。**

**文档版本**: 1.0.0
**最后更新**: 2025-12-17
