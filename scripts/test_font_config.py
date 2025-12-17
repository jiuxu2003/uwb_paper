#!/usr/bin/env python3
"""测试字体配置是否生效"""

# 先导入 matplotlib 并清除缓存
import matplotlib
import shutil
cache_dir = matplotlib.get_cachedir()
shutil.rmtree(cache_dir, ignore_errors=True)
print(f"✓ 清除缓存: {cache_dir}")

# 现在导入 waveform 模块（会执行字体配置）
from src.visualization import waveform  # noqa: F401
import matplotlib as mpl

# 检查配置是否生效
print("\n当前 matplotlib 字体配置:")
print(f"  font.family: {mpl.rcParams['font.family']}")
print(f"  font.sans-serif: {mpl.rcParams['font.sans-serif'][:5]}")
print(f"  font.serif: {mpl.rcParams['font.serif'][:4]}")
print(f"  mathtext.fontset: {mpl.rcParams['mathtext.fontset']}")

# 创建测试图表
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.text(0.5, 0.7, 'English: TH-UWB Signal 123.45 ns', ha='center', fontsize=16, family='sans-serif')
ax.text(0.5, 0.5, '中文：时域波形（帧）', ha='center', fontsize=16, family='sans-serif')
ax.text(0.5, 0.3, 'Mixed: 时间 (ns) Time', ha='center', fontsize=16, family='sans-serif')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('字体测试 Font Test', fontsize=18)

plt.savefig('outputs/font_test.png', dpi=150, bbox_inches='tight')
print('\n✓ 测试图表已保存到 outputs/font_test.png')
