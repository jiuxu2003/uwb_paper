"""
matplotlib 学术标准配置模块

本模块提供符合 IEEE 学术出版标准的 matplotlib 配置。

配置标准：
1. 字体：中文使用 Noto Serif CJK（SimSun 替代），英文使用 Times New Roman
2. 分辨率：≥300 DPI（适合学术论文打印）
3. 图表尺寸：7x5.25 英寸（IEEE 双栏图标准）
4. 字号：10-12 pt（清晰可读）

使用方法：
```python
from src.visualization import config
config.setup_academic_style()

# 或者使用上下文管理器
with config.academic_style():
    fig, ax = plt.subplots()
    # 绘图代码
```
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Optional, Tuple


# IEEE 学术论文标准配置
ACADEMIC_CONFIG = {
    # 字体配置（Constitution v1.2.0 Principle X）
    # 中文字体优先，确保中文文本不会fallback到Times New Roman
    "font.serif": [
        "Noto Serif CJK JP",    # 中文衬线字体（优先，避免Times New Roman缺少中文字符警告）
        "Times New Roman",      # 英文衬线字体（标准）
        "SimSun",               # 宋体（如果可用）
        "STSong",               # 华文宋体（macOS）
        "DejaVu Serif",         # 降级方案
    ],
    "font.sans-serif": [
        "Noto Sans CJK JP",     # 中文无衬线字体（优先）
        "Arial",                # 英文无衬线字体
        "SimHei",               # 黑体
        "DejaVu Sans",          # 降级方案
    ],
    "font.family": "serif",     # 默认使用衬线字体（更正式）
    "font.size": 12,            # 基础字号（10-12 pt 符合 IEEE 标准）

    # 数学符号配置
    "mathtext.fontset": "stix",           # STIX 字体（类似 Times New Roman）
    "mathtext.default": "regular",        # 数学文本默认样式
    "axes.unicode_minus": False,          # 解决负号显示问题

    # DPI 配置（≥300 DPI 符合学术出版要求）
    "figure.dpi": 100,                    # 屏幕显示 DPI
    "savefig.dpi": 300,                   # 保存图表 DPI（高清打印）
    "savefig.bbox": "tight",              # 自动裁剪空白
    "savefig.pad_inches": 0.05,           # 边距（英寸）

    # 图表尺寸配置（IEEE 标准）
    "figure.figsize": (7, 5.25),          # 双栏图尺寸（7x5.25 英寸）
    "figure.constrained_layout.use": True, # 自动调整布局

    # 轴标签和标题配置
    "axes.labelsize": 12,                 # 轴标签字号
    "axes.titlesize": 14,                 # 标题字号
    "axes.labelweight": "normal",         # 标签字重
    "axes.linewidth": 1.0,                # 轴线宽度
    "axes.grid": True,                    # 默认显示网格
    "axes.grid.which": "major",           # 主网格线
    "axes.axisbelow": True,               # 网格线在数据下方

    # 网格线配置
    "grid.alpha": 0.3,                    # 网格透明度
    "grid.linestyle": "--",               # 网格线样式
    "grid.linewidth": 0.6,                # 网格线宽度

    # 图例配置
    "legend.fontsize": 10,                # 图例字号
    "legend.framealpha": 0.9,             # 图例背景透明度
    "legend.edgecolor": "gray",           # 图例边框颜色
    "legend.fancybox": False,             # 禁用圆角（更正式）

    # 刻度配置
    "xtick.labelsize": 10,                # X 轴刻度字号
    "ytick.labelsize": 10,                # Y 轴刻度字号
    "xtick.direction": "in",              # X 轴刻度方向（向内）
    "ytick.direction": "in",              # Y 轴刻度方向（向内）
    "xtick.major.size": 4,                # X 轴主刻度长度
    "ytick.major.size": 4,                # Y 轴主刻度长度
    "xtick.minor.size": 2,                # X 轴次刻度长度
    "ytick.minor.size": 2,                # Y 轴次刻度长度

    # 线条和标记配置
    "lines.linewidth": 1.5,               # 线条宽度
    "lines.markersize": 6,                # 标记大小
    "lines.markeredgewidth": 0.5,         # 标记边框宽度

    # 颜色配置（使用 IEEE 推荐的色盲友好配色）
    "axes.prop_cycle": mpl.cycler(
        color=[
            "#0173B2",  # 蓝色（主色）
            "#DE8F05",  # 橙色
            "#029E73",  # 绿色
            "#D55E00",  # 红橙色
            "#CC78BC",  # 紫色
            "#CA9161",  # 棕色
            "#949494",  # 灰色
        ]
    ),
}


def setup_academic_style(
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    font_size: Optional[int] = None
) -> None:
    """
    应用学术标准 matplotlib 配置

    参数:
        figsize: 图表尺寸（英寸），默认 (7, 5.25) 双栏图
        dpi: 保存 DPI，默认 300
        font_size: 基础字号，默认 12

    示例:
        >>> from src.visualization import config
        >>> config.setup_academic_style()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> fig.savefig("plot.png")  # 自动使用 300 DPI
    """
    # 应用基础配置
    for key, value in ACADEMIC_CONFIG.items():
        mpl.rcParams[key] = value

    # 应用用户自定义配置
    if figsize is not None:
        mpl.rcParams["figure.figsize"] = figsize
    if dpi is not None:
        mpl.rcParams["savefig.dpi"] = dpi
    if font_size is not None:
        mpl.rcParams["font.size"] = font_size
        mpl.rcParams["axes.labelsize"] = font_size
        mpl.rcParams["axes.titlesize"] = font_size + 2
        mpl.rcParams["legend.fontsize"] = font_size - 2
        mpl.rcParams["xtick.labelsize"] = font_size - 2
        mpl.rcParams["ytick.labelsize"] = font_size - 2


@contextmanager
def academic_style(
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    font_size: Optional[int] = None
):
    """
    上下文管理器：临时应用学术标准配置

    参数:
        figsize: 图表尺寸（英寸），默认 (7, 5.25)
        dpi: 保存 DPI，默认 300
        font_size: 基础字号，默认 12

    使用示例:
        >>> from src.visualization import config
        >>> with config.academic_style(figsize=(5, 4), dpi=600):
        ...     fig, ax = plt.subplots()
        ...     ax.plot([1, 2, 3], [1, 4, 9])
        ...     fig.savefig("high_res_plot.png")  # 使用 600 DPI
    """
    # 保存当前配置
    original_config = {key: mpl.rcParams[key] for key in ACADEMIC_CONFIG.keys()}

    try:
        # 应用学术标准配置
        setup_academic_style(figsize=figsize, dpi=dpi, font_size=font_size)
        yield
    finally:
        # 恢复原始配置
        for key, value in original_config.items():
            mpl.rcParams[key] = value


def get_ieee_figsize(columns: int = 2) -> Tuple[float, float]:
    """
    获取 IEEE 标准图表尺寸

    参数:
        columns: 栏数，1（单栏）或 2（双栏）

    返回:
        (width, height): 图表尺寸（英寸）

    IEEE Transactions 标准尺寸：
        - 单栏图：3.5 x 2.625 英寸
        - 双栏图：7 x 5.25 英寸（默认）
    """
    if columns == 1:
        return (3.5, 2.625)  # 单栏图
    elif columns == 2:
        return (7, 5.25)     # 双栏图
    else:
        raise ValueError(f"columns 必须是 1 或 2，当前值: {columns}")


def verify_font_availability() -> dict:
    """
    验证所需字体是否可用

    返回:
        availability: 字体可用性字典
            - "Times New Roman": bool
            - "Noto Serif CJK JP": bool
            - "DejaVu Serif": bool

    示例:
        >>> from src.visualization import config
        >>> fonts = config.verify_font_availability()
        >>> if not fonts["Times New Roman"]:
        ...     print("警告：Times New Roman 字体不可用")
    """
    from matplotlib.font_manager import fontManager

    # 获取所有可用字体名称
    available_fonts = set([f.name for f in fontManager.ttflist])

    fonts_to_check = {
        "Times New Roman": "Times New Roman",
        "Noto Serif CJK JP": "Noto Serif CJK JP",
        "DejaVu Serif": "DejaVu Serif",
        "SimSun": "SimSun",
    }

    availability = {}
    for display_name, font_name in fonts_to_check.items():
        availability[display_name] = font_name in available_fonts

    return availability


# 模块加载时自动应用学术标准配置
# 注释掉以避免全局影响，用户需要显式调用 setup_academic_style()
# setup_academic_style()


if __name__ == "__main__":
    # 验证字体可用性
    print("=== matplotlib 学术标准配置 ===\n")

    print("字体可用性检查：")
    fonts = verify_font_availability()
    for font_name, available in fonts.items():
        status = "✓ 可用" if available else "✗ 不可用"
        print(f"  {font_name}: {status}")

    print("\n当前配置：")
    print(f"  图表尺寸: {mpl.rcParams['figure.figsize']}")
    print(f"  保存 DPI: {mpl.rcParams['savefig.dpi']}")
    print(f"  字体族: {mpl.rcParams['font.family']}")
    print(f"  基础字号: {mpl.rcParams['font.size']}")

    print("\n应用学术标准配置...")
    setup_academic_style()

    print("✓ 配置完成！")
    print(f"  图表尺寸: {mpl.rcParams['figure.figsize']}")
    print(f"  保存 DPI: {mpl.rcParams['savefig.dpi']}")
