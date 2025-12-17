"""
波形可视化模块

提供学术论文级别的时域波形图生成功能（300 DPI）。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from typing import Optional
from pathlib import Path

# 导入学术标准配置（与 compliance.py 保持一致）
from src.visualization import config


def plot_waveform(
    time_axis: np.ndarray,
    signal: np.ndarray,
    title: str = "TH-UWB 信号时域波形",
    xlabel: str = "时间 (ns)",
    ylabel: str = "幅度",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    绘制时域波形图（学术论文级别）

    生成符合学术论文标准的时域波形图，包含网格、图例、标签等元素。

    参数:
        time_axis: 时间轴数组（秒），shape (N,)
        signal: 信号数组（幅度），shape (N,)
        title: 图表标题，默认 "TH-UWB 信号时域波形"
        xlabel: X 轴标签，默认 "时间 (ns)"
        ylabel: Y 轴标签，默认 "幅度"
        figsize: 图表尺寸 (width, height)，默认 (12, 4)
        save_path: 保存路径（若为 None 则不保存），支持 .png 或 .pdf
        show: 是否显示图表窗口，默认 False（适合无头环境）

    前置条件:
        - len(time_axis) == len(signal)
        - time_axis 单位为秒，xlabel 指定显示单位

    后置条件:
        - 生成符合学术标准的图表（≥300 DPI）
        - 如果 save_path 指定，图表保存到文件
        - 图表包含网格、图例、轴标签、标题

    示例:
        >>> import numpy as np
        >>> t = np.linspace(0, 300e-9, 15000)  # 0-300 ns
        >>> signal = np.sin(2 * np.pi * 1e9 * t)  # 1 GHz 正弦波
        >>> plot_waveform(
        ...     t, signal,
        ...     title="示例信号",
        ...     save_path="outputs/waveforms/example.png"
        ... )

    性能:
        - 对于 N=500,000 采样点：< 2 秒
        - PNG 保存（300 DPI）：< 1 秒
        - PDF 保存：< 2 秒

    参考:
        - research.md Section 6.2: Matplotlib 学术配置标准
        - quickstart.md Step 1: 波形图示例
    """
    # 验证输入
    if len(time_axis) != len(signal):
        raise ValueError(
            f"time_axis 和 signal 长度必须一致，"
            f"当前: len(time_axis)={len(time_axis)}, len(signal)={len(signal)}"
        )

    # 应用学术标准配置（与 compliance.py 保持一致）
    config.setup_academic_style()

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 转换时间轴单位为纳秒（ns）以便显示
    time_ns = time_axis * 1e9

    # 绘制波形
    ax.plot(
        time_ns,
        signal,
        linewidth=1.0,
        label="TH-PPM 信号",
        color="#1f77b4",  # Matplotlib 默认蓝色
    )

    # 设置轴标签和标题
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle="--")

    # 添加图例
    ax.legend(loc="upper right")

    # 紧凑布局（避免标签被裁剪）
    plt.tight_layout()

    # 保存图表
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 图表已保存到 {save_path}")

    # 显示图表
    if show:
        plt.show()

    # 关闭图表释放内存
    plt.close(fig)


def plot_multi_waveforms(
    time_axis: np.ndarray,
    signals: dict,
    title: str = "多用户 TH-UWB 信号对比",
    xlabel: str = "时间 (ns)",
    ylabel: str = "幅度",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    绘制多条波形对比图（子图形式）

    用于对比多个用户或不同参数下的信号波形。

    参数:
        time_axis: 时间轴数组（秒），shape (N,)
        signals: 信号字典 {label: signal_array}，每个 signal 的 shape (N,)
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        figsize: 图表尺寸
        save_path: 保存路径（若为 None 则不保存）
        show: 是否显示图表窗口

    前置条件:
        - 所有 signal 长度与 time_axis 一致
        - signals 字典至少包含 1 个信号

    示例:
        >>> signals = {
        ...     "用户 0": user0_signal,
        ...     "用户 1": user1_signal,
        ...     "用户 2": user2_signal
        ... }
        >>> plot_multi_waveforms(t, signals, save_path="outputs/multi_users.png")

    参考:
        - research.md Section 6.2: 多子图布局标准
    """
    # 验证输入
    if not signals:
        raise ValueError("signals 字典不能为空")

    num_signals = len(signals)

    # 应用学术标准配置（与 compliance.py 保持一致）
    config.setup_academic_style()

    # 创建子图
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize, sharex=True)

    # 如果只有一个信号，axes 不是数组，需要包装
    if num_signals == 1:
        axes = [axes]

    # 转换时间轴单位
    time_ns = time_axis * 1e9

    # 绘制每个信号
    for ax, (label, signal) in zip(axes, signals.items()):
        if len(signal) != len(time_axis):
            raise ValueError(
                f"信号 '{label}' 长度与 time_axis 不一致："
                f"len(signal)={len(signal)}, len(time_axis)={len(time_axis)}"
            )

        ax.plot(time_ns, signal, linewidth=0.8, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right")

    # 设置 X 轴标签（只在最底部子图）
    axes[-1].set_xlabel(xlabel)

    # 设置总标题
    fig.suptitle(title, fontsize=14)

    # 紧凑布局
    plt.tight_layout()

    # 保存图表
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 图表已保存到 {save_path}")

    # 显示图表
    if show:
        plt.show()

    # 关闭图表
    plt.close(fig)
