#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理gradients文件夹中的.npy文件，对它们进行求和、平均并可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import torch


# 全局归一化
def global_normalization(data):
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = (data - data_min) / (data_max - data_min + 1e-8)
    return normalized


# 逐通道归一化
def channelwise_normalization(data):
    # 对每个通道单独归一化
    normalized = np.zeros_like(data)
    for c in range(data.shape[0]):
        channel_data = data[c]
        data_min = np.min(channel_data)
        data_max = np.max(channel_data)
        normalized[c] = (channel_data - data_min) / (data_max - data_min + 1e-8)
    return normalized


def plot_channel_gradients(
    grad_data, channel_names=None, save_dir="./", filename="average_gradients"
):
    """
    可视化梯度数据
    参数：
        grad_data     : numpy数组，形状为 (通道数, 高, 宽)
        channel_names : 各通道名称列表
        save_dir      : 保存路径
        filename      : 保存的文件名
    """
    os.makedirs(save_dir, exist_ok=True)

    if channel_names is None:
        num_channels = grad_data.shape[0]
        channel_names = [f"Channel_{i + 1}" for i in range(num_channels)]

    # 创建单张大图
    plt.figure(figsize=(6 * grad_data.shape[0], 6))

    for ch_idx in range(grad_data.shape[0]):
        ax = plt.subplot(1, grad_data.shape[0], ch_idx + 1)
        data = grad_data[ch_idx]  # 直接取通道数据

        # 绘制热力图
        im = ax.imshow(data, cmap="viridis", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 统计信息
        stats_text = f"""Mean: {data.mean():.2e}
        Max: {data.max():.2e}
        Min: {data.min():.2e}"""
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        ax.set_title(f"{channel_names[ch_idx]}")
        ax.axis("off")

    plt.savefig(f"{save_dir}/{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()


def gradients_pocess():
    # 设置路径
    # 使用绝对路径确保能找到文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gradients_dir = os.path.join(current_dir, "gradients")
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有梯度文件
    gradient_files = glob.glob(os.path.join(gradients_dir, "gradients_day*.npy"))
    print(f"找到 {len(gradient_files)} 个梯度文件")

    if not gradient_files:
        print("未找到梯度文件，请检查路径")
        return

    # 加载第一个文件以获取形状信息
    sample_data = np.load(gradient_files[0])
    print(f"梯度文件形状: {sample_data.shape}")

    # 初始化累加数组
    # 预期形状为 [1, 7, 6, 448, 304]
    # 我们需要对所有文件求和，所以初始化一个相同形状的零数组
    accumulated_gradients = np.zeros_like(sample_data)

    # 累加所有梯度文件
    print("正在累加梯度文件...")
    for file_path in tqdm(gradient_files):
        gradient_data = np.load(file_path)
        accumulated_gradients += gradient_data

    # 计算平均梯度
    average_gradients = accumulated_gradients / len(gradient_files)

    # 对平均梯度进行归一化
    average_gradients = channelwise_normalization(average_gradients)
    print(f"平均梯度形状: {average_gradients.shape}")

    # 保存平均梯度
    avg_save_path = os.path.join(output_dir, "average_gradients.npy")
    np.save(avg_save_path, average_gradients)
    print(f"平均梯度已保存至: {avg_save_path}")

    # 可视化平均梯度
    # 假设形状为 [1, 7, 6, 448, 304]，我们需要对每个时间步的每个通道进行可视化
    # 首先对整个序列求平均，得到 [1, 6, 448, 304]
    sequence_avg = np.mean(average_gradients, axis=1)

    # 去掉批次维度，得到 [6, 448, 304]
    sequence_avg = sequence_avg.squeeze(0)

    # 定义通道名称
    channel_names = ["sic", "si_u", "si_v", "t2m", "u10", "v10"]

    # 可视化平均梯度
    print("正在可视化平均梯度...")
    plot_channel_gradients(
        grad_data=sequence_avg,
        channel_names=channel_names,
        save_dir=output_dir,
        filename="average_gradients_all_timesteps",
    )

    # 对每个时间步分别可视化
    for t in range(average_gradients.shape[1]):
        # 提取当前时间步的数据 [1, 6, 448, 304]
        timestep_data = average_gradients[:, t, :, :, :]
        # 去掉批次维度 [6, 448, 304]
        timestep_data = timestep_data.squeeze(0)

        # 可视化当前时间步
        plot_channel_gradients(
            grad_data=timestep_data,
            channel_names=channel_names,
            save_dir=output_dir,
            filename=f"average_gradients_timestep_{t}",
        )

    print("处理完成！")


if __name__ == "__main__":
    gradients_pocess()
