import datetime
import io
from pathlib import Path
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def plot_channel_gradients(grad_data, channel_names=None, save_dir="./grad_plots"):
    """
    多通道梯度可视化函数
    参数：
        grad_data      : numpy数组，形状为 (样本数, 时间步长, 通道数, 高, 宽)
        channel_names  : 列表，各通道的名称（可选）
        save_dir       : 图片保存路径
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 遍历所有样本和时间步
    for sample_idx in tqdm(range(grad_data.shape[0]), desc="处理样本"):
        for time_step in range(grad_data.shape[1]):
            # 创建画布
            fig, axes = plt.subplots(
                1,
                grad_data.shape[2],
                figsize=(6 * grad_data.shape[2], 6),
                constrained_layout=True,
            )

            # 如果单通道情况处理
            if grad_data.shape[2] == 1:
                axes = [axes]

            # 绘制每个通道
            for ch_idx, ax in enumerate(axes):
                # 获取当前通道数据
                data = grad_data[sample_idx, time_step, ch_idx]

                # 绘制热力图
                im = ax.imshow(data, cmap="viridis", origin="lower")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 添加统计信息
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

                # 设置标题
                ax.set_title(
                    f"{channel_names[ch_idx]}\nSample {sample_idx} | Step {time_step}"
                )
                ax.axis("off")

            # 保存并关闭
            plt.savefig(f"{save_dir}/sample{sample_idx}_step{time_step}.png", dpi=150)
            plt.close()


def global_normalization(data):
    """整个数据集统一归一化到[0,1]范围"""
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8)  # 防止除零


def global_normalization(data):
    """整个数据集统一归一化到[0,1]范围"""
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8)  # 防止除零


def channelwise_normalization(data):
    """每个通道独立归一化"""
    # 保持通道维度：(样本, 时间步, 通道, 高, 宽)
    data_min = np.min(data, axis=(0, 1, 3, 4), keepdims=True)
    data_max = np.max(data, axis=(0, 1, 3, 4), keepdims=True)
    return (data - data_min) / (data_max - data_min + 1e-8)


def plot_single_sample_timestep(args):
    """
    为单个样本和时间步处理绘图，用于并行化
    """
    sample_idx, time_step, grad_data, channel_names, save_dir, dpi = args

    # 创建画布
    fig, axes = plt.subplots(
        1,
        grad_data.shape[2],
        figsize=(6 * grad_data.shape[2], 6),
        constrained_layout=True,
    )

    # 如果单通道情况处理
    if grad_data.shape[2] == 1:
        axes = [axes]

    # 绘制每个通道
    for ch_idx, ax in enumerate(axes):
        # 获取当前通道数据
        data = grad_data[sample_idx, time_step, ch_idx]

        # 绘制热力图
        im = ax.imshow(data, cmap="viridis", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 添加统计信息
        stats_text = (
            f"Mean: {data.mean():.2e}\nMax: {data.max():.2e}\nMin: {data.min():.2e}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # 设置标题
        ax.set_title(f"{channel_names[ch_idx]}\nSample {sample_idx} | Step {time_step}")
        ax.axis("off")

    # 保存并关闭
    plt.savefig(f"{save_dir}/sample{sample_idx}_step{time_step}.png", dpi=dpi)
    plt.close()

    return f"Processed sample {sample_idx}, step {time_step}"


def plot_channel_gradients_optimized(
    grad_data,
    channel_names=None,
    save_dir="./grad_plots",
    max_workers=None,
    dpi=100,
    batch_samples=None,
    batch_timesteps=None,
):
    """
    优化的多通道梯度可视化函数

    参数：
        grad_data      : numpy数组，形状为 (样本数, 时间步长, 通道数, 高, 宽)
        channel_names  : 列表，各通道的名称（可选）
        save_dir       : 图片保存路径
        max_workers    : 并行处理的工作进程数，默认为CPU核心数-1
        dpi            : 图像分辨率，较低值可加快处理
        batch_samples  : 每次处理的样本数，None表示处理所有样本
        batch_timesteps: 每次处理的时间步数，None表示处理所有时间步
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 如果通道名称未提供，创建默认名称
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(grad_data.shape[2])]

    # 设置默认的工作进程数
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    # 限制批次大小
    if batch_samples is None:
        batch_samples = grad_data.shape[0]
    if batch_timesteps is None:
        batch_timesteps = grad_data.shape[1]

    # 准备并行处理的任务
    tasks = []
    for sample_idx in range(min(batch_samples, grad_data.shape[0])):
        for time_step in range(min(batch_timesteps, grad_data.shape[1])):
            tasks.append(
                (sample_idx, time_step, grad_data, channel_names, save_dir, dpi)
            )

    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(plot_single_sample_timestep, tasks),
                total=len(tasks),
                desc="处理样本和时间步",
            )
        )

    return f"已完成 {len(tasks)} 个子图的处理"


def memory_efficient_processing(
    grad_data, channel_names=None, save_dir="./grad_plots", slice_size=2, dpi=100
):
    """
    内存高效的处理方法 - 每次只加载部分数据到内存

    参数：
        grad_data    : numpy数组，形状为 (样本数, 时间步长, 通道数, 高, 宽)
        channel_names: 列表，各通道的名称（可选）
        save_dir     : 图片保存路径
        slice_size   : 每次处理的样本数
        dpi          : 图像分辨率
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 如果通道名称未提供，创建默认名称
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(grad_data.shape[2])]

    num_samples = grad_data.shape[0]
    num_steps = grad_data.shape[1]

    # 分批处理样本
    for start_sample in tqdm(range(0, num_samples, slice_size), desc="处理样本批次"):
        end_sample = min(start_sample + slice_size, num_samples)

        # 只获取当前需要的数据切片
        current_data = grad_data[start_sample:end_sample]

        # 为当前批次的每个样本和时间步绘图
        for s_idx, sample_idx in enumerate(range(start_sample, end_sample)):
            for time_step in range(num_steps):
                # 创建画布
                fig, axes = plt.subplots(
                    1,
                    grad_data.shape[2],
                    figsize=(5 * grad_data.shape[2], 5),
                    constrained_layout=True,
                )

                # 如果单通道情况处理
                if grad_data.shape[2] == 1:
                    axes = [axes]

                # 绘制每个通道
                for ch_idx, ax in enumerate(axes):
                    # 获取当前通道数据
                    data = current_data[s_idx, time_step, ch_idx]

                    # 绘制热力图
                    im = ax.imshow(data, cmap="viridis", origin="lower")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    # 添加统计信息 (精简版)
                    stats_text = f"Mean: {data.mean():.2e}"
                    ax.text(
                        0.05,
                        0.95,
                        stats_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.7),
                    )

                    # 设置标题
                    ax.set_title(f"{channel_names[ch_idx]}")
                    ax.set_xlabel(f"Sample {sample_idx} | Step {time_step}")
                    ax.axis("off")

                # 保存并关闭
                plt.savefig(
                    f"{save_dir}/sample{sample_idx}_step{time_step}.png", dpi=dpi
                )
                plt.close()

    return "处理完成"


def plot_batch_figures(
    grad_data,
    channel_names=None,
    samples_per_figure=7,
    steps_per_figure=7,
    dpi=100,
):
    """
    批量绘制多个样本和时间步在一个大图中，减少IO操作

    参数：
        grad_data         : numpy数组，形状为 (样本数, 时间步长, 通道数, 高, 宽)
        channel_names     : 列表，各通道的名称（可选）
        save_dir          : 图片保存路径
        samples_per_figure: 每个大图中包含的样本数
        steps_per_figure  : 每个大图中包含的时间步数
        dpi               : 图像分辨率
    """

    # 如果通道名称未提供，创建默认名称
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(grad_data.shape[2])]

    num_samples = grad_data.shape[0]
    num_steps = grad_data.shape[1]
    num_channels = grad_data.shape[2]

    # 计算需要创建的批次数
    sample_batches = (num_samples + samples_per_figure - 1) // samples_per_figure
    step_batches = (num_steps + steps_per_figure - 1) // steps_per_figure
    with io.BytesIO() as buf:
        for sample_batch in tqdm(range(sample_batches), desc="处理样本批次"):
            for step_batch in range(step_batches):
                # 计算当前批次的样本和时间步范围
                sample_start = sample_batch * samples_per_figure
                sample_end = min(sample_start + samples_per_figure, num_samples)

                step_start = step_batch * steps_per_figure
                step_end = min(step_start + steps_per_figure, num_steps)

                # 计算当前批次实际的样本和时间步数
                curr_samples = sample_end - sample_start
                curr_steps = step_end - step_start

                # 创建包含多个子图的大图
                # 布局: 每行是一个样本的不同时间步，每列是同一时间步的不同通道
                fig, axes = plt.subplots(
                    curr_samples * curr_steps,  # 行数 = 样本数 * 时间步数
                    num_channels,  # 列数 = 通道数
                    figsize=(3 * num_channels, 3 * curr_samples * curr_steps),
                    constrained_layout=True,
                )

                # 如果只有一个子图，需要特殊处理
                if curr_samples * curr_steps == 1 and num_channels == 1:
                    axes = np.array([[axes]])
                elif curr_samples * curr_steps == 1:
                    axes = np.array([axes])
                elif num_channels == 1:
                    axes = axes.reshape(-1, 1)

                # 遍历批次中的所有样本和时间步
                for s_idx, sample_idx in enumerate(range(sample_start, sample_end)):
                    for t_idx, time_step in enumerate(range(step_start, step_end)):
                        # 计算当前样本和时间步在图中的行索引
                        row_idx = s_idx * curr_steps + t_idx

                        for ch_idx in range(num_channels):
                            # 获取当前通道数据
                            data = grad_data[sample_idx, time_step, ch_idx]

                            # 绘制热力图
                            im = axes[row_idx, ch_idx].imshow(
                                data, cmap="viridis", origin="lower"
                            )

                            # 在第一行每个通道上方添加通道名称
                            if row_idx == 0:
                                axes[row_idx, ch_idx].set_title(
                                    f"{channel_names[ch_idx]}"
                                )

                            # 在每行左侧添加样本和时间步信息
                            if ch_idx == 0:
                                axes[row_idx, ch_idx].set_ylabel(
                                    f"S{sample_idx} T{time_step}"
                                )

                            # 关闭坐标轴
                            # axes[row_idx, ch_idx].axis("off")

                # 为整个图添加colorbar
                plt.colorbar(im, ax=axes.ravel().tolist())

                # 保存大图
                plt.savefig(
                    buf,
                    dpi=dpi,
                )
                # plt.show()
                plt.close()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"grad_day_{timestamp}.png"
        file_path = Path("grads") / file_name
        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                str(file_path), ContentFile(buf.getvalue())
            )
            url = default_storage.url(file_path)
    return url

