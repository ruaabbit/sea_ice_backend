import datetime
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from torch.utils.data import DataLoader

from .config import Configs
from .dataset.dataset import SIC_dataset
from .train import MyLightningModule
from .utils.metrics import *


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


def calculate_daily_gradients(
    model,
    dataloader,
    device,
    pred_gap,
    grad_type="sum",
    position=None,
    variable=None,
):
    """逐日预测梯度计算，支持区域分析和特定像素位置分析"""
    model.eval()
    pptvs = []

    # 创建像素位置掩码
    if position is not None and len(position) >= 2:
        pixel_mask = torch.zeros((1, 1, 1, 448, 304), device=device)
        try:
            # 计算矩形区域的边界
            x_coords = [x for x, y in position]
            y_coords = [y for x, y in position]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            # 将矩形区域内的所有像素点设置为1
            pixel_mask[..., x_min : x_max + 1, y_min : y_max + 1] = 1.0
        except Exception as e:
            # print(f"创建像素掩码时出错: {e}")
            # 如果出错，使用全1掩码（不进行区域限制）
            pixel_mask = torch.ones((1, 1, 1, 448, 304), device=device)

    for inputs, targets, inputs_mark, targets_mark in dataloader:
        # 数据准备
        inputs = inputs.to(device).requires_grad_(True)
        targets = targets.to(device)
        mask = model.mask.to(device)

        # 前向传播
        pred = model(inputs, inputs_mark, targets_mark)
        # 不打印每个批次的形状，减少输出冗余
        # # print(pred.shape)

        # 梯度计算
        if position is not None:
            # 只关注矩形区域内的预测
            pred = pred * pixel_mask

        # 选择特定的预测时间步长
        if pred_gap is not None:
            pred = pred[:, pred_gap - 1, :, :, :]

        # 选择特定的变量
        if variable is not None:
            pred = pred[:, variable - 1, :, :]
        else:
            # 当variable为None时，保持pred的原始形状
            pred = pred.reshape(pred.shape[0], -1, pred.shape[-2], pred.shape[-1])

        # 计算梯度
        if grad_type == "sum":
            f = torch.sum(torch.abs(pred) * mask)
        else:  # l2
            f = torch.sum((pred * mask) ** 2)

        # 反向传播
        f.backward()

        # 收集梯度绝对值
        grad = inputs.grad
        pptv = grad.abs().cpu().numpy()
        pptvs.append(pptv)

        # 清除梯度
        inputs.grad = None

    return np.concatenate(pptvs, axis=0)


def plot_channel_gradients(grad_data, channel_names=None):
    """
    可视化梯度数据
    参数：
        grad_data     : numpy数组，形状为 (通道数, 高, 宽)
        channel_names : 各通道名称列表
        save_dir      : 保存路径
        filename      : 保存的文件名
    """

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
    with io.BytesIO() as buffer:

        plt.savefig(buffer, dpi=150, bbox_inches="tight")
        plt.close()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"average_gradients_day_{timestamp}.png"
        file_path = Path("grads") / file_name
        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                str(file_path), ContentFile(buffer.getvalue())
            )
            url = default_storage.url(file_path)
    return url


def grad_nb(
    start_time: int,
    end_time: int,
    pred_gap: int,
    grad_type: str,
    position: str = None,
    variable: int = None,
):
    """计算指定时间范围内的梯度，可选择保存或直接可视化结果"""

    Model_Type = "SimVP_7_7"
    checkpoint_path = f"seaice/cross_modality/checkpoints/{Model_Type}.ckpt"

    config = Configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    ).to(device)

    # 创建专用数据加载器
    grad_dataset = SIC_dataset(
        config.full_data_path,
        start_time,
        end_time,
        config.input_gap,
        config.input_length,
        config.pred_shift,
        config.pred_gap,
        config.pred_length,
        samples_gap=1,
    )
    grad_loader = DataLoader(grad_dataset, batch_size=1, shuffle=False)

    # 解析像素位置

    pixel_positions = (
        [tuple(map(int, pos.split(","))) for pos in position.split(";")]
        if position
        else None
    )

    # 执行梯度计算
    gradients = calculate_daily_gradients(
        model,
        grad_loader,
        device,
        pred_gap=pred_gap,
        grad_type=grad_type,
        position=pixel_positions,
        variable=variable,
    )

    return gradients


def batch_process_gradients(
    start_date, end_date, pred_gap, grad_type="sum", position=None, variable=None
):
    """
    批量处理从起始日期到结束日期的每一天的梯度计算

    参数：
        start_date  : 起始日期，格式为YYYYMMDD
        end_date    : 结束日期，格式为YYYYMMDD
        pred_gap    : 预测提前期，范围是1-7
        grad_type   : 梯度计算方式，"sum"或"l2"
        position    : 要分析的像素位置，格式为'x1,y1;x2,y2;x3,y3;x4,y4'
        variable    : 要分析的变量，分别为1-6[SIC,SI_U,SI_V,T2M,U10M,V10M]
    """

    # 配置参数
    input_length = 7
    prediction_length = 7
    days_to_add = input_length + prediction_length - 1

    # 将日期字符串转换为datetime对象
    start_date_obj = datetime.datetime.strptime(str(start_date), "%Y%m%d")
    end_date_obj = datetime.datetime.strptime(str(end_date), "%Y%m%d")

    # 计算调整后的结束日期
    adjusted_end_date_obj = end_date_obj - datetime.timedelta(days=days_to_add)

    # 初始化累积梯度数组
    accumulated_gradients = None
    gradient_count = 0
    result_urls = []

    # 在日期范围内循环
    current_date_obj = start_date_obj
    while current_date_obj <= adjusted_end_date_obj:
        # 获取当前日期的字符串表示
        current_start_date = current_date_obj.strftime("%Y%m%d")
        current_end_date_obj = current_date_obj + datetime.timedelta(days=days_to_add)
        current_end_date = current_end_date_obj.strftime("%Y%m%d")

        # 执行梯度计算
        gradients = grad_nb(
            start_time=int(current_start_date),
            end_time=int(current_end_date),
            pred_gap=pred_gap,
            grad_type=grad_type,
            position=position,
            variable=variable,
        )

        # 如果需要累积结果
        if accumulated_gradients is None:
            accumulated_gradients = gradients.copy()
        else:
            accumulated_gradients += gradients

        gradient_count += 1

        # 移动到下一天
        current_date_obj += datetime.timedelta(days=1)

    # 计算平均梯度
    average_gradients = accumulated_gradients / gradient_count

    # 定义通道名称
    channel_names = ["sic", "si_u", "si_v", "t2m", "u10", "v10"]

    # 对梯度数据进行归一化
    normalized_gradients = channelwise_normalization(average_gradients.squeeze(0))

    # 对每个时间步分别可视化
    for t in range(normalized_gradients.shape[0]):
        # 提取当前时间步的数据
        timestep_data = normalized_gradients[t]

        # 可视化当前时间步
        result_url = plot_channel_gradients(
            grad_data=timestep_data, channel_names=channel_names
        )
        result_urls.append(result_url)
    torch.cuda.empty_cache()

    return result_urls
