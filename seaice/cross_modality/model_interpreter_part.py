import numpy as np
from torch.utils.data import DataLoader

from .config import Configs
from .dataset import SIC_dataset
from .train import MyLightningModule
from .utils.metrics import *


def calculate_daily_gradients(
    model,
    dataloader,
    device,
    pred_gap,
    grad_type="sum",
    position=None,
    pixel_positions=None,
    variable=None,
):
    """逐日预测梯度计算，支持区域分析和特定像素位置分析"""
    model.eval()
    pptvs = []

    # 创建像素位置掩码
    if position is not None:
        pixel_mask = torch.zeros((1, 1, 1, 448, 304), device=device)
        # 计算矩形区域的边界
        x_coords = [x for x, y in position]
        y_coords = [y for x, y in position]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # 将矩形区域内的所有像素点设置为1
        pixel_mask[..., x_min : x_max + 1, y_min : y_max + 1] = 1.0

    for inputs, targets, inputs_mark, targets_mark in dataloader:
        # 数据准备
        inputs = inputs.to(device).requires_grad_(True)
        targets = targets.to(device)
        mask = model.mask.to(device)

        # 前向传播
        pred = model(inputs, inputs_mark, targets_mark)
        print(pred.shape)

        # 梯度计算
        if pixel_positions is not None:
            # 只关注矩形区域内的预测
            pred = pred * pixel_mask

        # 选择特定的预测时间步长
        if pred_gap is not None:
            pred = pred[:, pred_gap - 1, :, :, :]

        # 选择特定的变量
        if variable is not None:
            pred = pred[:, variable - 1, :, :]

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


def grad_nb(
    start_time: int,
    end_time: int,
    pred_gap: int,
    grad_type: str,
    position: str,
):

    Model_Type = "SimVP_7_7"

    checkpoint_path = f"checkpoints/{Model_Type}.ckpt"
    variables = ["SIC", "SI_U", "SI_V", "T2M", "U10M", "V10M"]

    config = Configs()

    device = torch.device("cuda")

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
    pixel_positions = [tuple(map(int, pos.split(","))) for pos in position.split(";")]

    # 执行梯度计算
    gradients = calculate_daily_gradients(
        model,
        grad_loader,
        device,
        pred_gap=pred_gap,
        grad_type=grad_type,
        position=pixel_positions,
    )
    # 结果保存
    save_path = f"gradient_analysis/gradients/gradients_day{start_time}.npy"
    np.save(save_path, gradients)
