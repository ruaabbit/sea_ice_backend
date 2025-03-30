import numpy as np
from torch.utils.data import DataLoader


from .config import Configs
from .view_my_result import plot_batch_figures
from .dataset import SIC_dataset
from .train import MyLightningModule
from .utils.metrics import *


def calculate_daily_gradients(model, dataloader, device, target_day=0, grad_type="sum"):
    """逐日预测梯度计算"""
    model.eval()
    gradients = []

    for inputs, targets, inputs_mark, targets_mark in dataloader:
        # 数据准备
        inputs = inputs.to(device).requires_grad_(True)
        targets = targets.to(device)
        mask = model.mask.to(device)

        # 前向传播
        pred = model(inputs, inputs_mark, targets_mark)

        # 提取目标日的预测和对应输入时间步
        daily_pred = pred[:, target_day, :, :, :]  # [batch, H, W]
        input_slice = inputs[:, target_day, :, :, :]  # 假设输入与预测时间步对齐

        # 计算输入与预测的差异
        diff = input_slice - daily_pred

        # 梯度计算逻辑（基于差异和掩码）
        if grad_type == "sum":
            loss = torch.sum(torch.abs(diff) * mask)
        else:  # L2范数
            loss = torch.sqrt(torch.sum((diff * mask) ** 2))

        # 反向传播
        loss.backward()

        # 收集梯度
        batch_grad = inputs.grad.detach().abs().cpu().numpy()
        gradients.append(batch_grad)

        # 清除梯度
        inputs.grad = None

    return np.concatenate(gradients, axis=0)


def grad_nb(start_time: int, end_time: int, grad_day: int, grad_type: str):
    Model_Type = "SimVP_7_7"

    checkpoint_path = f"seaice/cross_modality/checkpoints/{Model_Type}.ckpt"

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

    # 执行梯度计算
    gradients = calculate_daily_gradients(
        model,
        grad_loader,
        device,
        target_day=grad_day,
        grad_type=grad_type,
    )

    # 结果保存
    channel_labels = ["sic", "si_u", "si_v", "t2m", "u10", "v10"]

    url = plot_batch_figures(
        grad_data=gradients,
        channel_names=channel_labels,
        samples_per_figure=7,
        steps_per_figure=7,
        dpi=100,
    )
    return [url]
