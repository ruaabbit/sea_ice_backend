import datetime
import io
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
import xarray as xr
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .model_factory import IceNet
from .utils import SIC_dataset


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "config":  # 替换成文件中类的实际名称
            module = "seaice.osi_450_a.config"
        return super().find_class(module, name)


# 从 pkl 文件加载 configs
with open("seaice/osi_450_a/pkls/train_config_SICTeDev_update.pkl", "rb") as f:
    configs = CustomUnpickler(f).load()

model_path: str = "seaice/osi_450_a/checkpoints/checkpoint_SICTeDev_update.chk"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arctic_mask = np.load("seaice/osi_450_a/data/arctic_mask.npy")
arctic_mask = torch.tensor(arctic_mask, dtype=torch.float32).to(device)

land_mask = np.load("seaice/osi_450_a/data/land_mask.npy")

model = IceNet(configs).to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["net"])
model.eval()


def predict_ice_concentration_from_images(
        image_list: List[Image.Image], input_times: List[int],
) -> np.ndarray:
    """
    从图像列表预测未来的海冰浓度。

    参数:
        image_list: 包含14个PIL图像的列表

    返回:
        np.ndarray: 预测的海冰浓度图，形状为 (12, H, W)
    """
    processed_images = []
    for img in image_list:
        if img.mode != "L":
            img = img.convert("L")
        img_array = np.array(img)
        processed_images.append(img_array)
    input_array = np.stack(processed_images)
    # 针对图片的数据预处理，除以255
    input_array = input_array / 255.0
    return _predict(input_array, input_times)


def predict_ice_concentration_from_nc_files(
        nc_file_paths: List[str], input_times: List[int], variable_name: str = "ice_conc",
) -> np.ndarray:
    """
    从多个 .nc 文件预测未来的海冰浓度。

    参数：
        nc_file_path: .nc 文件的路径
        variable_name: 要读取的变量名称，默认为 'sea_ice_concentration'

    返回：
        np.ndarray: 预测的海冰浓度图，形状为 (12, H, W)
    """
    nc_data = []
    # 读取 nc 文件
    for nc_file_path in nc_file_paths:
        with xr.open_dataset(nc_file_path) as xr_data:
            data = xr_data["ice_conc"][0]
        input_array = np.array(data)
        np.nan_to_num(input_array, nan=0, copy=False)
        input_array = np.where(input_array == -32767, 0, input_array)
        input_array = input_array / 100.0
        nc_data.append(input_array)
    return _predict(np.stack(nc_data), input_times)


def _predict(input_array: np.ndarray, input_times: List[int]) -> np.ndarray:
    """
    私有函数，执行实际的预测工作。

    参数:
        input_array: 经过预处理的输入数组，形状为 (12, H, W)

    返回:
        np.ndarray: 预测的海冰浓度图，形状为 (12, H, W)
    """
    # 确保输入数据在GPU上
    input_tensor = torch.from_numpy(input_array).float()
    input_tensor = input_tensor[None, :, None, :, :].to(device)
    dummy_input_tensor = torch.zeros_like(input_tensor).to(device)
    input_times = torch.tensor(input_times, dtype=torch.long, device=device)

    with torch.no_grad():
        output, _ = model(input_tensor, dummy_input_tensor, input_times)

        # 转换为numpy数组
        prediction = output.cpu().numpy()[0]
    torch.cuda.empty_cache()
    return prediction


def _get_grad(dataloader: DataLoader, grad_month: int, grad_type: str):
    sic_pred_list = []
    a = grad_month - 1
    b = grad_month
    for inputs, targets, input_times in dataloader:
        inputs = inputs.float().to(device)
        inputs.requires_grad = True
        targets = targets.float().to(device)

        input_times = torch.tensor(input_times, dtype=torch.float32).to(device)

        outputgrad, _ = model(inputs, targets, input_times)

        outputgrad = outputgrad[0, a:b, 0, :, :]
        outputgrad = outputgrad[0, :, :]

        # print("target shape:", targets.shape)
        input = inputs[0, a:b, 0, :, :]
        input = input[0, :, :]

        if grad_type == "sum":
            outputgrad_1 = torch.sum(abs(input - outputgrad) * arctic_mask)
        else:
            outputgrad_1 = torch.sqrt(
                torch.sum(((input - outputgrad) * arctic_mask) ** 2)
            )

        outputgrad_1.backward()
        grads = inputs.grad.cpu().numpy()
        grads_abs = abs(grads)
        sic_pred_list.append(grads_abs)
    torch.cuda.empty_cache()
    # model.zero_grad()
    return sic_pred_list


def grad(start_time: int, end_time: int, grad_month: int, grad_type: str):
    # start_time = 201801
    # end_time = 201812
    # grad_month = 1
    # grad_type = "variation"  # or "sum"

    end_time_str = str(end_time)

    year = int(end_time_str[:4])
    month = int(end_time_str[4:])

    year += 1

    adjusted_end_time = year * 100 + month
    end_time = adjusted_end_time

    dataset_test = SIC_dataset(
        "seaice/osi_450_a/data/full_sic_update.nc",
        start_time,
        end_time,
        configs.input_gap,
        configs.input_length,
        configs.pred_shift,
        configs.pred_gap,
        configs.pred_length,
        samples_gap=12,
    )

    dataloader_test = DataLoader(dataset_test, shuffle=False)

    start_time_str = str(start_time)
    start_month = int(start_time_str[4:6])
    month_sequence = []
    for i in range(12):
        current_month = (start_month + i - 1) % 12 + 1
        month_sequence.append(current_month)

    month_position = month_sequence.index(grad_month)

    sic_grad = _get_grad(dataloader_test, month_position, grad_type)

    sic_grad = np.array(sic_grad)

    existing_array = sic_grad

    # 对数组进行处理，按第一个维度求平均
    mean_array = np.mean(existing_array, axis=0)

    if grad_type == "sum":
        mean_array[mean_array > 4] = 4
    else:
        mean_array[mean_array > 0.05] = 0.05
    mean_array[mean_array < 4e-06] = 4e-06
    max_value = np.max(mean_array)
    min_value = np.min(mean_array)

    mean_array = (mean_array - min_value) / (max_value - min_value)

    grad_month_np = mean_array

    grad_month_np = grad_month_np[0, :, 0, :]

    def _percentage(x, pos):
        return f"{x:.0%}"

    # 创建渐变色的colormap
    colors = [(0, "white"), (1, "#D50103")]  # 定义颜色映射
    cmap_ice_conc1 = LinearSegmentedColormap.from_list("cmap_ice_conc", colors)

    sic = grad_month_np
    start_time_str = str(start_time)
    month = int(start_time_str[4:6])

    result_urls = []
    for i in range(12):
        fig, ax = plt.subplots()
        sic[i] = gaussian_filter(sic[i], sigma=3)
        img = ax.imshow(sic[i], cmap=cmap_ice_conc1, vmin=0, vmax=1)

        # 绘制陆地
        land_color = "#c0c0c0"
        land = np.ma.masked_where(land_mask == False, land_mask)  # 创建陆地的掩模
        ax.imshow(land, cmap=ListedColormap([land_color]))

        # 关闭坐标轴和标签
        ax.set_title("")  # 禁用标题
        ax.set_xlabel("")  # 禁用 x 轴标签
        ax.set_ylabel("")  # 禁用 y 轴标签
        ax.set_xticks([])  # 禁用 x 轴刻度
        ax.set_yticks([])  # 禁用 y 轴刻度

        # 显示colorbar
        cbar = plt.colorbar(img)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_percentage))
        month_n = (month + i - 1) % 12 + 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"grad_month{month_n}_{timestamp}.png"
        file_path = Path("grads") / file_name
        with io.BytesIO() as buffer:
            plt.savefig(buffer, bbox_inches="tight")
            if not default_storage.exists(str(file_path)):
                file_path = default_storage.save(str(file_path), ContentFile(buffer.getvalue()))
            result_urls.append(default_storage.url(file_path))
        plt.close()
    return result_urls
