import datetime
import io
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .config import configs
from .trainer import Trainer
from .utils import SIC_dataset


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "config":  # 替换成文件中类的实际名称
            module = "seaice.osi_450_a.config"
        return super().find_class(module, name)


# 从 pkl 文件加载 configs
with open("seaice/osi_450_a/pkls/train_config_SICTeDev_update.pkl", "rb") as f:
    model_configs = CustomUnpickler(f).load()


def adjust_end_time(end_time):
    # 将输入的 YYYYMM 格式时间转换为字符串
    end_time_str = str(end_time)

    # 获取年份和月份部分
    year = int(end_time_str[:4])
    month = int(end_time_str[4:])

    year += 1

    # 格式化调整后的日期为YYYYMM格式
    adjusted_end_time = year * 100 + month
    return adjusted_end_time


def grad_nb(start_time: int, end_time: int, grad_month: int, grad_type: str):
    end_time = adjust_end_time(end_time)
    train_save_dir = "seaice/osi_450_a"

    model_configs.model = configs.model

    dataset_test = SIC_dataset(
        "seaice/osi_450_a/data/full_sic_update.nc",
        start_time,
        end_time,
        model_configs.input_gap,
        model_configs.input_length,
        model_configs.pred_shift,
        model_configs.pred_gap,
        model_configs.pred_length,
        samples_gap=12,
    )

    dataloader_test = DataLoader(dataset_test, shuffle=False)

    tester = Trainer(model_configs)

    tester.network.load_state_dict(
        torch.load(
            f"{train_save_dir}/checkpoints/checkpoint_{configs.model}_update.chk"
        )["net"]
    )

    start_time_str = str(start_time)
    start_month = int(start_time_str[4:6])
    month_sequence = []
    for i in range(12):
        current_month = (start_month + i - 1) % 12 + 1
        month_sequence.append(current_month)

    month_position = month_sequence.index(grad_month)

    sic_grad = tester.get_grad(dataloader_test, month_position, grad_type)

    sic_grad = np.array(sic_grad)

    existing_array = sic_grad

    # 对数组进行处理，按第一个维度求平均
    mean_array = np.mean(existing_array, axis=0)

    ################
    if grad_type == "sum":
        mean_array[mean_array > 4] = 4
    else:
        mean_array[mean_array > 0.05] = 0.05
    mean_array[mean_array < 4e-06] = 4e-06
    max_value = np.max(mean_array)
    min_value = np.min(mean_array)

    mean_array = (mean_array - min_value) / (max_value - min_value)

    land_mask = np.load("seaice/osi_450_a/data/land_mask.npy")
    times = np.load("seaice/osi_450_a/data/times.npy")

    grad_month_np = mean_array

    pred_length = times.shape[1] // 2
    pred_times = times[0, pred_length:]

    grad_month_np = grad_month_np[0, :, 0, :]

    def percentage(x, pos):
        return f"{x:.0%}"

    def plot_sic1(sic, cmap):
        result_urls = []
        # 绘制海冰浓度图
        start_time_str = str(start_time)
        month = int(start_time_str[4:6])
        for i, time in enumerate(pred_times):
            fig, ax = plt.subplots()
            sic[i] = gaussian_filter(sic[i], sigma=3)
            img = ax.imshow(sic[i], cmap=cmap, vmin=0, vmax=1)

            # 绘制陆地
            #         land_color = "#d2b48c"  # 陆地的颜色
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
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage))
            month_n = (month + i - 1) % 12 + 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"grad_month{month_n}_{timestamp}.png"
            file_path = Path("grads") / file_name
            with io.BytesIO() as buffer:
                plt.savefig(buffer, bbox_inches="tight")
                plt.close()
                if not default_storage.exists(str(file_path)):
                    file_path = default_storage.save(
                        str(file_path), ContentFile(buffer.getvalue())
                    )
                result_urls.append(default_storage.url(file_path))
        return result_urls

    # 创建渐变色的colormap
    colors = [(0, "white"), (1, "#D50103")]  # 定义颜色映射
    cmap_ice_conc1 = LinearSegmentedColormap.from_list("cmap_ice_conc", colors)

    result_urls = plot_sic1(grad_month_np, cmap_ice_conc1)
    torch.cuda.empty_cache()
    return result_urls
