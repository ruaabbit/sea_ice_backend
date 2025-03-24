""" 
实现了从 Arctic NetCDF 文件中提取 mask 的功能。
提取的arctic_mask将作为模型关注的海洋区域，1代表海洋区域，0代表非海洋区域。
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# 打开 NetCDF 文件
file_path = "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Concentration/2023/12/sic_psn25_20231231_F17_v05r00.nc"
dataset = xr.open_dataset(file_path)

# 打印数据集的概览
print(dataset)

# 选择一个变量进行处理，例如 'cdr_seaice_conc'
if "cdr_seaice_conc" in dataset:
    data_var = dataset["cdr_seaice_conc"][0]

    # 将数据转换为 numpy 数组
    data_array = data_var.values

    # 提取 NaN 区域作为 mask
    mask = ~np.isnan(data_array)  # True 表示 NaN，False 表示非 NaN

    # 保存 mask 到文件
    np.save("arctic_mask.npy", mask)

    # 可视化 mask
    plt.imshow(mask, cmap="gray")  # 使用灰度 colormap，True 为白色，False 为黑色
    plt.axis("off")  # 关闭坐标轴显示
    plt.savefig("arctic_mask.png", bbox_inches="tight", pad_inches=0)

else:
    print("Variable 'cdr_seaice_conc' not found in the dataset.")
