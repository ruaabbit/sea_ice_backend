import os

import numpy as np
import xarray as xr
from tqdm import tqdm  # 导入 tqdm

# 读取txt文件中的路径
file_paths = np.genfromtxt("nc_sic_path.txt", dtype=str)

# 遍历每个文件路径，并使用 tqdm 显示进度条
for file_path in tqdm(file_paths, desc="Processing files"):
    # 读取.nc文件
    sea_ice_conc = xr.open_dataset(file_path)["cdr_seaice_conc"].values[0]
    sea_ice_conc = np.nan_to_num(sea_ice_conc)  # 替换 NaN 值为 0

    # 将数据转换为 float32 类型
    sea_ice_conc = sea_ice_conc.astype(np.float32)

    # 从文件路径中提取日期信息
    # 文件名格式为 "sic_psn25_YYYYMMDD_n07_v05r00.nc"
    file_name = os.path.basename(file_path)
    date = file_name.split("_")[2]  # 提取日期部分，例如 "19790101"

    # 构建输出文件路径
    output_dir = os.path.dirname(file_path)  # 获取文件所在目录
    output_file = os.path.join(output_dir, f"sea_ice_conc_{date}.npy")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存为.npy文件
    np.save(output_file, sea_ice_conc)
