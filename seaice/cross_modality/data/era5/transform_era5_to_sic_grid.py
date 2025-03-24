from datetime import datetime
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata
import os
from tqdm import tqdm

# 定义 EPSG:3411 投影
proj_3411 = "EPSG:3411"  # NSIDC 北极极射投影
proj_wgs84 = "EPSG:4326"  # WGS84 经纬度坐标系

# 创建转换器
transformer = Transformer.from_crs(proj_3411, proj_wgs84)


def save_data(data, output_dir, filename):
    """保存数据到指定目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, filename)
    np.save(output_file, data)


# 打开 SIC NetCDF 文件
sic_file = "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Concentration/2023/12/sic_psn25_20231231_F17_v05r00.nc"
sic_data = xr.open_dataset(sic_file)

# 获取 SIC 的网格坐标
x_coords = sic_data["x"].values  # 448 个 x 坐标
y_coords = sic_data["y"].values  # 304 个 y 坐标

# 创建网格
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

# 转换网格坐标为经纬度
sic_lat, sic_lon = transformer.transform(x_grid, y_grid)

# 将 SIC 的经纬度展平
grid_points = np.column_stack((sic_lon.flatten(), sic_lat.flatten()))

# 打开 ERA5 NetCDF 文件
era5_file = "/data1/Arctic_Ice_Forecasting_Datasets/ERA5/era5.nc"
era5_data = xr.open_dataset(era5_file)

# 获取 ERA5 的经纬度和时间
era5_lat = era5_data["latitude"].values  # 721 个 latitude 坐标
era5_lon = era5_data["longitude"].values  # 1440 个 longitude 坐标
era5_time = era5_data["valid_time"].values  # 时间维度，16436个时间戳

# 将 ERA5 的经度从 0-360 转换为 -180 到 180
era5_lon = np.where(era5_lon > 180, era5_lon - 360, era5_lon)

# 创建 ERA5 的二维经纬度网格
era5_lon_grid, era5_lat_grid = np.meshgrid(era5_lon, era5_lat)

# 将 ERA5 的经纬度展平
era5_points = np.column_stack((era5_lon_grid.flatten(), era5_lat_grid.flatten()))

# 定义要处理的变量列表
variables = ["u10", "v10", "t2m"]

# 按顺序处理每个变量
for var in variables:
    print(f"Processing variable: {var}")
    # 获取当前变量的数据
    var_data = era5_data[var].values  # 时间 x 纬度 x 经度

    # 遍历 ERA5 的每一天数据
    for i in tqdm(range(len(era5_time)), desc=f"Processing {var}"):
        # 获取当前日期的变量值
        var_day = var_data[i]

        # 将变量值展平
        var_values = var_day.flatten()

        # 使用 griddata 进行插值
        var_result = griddata(
            era5_points, var_values, grid_points, method="nearest"
        ).reshape(448, 304)

        # 替换 NaN 值为 0
        var_result = np.nan_to_num(var_result)

        # 将数据转换为 float32 类型
        var_result = var_result.astype(np.float32)

        # 获取当前日期
        current_time = era5_time[i].astype("datetime64[s]").astype(datetime)
        date_str = (
            f"{current_time.year:04d}{current_time.month:02d}{current_time.day:02d}"
        )

        # 保存变量数据
        var_output_dir = os.path.join(
            f"/data1/Arctic_Ice_Forecasting_Datasets/ERA5/{var}",
            f"{current_time.year:04d}",
            f"{current_time.month:02d}",
        )
        save_data(var_result, var_output_dir, f"{var}_{date_str}.npy")

    # 释放当前变量的内存
    del var_data
    era5_data[var].close()

# 关闭 ERA5 文件
era5_data.close()
