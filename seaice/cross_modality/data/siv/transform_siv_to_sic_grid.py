import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata
import os
import glob
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

# 获取所有年份的 SIV 文件，在你的一个目录下的同类型命名的文件
siv_files = glob.glob(
    "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Velocity/icemotion_daily_nh_25km_*_v4.1.nc"
)

# 定义要处理的变量列表
variables = ["u", "v"]

# 遍历每个 SIV 文件
for siv_file in tqdm(siv_files, desc="Processing SIV files"):  # 添加进度条
    # 打开 SIV NetCDF 文件
    siv_data = xr.open_dataset(siv_file)

    # 获取 SIV 的经纬度和变量值
    u_lat = siv_data["latitude"].values  # 361x361 纬度
    u_lon = siv_data["longitude"].values  # 361x361 经度
    u = siv_data["u"].values  # 时间 x 361x361 变量值
    v = siv_data["v"].values  # 时间 x 361x361 变量值

    # 获取时间维度
    times = siv_data["time"].values  # cftime.DatetimeJulian 格式

    # 逐个处理变量
    for var_name in variables:
        # 获取当前变量的数据
        var_data = siv_data[var_name].values  # 时间 x 361x361 变量值

        # 遍历每一天的数据
        for i in tqdm(
                range(var_data.shape[0]),
                desc=f"Processing days in {os.path.basename(siv_file)} for {var_name}",
                leave=False,
        ):  # 添加进度条
            # 获取当前日期的变量值
            var_day = var_data[i]

            # 将 SIV 的经纬度展平
            points = np.column_stack((u_lon.flatten(), u_lat.flatten()))

            # 处理变量
            var_values = var_day.flatten()
            var_result = griddata(
                points, var_values, grid_points, method="nearest"
            ).reshape(448, 304)
            var_result = np.nan_to_num(var_result)  # 替换 NaN 值为 0
            var_result = var_result.astype(np.float32)  # 将数据转换为 float32 类型

            # 获取当前日期（cftime.DatetimeJulian 格式）
            current_time = times[i]
            date_str = (
                f"{current_time.year:04d}{current_time.month:02d}{current_time.day:02d}"
            )

            # 保存变量数据
            var_output_dir = os.path.join(
                f"/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Velocity/sea_ice_{var_name}_velocity",
                f"{current_time.year:04d}",
                f"{current_time.month:02d}",
            )
            save_data(
                var_result,
                var_output_dir,
                f"sea_ice_{var_name}_velocity_{date_str}.npy",
            )

    # 关闭 SIV 文件
    siv_data.close()
