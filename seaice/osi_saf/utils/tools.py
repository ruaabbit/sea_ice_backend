import datetime
import logging

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import xarray as xr
from PIL import Image
from dask.diagnostics import ProgressBar
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset


def setup_logging(log_file):
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器并设置日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建流处理器 (StreamHandler) 用于输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def generate_date_list(start_time, end_time):
    """
    Generate a list of dates.

    Args:
        start_time: Start date in the format YYYYMMDD.
        end_time: End date in the format YYYYMMDD.

    Returns:
        List of dates in the format of list[YYYYMMDD].
    """
    start = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    return [
        int(dt.strftime("%Y%m%d"))
        for dt in [start + relativedelta(days=i) for i in range((end - start).days + 1)]
    ]


def write_processed_data_to_netcdf(data_paths_file, start_time, end_time, out_path):
    """
    Write processed data to a .nc file for future use.

    Args:
        data_paths_file: Path to a text file containing the file paths.
        start_time: Start time for data filtering.
        end_time: End time for data filtering.
        out_path: Path of the output .nc file.
    """
    data_paths = np.genfromtxt(data_paths_file, dtype=str)
    times = np.array(
        [int(path.split("/")[-1].split("_")[5][0:8]) for path in data_paths]
    )
    indices = np.where((times >= start_time) & (times <= end_time))[0]
    data_paths = data_paths[indices]
    times = times[indices]

    data_arrays = [
        create_xarray_data_array(path, time) for path, time in zip(data_paths, times)
    ]

    dataset = xr.concat(data_arrays, dim="time").to_dataset(name="imgs")
    with ProgressBar():
        dataset.to_netcdf(
            out_path,
            encoding={"imgs": {"dtype": "f4", "zlib": True, "complevel": 4}},
            compute=True,
            engine="h5netcdf",
        )


def create_xarray_data_array(data_path, time):
    """
    Load data from a path and return a processed xarray DataArray.

    Args:
        data_path: Path to the data file.
        time: Time associated with the data.

    Returns:
        xarray DataArray with added time dimension.
    """
    data = da.from_delayed(
        dask.delayed(load_and_process_data)(data_path),
        shape=(432, 432),
        dtype="float32",
    )
    x_coords = da.arange(data.shape[1])  # x coordinates
    y_coords = da.arange(data.shape[0])  # y coordinates

    return xr.DataArray(
        data, dims=["yc", "xc"], coords={"yc": y_coords, "xc": x_coords}
    ).expand_dims({"time": [time]})


def load_and_process_data(data_path):
    """
    Load and process data from a given path.

    Args:
        data_path: Path to the data file.

    Returns:
        Processed data.
    """
    data = xr.open_dataset(data_path)
    return process_sea_ice_data(data)


def process_sea_ice_data(data):
    """
    0 - 100 Sea ice concentration %
    -32767 Land

    处理数据，包括归一化、处理缺失数据、陆地屏蔽等
    Args:
        data: 输入的海冰数据
    Returns:
        ice_conc: 处理后的海冰密集度数据
    """
    ice_conc = np.array(data["ice_conc"][:][0])
    ice_conc = np.nan_to_num(ice_conc, nan=0)

    # 处理陆地
    ice_conc[ice_conc == -32767] = 0

    # 归一化至[0-1]
    ice_conc = ice_conc / 100

    # 确保没有超出范围的值
    # assert not np.any(ice_conc > 1)

    return ice_conc


def unfold_stack_over_channel(image, patch_size):
    """
    将图像切分成多个小块并沿着通道堆叠
    Args:
        image (N, *, C, H, W): 最后两个维度必须是空间维度
        patch_size(k_h,k_w): 长度为2的元组，就是configs.patch_size
    Returns:
        output (N, *, C*k_h*k_w, H/k_h, W/k_w)
    """
    n_dims = len(image.shape)
    # assert n_dims == 4 or n_dims == 5
    if patch_size[0] == 1 and patch_size[1] == 1:
        return image

    patch = image.unfold(-2, size=patch_size[0], step=patch_size[0])
    patch = patch.unfold(-2, size=patch_size[1], step=patch_size[1]).flatten(-2)
    if n_dims == 4:  # (N, C, H, W)
        patch = patch.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dims == 5:  # (N, T, C, H, W)
        patch = patch.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    # assert patch.shape[-3] == image.shape[-3] * patch_size[0] * patch_size[1]
    return patch


def fold_tensor(tensor, output_size, patch_size):
    """
    用non-overlapping的块重建图像
    Args:
        tensor shape (N, *, C*k_h*k_w, h, w)
        output_size: (H, W)，要重建的原始图像的大小
        patch_size: (k_h, k_w)
        请注意，对于non-overlapping的滑动窗口，通常stride等于patch_size
    Returns:
        output (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    n_dims = len(tensor.shape)
    # assert n_dims == 4 or n_dims == 5

    if patch_size[0] == 1 and patch_size[1] == 1:
        return tensor

    # 展平输入
    flattened = tensor.flatten(0, 1) if n_dims == 5 else tensor

    # 使用 F.fold 函数进行重建
    folded = F.fold(
        flattened.flatten(-2),
        output_size=output_size,
        kernel_size=patch_size,
        stride=patch_size,
    )

    if n_dims == 5:
        folded = folded.view(tensor.size(0), tensor.size(1), *folded.shape[1:])

    return folded


def prepare_input_target_indices(
        time_length,
        input_gap,
        input_length,
        prediction_shift,
        prediction_gap,
        prediction_length,
        sample_gap,
):
    """
    Args:
        input_gap: 两个连续输入帧之间的时间间隔
        input_length: 输入帧的数量
        prediction_shift: 最后一个目标预测的前导时间
        prediction_gap: 两个连续输出帧之间的时间间隔
        prediction_length: 输出帧的数量
        sample_gap: 两个检索样本的起始时间之间的间隔
    Returns:
        input_indices: 指向输入样本位置的索引
        target_indices: 指向目标样本位置的索引
    """
    # assert prediction_shift >= prediction_length
    input_span = input_gap * (input_length - 1) + 1
    input_index = np.arange(0, input_span, input_gap)
    target_index = (
            np.arange(0, prediction_shift, prediction_gap) + input_span + prediction_gap - 1
    )
    indices = np.concatenate([input_index, target_index]).reshape(
        1, input_length + prediction_length
    )
    max_sample_count = time_length - (input_span + prediction_shift - 1)
    indices = indices + np.arange(max_sample_count)[:, np.newaxis] @ np.ones(
        (1, input_length + prediction_length), dtype=int
    )
    input_indices = indices[::sample_gap, :input_length]
    target_indices = indices[::sample_gap, input_length:]
    # assert len(input_indices) == len(target_indices)
    return input_indices, target_indices


def nc_to_png(nc_file_path, png_file_path, variable_name='ice_conc'):
    """
    将 nc 文件中的数据读取并保存为精确 432x432 像素的 PNG 图片。

    参数：
    - nc_file_path: str，nc 文件的路径。
    - png_file_path: str，保存 PNG 图片的路径。
    - variable_name: str，nc 文件中表示海冰数据的变量名。
    """
    # 打开 nc 文件
    with Dataset(nc_file_path, 'r') as nc_file:
        # 读取变量数据
        print(nc_file.variables)
        if variable_name in nc_file.variables:
            sea_ice_data = nc_file.variables[variable_name][:]
        else:
            raise KeyError(f"变量 '{variable_name}' 不存在于 nc 文件中。")

    # 检查数据的形状是否符合要求
    if sea_ice_data.shape[-2:] != (432, 432):
        raise ValueError("数据的空间分辨率不是 432x432，请检查文件内容。")

    # 进行数据预处理
    sea_ice_data = np.squeeze(sea_ice_data)  # 去掉单通道维度
    sea_ice_data = np.where(sea_ice_data == -32767, 0, sea_ice_data)


    sea_ice_data = sea_ice_data / 100

    # 创建图像并设置精确的DPI
    dpi = 72  # 标准显示DPI
    figsize = (432 / dpi, 432 / dpi)  # 根据目标像素和DPI计算图像尺寸

    # 创建图像
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 创建填充整个图像的轴
    ax.set_axis_off()
    fig.add_axes(ax)

    # 显示图像
    ax.imshow(sea_ice_data, cmap='plasma', vmin=0, vmax=1)

    # 保存为精确尺寸的PNG
    plt.savefig(png_file_path,
                dpi=dpi,
                format='png',
                bbox_inches=None,
                pad_inches=0)
    plt.close()

    # 验证保存的图片尺寸
    from PIL import Image
    with Image.open(png_file_path) as img:
        print(f"保存的图片尺寸为: {img.size}")

    print(f"图片已成功保存到 {png_file_path}")


def png_to_nc(png_file_path, nc_file_path, variable_name='ice_conc'):
    """
    将 PNG 图片中的数据读取并保存为 .nc 文件。

    参数：
    - png_file_path: str，PNG 图片的路径。
    - nc_file_path: str，保存的 .nc 文件的路径。
    - variable_name: str，.nc 文件中表示海冰数据的变量名。
    """
    # 打开 PNG 图片并转换为灰度模式（如果需要）
    image = Image.open(png_file_path).convert('L')
    image_data = np.array(image).astype(np.float32) / 255.0  # 将数据归一化到 [0, 1] 范围

    # 恢复到原始比例（假设 PNG 中的数据是通过将数值乘以100得到的）
    image_data = image_data * 100.0

    # 创建一个新的 nc 文件
    with Dataset(nc_file_path, 'w', format='NETCDF4') as nc_file:
        # 创建维度
        nc_file.createDimension('x', 432)
        nc_file.createDimension('y', 432)

        # 创建变量
        sea_ice_var = nc_file.createVariable(variable_name, 'f4', ('x', 'y'))

        # 将数据写入变量
        sea_ice_var[:] = image_data

        # 添加简单的描述
        sea_ice_var.units = "percentage"
        sea_ice_var.long_name = "Sea Ice Concentration"

    print(f"数据已成功保存到 {nc_file_path}")


if __name__ == '__main__':
    pub_path = 'media/downloads/'
    # for i in range(1, 20):
    #     nc_path = f'2020/01/ice_conc_nh_ease2-250_cdr-v3p0_202001{str(i).zfill(2)}1200.nc'
    #     png_path = 'png/' + nc_path.replace('.nc', '.png')
    #     nc_to_png(pub_path + nc_path, pub_path + png_path)
    nc_path = r'2020/01/asi-AMSR2-n3125-20241105-v5.4.nc'
    png_path = 'png/' + nc_path.replace('.nc', '.png')
    nc_to_png(pub_path + nc_path, pub_path + png_path)