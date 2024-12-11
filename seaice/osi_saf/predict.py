from typing import List

import numpy as np
import torch
import xarray as xr
from PIL import Image

from .utils.model_factory import IceNet

model_path: str = "seaice/osi_saf/checkpoints/checkpoint_SICFN_14.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


def load_model():
    global model
    if model is None:
        model = IceNet().to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["net"])
        model.eval()


def predict_ice_concentration_from_images(
        image_list: List[Image.Image],
) -> np.ndarray:
    """
    从图像列表预测未来的海冰浓度。

    参数:
        image_list: 包含14个PIL图像的列表

    返回:
        np.ndarray: 预测的海冰浓度图，形状为 (14, H, W)
    """
    load_model()

    processed_images = []
    for img in image_list:
        if img.mode != "L":
            img = img.convert("L")
        img_array = np.array(img)
        processed_images.append(img_array)
    input_array = np.stack(processed_images)
    # 针对图片的数据预处理，除以255
    input_array = input_array / 255.0
    return _predict(model, input_array)


def predict_ice_concentration_from_nc_files(
        nc_file_paths: List[str], variable_name: str = "ice_conc"
) -> np.ndarray:
    """
    从多个 .nc 文件预测未来的海冰浓度。

    参数：
        nc_file_path: .nc 文件的路径
        variable_name: 要读取的变量名称，默认为 'sea_ice_concentration'

    返回：
        np.ndarray: 预测的海冰浓度图，形状为 (14, H, W)
    """
    load_model()

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
    return _predict(model, np.stack(nc_data))


def _predict(net, input_array: np.ndarray) -> np.ndarray:
    """
    私有函数，执行实际的预测工作。

    参数:
        input_array: 经过预处理的输入数组，形状为 (14, H, W)

    返回:
        np.ndarray: 预测的海冰浓度图，形状为 (14, H, W)
    """
    try:
        # 转换为张量并调整形状，确保数据在GPU上
        input_tensor = torch.from_numpy(input_array).float()
        input_tensor = input_tensor[None, :, None, :, :].to(device)

        # 模型预测
        with torch.no_grad():
            output = net(input_tensor)

        # 转换为 numpy 数组
        prediction = output.cpu().numpy()[0]

        return prediction
    finally:
        # 清理GPU缓存
        torch.cuda.empty_cache()
