import datetime
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# 全局常量
LAND_MASK_DATA = np.load("seaice/common/data/land_mask.npy")
WATER_MASK = (LAND_MASK_DATA == 0)
LAND_MASK = (LAND_MASK_DATA == 1)

WATER_COLOR = np.array([0, 0.3, 0.8])
LAND_COLOR = np.array([0.9, 0.8, 0.2])
ICE_COLOR_BASE = np.array([0.85, 0.85, 0.9])
ICE_COLOR_RANGE = np.array([0.15, 0.15, 0.1])


def prediction_result_to_image(prediction_result: np.ndarray):
    """
    使用PIL直接生成图像的备选方案，适用于不需要matplotlib特性的情况
    """

    # 数据预处理部分相同...
    data = np.clip(prediction_result, 0, 1)
    rgb_image = np.zeros((432, 432, 3), dtype=np.float32)
    ice_mask = WATER_MASK & (data > 0)
    rgb_image[WATER_MASK & (data == 0)] = WATER_COLOR
    rgb_image[LAND_MASK] = LAND_COLOR

    if np.any(ice_mask):
        ice_colors = ICE_COLOR_BASE + data[ice_mask, np.newaxis] * ICE_COLOR_RANGE
        rgb_image[ice_mask] = ice_colors

    # 转换为PIL图像并直接保存
    rgb_image = (rgb_image * 255).astype(np.uint8)
    image = Image.fromarray(rgb_image)

    with io.BytesIO() as buffer:
        image.save(buffer,
                   format='PNG',
                   optimize=True)  # 启用PNG优化

        # 文件保存部分相
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"predict_task_{timestamp}_{random.randint(10000, 99999)}.png"
        file_path = Path("predicts") / file_name

        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(str(file_path), ContentFile(buffer.getvalue()))

    return default_storage.url(file_path)
