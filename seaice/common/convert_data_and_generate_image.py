import datetime
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

land_mask = np.load("seaice/common/data/land_mask.npy")  # 1: land, 0: ocean


def prediction_result_to_image(prediction_result: np.ndarray):
    # 生成带时间戳的随机文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_num = random.randint(10000, 99999)
    file_name = f"predict_task_{timestamp}_{random_num}.png"

    buffer = io.BytesIO()
    data = np.array(prediction_result)

    # 将数据归一化到0-255范围内
    data = (data * 255).astype(np.uint8)

    # 初始化RGB图像为白色
    rgb_data = np.ones((data.shape[0], data.shape[1], 3), dtype=np.uint8) * 255

    # land_mask部分设为黄色
    rgb_data[land_mask == 1] = [200, 200, 0]

    # data为0且land_mask为0的部分设为蓝色
    rgb_data[(data == 0) & (land_mask == 0)] = [0, 0, 200]

    # 创建图片对象
    image = Image.fromarray(rgb_data)

    # 保存为 PNG 图片
    image.save(buffer, format='PNG')

    # 保存文件
    file_path = Path("predicts") / file_name
    if not default_storage.exists(str(file_path)):
        file_path = default_storage.save(
            str(file_path), ContentFile(buffer.getvalue())
        )
    file_url = default_storage.url(file_path)
    buffer.close()

    return file_url
