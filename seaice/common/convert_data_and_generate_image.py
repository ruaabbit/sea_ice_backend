import datetime
import io
import random
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from matplotlib.colors import LinearSegmentedColormap

# 全局常量
LAND_MASK_DATA = np.load("seaice/common/data/land_mask.npy")
WATER_MASK = LAND_MASK_DATA == 0
LAND_MASK = LAND_MASK_DATA == 1

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
        image.save(buffer, format="PNG", optimize=True)  # 启用PNG优化

        # 文件保存部分相
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"predict_task_{timestamp}_{random.randint(10000, 99999)}.png"
        file_path = Path("predicts") / file_name

        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                str(file_path), ContentFile(buffer.getvalue())
            )

    return default_storage.url(file_path)


def create_ice_colormap(cmap_style="default", alpha=1.0, n_colors=256):
    """创建渐变色映射，用于表示海冰浓度

    参数:
        cmap_style: 颜色映射样式 ('default', 'blue_white_red', 'rainbow', 'ice')
        alpha: 透明度 (0.0-1.0)
        n_colors: 颜色分级数量

    返回:
        matplotlib颜色映射对象
    """
    # 不同的预设颜色方案
    if cmap_style == "default":
        # 原始的深蓝到红色渐变
        ocean_color = "#0077be"  # 深蓝色
        colors = [ocean_color, "red"]

    elif cmap_style == "blue_white_red":
        # 蓝-白-红渐变，更好地反映冷热变化
        colors = ["#0077be", "#ffffff", "#ff0000"]

    elif cmap_style == "rainbow":
        # 从蓝色到红色的完整光谱
        colors = ["#0077be", "#00bfff", "#00ffff", "#bfff00", "#ffbf00", "#ff0000"]

    elif cmap_style == "ice":
        # 专为海冰设计的渐变，蓝色表示海洋，浅蓝表示新冰，白色表示厚冰
        colors = [
            "#0077be",  # 深蓝 (海洋)
            "#00a5db",  # 浅蓝 (低浓度)
            "#b1e4ff",  # 淡蓝 (中低浓度)
            "#e1f2fe",  # 非常浅的蓝 (中等浓度)
            "#ffffff",  # 白色 (高浓度)
        ]
    else:
        # 默认使用原始方案
        ocean_color = "#0077be"
        colors = [ocean_color, "red"]

    # 创建颜色映射
    cmap = LinearSegmentedColormap.from_list("ice_cmap", colors, N=n_colors)

    # 如果需要透明度，创建带alpha通道的版本
    if alpha < 1.0:
        # 获取原始颜色表
        cmap_colors = cmap(np.linspace(0, 1, n_colors))
        # 设置alpha通道
        cmap_colors[:, 3] = alpha
        # 用修改后的颜色创建新颜色映射
        cmap = LinearSegmentedColormap.from_list(
            f"ice_cmap_alpha{alpha}", cmap_colors, N=n_colors
        )

    return cmap


def prediction_result_to_globe_image(prediction_result: np.ndarray):
    """
    Convert prediction result to a globe-projected image using Cartopy.
    Returns the URL of the saved image.
    """
    lat = np.load("seaice/common/data/lat.npy")
    lon = np.load("seaice/common/data/lon.npy")
    lat_mask = np.isnan(lat)
    lon_mask = np.isnan(lon)
    lat = ma.masked_array(lat, mask=lat_mask)
    lon = ma.masked_array(lon, mask=lon_mask)

    ice_conc_mask = prediction_result == 0
    ice_conc = ma.masked_array(prediction_result, mask=ice_conc_mask)
    fig = plt.figure(figsize=(10, 5), facecolor="w")

    # 使用等距离圆柱投影
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=90))

    # 完全隐藏所有边框和轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axis_off()  # 完全隐藏坐标轴

    # 设置全球视图
    ax.set_global()

    ocean_color = "#0077be"
    ax.add_feature(cfeature.OCEAN, color=ocean_color, zorder=0)

    # 创建海冰颜色映射
    ice_cmap = create_ice_colormap(cmap_style="ice", n_colors=256)

    # --- 修改：简化掩码逻辑 ---
    # 准备用于 pcolormesh 的数据
    plot_data = ice_conc

    # --- 绘图顺序和方法 ---
    # 1. 绘制海冰和海洋
    im = ax.pcolormesh(
        lon,
        lat,
        plot_data,
        transform=ccrs.PlateCarree(),
        cmap=ice_cmap,
        vmin=0,
        vmax=1,
        zorder=1,
        shading="auto",
    )

    # 2. 绘制陆地
    ax.add_feature(cfeature.LAND, color="#c0c0c0", zorder=2)  # 灰色陆地

    # 3. 绘制海岸线
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3, color="black")

    # 设置图形边界紧凑
    fig.tight_layout(pad=0)

    # 保存图像
    with io.BytesIO() as buffer:
        plt.savefig(
            buffer,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="w",
            edgecolor="w",
            transparent=False,
        )
        plt.close()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"predict_globe_{timestamp}_{random.randint(10000, 99999)}.png"
        file_path = Path("predicts") / file_name

        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                str(file_path), ContentFile(buffer.getvalue()))

    return default_storage.url(file_path)
