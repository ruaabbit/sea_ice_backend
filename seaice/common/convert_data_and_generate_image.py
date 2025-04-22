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
        image.save(buffer, format="webp", lossless=True)

        # 文件保存部分相
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"predict_task_{timestamp}_{random.randint(10000, 99999)}.webp"
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
    if cmap_style == "ice":
        # 专为海冰设计的渐变，蓝色表示海洋，浅蓝表示新冰，白色表示厚冰
        colors = [
            "#0e2d59",  # 深蓝 (海洋) - 保持不变
            "#1a5c8c",  # 中深蓝 (优化过渡)
            "#4a8fc1",  # 中等蓝 (优化过渡)
            "#a0d0f0",  # 浅蓝白 (优化过渡)
            "#ffffff",  # 白色 (高浓度) - 保持不变
        ]
    else:
        # 默认使用原始方案
        ocean_color = "#0e2d59"
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


COASTLINE = cfeature.NaturalEarthFeature(
    "physical", "coastline", "50m", edgecolor="black", facecolor="never"
)
"""Automatically scaled coastline, including major islands."""

LAKES = cfeature.NaturalEarthFeature(
    "physical", "lakes", "50m", edgecolor="none", facecolor=cfeature.COLORS["water"]
)
"""Automatically scaled natural and artificial lakes."""

LAND = cfeature.NaturalEarthFeature(
    "physical",
    "land",
    "50m",
    edgecolor="none",
    facecolor=cfeature.COLORS["land"],
    zorder=-1,
)
"""Automatically scaled land polygons, including major islands."""

OCEAN = cfeature.NaturalEarthFeature(
    "physical",
    "ocean",
    "50m",
    edgecolor="none",
    facecolor=cfeature.COLORS["water"],
    zorder=-1,
)
"""Automatically scaled ocean polygons."""

RIVERS = cfeature.NaturalEarthFeature(
    "physical",
    "rivers_lake_centerlines",
    "50m",
    edgecolor=cfeature.COLORS["water"],
    facecolor="never",
)


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

    ax.add_feature(cfeature.OCEAN, color="#0e2d59", zorder=0)

    # 创建海冰颜色映射
    ice_cmap = create_ice_colormap(cmap_style="ice", n_colors=256)

    # --- 修改：简化掩码逻辑 ---
    # 准备用于 pcolormesh 的数据
    plot_data = ice_conc

    # --- 绘图顺序和方法 ---
    # 1. 绘制海冰和海洋
    ax.pcolormesh(
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
    ax.add_feature(cfeature.LAND, color="#e1e5cf", zorder=2)  # 灰色陆地

    # 3. 绘制海岸线
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3, color="black")

    # 4. 绘制湖泊和河流
    ax.add_feature(cfeature.LAKES, color="#4a8fc1", zorder=4)  # 浅蓝色湖泊
    ax.add_feature(cfeature.RIVERS, color="#0e2d59", zorder=5)  # 深蓝色河流

    # 设置图形边界紧凑
    fig.tight_layout(pad=0)

    # 保存图像
    with io.BytesIO() as buffer:
        plt.savefig(
            buffer,
            format="webp",
            pil_kwargs={"lossless": True},
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="w",
            edgecolor="w",
            transparent=False,
        )
        plt.close()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"predict_globe_{timestamp}_{random.randint(10000, 99999)}.webp"
        file_path = Path("predicts") / file_name

        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                str(file_path), ContentFile(buffer.getvalue())
            )

    return default_storage.url(file_path)
