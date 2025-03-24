import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from palettable.colorbrewer.diverging import RdYlBu_11, PuOr_9, BrBG_9
from palettable.colorbrewer.sequential import YlGnBu_9, YlOrRd_9, Blues_9
import os

# 加载 .npy 文件
sic = np.load(
    "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Concentration/1979/01/sea_ice_conc_19790101.npy"
)
siv_u = np.load(
    "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Velocity/sea_ice_x_velocity/1979/01/sea_ice_x_velocity_19790101.npy"
)
siv_v = np.load(
    "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Velocity/sea_ice_y_velocity/1979/01/sea_ice_y_velocity_19790101.npy"
)
t2m = np.load(
    "/data1/Arctic_Ice_Forecasting_Datasets/ERA5/t2m/1979/01/t2m_19790101.npy"
)
u10 = np.load(
    "/data1/Arctic_Ice_Forecasting_Datasets/ERA5/u10/1979/01/u10_19790101.npy"
)
v10 = np.load(
    "/data1/Arctic_Ice_Forecasting_Datasets/ERA5/v10/1979/01/v10_19790101.npy"
)

# 定义NSIDC Sea Ice Polar Stereographic North投影
nsidc_proj = ccrs.Stereographic(
    central_latitude=90, central_longitude=-45, true_scale_latitude=70
)

# 创建 examples 文件夹
save_dir = "examples"
os.makedirs(save_dir, exist_ok=True)

# 定义 50m 分辨率的陆地
land_50m = cfeature.NaturalEarthFeature(
    category="physical",
    name="land",
    scale="50m",
    edgecolor="none",
    facecolor=cfeature.COLORS["land"],
)

# 定义 50m 分辨率的海岸线
coastline_50m = cfeature.NaturalEarthFeature(
    category="physical",
    name="coastline",
    scale="50m",
    edgecolor="black",
    facecolor="none",
)

# 定义 50m 分辨率的河流
rivers_50m = cfeature.NaturalEarthFeature(
    category="physical",
    name="rivers_lake_centerlines",
    scale="50m",
    edgecolor="black",
    facecolor="none",
)

# 定义 50m 分辨率的湖泊
lakes_50m = cfeature.NaturalEarthFeature(
    category="physical",
    name="lakes",
    scale="50m",
    edgecolor="none",
    facecolor=cfeature.COLORS["water"],
)

# 定义降采样步长
step = 6

# 降采样数据
siv_u_subsampled = siv_u[::step, ::step]
siv_v_subsampled = siv_v[::step, ::step]
x_subsampled = np.linspace(-3850000, 3750000, siv_u.shape[1])[::step]
y_subsampled = np.linspace(-5350000, 5850000, siv_u.shape[0])[::step]
X_subsampled, Y_subsampled = np.meshgrid(x_subsampled, y_subsampled)

# 定义统一的布局参数
cbar_pad = 0.05  # colorbar 与图像的间距
title_pad = 15  # 标题与图像的间距

# 保存每张子图
# 1. Sea Ice Concentration (SIC)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
im0 = ax.imshow(
    sic,
    cmap=Blues_9.mpl_colormap.reversed(),
    transform=nsidc_proj,
    extent=[-3850000, 3750000, -5350000, 5850000],
)
ax.set_title("Sea Ice Concentration (SIC)", pad=title_pad)
ax.add_feature(land_50m)
ax.add_feature(coastline_50m)
ax.add_feature(lakes_50m)
gl0 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl0.rotate_labels = False
cbar0 = plt.colorbar(im0, ax=ax, pad=cbar_pad)
cbar0.set_label("Concentration")
plt.savefig(f"{save_dir}/sea_ice_concentration.png", bbox_inches="tight", dpi=300)
plt.close()

# 2. Sea Ice X Velocity (SIV U)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
im1 = ax.imshow(
    siv_u,
    cmap=BrBG_9.mpl_colormap.reversed(),
    transform=nsidc_proj,
    extent=[-3850000, 3750000, -5350000, 5850000],
)
ax.set_title("Sea Ice X Velocity (SIV U)", pad=title_pad)
ax.add_feature(land_50m)
ax.add_feature(coastline_50m)
ax.add_feature(lakes_50m)
gl1 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl1.rotate_labels = False
cbar1 = plt.colorbar(im1, ax=ax, pad=cbar_pad)
cbar1.set_label("Velocity (cm/s)")
plt.savefig(f"{save_dir}/sea_ice_x_velocity.png", bbox_inches="tight", dpi=300)
plt.close()

# 3. Sea Ice Y Velocity (SIV V)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
im2 = ax.imshow(
    siv_v,
    cmap=PuOr_9.mpl_colormap.reversed(),
    transform=nsidc_proj,
    extent=[-3850000, 3750000, -5350000, 5850000],
)
ax.set_title("Sea Ice Y Velocity (SIV V)", pad=title_pad)
ax.add_feature(land_50m)
ax.add_feature(coastline_50m)
ax.add_feature(lakes_50m)
gl2 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl2.rotate_labels = False
cbar2 = plt.colorbar(im2, ax=ax, pad=cbar_pad)
cbar2.set_label("Velocity (cm/s)")
plt.savefig(f"{save_dir}/sea_ice_y_velocity.png", bbox_inches="tight", dpi=300)
plt.close()

# 4. 2m Temperature (T2M)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
im3 = ax.imshow(
    t2m,
    cmap=RdYlBu_11.mpl_colormap.reversed(),
    transform=nsidc_proj,
    extent=[-3850000, 3750000, -5350000, 5850000],
)
ax.set_title("2m Temperature (T2M)", pad=title_pad)
ax.add_feature(coastline_50m)
gl3 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl3.rotate_labels = False
cbar3 = plt.colorbar(im3, ax=ax, pad=cbar_pad)
cbar3.set_label("Temperature (K)")
plt.savefig(f"{save_dir}/2m_temperature.png", bbox_inches="tight", dpi=300)
plt.close()

# 5. 10m U Wind (U10)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
im4 = ax.imshow(
    u10,
    cmap=YlGnBu_9.mpl_colormap,
    transform=nsidc_proj,
    extent=[-3850000, 3750000, -5350000, 5850000],
)
ax.set_title("10m U Wind (U10)", pad=title_pad)
ax.add_feature(coastline_50m)
gl4 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl4.rotate_labels = False
cbar4 = plt.colorbar(im4, ax=ax, pad=cbar_pad)
cbar4.set_label("Wind Speed (m/s)")
plt.savefig(f"{save_dir}/10m_u_wind.png", bbox_inches="tight", dpi=300)
plt.close()

# 6. 10m V Wind (V10)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
im5 = ax.imshow(
    v10,
    cmap=YlOrRd_9.mpl_colormap,
    transform=nsidc_proj,
    extent=[-3850000, 3750000, -5350000, 5850000],
)
ax.set_title("10m V Wind (V10)", pad=title_pad)
ax.add_feature(coastline_50m)
gl5 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl5.rotate_labels = False
cbar5 = plt.colorbar(im5, ax=ax, pad=cbar_pad)
cbar5.set_label("Wind Speed (m/s)")
plt.savefig(f"{save_dir}/10m_v_wind.png", bbox_inches="tight", dpi=300)
plt.close()

# 7. U 方向海冰轨迹（上下翻转）
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
ax.quiver(
    X_subsampled,
    Y_subsampled,
    np.flipud(siv_u_subsampled),
    np.zeros_like(siv_u_subsampled),
    scale=350,
    transform=nsidc_proj,
)
ax.set_title("Sea Ice Motion (U Direction)", pad=title_pad)
ax.add_feature(coastline_50m)
ax.add_feature(lakes_50m)
gl6 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl6.rotate_labels = False
plt.savefig(f"{save_dir}/sea_ice_motion_u.png", bbox_inches="tight", dpi=300)
plt.close()

# 8. V 方向海冰轨迹（上下翻转）
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
ax.quiver(
    X_subsampled,
    Y_subsampled,
    np.zeros_like(siv_v_subsampled),
    np.flipud(siv_v_subsampled),
    scale=350,
    transform=nsidc_proj,
)
ax.set_title("Sea Ice Motion (V Direction)", pad=title_pad)
ax.add_feature(coastline_50m)
ax.add_feature(lakes_50m)
gl7 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl7.rotate_labels = False
plt.savefig(f"{save_dir}/sea_ice_motion_v.png", bbox_inches="tight", dpi=300)
plt.close()

# 9. UV 方向海冰轨迹（合在一起，上下翻转）
plt.figure(figsize=(8, 7))
ax = plt.axes(projection=nsidc_proj)
ax.quiver(
    X_subsampled,
    Y_subsampled,
    np.flipud(siv_u_subsampled),
    np.flipud(siv_v_subsampled),
    scale=350,
    transform=nsidc_proj,
)
ax.set_title("Sea Ice Motion (U + V Direction)", pad=title_pad)
ax.add_feature(coastline_50m)
ax.add_feature(lakes_50m)
gl8 = ax.gridlines(draw_labels=True, linestyle="--", color="black")
gl8.rotate_labels = False
plt.savefig(f"{save_dir}/sea_ice_motion_uv.png", bbox_inches="tight", dpi=300)
plt.close()
