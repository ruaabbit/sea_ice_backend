""" 计算多模态数据集的最大值和最小值，为数据归一化做准备 """

import numpy as np
from tqdm import tqdm  # 导入 tqdm 库

# 加载路径文件
sic_path = np.genfromtxt("sic_path.txt", dtype=str)
siv_u_path = np.genfromtxt("siv_u_path.txt", dtype=str)
siv_v_path = np.genfromtxt("siv_v_path.txt", dtype=str)
u10_path = np.genfromtxt("u10_path.txt", dtype=str)
v10_path = np.genfromtxt("v10_path.txt", dtype=str)
t2m_path = np.genfromtxt("t2m_path.txt", dtype=str)

# 确保路径数量一致
assert (
    len(sic_path)
    == len(siv_u_path)
    == len(siv_v_path)
    == len(u10_path)
    == len(v10_path)
    == len(t2m_path)
)

# 初始化每个变量的最大值和最小值
sic_max_values = []
sic_min_values = []
siv_u_max_values = []
siv_u_min_values = []
siv_v_max_values = []
siv_v_min_values = []
u10_max_values = []
u10_min_values = []
v10_max_values = []
v10_min_values = []
t2m_max_values = []
t2m_min_values = []

# 使用 tqdm 添加进度条
for i in tqdm(range(len(sic_path)), desc="Processing files"):
    # 加载数据
    ice_conc = np.load(sic_path[i])  # sic 数据
    siv_u = np.load(siv_u_path[i])  # siv_u 数据
    siv_v = np.load(siv_v_path[i])  # siv_v 数据
    u10 = np.load(u10_path[i])  # u10 数据
    v10 = np.load(v10_path[i])  # v10 数据
    t2m = np.load(t2m_path[i])  # t2m 数据

    # 计算当前文件的最大值和最小值
    sic_max_values.append(ice_conc.max())  # sic 最大值
    sic_min_values.append(ice_conc.min())  # sic 最小值

    siv_u_max_values.append(siv_u.max())  # siv_u 最大值
    siv_u_min_values.append(siv_u.min())  # siv_u 最小值

    siv_v_max_values.append(siv_v.max())  # siv_v 最大值
    siv_v_min_values.append(siv_v.min())  # siv_v 最小值

    u10_max_values.append(u10.max())  # u10 最大值
    u10_min_values.append(u10.min())  # u10 最小值

    v10_max_values.append(v10.max())  # v10 最大值
    v10_min_values.append(v10.min())  # v10 最小值

    t2m_max_values.append(t2m.max())  # t2m 最大值
    t2m_min_values.append(t2m.min())  # t2m 最小值

# 计算每个变量的全局最大值和最小值
sic_global_max = max(sic_max_values)  # sic 全局最大值
sic_global_min = min(sic_min_values)  # sic 全局最小值

siv_u_global_max = max(siv_u_max_values)  # siv_u 全局最大值
siv_u_global_min = min(siv_u_min_values)  # siv_u 全局最小值

siv_v_global_max = max(siv_v_max_values)  # siv_v 全局最大值
siv_v_global_min = min(siv_v_min_values)  # siv_v 全局最小值

u10_global_max = max(u10_max_values)  # u10 全局最大值
u10_global_min = min(u10_min_values)  # u10 全局最小值

v10_global_max = max(v10_max_values)  # v10 全局最大值
v10_global_min = min(v10_min_values)  # v10 全局最小值

t2m_global_max = max(t2m_max_values)  # t2m 全局最大值
t2m_global_min = min(t2m_min_values)  # t2m 全局最小值

# 将最大值和最小值存储为 numpy 数组
max_values = np.array(
    [
        sic_global_max,
        siv_u_global_max,
        siv_v_global_max,
        u10_global_max,
        v10_global_max,
        t2m_global_max,
    ]
).astype(
    np.float32
)  # 最大值数组
min_values = np.array(
    [
        sic_global_min,
        siv_u_global_min,
        siv_v_global_min,
        u10_global_min,
        v10_global_min,
        t2m_global_min,
    ]
).astype(
    np.float32
)  # 最小值数组

# 保存最大值和最小值到 .npy 文件
np.save("max_values.npy", max_values)
np.save("min_values.npy", min_values)

print("最大值数组:", max_values)
print("最小值数组:", min_values)
print("最大值已保存到 max_values.npy")
print("最小值已保存到 min_values.npy")
