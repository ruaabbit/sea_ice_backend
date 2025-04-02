"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-08-30 11:04:51
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-08 15:45:33
FilePath: /root/amsr2-asi_daygrid_swath-n6250/data/MAX_SIE.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import numpy as np
import xarray as xr
from tqdm import tqdm

paths = np.genfromtxt("nc_sic_path.txt", dtype=str)

MAX_SIE = 0

loop = tqdm(paths, total=len(paths), leave=True)
for path in loop:
    ice_conc = xr.open_dataset(path)["cdr_seaice_conc"][0].values
    ice_conc = np.nan_to_num(ice_conc)
    ice_conc[ice_conc > 0.15] = 1
    ice_conc[ice_conc <= 0.15] = 0
    if np.sum(ice_conc) > MAX_SIE:
        MAX_SIE = np.sum(ice_conc)
    loop.set_postfix(
        MAX_SIE=MAX_SIE,
    )

print(MAX_SIE)
