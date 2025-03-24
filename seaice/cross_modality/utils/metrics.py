"""
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-01-06 15:52:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-01-10 17:13:40
FilePath: /Oscar/Cross-modality/utils/metrics.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
"""

import torch
from torch import nn

Max_SIE = 27402.0  # 通过data/MAX_SIE.py计算得到

""" 
下面函数传入的pred, true的shape为(B, T, H, W)，mask的shape为(H, W)
"""


def loss_func(pred, true, mask):
    loss = nn.MSELoss()
    masked_pred = pred * mask
    masked_true = true * mask
    return loss(masked_pred, masked_true)


def mae_func(pred, true, mask):
    # mae = torch.abs(pred - true) * mask
    # mae = torch.sum(mae, dim=[2, 3]) / torch.sum(mask)
    # print(mae)

    pred = pred * mask
    true = true * mask
    mae = torch.abs(pred - true)
    mae = torch.sum(mae, dim=[2, 3]).mean(dim=1) / torch.sum(mask)
    return mae.mean()


def rmse_func(pred, true, mask):
    # mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3]) / torch.sum(mask)
    # rmse = torch.sqrt(mse)
    # print(rmse)

    pred = pred * mask
    true = true * mask
    mse = torch.sum((pred - true) ** 2, dim=[2, 3]).mean(dim=1) / torch.sum(mask)
    rmse = torch.sqrt(mse)
    return rmse.mean()


def r2_func(pred, true, mask):
    pred = pred * mask
    true = true * mask
    residual = torch.sum((pred - true) ** 2, dim=[2, 3]).mean(dim=1)
    mean_true = torch.sum(true, dim=[2, 3]) / torch.sum(mask)
    mean_true = mean_true.unsqueeze(-1).unsqueeze(-1)
    total = torch.sum((true - mean_true) ** 2, dim=[2, 3]).mean(dim=1)
    r2 = 1 - (residual / total)
    return r2.mean()


def BACC_func(pred, true, mask):
    # a = torch.sum(torch.abs(pred - true) * mask, dim=[2, 3])
    # BACC = 1 - a / Max_SIE
    # print(BACC)

    pred = pred * mask
    true = true * mask
    IIEE = torch.sum(torch.abs(pred - true), dim=[2, 3]).mean(dim=1)
    BACC = 1 - IIEE.mean() / Max_SIE
    return BACC
