"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-11-23 12:45:21
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-11-25 00:23:25
FilePath: /arctic_sic_prediction/model_factory.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
import torch.nn as nn

from models.SICTeDev import SICTeDev
from utils import unfold_StackOverChannel, fold_tensor


class IceNet(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.base_net = SICTeDev(
            configs.input_length,
            configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
            configs.img_size,
            configs.hid_S,
            configs.hid_T,
            configs.N_S,
            configs.N_T,
            configs.spatio_kernel_enc,
            configs.spatio_kernel_dec,
            configs.act_inplace,
            configs.incep_ker,
            configs.groups

        )

        self.patch_size = configs.patch_size
        self.img_size = configs.img_size

    def forward(self, inputs, targets, input_times):
        outputs, loss = self.base_net(
            unfold_StackOverChannel(inputs, self.patch_size),
            unfold_StackOverChannel(targets, self.patch_size),
            input_times
        )
        outputs = fold_tensor(outputs, self.img_size, self.patch_size)

        return outputs, loss
