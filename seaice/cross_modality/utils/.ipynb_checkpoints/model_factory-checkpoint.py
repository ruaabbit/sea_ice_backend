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

from models.IceMamba import IceMamba
from models.SimVP import SimVP
from config import configs


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        if configs.model == "IceMamba":
            self.network = IceMamba(
                img_size=configs.img_size,
                patch_size=configs.patch_size,
                embed_dim=configs.embed_dim,
                in_channels=configs.input_dim,
                pred_length=configs.pred_length,
                N_S=configs.N_S,
                hid_S=configs.hid_S,
                kernels=configs.kernels,
                attn_drop=configs.attn_drop,
                pos_drop=configs.pos_drop,
                d_state=configs.d_state,
                d_conv=configs.d_conv,
                expand=configs.expand,
                headdim=configs.headdim,
            )
        elif configs.model == "SimVP":
            self.network = SimVP(
                time_steps=configs.input_length,
                channels=configs.input_dim,
                spatial_hidden=configs.hid_S,
                temporal_hidden=configs.hid_T,
                spatial_layers=configs.N_S,
                temporal_layers=configs.N_T,
                encoder_kernel=configs.spatio_kernel_enc,
                decoder_kernel=configs.spatio_kernel_dec,
            )
        else:
            raise ValueError("错误的网络名称，不存在%s这个网络" % configs.model)

    def forward(self, inputs, inputs_mark):
        B, T, C, H, W = inputs.shape

        padding_b = abs(configs.img_size[0] - H)  # 计算底部填充的数量
        padding_r = abs(configs.img_size[1] - W)  # 计算右侧填充的数量

        # 指定在 W 轴方向右侧和H轴方向底部各填充多少个零
        inputs = nn.functional.pad(inputs, (0, padding_r, 0, padding_b), value=0)

        if configs.model == "IceMamba":
            preds = self.network(inputs, inputs_mark)
        elif configs.model == "SimVP":
            preds = self.network(inputs)

        # 恢复输入图片尺寸
        preds = preds[:, :, :, :H, :W]

        return preds
