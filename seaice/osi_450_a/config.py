"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-12-01 19:02:29
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-12-02 12:09:37
FilePath: /arctic_sic_prediction/config.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# configs.model = "ConvLSTM"
# configs.model = "PredRNN"
# configs.model = "PredRNNv2"
# configs.model = "E3DLSTM"
# configs.model = "SimVP"
# configs.model = "TAU"
# configs.model = "ConvNeXt"
# configs.model = "InceptionNeXt"
# configs.model = "Swin_Transformer"
# configs.model = "SICFN"
# configs.model = "VMRNN_B_Model"
# configs.model = "WaST"
configs.model = "SICTeDev"

# trainer related
configs.device = torch.device("cuda:0")
# configs.batch_size_vali = 16
configs.batch_size_vali = 1
configs.batch_size = 1
# configs.lr = 1e-5
configs.lr = 1e-3
configs.weight_decay = 1e-2
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = configs.num_epochs // 10
configs.gradient_clipping = True
configs.clipping_threshold = 1.0
configs.layer_norm = False
configs.display_interval = 50

# data related
configs.img_size = (432, 432)
# configs.img_size = (448, 304)

configs.input_dim = 1  # input_dim: 输入张量对应的通道数，对于彩图为3，灰图为1。
configs.output_dim = 1  # output_dim: 输入张量对应的通道数，对于彩图为3，灰图为1。

configs.input_length = 12  # 每轮训练输入多少张数据
configs.pred_length = 12  # 每轮训练输出多少张数据

configs.input_gap = 1  # 每张输入数据之间的间隔
configs.pred_gap = 1  # 每张输出数据之间的间隔

configs.pred_shift = configs.pred_gap * configs.pred_length
# configs.pred_shift = 24
# configs.pred_shift = 4 + 12

# configs.train_period = (197901, 201012)
# configs.eval_period = (201101, 201912)


configs.train_period = (198801, 201412)
configs.eval_period = (201501, 202012)
# configs.train_period = (198801, 198912)
# configs.eval_period = (201501, 201712)
# if(stage.flag == 0):
# configs.train_period = (185001, 205012)
# configs.eval_period = (205101, 209012)
# else:
configs.FT_train_period = (198801, 201012)
configs.FT_eval_period = (201101, 201512)
#     configs.train_period = (185001, 205412)
#     configs.eval_period = (205501, 209012)

# model related
configs.kernel_size = (3, 3)
# configs.patch_size = (2, 2)
configs.patch_size = (1, 1)
configs.hidden_dim = (
    96,
    96,
    96,
    96,
)  # hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要设置。

configs.decouple_beta = 0.1  # PredRNNv2

configs.kernel_size_3D = (2, 2, 2)  # E3DLSTM

# SimVP
configs.hid_S = 64
configs.hid_T = 256
configs.N_T = 8
configs.N_S = 4
configs.spatio_kernel_enc = 3
configs.spatio_kernel_dec = 3
configs.act_inplace = False

# TAU
configs.mlp_ratio = 4
configs.drop = 0.0
configs.drop_path = 0.1

# ConvNeXt
configs.use_grn = True

# SICFN
configs.patch_embed_size = (8, 8)
configs.dropcls = 0.0
configs.fno_blocks = 8
configs.fno_bias = True
configs.fno_softshrink = 0.0

# VMRNN
# configs.patch_size = 4
configs.embed_dim = 128
configs.depths = 12
configs.num_heads = 4
configs.window_size = 8
configs.drop_rate = 0.0
configs.attn_drop_rate = 0.0
configs.drop_path_rate = 0.1

# WaST
configs.encoder_dim = 4
configs.block_list = [2, 8, 2]
configs.drop_path_rate = 0.1
configs.mlp_ratio = 2.0

# SICTeDev
configs.incep_ker = [3, 5, 7, 11]
configs.groups = 8

# paths
configs.fine_tune_data_path = "./data/full_sic_update.nc"
# 使用 full_sic_update 时保存的权重文件后缀为update
# if(stage.flag == 0):
# configs.full_data_path = "./data/siconca_new2.nc"
# configs.full_data_path = "../../root/autodl-tmp/NSIDC/full_sic.nc"
# else:
# configs.full_data_path = "./data/MRI_r1i1p1f1_all.nc"
configs.full_data_path = "./data/full_sic_update.nc"
configs.train_log_path = "train_logs"
configs.test_results_path = "test_results"
configs.grad_results_path = "grad_results"
configs.grad_month = 1
configs.grad_type = "variation"
configs.grad_save_dir = f"/root/autodl-tmp/grad_results/{configs.grad_type}"
