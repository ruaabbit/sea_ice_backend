import torch.nn as nn

from ..config import configs
from ..models.SICFN import SICFN
from ..utils.tools import unfold_stack_over_channel, fold_tensor


class IceNet(nn.Module):
    def __init__(self):
        super().__init__()
        if configs.model == "SICFN":
            self.net = SICFN(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.patch_embed_size,
                configs.fno_blocks,
                configs.fno_bias,
                configs.fno_softshrink,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
                configs.dropcls,
            )
        else:
            raise ValueError("错误的网络名称，不存在%s这个网络" % configs.model)

    def forward(self, inputs):
        # B, T, C, H, W = inputs.shape

        pred = self.net(
            unfold_stack_over_channel(inputs, configs.patch_size),
        )

        pred = fold_tensor(pred, configs.img_size, configs.patch_size)

        return pred
