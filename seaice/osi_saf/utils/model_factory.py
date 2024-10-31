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

    def forward(self, inputs, targets):
        B, T, C, H, W = inputs.shape
        padding = abs(H - W) // 2  # 两侧填充的数量
        if configs.model in ["SICFN", ]:
            if H > W:
                # 指定在 W 轴方向左侧和右侧各填充多少个零
                inputs = nn.functional.pad(inputs, (padding, padding, 0, 0), value=0)
                targets = nn.functional.pad(targets, (padding, padding, 0, 0), value=0)
            elif H < W:
                # 指定在 H 轴方向上侧和下侧各填充多少个零
                inputs = nn.functional.pad(inputs, (0, 0, padding, padding), value=0)
                targets = nn.functional.pad(targets, (0, 0, padding, padding), value=0)

        pred, loss = self.net(
            unfold_stack_over_channel(inputs, configs.patch_size),
            unfold_stack_over_channel(targets, configs.patch_size),
        )

        pred = fold_tensor(pred, configs.img_size, configs.patch_size)

        return pred, loss
