import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


class PatchEmbed(nn.Module):
    def __init__(
            self, img_size, patch_embed_size, input_dim, embed_dim, norm_layer=None
    ):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch = nn.Conv2d(
            input_dim,
            embed_dim,
            kernel_size=patch_embed_size,
            stride=patch_embed_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        """
        Patch: [B, C, H, W] -> [B, C*patch_embed_size[0]*patch_embed_size[1], img_size[0]//patch_embed_size[0], img_size[1]//patch_embed_size[1]]
        Flatten: [B, C, H, W] -> [B, C, HW]
        Transpose: [B, C, HW] -> [B, HW, C]
        """
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc3 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x


class AdativeFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h, w, fno_blocks, fno_bias, fno_softshrink):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size, self.block_size)
        )
        self.b1 = torch.nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )
        self.w2 = torch.nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size, self.block_size)
        )
        self.b2 = torch.nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )
        self.relu = nn.ReLU()

        if fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum("...bd, bdk->...bk", input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(
            self.multiply(x.real, self.w1[0])
            - self.multiply(x.imag, self.w1[1])
            + self.b1[0],
            inplace=True,
        )
        x_imag = F.relu(
            self.multiply(x.real, self.w1[1])
            + self.multiply(x.imag, self.w1[0])
            + self.b1[1],
            inplace=True,
        )
        x_real = (
                self.multiply(x_real, self.w2[0])
                - self.multiply(x_imag, self.w2[1])
                + self.b2[0]
        )
        x_imag = (
                self.multiply(x_real, self.w2[1])
                + self.multiply(x_imag, self.w2[0])
                + self.b2[1]
        )

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)

        return x + bias


class AFNOBlock(nn.Module):
    def __init__(
            self,
            dim,
            h,
            w,
            fno_blocks,
            fno_bias,
            fno_softshrink,
            mlp_ratio,
            drop,
            drop_path,
            act_layer,
            norm_layer=nn.LayerNorm,
    ):
        """
        AFNO Block
        """
        super(AFNOBlock, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(
            dim,
            h=h,
            w=w,
            fno_blocks=fno_blocks,
            fno_bias=fno_bias,
            fno_softshrink=fno_softshrink,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.normlayer2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.normlayer1(x)))
        x = x + self.drop_path(self.mlp(self.normlayer2(x)))
        return x
