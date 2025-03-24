"""SimVP
A PyTorch impl of : `SimVP: Simpler yet Better Video Prediction`
    - https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.pdf
"""

import torch
from torch import nn


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        act_norm=False,
    ):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class MultiScaleConvBlock(nn.Module):

    def __init__(self, C_in, C_hid, C_out, kernel_sizes=[3, 5, 7, 9], groups=4):
        super(MultiScaleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in kernel_sizes:
            layers.append(
                GroupConv2d(
                    C_hid,
                    C_out,
                    kernel_size=ker,
                    stride=1,
                    padding=ker // 2,
                    groups=groups,
                    act_norm=True,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        upsampling: bool = False,
        act_norm: bool = False,
    ):
        super().__init__()
        self.act_norm = act_norm

        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act_norm:
            x = self.activation(self.norm(x))
        return x


class ConvSC(nn.Module):
    """空间卷积模块，支持下采样和上采样"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
        act_norm: bool = True,
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def generate_sampling_flags(num_layers: int, reverse: bool = False) -> list[bool]:
    """生成采样标志列表"""
    flags = [False, True] * (num_layers // 2)
    return list(reversed(flags[:num_layers])) if reverse else flags[:num_layers]


class Encoder(nn.Module):
    """SimVP的3D编码器"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int,
    ):
        super().__init__()
        sampling_flags = generate_sampling_flags(num_layers)

        layers = [
            ConvSC(
                in_channels,
                hidden_channels,
                kernel_size,
                downsampling=sampling_flags[0],
            )
        ]
        layers.extend(
            ConvSC(hidden_channels, hidden_channels, kernel_size, downsampling=s)
            for s in sampling_flags[1:]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_layer_output = self.layers[0](x)
        latent = first_layer_output
        for layer in self.layers[1:]:
            latent = layer(latent)
        return latent, first_layer_output


class Decoder(nn.Module):
    """SimVP的3D解码器"""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int,
    ):
        super().__init__()
        sampling_flags = generate_sampling_flags(num_layers, reverse=True)

        layers = [
            ConvSC(hidden_channels, hidden_channels, kernel_size, upsampling=s)
            for s in sampling_flags[:-1]
        ]
        layers.append(
            ConvSC(
                hidden_channels,
                hidden_channels,
                kernel_size,
                upsampling=sampling_flags[-1],
            )
        )

        self.layers = nn.Sequential(*layers)
        self.readout = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, hidden: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers[:-1]:
            hidden = layer(hidden)
        output = self.layers[-1](hidden + skip)
        return self.readout(output)


class MidIncepNet(nn.Module):
    """SimVPv1的中间Inception网络"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernel_sizes: list[int] = [3, 5, 7, 11],
        groups: int = 8,
    ):
        super().__init__()
        assert num_layers >= 2 and len(kernel_sizes) > 1
        self.num_layers = num_layers

        # 构建编码器层
        self.encoder = nn.ModuleList(
            [
                MultiScaleConvBlock(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels // 2,
                    hidden_channels,
                    kernel_sizes=kernel_sizes,
                    groups=groups,
                )
                for i in range(num_layers)
            ]
        )

        # 构建解码器层
        self.decoder = nn.ModuleList(
            [
                MultiScaleConvBlock(
                    hidden_channels if i == 0 else 2 * hidden_channels,
                    hidden_channels // 2,
                    hidden_channels if i < num_layers - 1 else in_channels,
                    kernel_sizes=kernel_sizes,
                    groups=groups,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        x = x.reshape(batch_size, time_steps * channels, height, width)

        # 编码器
        skip_connections = []
        z = x
        for i in range(self.num_layers):
            z = self.encoder[i](z)
            if i < self.num_layers - 1:
                skip_connections.append(z)

        # 解码器
        for i in range(self.num_layers):
            if i > 0:
                z = torch.cat([z, skip_connections[-i]], dim=1)
            z = self.decoder[i](z)

        return z.reshape(batch_size, time_steps, channels, height, width)


class SimVP(nn.Module):
    """SimVP主模型"""

    def __init__(
        self,
        time_steps: int,
        channels: int,
        spatial_hidden: int = 16,
        temporal_hidden: int = 256,
        spatial_layers: int = 4,
        temporal_layers: int = 4,
        encoder_kernel: int = 3,
        decoder_kernel: int = 3,
    ):
        super().__init__()
        self.encoder = Encoder(channels, spatial_hidden, spatial_layers, encoder_kernel)
        self.decoder = Decoder(spatial_hidden, channels, spatial_layers, decoder_kernel)
        self.hidden = MidIncepNet(
            time_steps * spatial_hidden, temporal_hidden, temporal_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 5, "输入张量维度必须为5 (B, T, C, H, W)"

        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)

        # 编码
        encoded, skip = self.encoder(x)
        _, encoded_channels, encoded_height, encoded_width = encoded.shape

        # 中间处理
        z = encoded.view(
            batch_size, time_steps, encoded_channels, encoded_height, encoded_width
        )
        hidden = self.hidden(z)
        hidden = hidden.reshape(
            batch_size * time_steps, encoded_channels, encoded_height, encoded_width
        )

        # 解码
        output = self.decoder(hidden, skip)
        return output.reshape(batch_size, time_steps, channels, height, width)
