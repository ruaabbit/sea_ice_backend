from functools import partial

# from layers.TAU.TAUSubBlock import TAUSubBlock
from ..layers.SICTeDev.Fourier_computing_unit import *
from ..layers.TAU.TAUSubBlock import TAUSubBlock


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class BasicConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            transpose=False,
            act_norm=False,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2,
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm,
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


# class BasicConv2d(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=1,
#         padding=0,
#         dilation=1,
#         upsampling=False,
#         act_norm=False,
#         act_inplace=True,
#     ):
#         super(BasicConv2d, self).__init__()
#         self.act_norm = act_norm
#         if upsampling is True:
#             self.conv = nn.Sequential(
#                 *[
#                     nn.Conv2d(
#                         in_channels,
#                         out_channels * 4,
#                         kernel_size=kernel_size,
#                         stride=1,
#                         padding=padding,
#                         dilation=dilation,
#                     ),
#                     nn.PixelShuffle(2),
#                 ]
#             )
#         else:
#             self.conv = nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#             )

#         self.norm = nn.GroupNorm(2, out_channels)
#         self.act = nn.SiLU(inplace=act_inplace)

#     def forward(self, x):
#         y = self.conv(x)
#         if self.act_norm:
#             y = self.act(self.norm(y))
#         return y


# class ConVSC(nn.Module):
#     def __init__(
#         self,
#         C_in,
#         C_out,
#         kernel_size=3,
#         downsampling=False,
#         upsampling=False,
#         act_norm=True,
#         act_inplace=True,
#     ):
#         super(ConVSC, self).__init__()
#         stride = 2 if downsampling is True else 1
#         padding = (kernel_size - stride + 1) // 2

#         self.conv = BasicConv2d(
#             C_in,
#             C_out,
#             kernel_size=kernel_size,
#             stride=stride,
#             upsampling=upsampling,
#             padding=padding,
#             act_norm=act_norm,
#             act_inplace=act_inplace,
#         )

#     def forward(self, x):
#         y = self.conv(x)
#         return y


# def sampling_generator(N, reverse=False):
#     samplings = [False, True] * (N // 2)
#     if reverse:
#         return list(reversed(samplings[:N]))
#     else:
#         return samplings[:N]


# class Encoder(nn.Module):
#     """3D Encoder for SimVP"""

#     def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
#         samplings = sampling_generator(N_S)
#         super(Encoder, self).__init__()
#         self.enc = nn.Sequential(
#             ConVSC(
#                 C_in,
#                 C_hid,
#                 spatio_kernel,
#                 downsampling=samplings[0],
#                 act_inplace=act_inplace,
#             ),
#             *[
#                 ConVSC(
#                     C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace
#                 )
#                 for s in samplings[1:]
#             ]
#         )

#     def forward(self, x):  # B*T, C, H, W
#         enc1 = self.enc[0](x)
#         latent = enc1
#         for i in range(1, len(self.enc)):
#             latent = self.enc[i](latent)
#         return latent, enc1


# class Decoder(nn.Module):
#     """3D Decoder for SimVP"""

#     def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
#         samplings = sampling_generator(N_S, reverse=True)
#         super(Decoder, self).__init__()
#         self.dec = nn.Sequential(
#             *[
#                 ConVSC(
#                     C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace
#                 )
#                 for s in samplings[:-1]
#             ],
#             ConVSC(
#                 C_hid,
#                 C_hid,
#                 spatio_kernel,
#                 upsampling=samplings[-1],
#                 act_inplace=act_inplace,
#             )
#         )
#         self.readout = nn.Conv2d(C_hid, C_out, 1)

#     def forward(self, hid, enc1=None):
#         for i in range(0, len(self.dec) - 1):
#             hid = self.dec[i](hid)
#         Y = self.dec[-1](hid + enc1)
#         Y = self.readout(Y)
#         return Y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(
            self,
            in_channels,
            out_channels,
            mlp_ratio=8.0,
            drop=0.0,
            drop_path=0.0,
    ):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = TAUSubBlock(
            in_channels,
            kernel_size=21,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
        )

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(
            self,
            channel_in,
            channel_hid,
            N2,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.1,
    ):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)
        ]

        # downsample
        enc_layers = [
            MetaBlock(
                channel_in,
                channel_hid,
                mlp_ratio,
                drop,
                drop_path=dpr[0],
            )
        ]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(
                MetaBlock(
                    channel_hid,
                    channel_hid,
                    mlp_ratio,
                    drop,
                    drop_path=dpr[i],
                )
            )
        # upsample
        enc_layers.append(
            MetaBlock(
                channel_hid,
                channel_in,
                mlp_ratio,
                drop,
                drop_path=drop_path,
            )
        )
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


# Temporal_block
class GroupConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
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
        self.activate = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
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
            y = y + layer(x)
        return y


######
class invertedBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(
            internal_channel, internal_channel, 7, 1, 3, groups=in_channel, bias=False
        )

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=internal_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.pw2 = nn.Conv2d(
            in_channels=internal_channel,
            out_channels=in_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )

    def hifi(self, x):
        x1 = self.pw1(x)
        x1 = self.relu(x1)
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.pw2(x1)
        x1 = self.relu(x1)
        # x2 = self.conv2(x)
        x3 = x1 + x

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)

        return x4

    def forward(self, x):
        return self.hifi(x) + x


class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.pw2 = nn.Conv2d(
            in_channels=internal_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x


#########


class TeDev(nn.Module):
    def __init__(
            self, channel_in, channel_hid, N_T, h, w, incep_ker=[3, 5, 7, 11], groups=8
    ):
        super(TeDev, self).__init__()

        self.N_T = N_T
        enc_layers = [
            Inception(
                channel_in,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for i in range(1, N_T - 1):
            enc_layers.append(
                Inception(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        enc_layers.append(
            Inception(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        )

        dec_layers = [
            Inception(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            Inception(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                incep_ker=incep_ker,
                groups=groups,
            )
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)

        self.enc = nn.Sequential(*enc_layers)
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.h = h
        self.w = w
        # self.blocks = nn.ModuleList(
        #     [
        #         FourierNetBlock(
        #             dim=channel_hid,
        #             mlp_ratio=4,
        #             drop=0.0,
        #             drop_path=dpr[i],
        #             act_layer=nn.GELU,
        #             norm_layer=norm_layer,
        #             h=self.h,
        #             w=self.w,
        #         )
        #         for i in range(12)
        #     ]
        # )
        self.blocks = invertedBlock(
            in_channel=channel_in // 3, out_channel=channel_in // 3
        )
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        bias = x
        x = x.reshape(B, T * C, H, W)

        # downsampling
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # Spectral Domain
        # B, D, H, W = z.shape
        # N = H * W
        # z = z.permute(0, 2, 3, 1)
        # z = z.view(B, N, D)
        # for blk in self.blocks:
        #     z = blk(z)
        # z = self.norm(z).permute(0, 2, 1)

        # z = z.reshape(B, D, H, W)
        z = self.blocks(z)
        # upsampling
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y + bias


class SICTeDev(nn.Module):
    def __init__(
            self,
            T,
            C,
            img_size,
            hid_S=16,
            hid_T=256,
            N_S=4,
            N_T=8,
            spatio_kernel_enc=3,
            spatio_kernel_dec=3,
            act_inplace=True,
            incep_ker=[3, 5, 7, 11],
            groups=8,
    ):
        super(SICTeDev, self).__init__()
        H, W = img_size
        self.H1 = int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))

        self.enc = Encoder(C, hid_S, N_S)
        self.dec = Decoder(hid_S, C, N_S)

        self.hid = TeDev(T * hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups)

        self.criterion = nn.SmoothL1Loss()

        # 定义月份位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(12, 64))

        self.month_conv_layers = nn.Conv2d(1, 64, kernel_size=8, stride=8, padding=0)

        self.month_deconv_layers = nn.ConvTranspose2d(
            in_channels=64, out_channels=1, kernel_size=8, stride=8, padding=0
        )
        # 初始化位置编码
        # nn.init.uniform_(self.positional_encoding, -0.1, 0.1)

    def forward(self, input_x, targets, input_times):
        with torch.autograd.set_detect_anomaly(True):
            assert len(input_x.shape) == 5

            B, T, C, H, W = input_x.shape
            # print(input_x.shape)
            # positional_encoding = generate_positional_encoding(input_times, 432*432)

            assert T == len(
                input_times
            ), "The length of the months list should match the time dimension of the input."
            # print(input_times)

            # 将 input_times 转换为整型（独热编码需要整型输入）
            input_times_int = input_times.to(torch.long) - 1  # 减 1 使其从 0 开始
            # 生成独热编码
            one_hot_encoding = F.one_hot(input_times_int, num_classes=12)
            # 将独热编码从 GPU 转移回 CPU 并转换为浮点型（如果需要）
            one_hot_encoding = one_hot_encoding.float().to(input_times.device)
            # print(one_hot_encoding)
            month_encoding = torch.matmul(one_hot_encoding, self.positional_encoding)
            outputs = []
            conv_layer = self.month_conv_layers
            deconv_layer = self.month_deconv_layers

            image = input_x[0, :, :, :, :]
            month_encoding = month_encoding.unsqueeze(2).unsqueeze(3)
            month_encoding = month_encoding.expand([12, 64, 54, 54])
            image = conv_layer(image)
            image = image + month_encoding
            image = deconv_layer(image)
            # image = image.reshape(B, T, C, H, W)
            # for i, month in enumerate(input_times):
            #     month = int(month.item())
            #     image = input_x[:, i, :, :, :]
            #     B, C, H, W = image.shape
            #     image = image.reshape(B * C, H, W)
            #     month_row = month_encoding[i]
            #     month_row = month_row.unsqueeze(1).unsqueeze(2)
            #     month_row = month_row.expand([64, 54, 54])

            #     image = conv_layer(image)
            #     image = image + month_row

            #     image = deconv_layer(image)
            #     image = image.reshape(B, C, H, W)

            #     outputs.append(image)
            # images = torch.stack(outputs, dim=1)

            # x = images.view(B * T, C, H, W)
            x = image

            embed, skip = self.enc(x)
            _, C_, H_, W_ = embed.shape

            z = embed.view(B, T, C_, H_, W_)
            # print("Z:",z.shape)
            # print("H1:", self.H1)
            # print("W1:", self.W1)
            hid = self.hid(z)
            # print("hid :", hid.shape)
            hid = hid.reshape(B * T, C_, H_, W_)

            next_frames = self.dec(hid, skip)
            next_frames = next_frames.reshape(B, T, C, H, W)
            next_frames = torch.clamp(next_frames, 0, 1)

            loss = self.criterion(next_frames, targets)

            return next_frames, loss
