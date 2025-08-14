import itertools
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.profiler import profile, record_function, ProfilerActivity
# torch.set_default_dtype(torch.float64)

from einops import rearrange


##########################################################################
class CustomLayerNorm(nn.Module):
    def __init__(self, nchannels):
        super(CustomLayerNorm, self).__init__()
        
        self.nchannels = nchannels
        self.weighted_transform = nn.Conv2d(nchannels, nchannels, kernel_size=1, stride=1, groups=nchannels, bias=False)

    def forward(self, x):
        # bz, nchannels, h, w = x.shape
        sigma = x.var(dim=1, keepdim=True, correction=1)
        # bz, 1, h, w = sigma.shape
        return self.weighted_transform(x / torch.sqrt(sigma+1e-5))



# ##########################################################################
# class LocalGatedLinearBlock(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super(LocalGatedLinearBlock, self).__init__()

#         self.channels_linear_op       = nn.Conv2d(dim, hidden_dim*2, kernel_size=1, bias=False)
#         self.channels_local_linear_op = nn.Conv2d(
#             hidden_dim*2, hidden_dim*2, 
#             kernel_size=3, stride=1, 
#             padding=1, padding_mode="replicate",
#             groups=hidden_dim*2, 
#             bias=False
#         )
#         self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

#     def forward(self, x):
#         x = self.channels_linear_op(x)
#         mask, x = self.channels_local_linear_op(x).chunk(2, dim=1)
#         x = nn.functional.sigmoid(mask) * mask * x
#         x = self.project_out(x)
#         return x

##########################################################################
class LocalGatedLinearBlock(nn.Module):
    def __init__(self, dim, hidden_dim, ngraphs):
        super(LocalGatedLinearBlock, self).__init__()

        self.channels_linear_op       = nn.Conv2d(dim, hidden_dim*2, kernel_size=1, bias=False, groups=ngraphs)
        self.channels_local_linear_op = nn.Conv2d(
            hidden_dim*2, hidden_dim*2, 
            kernel_size=3, stride=1, 
            padding=1, padding_mode="replicate",
            groups=hidden_dim*2, 
            bias=False
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False, groups=ngraphs)

    def forward(self, x):
        x = self.channels_linear_op(x)
        mask, x = self.channels_local_linear_op(x).chunk(2, dim=1)
        x = nn.functional.sigmoid(mask) * mask * x
        x = self.project_out(x)
        return x

# ##########################################################################
# class LocalGatedLinearBlock(nn.Module):
#     def __init__(self, dim, hidden_dim, ngraphs):
#         super(LocalGatedLinearBlock, self).__init__()

#         self.channels_local_linear_op = nn.Conv2d(
#             dim, hidden_dim*2, 
#             kernel_size=3, stride=1, 
#             padding=1, padding_mode="replicate",
#             groups=ngraphs, 
#             bias=False
#         )
#         self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, groups=ngraphs, bias=False)

#     def forward(self, x):
#         mask, x = self.channels_local_linear_op(x).chunk(2, dim=1)
#         x = nn.functional.sigmoid(mask) * mask * x
#         x = self.project_out(x)
#         return x


##########################################################################
class LocalLowpassFilteringBlock(nn.Module):
    def __init__(self, dim, ngraphs):
        super(LocalLowpassFilteringBlock, self).__init__()

        # Filter Layer
        # self.norm_filter = CustomLayerNorm(dim)
        # self.local_linear_filter = nn.Conv2d(
        #     dim, dim, 
        #     kernel_size=3, stride=1, 
        #     padding=1, padding_mode="replicate",
        #     groups=dim, 
        #     bias=False
        # )
        # self.skip_weight_filter = Parameter(
        #     torch.tensor([0.5, 0.5], dtype=torch.float32),
        #     requires_grad=True
        # )

        # Linear Layer
        self.norm = CustomLayerNorm(dim)
        self.local_linear = LocalGatedLinearBlock(dim, dim*2, ngraphs)
        self.skip_weight= Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            requires_grad=True
        )
    def forward(self, x):
        # x = self.skip_weight_filter[0] * x + self.skip_weight_filter[1] * self.local_linear_filter(self.norm_filter(x))
        x = self.skip_weight[0] * x + self.skip_weight[1] * self.local_linear(self.norm(x))
        return x


##########################################################################
class ReginalPixelEmbeding(nn.Module):
    def __init__(self, n_channels_in=3, dim=48, bias=False):
        super(ReginalPixelEmbeding, self).__init__()

        self.channels_local_linear_op01 = nn.Conv2d(
            n_channels_in, dim, 
            kernel_size=3, stride=1, 
            padding=1, padding_mode="replicate",
            bias=False
        )
        # self.project_out = nn.Conv2d(dim//2, dim, kernel_size=1, bias=False)

    def forward(self, x):

        # mask, x = self.channels_local_linear_op01(x).chunk(2, dim=1)
        # x = nn.functional.sigmoid(mask) * mask * x
        x = self.channels_local_linear_op01(x)

        return x


##########################################################################
## Down/Up Sampling
class Downsampling(nn.Module):
    def __init__(self, dim_in, dim_out, ngraphs):
        super(Downsampling, self).__init__()

        self.local_linear = nn.Conv2d(dim_in, dim_out//4, kernel_size=3, stride=1, padding=1, padding_mode="replicate", groups=ngraphs, bias=False)
        self.local_concat = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.local_concat(self.local_linear(x))
        return x

class Upsampling(nn.Module):
    def __init__(self, dim_in, dim_out, ngraphs):
        super(Upsampling, self).__init__()

        self.local_linear = nn.Conv2d(dim_in, dim_out*4, kernel_size=3, stride=1, padding=1, padding_mode="replicate", groups=ngraphs, bias=False)
        self.local_concat = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.local_concat(self.local_linear(x))
        return x

# ##########################################################################
# ## Down/Up Sampling
# class Downsampling(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(Downsampling, self).__init__()

#         self.local_linear = nn.Conv2d(dim_in, dim_out//2, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
#         self.local_concat = nn.PixelUnshuffle(2)

#     def forward(self, x):
#         mask, x = self.local_linear(x).chunk(2, dim=1)
#         x = nn.functional.sigmoid(mask) * mask * x
#         x = self.local_concat(x)
#         return x

# class Upsampling(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(Upsampling, self).__init__()

#         self.local_linear = nn.Conv2d(dim_in, dim_out*8, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
#         self.local_concat = nn.PixelShuffle(2)

#     def forward(self, x):
#         # print(f"Upsampling in:{x.shape}")
#         mask, x = self.local_linear(x).chunk(2, dim=1)
#         # print(f"Upsampling out//4:{x.shape}")
#         x = nn.functional.sigmoid(mask) * mask * x
#         x = self.local_concat(x)
#         # print(f"Upsampling out:{x.shape}")
#         return x


class AbtractMultiScaleGraphFilter(nn.Module):
    def __init__(self, 
        n_channels_in=3, 
        n_channels_out=3, 
        dims=[48, 64, 96, 128],
        ngraphs=[4, 4, 4, 4],
        num_blocks=[4, 6, 6, 8], 
        num_blocks_out=4
    ):

        super(AbtractMultiScaleGraphFilter, self).__init__()

        self.patch_3x3_embeding = ReginalPixelEmbeding(n_channels_in, dims[0])
        self.encoder_scale_00 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[0], ngraphs=ngraphs[0]) for i in range(num_blocks[0])
        ])
        
        self.down_sample_00_01 = Downsampling(dim_in=dims[0], dim_out=dims[1], ngraphs=ngraphs[0]) 
        self.encoder_scale_01 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[1], ngraphs=ngraphs[1]) for i in range(num_blocks[1])
        ])

        self.down_sample_01_02 = Downsampling(dim_in=dims[1], dim_out=dims[2], ngraphs=ngraphs[1]) 
        self.encoder_scale_02 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[2], ngraphs=ngraphs[2]) for i in range(num_blocks[2])
        ])

        self.down_sample_02_03 = Downsampling(dim_in=dims[2], dim_out=dims[3], ngraphs=ngraphs[2]) 
        self.encoder_scale_03 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[3], ngraphs=ngraphs[3]) for i in range(num_blocks[3])
        ])


        self.up_sample_03_02 = Upsampling(dim_in=dims[3], dim_out=dims[2], ngraphs=ngraphs[3])
        self.combine_channels_02 = nn.Conv2d(dims[2]*2, dims[2], kernel_size=1, bias=False, groups=ngraphs[2])
        self.decoder_scale_02 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[2], ngraphs=ngraphs[2]) for i in range(num_blocks[2])
        ])

        self.up_sample_02_01 = Upsampling(dim_in=dims[2], dim_out=dims[1], ngraphs=ngraphs[2])
        self.combine_channels_01 = nn.Conv2d(dims[1]*2, dims[1], kernel_size=1, bias=False, groups=ngraphs[1])
        self.decoder_scale_01 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[1], ngraphs=ngraphs[1]) for i in range(num_blocks[1])
        ])

        self.up_sample_01_00 = Upsampling(dim_in=dims[1], dim_out=dims[0], ngraphs=ngraphs[1])
        self.combine_channels_00 = nn.Conv2d(dims[0]*2, dims[0], kernel_size=1, bias=False, groups=ngraphs[0])
        self.decoder_scale_00 = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[0], ngraphs=ngraphs[0]) for i in range(num_blocks[0])
        ])

        self.refining_block = nn.Sequential(*[
            LocalLowpassFilteringBlock(dim=dims[0], ngraphs=ngraphs[0]) for i in range(num_blocks_out)
        ])
        self.linear_output = nn.Conv2d(dims[0], n_channels_out, kernel_size=1, bias=False)
        self.skip_weight_output = Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            requires_grad=True
        )
    def forward(self, img):
        
        # Downward
        inp_enc_scale_00 = self.patch_3x3_embeding(img)
        out_enc_scale_00 = self.encoder_scale_00(inp_enc_scale_00)
        # print(f"out_enc_scale_00:{out_enc_scale_00.shape}")
        
        inp_enc_scale_01 = self.down_sample_00_01(out_enc_scale_00)
        out_enc_scale_01 = self.encoder_scale_01(inp_enc_scale_01)
        # print(f"inp_enc_scale_01:{inp_enc_scale_01.shape}")

        inp_enc_scale_02 = self.down_sample_01_02(out_enc_scale_01)
        out_enc_scale_02 = self.encoder_scale_02(inp_enc_scale_02)
        # print(f"inp_enc_scale_02:{inp_enc_scale_02.shape}")

        inp_enc_scale_03 = self.down_sample_02_03(out_enc_scale_02)
        out_enc_scale_03 = self.encoder_scale_03(inp_enc_scale_03)
        # print(f"inp_enc_scale_03:{inp_enc_scale_03.shape}")

        # Upward
        inp_dec_scale_02 = self.up_sample_03_02(out_enc_scale_03)
        out_dec_scale_02 = torch.cat([inp_dec_scale_02, out_enc_scale_02], 1)
        out_dec_scale_02 = self.combine_channels_02(out_dec_scale_02)
        out_dec_scale_02 = self.decoder_scale_02(out_dec_scale_02)
        # print(f"out_dec_scale_02={out_dec_scale_02.shape}")

        inp_dec_scale_01 = self.up_sample_02_01(out_dec_scale_02)
        out_dec_scale_01 = torch.cat([inp_dec_scale_01, out_enc_scale_01], 1)
        out_dec_scale_01 = self.combine_channels_01(out_dec_scale_01)
        out_dec_scale_01 = self.decoder_scale_01(out_dec_scale_01)
        # print(f"out_dec_scale_01={out_dec_scale_01.shape}")

        inp_dec_scale_00 = self.up_sample_01_00(out_dec_scale_01)
        out_dec_scale_00 = torch.cat([inp_dec_scale_00, out_enc_scale_00], 1)
        out_dec_scale_00 = self.combine_channels_00(out_dec_scale_00)
        out_dec_scale_00 = self.decoder_scale_00(out_dec_scale_00)
        # print(f"out_dec_scale_00={out_dec_scale_00.shape}")
 
        output = self.refining_block(out_dec_scale_00)
        output = self.linear_output(output) 
        output = self.skip_weight_output[0] * img + self.skip_weight_output[1] * output
        # print(f"output={output.shape}")

        return output