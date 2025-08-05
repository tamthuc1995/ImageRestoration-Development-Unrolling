import itertools
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.profiler import profile, record_function, ProfilerActivity
# torch.set_default_dtype(torch.float64)

from einops import rearrange


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
    


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        # hidden_features = dim

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
class FFBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FFBlock, self).__init__()

        self.norm = CustomLayerNorm(dim)

        self.skip_connect_weight_final = Parameter(
            torch.ones((2), dtype=torch.float32) * torch.tensor([0.5, 0.5]),
            requires_grad=True
        )
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.skip_connect_weight_final[0]*x + self.skip_connect_weight_final[1]*self.ffn(self.norm(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelUnshuffle(2))
        
        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, padding_mode="replicate", bias=False))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelShuffle(2))

        # self.body = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))


    def forward(self, x):
        return self.body(x)

##########################################################################
# class FeatureExtraction(nn.Module):
#     def __init__(self, 
#         inp_channels=3, 
#         out_channels=48, 
#         dim = 48,
#         num_blocks = [2,2,2,2], 
#         num_refinement_blocks = 4,
#         ffn_expansion_factor = 2.0,
#         bias = False,
#     ):

#         super(FeatureExtraction, self).__init__()

#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

#         self.encoder_level1 = nn.Sequential(*[FFBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])
        
#         self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
#         self.encoder_level2 = nn.Sequential(*[FFBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[1])])
        
#         self.down2_3 = Downsample(int(dim)) ## From Level 2 to Level 3
#         self.encoder_level3 = nn.Sequential(*[FFBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])

#         self.down3_4 = Downsample(int(dim)) ## From Level 3 to Level 4
#         self.latent = nn.Sequential(*[FFBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[3])])
        
#         self.up4_3 = Upsample(int(dim)) ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
#         self.decoder_level3 = nn.Sequential(*[FFBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])

#         self.up3_2 = Upsample(int(dim)) ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
#         self.decoder_level2 = nn.Sequential(*[FFBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[1])])
        
#         self.up2_1 = Upsample(int(dim))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

#         self.decoder_level1 = nn.Sequential(*[FFBlock(dim=int(dim*2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])
        
#         self.refinement = nn.Sequential(*[FFBlock(dim=int(dim*2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_refinement_blocks)])

#         ###########################
#         self.output = nn.Conv2d(int(dim*2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

#     def forward(self, inp_img):

#         inp_enc_level1 = self.patch_embed(inp_img)
#         out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
#         inp_enc_level2 = self.down1_2(out_enc_level1)
#         out_enc_level2 = self.encoder_level2(inp_enc_level2)

#         inp_enc_level3 = self.down2_3(out_enc_level2)
#         out_enc_level3 = self.encoder_level3(inp_enc_level3) 

#         inp_enc_level4 = self.down3_4(out_enc_level3)        
#         latent = self.latent(inp_enc_level4) 
                        
#         inp_dec_level3 = self.up4_3(latent)
#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
#         out_dec_level3 = self.decoder_level3(inp_dec_level3) 

#         inp_dec_level2 = self.up3_2(out_dec_level3)
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
#         out_dec_level2 = self.decoder_level2(inp_dec_level2) 

#         inp_dec_level1 = self.up2_1(out_dec_level2)
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
#         out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
#         out_dec_level1 = self.refinement(out_dec_level1)
#         out_dec_level1 = self.output(out_dec_level1)

#         # return [latent, out_dec_level3, out_dec_level2, out_dec_level1]
#         return [out_dec_level1]

class FeatureExtraction(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=48, 
        dim = 48,
        num_blocks = [1,2,2,4], 
        num_refinement_blocks = 4,
        ffn_expansion_factor = 2.66,
        bias = False,
    ):

        super(FeatureExtraction, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[FFBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[FFBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[FFBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])

        # self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        # self.latent = nn.Sequential(*[FFBlock(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[3])])
        
        # self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[FFBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[FFBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[FFBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[FFBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_refinement_blocks)])

        ###########################
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.encoder_level3(inp_enc_level3) 

        # inp_enc_level4 = self.down3_4(out_enc_level3)        
        # latent = self.latent(inp_enc_level4) 
                        
        # inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return [out_dec_level1]
    



class GLRFast(nn.Module):
    def __init__(self, 
            n_channels, n_node_fts, n_graphs, connection_window, device, M_diag_init=0.4
        ):
        super(GLRFast, self).__init__()

        self.device = device
        self.n_channels        = n_channels
        self.n_node_fts        = n_node_fts
        self.n_graphs          = n_graphs
        self.n_edges           = (connection_window == 1).sum()
        self.connection_window = connection_window
        self.buffer_size       = connection_window.sum()

        # edges type from connection_window
        window_size = connection_window.shape[0]
        connection_window = connection_window.reshape((-1))
        m = np.arange(window_size)-window_size//2
        edge_delta = np.array(
            list(itertools.product(m, m)),
            dtype=np.int32
        )
        self.edge_delta = edge_delta[connection_window == 1]
        
        self.pad_dim_hw = np.abs(self.edge_delta.min(axis=0))


        kernel01 = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel01[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p01 = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 1.0,
            requires_grad=True,
        )
        self.stats_kernel01 = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        kernel02a = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0,-1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel02a[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p02a = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 0.5,
            requires_grad=True,
        )
        self.stats_kernel02a = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        kernel02b = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0,-1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel02b[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p02b = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 0.5,
            requires_grad=True,
        )
        self.stats_kernel02b = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        kernel03 = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel03[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p03 = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 0.5,
            requires_grad=True,
        )
        self.stats_kernel03 = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        ### Trainable parameters
        # features on nodes #self.n_node_fts
        self.multiM = Parameter(
            torch.ones((self.n_graphs, self.n_node_fts), device=self.device, dtype=torch.float32)*M_diag_init,
            requires_grad=True,
        )



    def get_neighbors_pixels(self, img_features):
        _, _, H, W = img_features.shape
        padH, padW = self.pad_dim_hw
        img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
        # neighbors_pixels = []
        # for shift_h, shift_w in self.edge_delta:
        #     fromh = padH + shift_h
        #     toh = padH + shift_h + H
        #     fromw = padW + shift_w
        #     tow = padW + shift_w + W
            
        #     neighbors_pixels.append(
        #         img_features_frame[:, :, fromh:toh, fromw:tow]
        #     )
        # neighbors_pixels_features = torch.stack(neighbors_pixels, axis=-3)

        neighbors_pixels_features = torch.stack([
            img_features_frame[:, :, padH + self.edge_delta[0, 0]:padH + self.edge_delta[0, 0] + H, padW + self.edge_delta[0, 1]:padW + self.edge_delta[0, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[1, 0]:padH + self.edge_delta[1, 0] + H, padW + self.edge_delta[1, 1]:padW + self.edge_delta[1, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[2, 0]:padH + self.edge_delta[2, 0] + H, padW + self.edge_delta[2, 1]:padW + self.edge_delta[2, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[3, 0]:padH + self.edge_delta[3, 0] + H, padW + self.edge_delta[3, 1]:padW + self.edge_delta[3, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[4, 0]:padH + self.edge_delta[4, 0] + H, padW + self.edge_delta[4, 1]:padW + self.edge_delta[4, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[5, 0]:padH + self.edge_delta[5, 0] + H, padW + self.edge_delta[5, 1]:padW + self.edge_delta[5, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[6, 0]:padH + self.edge_delta[6, 0] + H, padW + self.edge_delta[6, 1]:padW + self.edge_delta[6, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[7, 0]:padH + self.edge_delta[7, 0] + H, padW + self.edge_delta[7, 1]:padW + self.edge_delta[7, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[8, 0]:padH + self.edge_delta[8, 0] + H, padW + self.edge_delta[8, 1]:padW + self.edge_delta[8, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[9, 0]:padH + self.edge_delta[9, 0] + H, padW + self.edge_delta[9, 1]:padW + self.edge_delta[9, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[10, 0]:padH + self.edge_delta[10, 0] + H, padW + self.edge_delta[10, 1]:padW + self.edge_delta[10, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[11, 0]:padH + self.edge_delta[11, 0] + H, padW + self.edge_delta[11, 1]:padW + self.edge_delta[11, 1] + W]
            # img_features_frame[:, :, padH + self.edge_delta[12, 0]:padH + self.edge_delta[12, 0] + H, padW + self.edge_delta[12, 1]:padW + self.edge_delta[12, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[13, 0]:padH + self.edge_delta[13, 0] + H, padW + self.edge_delta[13, 1]:padW + self.edge_delta[13, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[14, 0]:padH + self.edge_delta[14, 0] + H, padW + self.edge_delta[14, 1]:padW + self.edge_delta[14, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[15, 0]:padH + self.edge_delta[15, 0] + H, padW + self.edge_delta[15, 1]:padW + self.edge_delta[15, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[16, 0]:padH + self.edge_delta[16, 0] + H, padW + self.edge_delta[16, 1]:padW + self.edge_delta[16, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[17, 0]:padH + self.edge_delta[17, 0] + H, padW + self.edge_delta[17, 1]:padW + self.edge_delta[17, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[18, 0]:padH + self.edge_delta[18, 0] + H, padW + self.edge_delta[18, 1]:padW + self.edge_delta[18, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[19, 0]:padH + self.edge_delta[19, 0] + H, padW + self.edge_delta[19, 1]:padW + self.edge_delta[19, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[20, 0]:padH + self.edge_delta[20, 0] + H, padW + self.edge_delta[20, 1]:padW + self.edge_delta[20, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[21, 0]:padH + self.edge_delta[21, 0] + H, padW + self.edge_delta[21, 1]:padW + self.edge_delta[21, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[22, 0]:padH + self.edge_delta[22, 0] + H, padW + self.edge_delta[22, 1]:padW + self.edge_delta[22, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[23, 0]:padH + self.edge_delta[23, 0] + H, padW + self.edge_delta[23, 1]:padW + self.edge_delta[23, 1] + W]
        ], axis=-3)
        return neighbors_pixels_features

    def normalize_and_transform_features(self, img_features):
        batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape
        # img_features = img_features.view(batch_size, self.n_graphs, self.n_node_fts, h_size, w_size)
        img_features = torch.nn.functional.normalize(img_features, dim=2)

        img_features_transform = torch.einsum(
            "bhcHW, hc -> bhcHW", img_features, self.multiM
        )
    
        img_features_transform = img_features_transform.view(batch_size, self.n_graphs*self.n_node_fts, h_size, w_size)

        return img_features_transform


    def extract_edge_weights(self, img_features):
        
        batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape

        img_features = self.normalize_and_transform_features(img_features)
        img_features_neighbors = self.get_neighbors_pixels(img_features)

        features_similarity = (img_features[:, :, None, :, :] * img_features_neighbors)
        features_similarity = features_similarity.view(
            batch_size, self.n_graphs, self.n_node_fts, self.n_edges, h_size, w_size
        ).sum(axis=2)

        edge_weights_norm = nn.functional.softmax(features_similarity, dim=2) 
        node_degree = edge_weights_norm.sum(axis=2)

        return edge_weights_norm, node_degree
    
    def stats_conv(self, patchs):
        stats_kernel = (
            self.stats_kernel_p01 * self.stats_kernel01
            + self.stats_kernel_p02a * self.stats_kernel02a
            + self.stats_kernel_p02b * self.stats_kernel02b
            + self.stats_kernel_p03 * self.stats_kernel03
        )
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
        temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'reflect')
        temp_out_patch = nn.functional.conv2d(
            temp_patch,
            weight=stats_kernel,
            stride=1,
            padding=0,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
        return out_patch

    def stats_conv_transpose(self, patchs):

        stats_kernel = (
            self.stats_kernel_p01 * self.stats_kernel01
            + self.stats_kernel_p02a * self.stats_kernel02a
            + self.stats_kernel_p02b * self.stats_kernel02b
            + self.stats_kernel_p03 * self.stats_kernel03
        )
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.reshape(batch_size*n_graphs, c_size, h_size, w_size)
        temp_out_patch = nn.functional.conv_transpose2d(
            temp_patch,
            weight=stats_kernel,
            stride=1,
            padding=1,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
        return out_patch


    def op_L_norm(self, img_signals, edge_weights, node_degree):

        batch_size, n_graphs, n_channels, h_size, w_size = img_signals.shape
        img_features_neighbors = self.get_neighbors_pixels(
            img_signals.view(batch_size, n_graphs*n_channels, h_size, w_size)
        ).view(batch_size, n_graphs, n_channels, self.n_edges, h_size, w_size)
        Wx = torch.einsum(
            "bhceHW, bheHW -> bhcHW", img_features_neighbors, edge_weights
        )
        output = img_signals - Wx
        return output


    def forward(self, patchs, edge_weights, node_degree):
        # output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)
        patchs = self.stats_conv(patchs)
        output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)
        output_patchs = self.stats_conv_transpose(output_patchs)

        return output_patchs
    



class GTVFast(nn.Module):
    def __init__(self, 
            n_channels, n_node_fts, n_graphs, connection_window, device, M_diag_init=0.4
        ):
        super(GTVFast, self).__init__()

        self.device = device
        self.n_channels        = n_channels
        self.n_node_fts        = n_node_fts
        self.n_graphs          = n_graphs
        self.n_edges           = (connection_window == 1).sum()
        self.connection_window = connection_window
        self.buffer_size       = connection_window.sum()

        # edges type from connection_window
        window_size = connection_window.shape[0]
        connection_window = connection_window.reshape((-1))
        m = np.arange(window_size)-window_size//2
        edge_delta = np.array(
            list(itertools.product(m, m)),
            dtype=np.int32
        )
        self.edge_delta = edge_delta[connection_window == 1]
        
        self.pad_dim_hw = np.abs(self.edge_delta.min(axis=0))

        kernel01 = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel01[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p01 = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 1.0,
            requires_grad=True,
        )
        self.stats_kernel01 = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        kernel02a = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0,-1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel02a[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p02a = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 0.5,
            requires_grad=True,
        )
        self.stats_kernel02a = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        kernel02b = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0,-1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel02b[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p02b = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 0.5,
            requires_grad=True,
        )
        self.stats_kernel02b = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        kernel03 = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(kernel03[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel_p03 = Parameter(
            torch.ones((1), device=self.device, dtype=torch.float32) * 0.5,
            requires_grad=True,
        )
        self.stats_kernel03 = torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel

        ### Trainable parameters
        # features on nodes #self.n_node_fts
        self.multiM = Parameter(
            torch.ones((self.n_graphs, self.n_node_fts), device=self.device, dtype=torch.float32)*M_diag_init,
            requires_grad=True,
        )


    def get_neighbors_pixels(self, img_features):
        _, _, H, W = img_features.shape
        padH, padW = self.pad_dim_hw
        img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
        # neighbors_pixels = []
        # for shift_h, shift_w in self.edge_delta:
        #     fromh = padH + shift_h
        #     toh = padH + shift_h + H
        #     fromw = padW + shift_w
        #     tow = padW + shift_w + W
            
        #     neighbors_pixels.append(
        #         img_features_frame[:, :, fromh:toh, fromw:tow]
        #     )
        # neighbors_pixels_features = torch.stack(neighbors_pixels, axis=-3)
        neighbors_pixels_features = torch.stack([
            img_features_frame[:, :, padH + self.edge_delta[0, 0]:padH + self.edge_delta[0, 0] + H, padW + self.edge_delta[0, 1]:padW + self.edge_delta[0, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[1, 0]:padH + self.edge_delta[1, 0] + H, padW + self.edge_delta[1, 1]:padW + self.edge_delta[1, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[2, 0]:padH + self.edge_delta[2, 0] + H, padW + self.edge_delta[2, 1]:padW + self.edge_delta[2, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[3, 0]:padH + self.edge_delta[3, 0] + H, padW + self.edge_delta[3, 1]:padW + self.edge_delta[3, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[4, 0]:padH + self.edge_delta[4, 0] + H, padW + self.edge_delta[4, 1]:padW + self.edge_delta[4, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[5, 0]:padH + self.edge_delta[5, 0] + H, padW + self.edge_delta[5, 1]:padW + self.edge_delta[5, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[6, 0]:padH + self.edge_delta[6, 0] + H, padW + self.edge_delta[6, 1]:padW + self.edge_delta[6, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[7, 0]:padH + self.edge_delta[7, 0] + H, padW + self.edge_delta[7, 1]:padW + self.edge_delta[7, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[8, 0]:padH + self.edge_delta[8, 0] + H, padW + self.edge_delta[8, 1]:padW + self.edge_delta[8, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[9, 0]:padH + self.edge_delta[9, 0] + H, padW + self.edge_delta[9, 1]:padW + self.edge_delta[9, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[10, 0]:padH + self.edge_delta[10, 0] + H, padW + self.edge_delta[10, 1]:padW + self.edge_delta[10, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[11, 0]:padH + self.edge_delta[11, 0] + H, padW + self.edge_delta[11, 1]:padW + self.edge_delta[11, 1] + W]
            # img_features_frame[:, :, padH + self.edge_delta[12, 0]:padH + self.edge_delta[12, 0] + H, padW + self.edge_delta[12, 1]:padW + self.edge_delta[12, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[13, 0]:padH + self.edge_delta[13, 0] + H, padW + self.edge_delta[13, 1]:padW + self.edge_delta[13, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[14, 0]:padH + self.edge_delta[14, 0] + H, padW + self.edge_delta[14, 1]:padW + self.edge_delta[14, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[15, 0]:padH + self.edge_delta[15, 0] + H, padW + self.edge_delta[15, 1]:padW + self.edge_delta[15, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[16, 0]:padH + self.edge_delta[16, 0] + H, padW + self.edge_delta[16, 1]:padW + self.edge_delta[16, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[17, 0]:padH + self.edge_delta[17, 0] + H, padW + self.edge_delta[17, 1]:padW + self.edge_delta[17, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[18, 0]:padH + self.edge_delta[18, 0] + H, padW + self.edge_delta[18, 1]:padW + self.edge_delta[18, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[19, 0]:padH + self.edge_delta[19, 0] + H, padW + self.edge_delta[19, 1]:padW + self.edge_delta[19, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[20, 0]:padH + self.edge_delta[20, 0] + H, padW + self.edge_delta[20, 1]:padW + self.edge_delta[20, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[21, 0]:padH + self.edge_delta[21, 0] + H, padW + self.edge_delta[21, 1]:padW + self.edge_delta[21, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[22, 0]:padH + self.edge_delta[22, 0] + H, padW + self.edge_delta[22, 1]:padW + self.edge_delta[22, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[23, 0]:padH + self.edge_delta[23, 0] + H, padW + self.edge_delta[23, 1]:padW + self.edge_delta[23, 1] + W]
        ], axis=-3)
        return neighbors_pixels_features
    

    def normalize_and_transform_features(self, img_features):
        batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape
        # img_features = img_features.view(batch_size, self.n_graphs, self.n_node_fts, h_size, w_size)
        img_features = torch.nn.functional.normalize(img_features, dim=2)

        img_features_transform = torch.einsum(
            "bhcHW, hc -> bhcHW", img_features, self.multiM
        )
    
        img_features_transform = img_features_transform.view(batch_size, self.n_graphs*self.n_node_fts, h_size, w_size)

        return img_features_transform


    def extract_edge_weights(self, img_features):
        
        batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape

        img_features = self.normalize_and_transform_features(img_features)
        img_features_neighbors = self.get_neighbors_pixels(img_features)

        features_similarity = (img_features[:, :, None, :, :] * img_features_neighbors)
        features_similarity = features_similarity.view(
            batch_size, self.n_graphs, self.n_node_fts, self.n_edges, h_size, w_size
        ).sum(axis=2)

        edge_weights_norm = nn.functional.softmax(features_similarity, dim=2) 
        node_degree = edge_weights_norm.sum(axis=2)

        return edge_weights_norm, node_degree


    def stats_conv(self, patchs):
        stats_kernel = (
            self.stats_kernel_p01 * self.stats_kernel01
            + self.stats_kernel_p02a * self.stats_kernel02a
            + self.stats_kernel_p02b * self.stats_kernel02b
            + self.stats_kernel_p03 * self.stats_kernel03
        )
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
        temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'reflect')
        temp_out_patch = nn.functional.conv2d(
            temp_patch,
            weight=stats_kernel,
            stride=1,
            padding=0,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
        return out_patch

    def stats_conv_transpose(self, patchs):

        stats_kernel = (
            self.stats_kernel_p01 * self.stats_kernel01
            + self.stats_kernel_p02a * self.stats_kernel02a
            + self.stats_kernel_p02b * self.stats_kernel02b
            + self.stats_kernel_p03 * self.stats_kernel03
        )
        
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.reshape(batch_size*n_graphs, c_size, h_size, w_size)
        temp_out_patch = nn.functional.conv_transpose2d(
            temp_patch,
            weight=stats_kernel,
            stride=1,
            padding=1,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
        return out_patch


    def op_C(self, img_signals, edge_weights, node_degree):


        batch_size, n_graphs, n_channels, h_size, w_size = img_signals.shape
        # img_signals = self.stats_conv(img_signals)

        img_signals = self.stats_conv(img_signals)

        img_features_neighbors = self.get_neighbors_pixels(
            img_signals.view(batch_size, n_graphs*n_channels, h_size, w_size)
        ).view(batch_size, n_graphs, n_channels, self.n_edges, h_size, w_size)
        Cx1 = img_signals[:, :, :, None, :, :] * edge_weights[:, :, None, :, :, :]
        Cx2 = img_features_neighbors * edge_weights[:, :, None, :, :, :]
        
        output = Cx1 - Cx2

        return output
    
    def op_C_transpose(self, edge_signals, edge_weights, node_degree):

        batch_size, n_graphs, n_channels, n_edges, H, W = edge_signals.shape
        edge_signals = edge_signals * edge_weights[:, :, None, :, :, :]

        output = edge_signals.sum(axis=3)

        padH, padW = self.pad_dim_hw
        output = nn.functional.pad(
            output.view(batch_size, n_graphs*n_channels, H, W),
            (padW, padW, padH, padH), "replicate"
        ).view(batch_size, n_graphs, n_channels, H + 2*padH, W + 2*padW)

        i=0
        for shift_h, shift_w in self.edge_delta:
            fromh = padH + shift_h
            toh = padH + shift_h + H
            fromw = padW + shift_w
            tow = padW + shift_w + W
            
            output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]
            i+=1

        output = output[:, :, :, padH:-padH, padW:-padW]

        output = self.stats_conv_transpose(output)
        return output

    def forward(self, patchs, edge_weights, node_degree):
        # C^T C
        edges_signals = self.op_C(patchs, edge_weights, node_degree)
        output_patchs = self.op_C_transpose(edges_signals, edge_weights, node_degree)

        return output_patchs


#########################################################################
class DCestimator(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_features):
        super(DCestimator, self).__init__()

        self.project_in = nn.Conv2d(dim_in, hidden_features*2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=False)

    def forward(self, patchs):

        out = self.project_in(patchs)
        out01, out02 = self.dwconv(out).chunk(2, dim=1)
        out = nn.functional.gelu(out01) * out02
        out = self.project_out(out)
        return out
    

class MixtureGTV(nn.Module):
    def __init__(self, 
            nchannels_in,
            n_graphs,
            n_node_fts,
            n_cnn_fts,
            connection_window,
            n_cgd_iters,
            alpha_init,
            beta_init,
            muy_init, ro_init, gamma_init,
            device
        ):
        super(MixtureGTV, self).__init__()

        self.device       = device
        self.n_graphs     = n_graphs
        self.n_node_fts   = n_node_fts
        self.n_total_fts  = n_graphs * n_node_fts
        self.n_cnn_fts    = n_cnn_fts
        self.n_levels     = 4
        self.n_cgd_iters  = n_cgd_iters
        self.nchannels_in = nchannels_in
        self.connection_window = connection_window

        self.alphaCGD =  Parameter(
            torch.ones((self.n_cgd_iters, n_graphs), device=self.device, dtype=torch.float32) * alpha_init,
            requires_grad=True
        )

        self.betaCGD =  Parameter(
            torch.ones((self.n_cgd_iters, n_graphs), device=self.device, dtype=torch.float32) * beta_init,
            requires_grad=True
        )

        self.patchs_features_extraction = FeatureExtraction(
            inp_channels=3, 
            out_channels=self.n_total_fts + 12, 
            dim=self.n_cnn_fts,
            num_blocks = [2, 3, 3, 4], 
            num_refinement_blocks = 4,
            ffn_expansion_factor = 2.6666,
            bias = False,
        ).to(self.device)

        self.combination_weight = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_total_fts, 
                out_channels=self.n_graphs, 
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
                bias=False
            ),
            nn.Softmax(dim=1)
        ).to(self.device)

        self.dc_estimator = DCestimator(12, 3, 12*2).to(self.device)

        self.ro00 = Parameter(
            torch.ones((n_graphs), device=self.device, dtype=torch.float32) * ro_init[0],
            requires_grad=True,
        )
        self.gamma00 = Parameter(
            torch.ones((n_graphs), device=self.device, dtype=torch.float32) * torch.log(gamma_init[0]),
            requires_grad=True,
        )
        self.GTVmodule00 = GTVFast(
            n_channels=self.nchannels_in,
            n_node_fts=self.n_node_fts,
            n_graphs=self.n_graphs,
            connection_window=self.connection_window,
            device=self.device,
            M_diag_init=1.0
        )

        self.muys00 = Parameter(
            torch.ones((n_graphs), device=self.device, dtype=torch.float32) * muy_init[0],
            requires_grad=True,
        )
        self.GLRmodule00 = GLRFast(
            n_channels=self.nchannels_in,
            n_node_fts=self.n_node_fts,
            n_graphs=self.n_graphs,
            connection_window=self.connection_window,
            device=self.device,
            M_diag_init=1.0
        )

    def apply_lightweight_transformer(self, patchs, list_graph_weightGTV, list_graph_weightGLR):

        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
        patchs = patchs.contiguous()
        graph_weights, graph_degree = list_graph_weightGLR[0]

        Lpatchs = self.GLRmodule00(patchs, graph_weights, graph_degree)
        Lpatchs = torch.einsum(
            "bHchw, H -> bHchw", Lpatchs, self.muys00
        )

        graph_weights, graph_degree = list_graph_weightGTV[0]
        CtCpatchs = self.GTVmodule00(patchs, graph_weights, graph_degree)
        CtCpatchs = torch.einsum(
            "bHchw, H -> bHchw", CtCpatchs, self.ro00
        )

        output = patchs + Lpatchs + CtCpatchs

        return output
    
    def soft_threshold(self, delta, gamma):
        # batch_size, n_graphs, n_channels, n_edges, H, W = delta.shape
        # n_graphs = gamma.shape

        gamma = gamma[None, :, None, None, None, None]
        # print(f"Gamma.shape={gamma.shape}")

        condA = (delta < -gamma) 
        outputA = torch.where(
            condA,
            delta+gamma,
            0.0
        )
        condB = (delta > gamma) 
        outputB = torch.where(
            condB,
            delta-gamma,
            0.0
        )
        output = outputA + outputB
        return output


    def forward(self, patchs):
        # with record_function("MultiScaleMixtureGLR:forward"): 
        # print("#"*80)
        # patchs = patchs.permute(dims=(0, 3, 1, 2))
        # patchs = patchs.contiguous()
        # patchs = self.images_domain_to_abtract_domain(patchs)
        # print(f"patchs.shape={patchs.shape}")
        batch_size, c_size, h_size, w_size = patchs.shape

        #####
        ## Graph low pass filter
        
        list_features_patchs = self.patchs_features_extraction(patchs)
        list_graph_weightGTV = [None] * self.n_levels
        list_graph_weightGLR = [None] * self.n_levels
        bz, nfts, h, w = list_features_patchs[0].shape

        list_graph_weightGTV[0] = self.GTVmodule00.extract_edge_weights(
            list_features_patchs[0][:, :-12, :, :].view((bz, self.GTVmodule00.n_graphs, self.GTVmodule00.n_node_fts, h, w))
        )
        list_graph_weightGLR[0] = self.GLRmodule00.extract_edge_weights(
            list_features_patchs[0][:, :-12, :, :].view((bz, self.GLRmodule00.n_graphs, self.GLRmodule00.n_node_fts, h, w))
        )


        dc_term = self.dc_estimator(list_features_patchs[0][:, -12:, :, :])
        y_tilde = patchs - dc_term
        ###########################################################


        epsilon = self.GTVmodule00.op_C(y_tilde[:, None, :, :, :], list_graph_weightGTV[0][0], list_graph_weightGTV[0][1])
        bias    = torch.zeros_like(epsilon)

        left_hand_size = self.GTVmodule00.op_C_transpose(epsilon - bias, list_graph_weightGTV[0][0], list_graph_weightGTV[0][1]) 
        left_hand_size *= self.ro00[None, :, None, None, None]
        left_hand_size += y_tilde[:, None, :, :, :]
        ############################################################
        output = left_hand_size
        system_residual = left_hand_size -  self.apply_lightweight_transformer(output, list_graph_weightGTV, list_graph_weightGLR)
        update = system_residual
        output = output + self.alphaCGD[0, None, :, None, None, None] * update

        system_residual = left_hand_size -  self.apply_lightweight_transformer(output, list_graph_weightGTV, list_graph_weightGLR)
        update = system_residual + self.betaCGD[1, None, :, None, None, None] * update
        output = output + self.alphaCGD[1, None, :, None, None, None] * update

        epsilon = self.soft_threshold(
            self.GTVmodule00.op_C(output, list_graph_weightGTV[0][0], list_graph_weightGTV[0][1]) + bias,
            torch.exp(self.gamma00)
        )
        bias  = bias + (self.GTVmodule00.op_C(output, list_graph_weightGTV[0][0], list_graph_weightGTV[0][1]) - epsilon)

        left_hand_size = self.GTVmodule00.op_C_transpose(epsilon - bias, list_graph_weightGTV[0][0], list_graph_weightGTV[0][1]) 
        left_hand_size *= self.ro00[None, :, None, None, None]
        left_hand_size += y_tilde[:, None, :, :, :]
        ############################################################

        output = left_hand_size
        system_residual = left_hand_size -  self.apply_lightweight_transformer(output, list_graph_weightGTV, list_graph_weightGLR)
        update = system_residual
        output = output + self.alphaCGD[2, None, :, None, None, None] * update

        system_residual = left_hand_size -  self.apply_lightweight_transformer(output, list_graph_weightGTV, list_graph_weightGLR)
        update = system_residual + self.betaCGD[3, None, :, None, None, None] * update
        output = output + self.alphaCGD[3, None, :, None, None, None] * update

        # system_residual = left_hand_size -  self.apply_lightweight_transformer(output, list_graph_weightGTV, list_graph_weightGLR)
        # update = system_residual + self.betaCGD[4, None, :, None, None, None] * update
        # output = output + self.alphaCGD[4, None, :, None, None, None] * update
        
        # system_residual = left_hand_size -  self.apply_lightweight_transformer(output, list_graph_weightGTV, list_graph_weightGLR)
        # update = system_residual + self.betaCGD[5, None, :, None, None, None] * update
        # output = output + self.alphaCGD[5, None, :, None, None, None] * update
        

        score = self.combination_weight(list_features_patchs[0][:, :-12, :, :])
        output = torch.einsum(
            "bgchw, bghw -> bchw", output, score
        ) + dc_term

        return output


#########################################################################
class SharpeningBlock(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_features):
        super(SharpeningBlock, self).__init__()

        self.project_in = nn.Conv2d(dim_in, hidden_features*2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=False)
        self.skip_connect_weight = Parameter(
            torch.ones((2), dtype=torch.float32) * torch.tensor([0.5, 0.5]),
            requires_grad=True
        )
        
    def forward(self, patchs):

        out = self.project_in(patchs)
        out01, out02 = self.dwconv(out).chunk(2, dim=1)
        out = nn.functional.gelu(out01) * out02
        out = self.project_out(out)
        out = self.skip_connect_weight[0] * patchs + self.skip_connect_weight[1] * out
        return out

class MultiScaleSequenceDenoiser(nn.Module):
    def __init__(self, device):
        super(MultiScaleSequenceDenoiser, self).__init__()
        self.device = device

        # CONNECTION_FLAGS_5x5 = np.array([
        #     1,1,1,1,1,
        #     1,1,1,1,1,
        #     1,1,0,1,1,
        #     1,1,1,1,1,
        #     1,1,1,1,1,
        # ]).reshape((5,5))
        CONNECTION_FLAGS_5x5_small = np.array([
            0,0,1,0,0,
            0,1,1,1,0,
            1,1,0,1,1,
            0,1,1,1,0,
            0,0,1,0,0,
        ]).reshape((5,5))

        self.skip_connect_weight03 = Parameter(
            torch.ones((2), dtype=torch.float32, device=self.device) * torch.tensor([0.1, 0.9]).to(device),
            requires_grad=True
        )
        self.mixtureGLR_block03 = MixtureGTV(
            nchannels_in=3,
            n_graphs=24,
            n_node_fts=3,
            n_cnn_fts=72,
            connection_window=CONNECTION_FLAGS_5x5_small,
            n_cgd_iters=4,
            alpha_init=0.5,
            beta_init=0.1,
            muy_init=torch.tensor([[0.1], [0.0], [0.0], [0.0]]).to(self.device),
            ro_init=torch.tensor([[0.1], [0.0], [0.0], [0.0]]).to(self.device),
            gamma_init=torch.tensor([[0.001], [0.0], [0.0], [0.0]]).to(self.device),
            device=self.device
        )
    
    def forward(self, patchs):
        output = self.skip_connect_weight03[0] * patchs + self.skip_connect_weight03[1] * self.mixtureGLR_block03(patchs)
        return output

