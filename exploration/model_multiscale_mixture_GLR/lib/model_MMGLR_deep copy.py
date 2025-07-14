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
## Gated-Dconv Feed-Forward Network (GDFN)
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

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
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

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[FFBlock(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[FFBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])


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
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return [latent, out_dec_level3, out_dec_level2, out_dec_level1]



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


        ### Trainable parameters
        # features on nodes #self.n_node_fts
        self.multiM = Parameter(
            torch.ones((self.n_graphs, self.n_node_fts), device=self.device, dtype=torch.float32)*M_diag_init,
            requires_grad=True,
        )
        #############
        avg_kernel = 1.0 * torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        # delta_x = 0.5 * torch.tensor([
        #     [0.0, 0.0, 0.0],
        #     [0.0,-1.0, 1.0],
        #     [0.0, 0.0, 0.0],
        # ])
        # delta_y = 0.5 * torch.tensor([
        #     [0.0, 0.0, 0.0],
        #     [0.0,-1.0, 0.0],
        #     [0.0, 1.0, 0.0],
        # ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(avg_kernel[np.newaxis, np.newaxis, :, :])
            # kernel.append(delta_x[np.newaxis, np.newaxis, :, :])
            # kernel.append(delta_y[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel = Parameter(
            torch.ones((self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel,
            requires_grad=True,
        )


    def get_neighbors_pixels(self, img_features):
        _, _, H, W = img_features.shape
        padH, padW = self.pad_dim_hw
        img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
        neighbors_pixels = []
        for shift_h, shift_w in self.edge_delta:
            fromh = padH + shift_h
            toh = padH + shift_h + H
            fromw = padW + shift_w
            tow = padW + shift_w + W
            
            neighbors_pixels.append(
                img_features_frame[:, :, fromh:toh, fromw:tow]
            )
        neighbors_pixels_features = torch.stack(neighbors_pixels, axis=-3)

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

        # features_similarity = torch.clip(features_similarity, max=10, min=-10)

        edge_weights_norm = nn.functional.softmax(features_similarity, dim=2) 
        node_degree = edge_weights_norm.sum(axis=2)

        # edge_weights = torch.exp(features_similarity)
        # node_degree = edge_weights.sum(axis=2)
        
        # norm_factor = 1.0 / torch.sqrt(node_degree)
        # norm_factor_neighbors = self.get_neighbors_pixels(norm_factor)
        # edge_weights_norm = norm_factor[:, :, None, :, :] * edge_weights * norm_factor_neighbors

        # norm_factor = 1.0 / node_degree
        # edge_weights_norm = norm_factor[:, :, None, :, :] * edge_weights

        # edge_weights_norm = edge_weights

        return edge_weights_norm, node_degree
    


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

    def stats_conv(self, patchs):
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
        temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'reflect')
        temp_out_patch = nn.functional.conv2d(
            temp_patch,
            weight=self.stats_kernel,
            stride=1,
            padding=0,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
        return out_patch

    def stats_conv_transpose(self, patchs):
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.reshape(batch_size*n_graphs, c_size, h_size, w_size)
        temp_out_patch = nn.functional.conv_transpose2d(
            temp_patch,
            weight=self.stats_kernel,
            stride=1,
            padding=1,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, h_size)
        return out_patch

    def forward(self, patchs, edge_weights, node_degree):
        # with record_function("GLR:forward"): 
        # F
        # batch_size, n_graphs, c_size, h_size, w_size = patchs.shape

        patchs = self.stats_conv(patchs)
        # batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        # L
        output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)

        # F^T
        output_patchs = self.stats_conv_transpose(output_patchs)
        return output_patchs
    


class MixtureGGLR(nn.Module):
    def __init__(self, 
            nchannels_in,
            n_graphs,
            n_node_fts,
            connection_window,
            n_cgd_iters,
            alpha_init,
            beta_init,
            muy_init,
            device
        ):
        super(MixtureGGLR, self).__init__()

        self.device       = device
        self.n_graphs     = n_graphs
        self.n_node_fts   = n_node_fts
        self.n_total_fts  = n_graphs * n_node_fts
        self.n_cgd_iters  = n_cgd_iters
        self.nchannels_in = nchannels_in
        self.connection_window = connection_window

        self.muys = Parameter(
            torch.ones((n_graphs), device=self.device, dtype=torch.float32) * muy_init,
            requires_grad=True,
        )

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
            out_channels=self.n_total_fts, 
            dim = self.n_total_fts,
            num_blocks = [2, 2, 2, 2], 
            num_refinement_blocks = 4,
            ffn_expansion_factor = 1,
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

        self.GGLRmodule = GLRFast(
            n_channels=self.nchannels_in,
            n_node_fts=self.n_node_fts,
            n_graphs=self.n_graphs,
            connection_window=self.connection_window,
            device=self.device,
            M_diag_init=1.0
        )

    def apply_lightweight_transformer(self, patchs, graph_weights, graph_degree):

        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
        patchs = patchs.contiguous()

        Lpatchs = self.GGLRmodule(patchs, graph_weights, graph_degree)
        Lpatchs = torch.einsum(
            "bHchw, H -> bHchw", Lpatchs, self.muys
        )
        output = patchs + Lpatchs

        return output

    # def apply_lightweight_transformer01(self, patchs, graph_weights, graph_degree):

    #     batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
    #     patchs = patchs.contiguous()

    #     Lpatchs = self.GGLRmodule01(patchs, graph_weights, graph_degree)
    #     Lpatchs = torch.einsum(
    #         "bHchw, H -> bHchw", Lpatchs, self.muys01
    #     )
    #     output = patchs + Lpatchs

    #     return output

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
        features_patchs = self.patchs_features_extraction(patchs)[-1]
        bz, nfts, h, w = features_patchs.shape
        gW, gD = self.GGLRmodule.extract_edge_weights(
            features_patchs.view((bz, self.GGLRmodule.n_graphs, self.GGLRmodule.n_node_fts, h, w))
        )

        # 1st step
        output = patchs[:, None, :, :, :]
        system_residual = patchs[:, None, :, :, :] -  self.apply_lightweight_transformer(output, gW, gD)
        update = system_residual
        output = output + self.alphaCGD[0, None, :, None, None, None] * update

        # 2nd step
        system_residual = patchs[:, None, :, :, :] -  self.apply_lightweight_transformer(output, gW, gD)
        update = system_residual + self.betaCGD[1, None, :, None, None, None] * update
        output = output + self.alphaCGD[1, None, :, None, None, None] * update

        # 3rd step
        system_residual = patchs[:, None, :, :, :] -  self.apply_lightweight_transformer(output, gW, gD)
        update = system_residual + self.betaCGD[2, None, :, None, None, None] * update
        output = output + self.alphaCGD[2, None, :, None, None, None] * update

        # 4th step
        system_residual = patchs[:, None, :, :, :] -  self.apply_lightweight_transformer(output, gW, gD)
        update = system_residual + self.betaCGD[3, None, :, None, None, None] * update
        output = output + self.alphaCGD[3, None, :, None, None, None] * update

        # 5th step
        system_residual = patchs[:, None, :, :, :] -  self.apply_lightweight_transformer(output, gW, gD)
        update = system_residual + self.betaCGD[4, None, :, None, None, None] * update
        output = output + self.alphaCGD[4, None, :, None, None, None] * update

        score = self.combination_weight(features_patchs)
        output = torch.einsum(
            "bgchw, bghw -> bchw", output, score
        )

        return output

