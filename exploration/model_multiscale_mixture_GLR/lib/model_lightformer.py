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
    


# class GGLRFast(nn.Module):
#     def __init__(self, 
#             n_channels, n_node_fts, n_graphs, connection_window, device, M_diag_init=0.4
#         ):
#         super(GGLRFast, self).__init__()

#         self.device = device
#         self.n_channels        = n_channels
#         self.n_node_fts        = n_node_fts
#         self.n_graphs          = n_graphs
#         self.n_edges           = (connection_window == 1).sum()
#         self.connection_window = connection_window
#         self.buffer_size       = connection_window.sum()

#         # edges type from connection_window
#         window_size = connection_window.shape[0]
#         connection_window = connection_window.reshape((-1))
#         m = np.arange(window_size)-window_size//2
#         edge_delta = np.array(
#             list(itertools.product(m, m)),
#             dtype=np.int32
#         )
#         self.edge_delta = edge_delta[connection_window == 1]
        
#         self.pad_dim_hw = np.abs(self.edge_delta.min(axis=0))


#         ### Trainable parameters
#         # features on nodes #self.n_node_fts
#         self.multiM = Parameter(
#             (torch.ones((self.n_graphs, self.n_node_fts), device=self.device, dtype=torch.float32))*M_diag_init,
#             requires_grad=True,
#         )

#         #############
#         avg_kernel = 1.0 * torch.tensor([
#             [0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0],
#         ])
#         delta_x = 0.5 * torch.tensor([
#             [0.0, 0.0, 0.0],
#             [0.0,-1.0, 1.0],
#             [0.0, 0.0, 0.0],
#         ])
#         delta_y = 0.5 * torch.tensor([
#             [0.0, 0.0, 0.0],
#             [0.0,-1.0, 0.0],
#             [0.0, 1.0, 0.0],
#         ])
#         kernel = []
#         for r in range(self.n_channels):
#             kernel.append(avg_kernel[np.newaxis, np.newaxis, :, :])
#             kernel.append(delta_x[np.newaxis, np.newaxis, :, :])
#             kernel.append(delta_y[np.newaxis, np.newaxis, :, :])

#         kernel = torch.concat(kernel, axis=0).to(self.device)
#         self.stats_kernel = Parameter(
#             torch.ones((3*self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel,
#             requires_grad=True,
#         )

#     def get_neighbors_pixels(self, img_features):
#         _, _, H, W = img_features.shape
#         padH, padW = self.pad_dim_hw
#         img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
#         neighbors_pixels = []
#         for shift_h, shift_w in self.edge_delta:
#             fromh = padH + shift_h
#             toh = padH + shift_h + H
#             fromw = padW + shift_w
#             tow = padW + shift_w + W
            
#             neighbors_pixels.append(
#                 img_features_frame[:, :, fromh:toh, fromw:tow]
#             )
#         neighbors_pixels_features = torch.stack(neighbors_pixels, axis=-3)

#         return neighbors_pixels_features
    

#     def normalize_and_transform_features(self, img_features):
#         batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape
#         # img_features = img_features.view(batch_size, self.n_graphs, self.n_node_fts, h_size, w_size)
#         img_features = torch.nn.functional.normalize(img_features, dim=2)

#         img_features_transform = torch.einsum(
#             "bhcHW, hc -> bhcHW", img_features, self.multiM
#         )
    
#         img_features_transform = img_features_transform.view(batch_size, self.n_graphs*self.n_node_fts, h_size, w_size)

#         return img_features_transform


#     def extract_edge_weights(self, img_features):
        
#         batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape

#         img_features = self.normalize_and_transform_features(img_features)
#         img_features_neighbors = self.get_neighbors_pixels(img_features)

#         features_similarity = (img_features[:, :, None, :, :] * img_features_neighbors)
#         features_similarity = features_similarity.view(
#             batch_size, self.n_graphs, self.n_node_fts, self.n_edges, h_size, w_size
#         ).sum(axis=2)

#         features_similarity = torch.clip(features_similarity, max=10, min=-10)
        
#         edge_weights = torch.exp(features_similarity)
#         node_degree = edge_weights.sum(axis=2)
#         norm_factor = 1.0 / torch.sqrt(node_degree)
#         norm_factor_neighbors = self.get_neighbors_pixels(norm_factor)
#         edge_weights_norm = norm_factor[:, :, None, :, :] * edge_weights * norm_factor_neighbors

#         # norm_factor = 1.0 / node_degree
#         # edge_weights_norm = norm_factor[:, :, None, :, :] * edge_weights

#         return edge_weights_norm, node_degree
    


#     def op_L_norm(self, img_signals, edge_weights, node_degree):

#         batch_size, n_graphs, n_channels, h_size, w_size = img_signals.shape
#         img_features_neighbors = self.get_neighbors_pixels(
#             img_signals.view(batch_size, n_graphs*n_channels, h_size, w_size)
#         ).view(batch_size, n_graphs, n_channels, self.n_edges, h_size, w_size)
#         Wx = torch.einsum(
#             "bhceHW, bheHW -> bhcHW", img_features_neighbors, edge_weights
#         )
#         output = img_signals - Wx
#         return output

    
#     def stats_conv(self, patchs):
#         batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
#         temp_patch = patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
#         temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'reflect')
#         temp_out_patch = nn.functional.conv2d(
#             temp_patch,
#             weight=self.stats_kernel,
#             stride=1,
#             padding=0,
#             dilation=1,
#             groups=self.n_channels,
#         )
#         out_patch = temp_out_patch.view(batch_size, n_graphs, 3*c_size, h_size, w_size)
#         return out_patch

#     def stats_conv_transpose(self, patchs):
#         batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
#         temp_patch = patchs.reshape(batch_size*n_graphs, c_size, h_size, w_size)
#         temp_out_patch = nn.functional.conv_transpose2d(
#             temp_patch,
#             weight=self.stats_kernel,
#             stride=1,
#             padding=1,
#             dilation=1,
#             groups=self.n_channels,
#         )
#         out_patch = temp_out_patch.view(batch_size, n_graphs, c_size//3, h_size, h_size)
#         return out_patch


#     def forward(self, patchs, edge_weights, node_degree):
#         # with record_function("GLR:forward"): 
#         # F
#         # batch_size, n_graphs, c_size, h_size, w_size = patchs.shape

#         patchs = self.stats_conv(patchs)
#         # L
#         output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)
        
#         # F^T
#         output_patchs = self.stats_conv_transpose(output_patchs)
#         return output_patchs
    

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

        features_similarity = torch.clip(features_similarity, max=10, min=-10)
        
        edge_weights = torch.exp(features_similarity)
        node_degree = edge_weights.sum(axis=2)
        
        norm_factor = 1.0 / torch.sqrt(node_degree)
        norm_factor_neighbors = self.get_neighbors_pixels(norm_factor)
        edge_weights_norm = norm_factor[:, :, None, :, :] * edge_weights * norm_factor_neighbors

        # norm_factor = 1.0 / node_degree
        # edge_weights_norm = norm_factor[:, :, None, :, :] * edge_weights

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




    def forward(self, patchs, edge_weights, node_degree):
        # with record_function("GLR:forward"): 
        # F
        # batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        # L
        output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)

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



        self.skip_connect_weight_final = Parameter(
            torch.ones((2), device=self.device, dtype=torch.float32) * torch.tensor([0.5, 0.5]).to(self.device),
            requires_grad=True
        )

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

        self.patchs_features_extraction = nn.Sequential(
            CustomLayerNorm(self.nchannels_in),
            nn.Conv2d(
                self.nchannels_in,
                self.n_total_fts, 
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="replicate",
                bias=False
            ),
        ).to(self.device)

        self.combination_weight = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_total_fts, 
                out_channels=self.n_graphs, 
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
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
        features_patchs = self.patchs_features_extraction(patchs)
        bz, nfts, h, w = features_patchs.shape
        gW, gD = self.GGLRmodule.extract_edge_weights(
            features_patchs.view((bz, self.GGLRmodule.n_graphs, self.GGLRmodule.n_node_fts, h, w))
        )

        output = patchs[:, None, :, :, :]
        system_residual = output -  self.apply_lightweight_transformer(output, gW, gD)
        update = system_residual

        for iter in range(self.n_cgd_iters):
            A_mul_update = self.apply_lightweight_transformer(update, gW, gD)
            output = output + self.alphaCGD[iter, None, :, None, None, None] * update
            system_residual = system_residual - self.alphaCGD[iter, None, :, None, None, None] * A_mul_update
            update = system_residual + self.betaCGD[iter, None, :, None, None, None] * update

        score = self.combination_weight(features_patchs)
        output = torch.einsum(
            "bgchw, bghw -> bchw", output, score
        )

        final_output = self.skip_connect_weight_final[0] * output
        final_output += self.skip_connect_weight_final[1] * patchs
    
        return final_output
    
##########################################################################
##---------- MultiscaleLightWeightTransformer -----------------------
class MultiscaleLightWeightTransformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        n_abtract_channels=48,
        num_blocks_per_level=[4,6,6,8], 
        n_graphs_per_level=[1,2,4,8],
        n_cgd_per_level=[5,5,5,5],
        num_blocks_output = 4,
        device=torch.device("cpu")
    ):  
        
        super(MultiscaleLightWeightTransformer, self).__init__()
        self.device = device
        CONNECTION_FLAGS = np.array([
            1,1,1,
            1,0,1,
            1,1,1,
        ]).reshape((3,3))

        self.skip_connect_weight_final = Parameter(
            torch.ones((2), device=self.device, dtype=torch.float32) * torch.tensor([0.5, 0.5]).to(self.device),
            requires_grad=True
        )


        n_abtract_channels_01 = n_abtract_channels
        self.patch_embeding   = nn.Conv2d(
            inp_channels, n_abtract_channels_01, 
            kernel_size=3, stride=1, padding=1, bias=False
        ).to(self.device)

        self.denoise_level_01 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_01,
                "n_graphs": n_graphs_per_level[0],
                "n_node_fts": n_abtract_channels_01 // n_graphs_per_level[0],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[0],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[0])
        ]).to(self.device)

        self.denoise_decode_level_01 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_01,
                "n_graphs": n_graphs_per_level[0],
                "n_node_fts": n_abtract_channels_01 // n_graphs_per_level[0],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[0],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[0])
        ]).to(self.device)

        n_abtract_channels_02 = n_abtract_channels_01 * 2
        self.down_sampling_01_02 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_abtract_channels_01, 
                out_channels=n_abtract_channels_02//4, 
                kernel_size=3, stride=1, padding=1, dilation=1,
                padding_mode="replicate", bias=False
            ),
            nn.PixelUnshuffle(2)
        ).to(self.device)
        self.denoise_level_02 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_02,
                "n_graphs": n_graphs_per_level[1],
                "n_node_fts": n_abtract_channels_02 // n_graphs_per_level[1],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[1],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[1])
        ]).to(self.device)
        self.combine_channels_01 = nn.Conv2d(
            int(n_abtract_channels_01*2), int(n_abtract_channels_01), 
            kernel_size=1, bias=False
        ).to(self.device)
        self.up_sampling_02_01 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_abtract_channels_02,
                out_channels=n_abtract_channels_01*4,
                kernel_size=3, stride=1, padding=1, #output_padding=1, 
                padding_mode='zeros', bias=False,
            ),
            nn.PixelShuffle(2)
        ).to(self.device)
        self.denoise_decode_level_02 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_02,
                "n_graphs": n_graphs_per_level[1],
                "n_node_fts": n_abtract_channels_02 // n_graphs_per_level[1],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[1],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[1])
        ]).to(self.device)


        n_abtract_channels_03 = n_abtract_channels_02 * 2
        self.down_sampling_02_03 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_abtract_channels_02, 
                out_channels=n_abtract_channels_03//4, 
                kernel_size=3, stride=1, padding=1, dilation=1,
                padding_mode="replicate", bias=False
            ),
            nn.PixelUnshuffle(2)
        ).to(self.device)
        self.denoise_level_03 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_03,
                "n_graphs": n_graphs_per_level[2],
                "n_node_fts": n_abtract_channels_03 // n_graphs_per_level[2],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[2],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[2])
        ]).to(self.device)
        self.combine_channels_02 = nn.Conv2d(
            int(n_abtract_channels_02*2), int(n_abtract_channels_02), 
            kernel_size=1, bias=False
        ).to(self.device)
        self.up_sampling_03_02 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_abtract_channels_03,
                out_channels=n_abtract_channels_02*4,
                kernel_size=3, stride=1, padding=1, #output_padding=1, 
                padding_mode='zeros', bias=False,
            ),
            nn.PixelShuffle(2)
        ).to(self.device)
        self.denoise_decode_level_03 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_03,
                "n_graphs": n_graphs_per_level[2],
                "n_node_fts": n_abtract_channels_03 // n_graphs_per_level[2],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[2],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[2])
        ]).to(self.device)


        n_abtract_channels_04 = n_abtract_channels_03 * 2
        self.down_sampling_03_04 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_abtract_channels_03, 
                out_channels=n_abtract_channels_04//4, 
                kernel_size=3, stride=1, padding=1, dilation=1,
                padding_mode="replicate", bias=False
            ),
            nn.PixelUnshuffle(2)
        ).to(self.device)

        self.denoise_level_04 = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": n_abtract_channels_04,
                "n_graphs": n_graphs_per_level[3],
                "n_node_fts": n_abtract_channels_04 // n_graphs_per_level[3],
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": n_cgd_per_level[3],
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_per_level[3])
        ]).to(self.device)
        self.combine_channels_03 = nn.Conv2d(
            int(n_abtract_channels_03*2), int(n_abtract_channels_03), 
            kernel_size=1, bias=False
        ).to(self.device)
        self.up_sampling_04_03 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_abtract_channels_04,
                out_channels=n_abtract_channels_03*4,
                kernel_size=3, stride=1, padding=1, #output_padding=1, 
                padding_mode='zeros', bias=False,
            ),
            nn.PixelShuffle(2)
        ).to(self.device)


        ###################################################################
        self.final_combine_channel_output = nn.Conv2d(
            int(n_abtract_channels_01), int(out_channels),
            kernel_size=1, bias=False
        ).to(self.device)

        self.denoise_level_output = nn.Sequential(*[
            MixtureGGLR(**{
                "nchannels_in": 3,
                "n_graphs": 4,
                "n_node_fts": n_abtract_channels // 4,
                "connection_window": CONNECTION_FLAGS,
                "n_cgd_iters": 5,
                "alpha_init": 0.5,
                "beta_init": 0.1,
                "muy_init": 0.1,
                "device": self.device
            }) 
            for i in range(num_blocks_output)
        ]).to(self.device)

    def forward(self, patchs):

        encode_01 = self.patch_embeding(patchs)
        denoised_encode_01 = self.denoise_level_01(encode_01)

        encode_02 = self.down_sampling_01_02(denoised_encode_01)
        denoised_encode_02 = self.denoise_level_02(encode_02)

        encode_03 = self.down_sampling_02_03(denoised_encode_02)
        denoised_encode_03 = self.denoise_level_03(encode_03)

        encode_04 = self.down_sampling_03_04(denoised_encode_03)
        denoised_encode_04 = self.denoise_level_04(encode_04)

        decode_03 = self.up_sampling_04_03(denoised_encode_04)
        denoised_decode_03 = torch.cat([decode_03, denoised_encode_03], 1)
        denoised_decode_03 = self.combine_channels_03(denoised_decode_03)
        denoised_decode_03 = self.denoise_decode_level_03(denoised_decode_03)

        decode_02 = self.up_sampling_03_02(denoised_decode_03)
        denoised_decode_02 = torch.cat([decode_02, denoised_encode_02], 1)
        denoised_decode_02 = self.combine_channels_02(denoised_decode_02)
        denoised_decode_02 = self.denoise_decode_level_02(denoised_decode_02)


        decode_01 = self.up_sampling_02_01(denoised_decode_02)
        denoised_decode_01 = torch.cat([decode_01, denoised_encode_01], 1)
        denoised_decode_01 = self.combine_channels_01(denoised_decode_01)
        denoised_decode_01 = self.denoise_decode_level_01(denoised_decode_01)
        
        # final_output = self.final_combine_channel_output(denoised_decode_01)
        final_output = self.skip_connect_weight_final[0] * self.final_combine_channel_output(denoised_decode_01)
        final_output += self.skip_connect_weight_final[1] * patchs

        final_output = self.denoise_level_output(final_output)
    
        return final_output
        