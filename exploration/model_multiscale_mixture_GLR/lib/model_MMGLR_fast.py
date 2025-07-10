import itertools
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.profiler import profile, record_function, ProfilerActivity
# torch.set_default_dtype(torch.float64)

from einops import rearrange


class GLRFast(nn.Module):
    def __init__(self, 
            n_channels, n_node_fts, n_graphs, connection_window, device,
            M_diag_init=0.4
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
            (torch.ones((self.n_graphs, self.n_node_fts), device=self.device, dtype=torch.float32))*M_diag_init,
            requires_grad=True,
        )

        #############
        avg_kernel = 1.0 * torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        delta_x = 0.5 * torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0,-1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        delta_y = 0.5 * torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0,-1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        kernel = []
        for r in range(self.n_channels):
            kernel.append(avg_kernel[np.newaxis, np.newaxis, :, :])
            kernel.append(delta_x[np.newaxis, np.newaxis, :, :])
            kernel.append(delta_y[np.newaxis, np.newaxis, :, :])

        kernel = torch.concat(kernel, axis=0).to(self.device)
        self.stats_kernel = Parameter(
            torch.ones((3*self.n_channels, 1, 3, 3), device=self.device, dtype=torch.float32) * kernel,
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
    
        img_features_transform = img_features_transform.reshape(batch_size, self.n_graphs*self.n_node_fts, h_size, w_size)

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
        # edge_weights_norm = nn.functional.softmax(features_similarity, dim=2)
        # node_degree = edge_weights.sum(axis=2)
        
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

    
    def stats_conv(self, patchs):
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
        temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'reflect')
        temp_out_patch = nn.functional.conv2d(
            temp_patch,
            weight=self.stats_kernel,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, 3*c_size, h_size, w_size)
        return out_patch

    def stats_conv_transpose(self, patchs):
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        temp_patch = patchs.reshape(batch_size*n_graphs, c_size, h_size, w_size)
        temp_out_patch = nn.functional.conv_transpose2d(
            temp_patch,
            weight=self.stats_kernel,
            stride=1,
            padding=1,
            dilation=1,
            groups=self.n_channels,
        )
        out_patch = temp_out_patch.view(batch_size, n_graphs, c_size//3, h_size, h_size)
        return out_patch


    def forward(self, patchs, edge_weights, node_degree):
        # with record_function("GLR:forward"): 
        # F
        # batch_size, n_graphs, c_size, h_size, w_size = patchs.shape

        patchs = self.stats_conv(patchs)
        # L
        output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)
        
        # F^T
        output_patchs = self.stats_conv_transpose(output_patchs)
        return output_patchs


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Extractor(nn.Module):
    def __init__(self, n_features_in, n_features_out, n_channels_in, n_channels_out, device):
        super(Extractor, self).__init__()

        self.device         = device
        # self.kernel_size    = 3
        # self.stride         = 2
        # self.padding        = 1
        self.n_features_in  = n_features_in
        self.n_features_out = n_features_out
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        
        # Downsampler kernel
        self.down_sampling_kernel = Parameter(
            torch.ones((n_channels_out, 1, 3, 3), device=self.device, dtype=torch.float32) * (1/9),
            requires_grad=True,
        )
    
        # Features extractor
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=n_features_in, 
                out_channels=n_features_out, 
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                padding_mode="replicate",
                bias=False
            ),
            # nn.PixelUnshuffle(2),
            LayerNorm(n_features_out, 'BiasFree')
        ).to(self.device)

    def downsampling(self, input_patchs):
        output_patchs = nn.functional.conv2d(
            input_patchs,
            weight=self.down_sampling_kernel,
            stride=2,
            padding=1,
            dilation=1,
            groups=self.n_channels_in,
        )
        return output_patchs

    def upsampling(self, input_patchs):
        output_patchs = nn.functional.conv_transpose2d(
            input_patchs,
            weight=self.down_sampling_kernel,
            stride=2,
            padding=0,
            dilation=1,
            groups=self.n_channels_in,
        )[:, :, 1:, 1:]
        return output_patchs

    def forward(self, input_features_patchs):
        # with record_function("Extractor:forward"): 
        output_features_patchs = self.cnn_layer(input_features_patchs)
        return output_features_patchs


class MultiScaleMixtureGLR(nn.Module):
    def __init__(self, n_levels, n_graphs, n_cgd_iters, alpha_init, beta_init, muy_init, device, GLR_modules_conf=[],  Extractor_modules_conf=[]):
        super(MultiScaleMixtureGLR, self).__init__()

        self.device      = device
        self.n_levels    = n_levels
        self.n_graphs    = n_graphs
        self.n_cgd_iters = n_cgd_iters
        self.nchannels_images  = 3
        self.nchannels_abtract = Extractor_modules_conf[0]["ExtractorConf"]["n_channels_in"]


        self.muys = Parameter(
            torch.ones((self.n_levels, n_graphs), device=self.device, dtype=torch.float32) * muy_init,
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

        # self.combination_weight = Parameter(
        #     torch.ones((self.n_graphs), device=self.device, dtype=torch.float32)/self.n_graphs,
        #     requires_grad=True,
        # )

        self.combination_weight = nn.Sequential(
            nn.Conv2d(
                in_channels=Extractor_modules_conf[0]["ExtractorConf"]["n_features_in"], 
                out_channels=self.n_graphs, 
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
                bias=False
            ),
            nn.Softmax(dim=1)
        ).to(self.device)

        self.patchs_embeding = nn.Conv2d(
            Extractor_modules_conf[0]["ExtractorConf"]["n_channels_in"],
            Extractor_modules_conf[0]["ExtractorConf"]["n_features_in"], 
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
            bias=False
        ).to(self.device)

        self.list_mixtureGLR = nn.ModuleList([])
        for level in range(self.n_levels): 
            glr_module = GLRFast(**GLR_modules_conf[level]["GLRConf"]) 
            # glr_module.compile()
            self.list_mixtureGLR.append(glr_module)

        self.list_Extractor = nn.ModuleList([])
        for level in range(self.n_levels-1):
            extractor_module = Extractor(**Extractor_modules_conf[level]["ExtractorConf"]) 
            # extractor_module.compile()
            self.list_Extractor.append(extractor_module)


    def apply_multi_scale_lightweight_transformer(self, patchs, list_graph_weights):
        
        # with record_function("MultiScaleMixtureGLR:apply_multi_scale_lightweight_transformer"): 
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
        patchs = patchs.contiguous()
        
        list_2D_signals = [patchs]

        ###########################################################################
        # H
        # print(f"list_2D_signals[0].shape={patchs.shape}")
        downsampled_patchs = patchs
        for level in range(self.n_levels-1):

            extractor_module = self.list_Extractor[level]
            batch_size, n_graphs, c_size, h_size, w_size = downsampled_patchs.shape 

            downsampled_patchs = extractor_module.downsampling(
                downsampled_patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
            )
            
            _, c_size_new, h_size_new, w_size_new = downsampled_patchs.shape 
            downsampled_patchs = downsampled_patchs.view(batch_size, n_graphs, c_size_new, h_size_new, w_size_new)

            list_2D_signals.append(downsampled_patchs)
            # print(f"list_2D_signals[{level+1}].shape={list_2D_signals[level+1].shape}")

        ###########################################################################
        # L
        list_2D_signals_L_transform = []
        for level in range(self.n_levels):
            glr_module = self.list_mixtureGLR[level]
            laplacian_2D_signals = glr_module(list_2D_signals[level], list_graph_weights[level][0], list_graph_weights[level][1])
            list_2D_signals_L_transform.append(laplacian_2D_signals)
            # print(f"list_2D_signals_L_transform[{level}].shape={list_2D_signals_L_transform[level].shape}")



        ###########################################################################
        # H^T
        upsampled_patchs = list_2D_signals_L_transform[self.n_levels-1]
        # print(f"upsampled_patchs[{self.n_levels-1}].shape={upsampled_patchs.shape}")
        for level in range(self.n_levels-2, -1, -1):

            extractor_module = self.list_Extractor[level]
            batch_size, n_graphs, c_size, h_size, w_size = upsampled_patchs.shape 

            upsampled_patchs = extractor_module.upsampling(
                upsampled_patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
            )
            _, c_size_new, h_size_new, w_size_new = upsampled_patchs.shape
            upsampled_patchs = upsampled_patchs.view(batch_size, n_graphs, c_size_new, h_size_new, w_size_new)
            # print(f"upsampled_patchs[{level}].shape={upsampled_patchs.shape}")

            # print(f"concat({list_2D_signals_L_transform[level].shape}, {upsampled_patchs.shape})")
            upsampled_patchs = torch.concat([
                list_2D_signals_L_transform[level],
                upsampled_patchs,
            ], axis=1)
            # print(f"upsampled_patchs_cummulative.shape={upsampled_patchs.shape}")
    

        ###########################################################################
        # Sum of log experts <-> product of experts
        batch_size, _, c_size, h_size, w_size = upsampled_patchs.shape 
        upsampled_patchs_final = upsampled_patchs.view((batch_size, self.n_levels, self.n_graphs, c_size, h_size, w_size))
        # print(f"upsampled_patchs_final.shape={upsampled_patchs_final.shape}")
        
        outputLx = torch.einsum(
            "bLHchw, LH -> bHchw", upsampled_patchs_final, self.muys
        )
        output = patchs + outputLx
        # print(f"outputLx.shape={output.shape}")
        
        return output

    def forward(self, patchs):
        # with record_function("MultiScaleMixtureGLR:forward"): 
        # print("#"*80)
        patchs = patchs.permute(dims=(0, 3, 1, 2))
        # patchs = self.images_domain_to_abtract_domain(patchs)
        # print(f"patchs.shape={patchs.shape}")
        batch_size, c_size, h_size, w_size = patchs.shape
        features_patchs_core = self.patchs_embeding(patchs)

        features_patchs = features_patchs_core
        glr_module = self.list_mixtureGLR[0]

        # print(f"features_patchs.shape={features_patchs.shape}")
        bz, nfts, h, w = features_patchs.shape
        gW, gD = glr_module.extract_edge_weights(
            features_patchs.view((bz, glr_module.n_graphs, glr_module.n_node_fts, h, w))
        )
        # print(f"gW.shape={gW.shape} gD.shape={gD.shape}")
        list_graph_weights = [(gW, gD)]
        for level in range(self.n_levels-1):
            # print("#"*80)
            ###
            extractor_module = self.list_Extractor[level]
            glr_module = self.list_mixtureGLR[level+1]

            ###
            features_patchs = extractor_module(features_patchs)
            # print(f"features_patchs.shape={features_patchs.shape}")
            bz, nfts, h, w = features_patchs.shape
            gW, gD = glr_module.extract_edge_weights(
                features_patchs.view((bz, glr_module.n_graphs, glr_module.n_node_fts, h, w))
            )

            # print(f"gW.shape={gW.shape} gD.shape={gD.shape}")
            list_graph_weights.append((gW, gD))

        
        # print("#"*80)

        output = patchs[:, None, :, :, :]
        system_residual = output -  self.apply_multi_scale_lightweight_transformer(output, list_graph_weights)
        update = system_residual

        for iter in range(self.n_cgd_iters):
            A_mul_update = self.apply_multi_scale_lightweight_transformer(update, list_graph_weights)
            output = output + self.alphaCGD[iter, None, :, None, None, None] * update
            system_residual = system_residual - self.alphaCGD[iter, None, :, None, None, None] * A_mul_update
            update = system_residual + self.betaCGD[iter, None, :, None, None, None] * update

        score = self.combination_weight(features_patchs_core)
        output = torch.einsum(
            "bgchw, bghw -> bchw", output, score
        )

        # output = torch.clip(output, min=0.0, max=1.0).permute(dims=(0, 2, 3, 1))
        # print(f"output.shape={output.shape}")
        # output = self.abtract_domain_to_images_domain(output)
        # print(f"output_patch.shape={output.shape}")
        output = output.permute(dims=(0, 2, 3, 1))
        return output
    


class ModelLightWeightTransformerGLR(nn.Module):
    def __init__(self, img_height, img_width, n_blocks, n_graphs, n_levels, device, global_mmglr_confs={}):
        super(ModelLightWeightTransformerGLR, self).__init__()

        # self.nchannels_images = nchannels_images 
        # self.nchannels_abtract = nchannels_abtract 
        self.n_blocks = n_blocks
        self.n_graphs = n_graphs
        self.n_levels = n_levels
        self.device = device

        self.light_weight_transformer_blocks = nn.ModuleList([])
        for block_i in range(self.n_blocks):
            block = MultiScaleMixtureGLR(**global_mmglr_confs)
            self.light_weight_transformer_blocks.append(block)

        self.cumulative_result_weight = Parameter(
            torch.ones((n_blocks), device=self.device, dtype=torch.float32) * 0.99,
            requires_grad=True
        )

    def forward(self, input_patchs):
        output = input_patchs
        for block_i in range(0, self.n_blocks):
            block = self.light_weight_transformer_blocks[block_i]
            output_temp = block(output)

            p = self.cumulative_result_weight[block_i]
            output = p * output_temp + (1-p) * output
        
        return output


