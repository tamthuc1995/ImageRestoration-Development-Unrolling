import itertools
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.profiler import profile, record_function, ProfilerActivity
# torch.set_default_dtype(torch.float64)


def MortonFromPosition(position):
    """Convert integer (x,y,z) positions to Morton codes

    Args:
      positions: Nx3 np array (will be cast to int32)

    Returns:
      Length-N int64 np array
    """

    position = np.asarray(position, dtype=np.int32)
    morton_code = np.zeros(len(position), dtype=np.int64)
    coeff = np.asarray([4, 2, 1], dtype=np.int64)
    for b in range(21):
        morton_code |= ((position & (1 << b)) << (2 * b)) @ coeff
    assert morton_code.dtype == np.int64
    return morton_code


def PositionFromMorton(morton_code):
    """Convert int64 Morton code to int32 (x,y,z) positions

    Args:
      morton_code: int64 np array

    Returns:
      Nx3 int32 np array
    """

    morton_code = np.asarray(morton_code, dtype=np.int64)
    position = np.zeros([len(morton_code), 3], dtype=np.int32)
    shift = np.array([2, 1, 0], dtype=np.int64)
    for b in range(21):
        position |= ((morton_code[:, np.newaxis] >> shift[np.newaxis, :]) >> (2 * b)
                     ).astype(np.int32) & (1 << b)
    assert position.dtype == np.int32
    return position

def hash_to_index(hash_val, hash_table):
    if hash_val in hash_table:
        return hash_table[hash_val]
    else:
        return -1
hash_to_index_vec = np.vectorize(hash_to_index)


def create_sparse_structure_from_images(img_height, img_width, edge_delta):

    # CREATE NODES
    xindex, yindex = np.meshgrid(np.arange(img_width), np.arange(img_height))
    xy_location = np.stack([yindex, xindex], axis=2).reshape(-1, 2)
    hash_code = MortonFromPosition(
        np.concatenate([xy_location, np.zeros((xy_location.shape[0], 1))], axis=1)
    )
    order = np.argsort(hash_code)

    ## MUCH REMEMBER ORDER
    xy_location = xy_location[order]
    order_inverse = np.zeros(xy_location.shape[0], dtype=np.int32)
    order_inverse[order] = np.arange(xy_location.shape[0])
    
    hash_code = hash_code[order]
    hash_code_map = {code:i for i, code in enumerate(hash_code)}
    max_edge_type = edge_delta.shape[0]

    #
    possible_node_i_indx = np.arange(xy_location.shape[0], dtype=np.int32)[:, np.newaxis] + np.zeros([1, max_edge_type], dtype=np.int32)
    possible_node_i_indx = possible_node_i_indx.flatten()
    possible_edge_types  = np.repeat(np.arange(0, max_edge_type).reshape(1, max_edge_type), xy_location.shape[0], axis=0).flatten()

    #
    possible_node_j_location = xy_location[:, np.newaxis, :] + edge_delta[np.newaxis, :, :]
    possible_node_j_location = possible_node_j_location.reshape([-1, 2])
    possible_node_j_hash = MortonFromPosition(
        np.concatenate([possible_node_j_location, np.zeros((possible_node_j_location.shape[0], 1))], axis=1)
    )
    possible_node_j_indx = hash_to_index_vec(possible_node_j_hash, hash_code_map)

    #
    valid_edges = possible_node_j_indx >= 0
    node_i_indx = possible_node_i_indx[valid_edges]
    node_j_indx = possible_node_j_indx[valid_edges]
    edges_type  = possible_edge_types[valid_edges]
    edges = np.stack([
        node_i_indx, node_j_indx
    ], axis=1)

    ## Control meta information here
    sparse_image_obj = collections.OrderedDict()
    sparse_image_obj['order']          = order
    sparse_image_obj["node_locations"] = xy_location
    sparse_image_obj["order_inverse"]  = order_inverse
    sparse_image_obj['edges']          = edges
    sparse_image_obj["edges_type"]     = edges_type
    return sparse_image_obj


class GLR(nn.Module):
    def __init__(self, 
            input_width, input_height, n_channels, n_node_fts, n_graphs, connection_window, device,
            M_diag_init=0.4, Mxy_diag_init=1.0
        ):
        super(GLR, self).__init__()

        self.device = device
        self.n_channels        = n_channels
        self.n_node_fts        = n_node_fts
        self.n_graphs          = n_graphs
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
        self.edge_delta = torch.from_numpy(edge_delta[connection_window == 1]).to(self.device)

        ### CAN BE RECALIBRATED
        self.input_width  = None
        self.input_height = None
        self.graph_frame_recalibrate(input_height, input_width)

        ### Trainable parameters
        # features on nodes
        self.multiM = Parameter(
            torch.diag_embed(torch.ones((self.n_graphs, self.n_node_fts), device=self.device, dtype=torch.float32))*M_diag_init,
            requires_grad=True,
        )
        self.multiMxy = Parameter(
            torch.ones((self.n_graphs, 2), device=self.device, dtype=torch.float32)*Mxy_diag_init,
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

    # don't compile this
    @torch.compiler.disable(recursive=False)
    def graph_frame_recalibrate(self, input_height, input_width):

        if (self.input_width != input_width) or (self.input_height != input_height):

            self.input_width  = input_width
            self.input_height = input_height
            sparse_image_obj  = create_sparse_structure_from_images(input_height, input_width, self.edge_delta.cpu().numpy())
            self.graph_node_order          = torch.from_numpy(sparse_image_obj['order']).to(self.device)
            self.graph_node_order_inverse  = torch.from_numpy(sparse_image_obj['order_inverse']).to(self.device)
            self.graph_node_edges          = torch.from_numpy(sparse_image_obj['edges']).to(self.device)
            self.graph_node_edges_type     = torch.from_numpy(sparse_image_obj['edges_type']).to(self.device)
            # DEBUG info
            # self.graph_node_node_locations = torch.from_numpy(sparse_image_obj["node_locations"]).to(self.device)
        # else:
        #     print("No need to recalibrate")
        

    def sparse_sum_by_group(self, arr, groups, buffer_index, output_holder):
        # with record_function("GLR:sparse_sum_by_group"): 
        output_holder[:, :, :, groups, buffer_index] = arr
        output_sum = output_holder.sum(axis=-1)

        return output_sum


    @torch.compiler.disable(recursive=False)
    def extract_edge_weights(self, img_features):
        # with record_function("GLR:extract_edge_weights"): 
        batch_size, n_graphs, n_node_fts, h_size, w_size = img_features.shape
        # sigma = img_features.var(2, keepdim=True, unbiased=False)
        # img_features = img_features / torch.sqrt(sigma+0.00001)
        img_features = torch.nn.functional.normalize(img_features, dim=2)

        # print(f"sigma.shape={sigma.shape}")

        assert ((n_graphs == self.n_graphs) or (n_graphs == 1)), "n_graphs != self.n_graphs"
        assert (n_node_fts == self.n_node_fts), "n_graphs != self.n_graphs"
        assert (h_size == self.input_height), "h_size != self.input_height"
        assert (w_size == self.input_width), "w_size != self.input_width"
        node_features = self.signal2D_to_graph_signals(img_features)
        _, _, _, num_nodes = node_features.shape

        

        ## METHOD 02
        nodeI = self.graph_node_edges[:, 0]
        nodeJ = self.graph_node_edges[:, 1]
        node_featuresM = torch.einsum(
            "bhcn, hcv -> bhvn", node_features, self.multiM
        )
        features_node_i = node_featuresM[:, :, :, nodeI]
        features_node_j = node_featuresM[:, :, :, nodeJ]
        features_similarity = (features_node_i * features_node_j).sum(axis=2)
        # print(f"features_similarity.shape={features_similarity.shape}")
        # print(f"features_similarity.min={torch.min(features_similarity).detach().cpu().numpy()}")
        # print(f"features_similarity.max={torch.max(features_similarity).detach().cpu().numpy()}")

        features_similarity = torch.clip(features_similarity, max=10, min=-10)
        edge_weights = torch.exp(features_similarity)
        # print(f"edge_weights.shape={edge_weights.shape}")
        # print(f"edge_weights.min={torch.min(edge_weights).detach().cpu().numpy()}")
        # print(f"edge_weights.max={torch.max(edge_weights).detach().cpu().numpy()}")

        edge_weights = edge_weights[:, :, None, :] 
        # edge_weights = edge_weights[:, :, None, :] * base_edge_weights[None, :, None, self.graph_node_edges_type]
        # print(f"base_edge_weights.shape={base_edge_weights.shape}")
        # print(f"base_edge_weights.min={torch.min(base_edge_weights).detach().cpu().numpy()}")
        # print(f"base_edge_weights.max={torch.max(base_edge_weights).detach().cpu().numpy()}")

        #### Final edge weight as exp( features_part + location_part)
        # print(f"edge_weights.shape={edge_weights.shape}")
        # print(f"edge_weights.min={torch.min(edge_weights).detach().cpu().numpy()}")
        # print(f"edge_weights.max={torch.max(edge_weights).detach().cpu().numpy()}")

        #### Calculate node_degree for graph laplacian normalization
        node_degree = torch.zeros(
            (batch_size, self.n_graphs, 1, num_nodes, self.buffer_size),
            device=self.device
        )
        node_degree = self.sparse_sum_by_group(
            edge_weights, self.graph_node_edges[:, 0], 
            self.graph_node_edges_type, node_degree
        )
        # print(f"node_degree.shape={node_degree.shape}")
        # print(f"node_degree.min={torch.min(node_degree).detach().cpu().numpy()}")
        # print(f"node_degree.max={torch.max(node_degree).detach().cpu().numpy()}")

        # Normalized edge weights using square root inverse
        node_degree_sqrtinv = 1.0 / torch.sqrt(node_degree)
        edge_weights = (node_degree_sqrtinv[:, :, :, nodeI]) * edge_weights * (node_degree_sqrtinv[:, :, :, nodeJ])
        
        # edge_weights.shape -> (n_edges, batch_size, n_graphs)

        return edge_weights, node_degree
        

    @torch.compiler.disable(recursive=False)
    def op_L_norm(self, graph_signals, edge_weights, node_degree):

        # with record_function("GLR:op_L_norm"): 
        batch_size, n_graphs, n_signals, num_nodes = graph_signals.shape
        assert ((n_graphs == self.n_graphs) or (n_graphs == 1)), "n_graphs != self.n_graphs"

        output_holder = torch.zeros(
            (batch_size, self.n_graphs, n_signals, num_nodes, self.buffer_size),
            device=self.device
        )

        nodeI = self.graph_node_edges[:, 0]
        nodeJ = self.graph_node_edges[:, 1]
        edges_type = self.graph_node_edges_type

        signal_on_edges = graph_signals[:, :, :, nodeJ] * edge_weights
        output = self.sparse_sum_by_group(
            signal_on_edges, nodeI, 
            edges_type, output_holder
        )
        # normalized graph, so all degree are 1
        output = graph_signals - output

        return output


    def signal2D_to_graph_signals(self, patchs):
        # with record_function("GLR:signal2D_to_graph_signals"): 
        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
        assert (h_size == self.input_height), "h_size != self.input_height"
        assert (w_size == self.input_width), "w_size != self.input_width"
        
        graph_signals = patchs.view((batch_size, n_graphs, c_size, h_size * w_size))
        graph_signals = graph_signals[:, :, :, self.graph_node_order]
        # graph_signals.shape -> (batch_size, n_graphs, channel_size, n_nodes)

        return graph_signals

    def graph_signals_to_signal2D(self, graph_signals):
        # with record_function("GLR:graph_signals_to_signal2D"): 
        # graph_signals.shape -> (n_nodes, batch_size, n_graphs, channel_size)
        batch_size, n_graphs, c_size, n_nodes = graph_signals.shape

        patchs_size = (batch_size, n_graphs, c_size, self.input_height, self.input_width)
        patchs = graph_signals[:, :, :, self.graph_node_order_inverse].view(patchs_size)
            # patchs.shape -> (batch_size, n_graphs, channel_size, input_height, input_width)
        return patchs
    
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
        temp_patch = patchs.view(batch_size*n_graphs, c_size, h_size, w_size)
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
        patchs = self.stats_conv(patchs)
        # L
        graph_signals = self.signal2D_to_graph_signals(patchs)
        output_graph_signals = self.op_L_norm(graph_signals, edge_weights, node_degree)
        output_patchs = self.graph_signals_to_signal2D(output_graph_signals)
        # F^T
        output_patchs = self.stats_conv_transpose(output_patchs)
        return output_patchs



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
                out_channels=n_features_out//4, 
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                padding_mode="zeros",
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_features_out//4, 
                out_channels=n_features_out//4, 
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                padding_mode="zeros",
                bias=False
            ),
            nn.PixelUnshuffle(2)
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

        self.combination_weight = Parameter(
            torch.ones((self.n_graphs), device=self.device, dtype=torch.float32)/self.n_graphs,
            requires_grad=True,
        )

        self.patchs_embeding = nn.Conv2d(
            Extractor_modules_conf[0]["ExtractorConf"]["n_channels_in"],
            Extractor_modules_conf[0]["ExtractorConf"]["n_features_in"], 
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ).to(self.device)

        self.list_mixtureGLR = nn.ModuleList([])
        for level in range(self.n_levels): 
            glr_module = GLR(**GLR_modules_conf[level]["GLRConf"]) 
            # glr_module.compile()
            self.list_mixtureGLR.append(glr_module)

        self.list_Extractor = nn.ModuleList([])
        for level in range(self.n_levels-1):
            extractor_module = Extractor(**Extractor_modules_conf[level]["ExtractorConf"]) 
            # extractor_module.compile()
            self.list_Extractor.append(extractor_module)

    
    @torch.compiler.disable(recursive=False)
    def graph_frame_recalibrate(self, input_height, input_width):
        with torch.no_grad():
            dummy_features_patchs = self.patchs_embeding(
                torch.zeros(size=(1, self.nchannels_abtract, input_height, input_width), device=self.device, dtype=torch.float32)
            )
            # print(f"dummy_features_patchs.shape={dummy_features_patchs.shape}")
            self.list_mixtureGLR[0].graph_frame_recalibrate(input_height, input_width)

            for level in range(self.n_levels-1):
                dummy_features_patchs = self.list_Extractor[level](dummy_features_patchs)
                _, _, h, w = dummy_features_patchs.shape
                self.list_mixtureGLR[level+1].graph_frame_recalibrate(h, w)
        

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

        features_patchs = self.patchs_embeding(patchs)
        glr_module = self.list_mixtureGLR[0]

        # print(f"features_patchs.shape={features_patchs.shape}")
        bz, nfts, h, w = features_patchs.shape
        gW, gD = glr_module.extract_edge_weights(
            features_patchs.view((bz, glr_module.n_graphs, glr_module.n_node_fts, h, w))
        )

        # print(f"gW.shape={gW.shape} gD.shape={gD.shape}")
        list_graph_weights = [(gW, gD)]
        list_features_patchs = [features_patchs]

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
            list_features_patchs.append(features_patchs)

        
        # print("#"*80)

        output = patchs[:, None, :, :, :]
        system_residual = output -  self.apply_multi_scale_lightweight_transformer(output, list_graph_weights)
        update = system_residual

        for iter in range(self.n_cgd_iters):
            A_mul_update = self.apply_multi_scale_lightweight_transformer(update, list_graph_weights)
            output = output + self.alphaCGD[iter, None, :, None, None, None] * update
            system_residual = system_residual - self.alphaCGD[iter, None, :, None, None, None] * A_mul_update
            update = system_residual + self.betaCGD[iter, None, :, None, None, None] * update

        score = self.combination_weight
        output = torch.einsum(
            "bgchw, g -> bchw", output, score
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

        self.images_domain_to_abtract_domain = nn.Sequential(
            nn.Conv2d(self.nchannels_images, self.nchannels_abtract, kernel_size=1, bias=False),
            nn.Conv2d(self.nchannels_abtract, self.nchannels_abtract, kernel_size=3, stride=1, padding=1, groups=self.nchannels_abtract, bias=False),
        ).to(self.device)

        self.abtract_domain_to_images_domain = nn.Sequential(
            nn.Conv2d(self.nchannels_abtract, self.nchannels_abtract, kernel_size=3, stride=1, padding=1, groups=self.nchannels_abtract, bias=False),
            nn.Conv2d(self.nchannels_abtract, self.nchannels_images, kernel_size=1, bias=False),
        ).to(self.device)

        self.graph_frame_recalibrate(img_height, img_width)


    @torch.compiler.disable(recursive=False)
    def graph_frame_recalibrate(self, img_height, img_width):
        for block_i in range(self.n_blocks):
            block = self.light_weight_transformer_blocks[block_i]
            block.graph_frame_recalibrate(img_height, img_width)


    def forward(self, input_patchs):
        output = self.images_domain_to_abtract_domain(input_patchs)
        for block_i in range(0, self.n_blocks):
            block = self.light_weight_transformer_blocks[block_i]
            output_temp = block(output)

            p = self.cumulative_result_weight[block_i]
            output = p * output_temp + (1-p) * output


        output = self.abtract_domain_to_images_domain(output)
        return output
