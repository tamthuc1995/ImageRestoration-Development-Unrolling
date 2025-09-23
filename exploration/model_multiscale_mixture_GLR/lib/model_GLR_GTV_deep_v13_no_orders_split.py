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
            n_node_fts, n_graphs, M_diag_init=0.4
        ):
        super(GLRFast, self).__init__()

        # CONNECTION_FLAGS_5x5_small = np.array([
        #     0,0,1,0,0,
        #     0,1,1,1,0,
        #     1,1,0,1,1,
        #     0,1,1,1,0,
        #     0,0,1,0,0,
        # ]).reshape((5,5))
        CONNECTION_FLAGS_3x3_small = np.array([
            0,1,0,
            1,0,1,
            0,1,0,
        ]).reshape((3,3))

        self.n_channels        = n_node_fts * n_graphs
        self.n_node_fts        = n_node_fts
        self.n_graphs          = n_graphs

        connection_window = CONNECTION_FLAGS_3x3_small
        self.connection_window = connection_window
        self.n_edges           = (connection_window == 1).sum()
        self.buffer_size       = connection_window.sum()

        # edges type from connection_window
        window_size = connection_window.shape[0]
        connection_window = connection_window.reshape((-1))
        m = np.arange(window_size)-window_size//2
        edge_delta = np.array(
            list(itertools.product(m, m)),
            dtype=np.int32
        )
        edge_delta = edge_delta[connection_window == 1]
        pad_dim_hw = np.abs(edge_delta.min(axis=0))

        self.edge_delta = torch.tensor(edge_delta, dtype=torch.int32)
        self.pad_dim_hw = torch.tensor(pad_dim_hw, dtype=torch.int32)


        ### Trainable parameters
        # features on nodes #self.n_node_fts
        self.multiM = Parameter(
            torch.ones((self.n_graphs, self.n_node_fts), dtype=torch.float32)*M_diag_init,
            requires_grad=True,
        )


    def get_neighbors_pixels(self, img_features):
        _, _, H, W = img_features.shape
        padH, padW = self.pad_dim_hw
        img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
        neighbors_pixels_features = torch.stack([
            img_features_frame[:, :, padH + self.edge_delta[0, 0]:padH + self.edge_delta[0, 0] + H, padW + self.edge_delta[0, 1]:padW + self.edge_delta[0, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[1, 0]:padH + self.edge_delta[1, 0] + H, padW + self.edge_delta[1, 1]:padW + self.edge_delta[1, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[2, 0]:padH + self.edge_delta[2, 0] + H, padW + self.edge_delta[2, 1]:padW + self.edge_delta[2, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[3, 0]:padH + self.edge_delta[3, 0] + H, padW + self.edge_delta[3, 1]:padW + self.edge_delta[3, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[4, 0]:padH + self.edge_delta[4, 0] + H, padW + self.edge_delta[4, 1]:padW + self.edge_delta[4, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[5, 0]:padH + self.edge_delta[5, 0] + H, padW + self.edge_delta[5, 1]:padW + self.edge_delta[5, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[6, 0]:padH + self.edge_delta[6, 0] + H, padW + self.edge_delta[6, 1]:padW + self.edge_delta[6, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[7, 0]:padH + self.edge_delta[7, 0] + H, padW + self.edge_delta[7, 1]:padW + self.edge_delta[7, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[8, 0]:padH + self.edge_delta[8, 0] + H, padW + self.edge_delta[8, 1]:padW + self.edge_delta[8, 1] + W]
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
        output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)

        return output_patchs
    



class GTVFast(nn.Module):
    def __init__(self, 
            n_node_fts, n_graphs, M_diag_init=0.4
        ):
        super(GTVFast, self).__init__()


        # CONNECTION_FLAGS_5x5_small = np.array([
        #     0,0,1,0,0,
        #     0,1,1,1,0,
        #     1,1,0,1,1,
        #     0,1,1,1,0,
        #     0,0,1,0,0,
        # ]).reshape((5,5))

        CONNECTION_FLAGS_3x3_small = np.array([
            0,1,0,
            1,0,1,
            0,1,0,
        ]).reshape((3,3))

        self.n_channels        = n_node_fts * n_graphs
        self.n_node_fts        = n_node_fts
        self.n_graphs          = n_graphs

        connection_window = CONNECTION_FLAGS_3x3_small
        self.connection_window = connection_window
        self.n_edges           = (connection_window == 1).sum()
        self.buffer_size       = connection_window.sum()

        # edges type from connection_window
        window_size = connection_window.shape[0]
        connection_window = connection_window.reshape((-1))
        m = np.arange(window_size)-window_size//2
        edge_delta = np.array(
            list(itertools.product(m, m)),
            dtype=np.int32
        )
        edge_delta = edge_delta[connection_window == 1]
        pad_dim_hw = np.abs(edge_delta.min(axis=0))

        self.edge_delta = torch.tensor(edge_delta, dtype=torch.int32)
        self.pad_dim_hw = torch.tensor(pad_dim_hw, dtype=torch.int32)
        # print(f"edge_delta={self.edge_delta}, pad_dim_hw={self.pad_dim_hw}")

        ### Trainable parameters
        # features on nodes #self.n_node_fts
        self.multiM = Parameter(
            torch.ones((self.n_graphs, self.n_node_fts), dtype=torch.float32)*M_diag_init,
            requires_grad=True,
        )


    def get_neighbors_pixels(self, img_features):
        _, _, H, W = img_features.shape
        padH, padW = self.pad_dim_hw
        img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
        neighbors_pixels_features = torch.stack([
            img_features_frame[:, :, padH + self.edge_delta[0, 0]:padH + self.edge_delta[0, 0] + H, padW + self.edge_delta[0, 1]:padW + self.edge_delta[0, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[1, 0]:padH + self.edge_delta[1, 0] + H, padW + self.edge_delta[1, 1]:padW + self.edge_delta[1, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[2, 0]:padH + self.edge_delta[2, 0] + H, padW + self.edge_delta[2, 1]:padW + self.edge_delta[2, 1] + W],
            img_features_frame[:, :, padH + self.edge_delta[3, 0]:padH + self.edge_delta[3, 0] + H, padW + self.edge_delta[3, 1]:padW + self.edge_delta[3, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[4, 0]:padH + self.edge_delta[4, 0] + H, padW + self.edge_delta[4, 1]:padW + self.edge_delta[4, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[5, 0]:padH + self.edge_delta[5, 0] + H, padW + self.edge_delta[5, 1]:padW + self.edge_delta[5, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[6, 0]:padH + self.edge_delta[6, 0] + H, padW + self.edge_delta[6, 1]:padW + self.edge_delta[6, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[7, 0]:padH + self.edge_delta[7, 0] + H, padW + self.edge_delta[7, 1]:padW + self.edge_delta[7, 1] + W],
            # img_features_frame[:, :, padH + self.edge_delta[8, 0]:padH + self.edge_delta[8, 0] + H, padW + self.edge_delta[8, 1]:padW + self.edge_delta[8, 1] + W]
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

        # print(f"img_features={img_features.shape}, img_features_neighbors={img_features_neighbors.shape}, ")
        features_similarity = (img_features[:, :, None, :, :] * img_features_neighbors)
        features_similarity = features_similarity.view(
            batch_size, self.n_graphs, self.n_node_fts, self.n_edges, h_size, w_size
        ).sum(axis=2)

        edge_weights_norm = nn.functional.softmax(features_similarity, dim=2) 
        node_degree = edge_weights_norm.sum(axis=2)

        return edge_weights_norm, node_degree


    def op_C(self, img_signals, edge_weights, node_degree):


        batch_size, n_graphs, n_channels, h_size, w_size = img_signals.shape
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

        # edge00
        i = 0
        fromh, toh, fromw, tow = padH + self.edge_delta[i, 0], padH + self.edge_delta[i, 0] + H, padW + self.edge_delta[i, 1], padW + self.edge_delta[i, 1] + W
        output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]

        # edge01
        i = 1
        fromh, toh, fromw, tow = padH + self.edge_delta[i, 0], padH + self.edge_delta[i, 0] + H, padW + self.edge_delta[i, 1], padW + self.edge_delta[i, 1] + W
        output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]

        # edge02
        i = 2
        fromh, toh, fromw, tow = padH + self.edge_delta[i, 0], padH + self.edge_delta[i, 0] + H, padW + self.edge_delta[i, 1], padW + self.edge_delta[i, 1] + W
        output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]

        # edge03
        i = 3
        fromh, toh, fromw, tow = padH + self.edge_delta[i, 0], padH + self.edge_delta[i, 0] + H, padW + self.edge_delta[i, 1], padW + self.edge_delta[i, 1] + W
        output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]


        # i=0
        # for shift_h, shift_w in self.edge_delta:
        #     fromh = padH + shift_h
        #     toh = padH + shift_h + H
        #     fromw = padW + shift_w
        #     tow = padW + shift_w + W
            
        #     output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]
        #     i+=1

        output = output[:, :, :, padH:-padH, padW:-padW]

        return output

    def forward(self, patchs, edge_weights, node_degree):
        # C^T C
        edges_signals = self.op_C(patchs, edge_weights, node_degree)
        output_patchs = self.op_C_transpose(edges_signals, edge_weights, node_degree)

        return output_patchs

##########################################################################
class CustomLayerNorm(nn.Module):
    def __init__(self, nchannels, nsubnets):
        super(CustomLayerNorm, self).__init__()
        self.nsubnets = nsubnets
        self.nchannels = nchannels
        self.weighted_transform = nn.Conv2d(nchannels, nchannels, kernel_size=1, stride=1, groups=nchannels, bias=False)

    def forward(self, x):
        bz, nchannels, h, w = x.shape
        x = x.reshape(bz, self.nsubnets, self.nchannels//self.nsubnets, h, w)
        sigma = x.var(dim=2, keepdim=True, correction=1)
        x = x / torch.sqrt(sigma+1e-5)
        x = x.reshape(bz, self.nchannels, h, w)
        # bz, 1, h, w = sigma.shape
        return self.weighted_transform(x)

##########################################################################
class LocalGatedLinearBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nsubnets):
        super(LocalGatedLinearBlock, self).__init__()

        self.channels_linear_op       = nn.Conv2d(dim, hidden_dim*2, kernel_size=1, bias=False, groups=nsubnets)
        self.channels_local_linear_op = nn.Conv2d(
            hidden_dim*2, hidden_dim*2, 
            kernel_size=3, stride=1, 
            padding=1, padding_mode="replicate",
            groups=hidden_dim*2, 
            bias=False
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False, groups=nsubnets)

    def forward(self, x):
        x = self.channels_linear_op(x)
        mask, x = self.channels_local_linear_op(x).chunk(2, dim=1)
        x = nn.functional.sigmoid(mask) * mask * x
        x = self.project_out(x)
        return x

##########################################################################
class LocalNonLinearBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nsubnets):
        super(LocalNonLinearBlock, self).__init__()

        # Linear Layer
        self.norm = CustomLayerNorm(dim, nsubnets)
        self.local_linear = LocalGatedLinearBlock(dim, hidden_dim, nsubnets)
        self.skip_weight= Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            requires_grad=True
        )
    def forward(self, x):
        x = self.skip_weight[0] * x + self.skip_weight[1] * self.local_linear(self.norm(x))
        return x



class GTVGLR(nn.Module):
    def __init__(self, 
            n_graphs, n_node_fts,
            alpha_init, beta_init,
            muy_init, ro_init, gamma_init,
        ):
        super(GTVGLR, self).__init__()
        # MixtureGTVGLR( 
        #     n_graphs, n_node_fts,
        #     connection_window,
        #     n_cgd_iters, alpha_init, beta_init,
        #     muy_init, ro_init, gamma_init,
        #     device
        # )

        self.n_graphs     = n_graphs
        self.n_node_fts   = n_node_fts
        self.n_channels   = n_graphs * n_node_fts
        self.n_cgd_iters  = 3

        self.alphaCGD =  Parameter(
            torch.ones((self.n_cgd_iters, n_graphs), dtype=torch.float32) * alpha_init,
            requires_grad=True
        )

        self.betaCGD =  Parameter(
            torch.ones((self.n_cgd_iters, n_graphs), dtype=torch.float32) * beta_init,
            requires_grad=True
        )

        self.patchs_features_extractionGLR = nn.Sequential(
            LocalNonLinearBlock(
                dim=self.n_channels//2, 
                hidden_dim=int(self.n_channels//2 * 8/3), 
                nsubnets=1
            ),
            LocalNonLinearBlock(
                dim=self.n_channels//2, 
                hidden_dim=int(self.n_channels//2 * 8/3), 
                nsubnets=1
            ),
            LocalNonLinearBlock(
                dim=self.n_channels//2, 
                hidden_dim=int(self.n_channels//2 * 8/3), 
                nsubnets=1
            ),
            nn.Conv2d(
                in_channels=self.n_channels//2, 
                out_channels=self.n_channels, 
                kernel_size=1,
                stride=1,
                padding=0, # padding_mode="replicate",
                groups=1,
                bias=False
            )
        )
        self.patchs_features_extractionGTV = nn.Sequential(
            LocalNonLinearBlock(
                dim=self.n_channels//2, 
                hidden_dim=int(self.n_channels//2 * 8/3), 
                nsubnets=1
            ),
            LocalNonLinearBlock(
                dim=self.n_channels//2, 
                hidden_dim=int(self.n_channels//2 * 8/3), 
                nsubnets=1
            ),
            LocalNonLinearBlock(
                dim=self.n_channels//2, 
                hidden_dim=int(self.n_channels//2 * 8/3), 
                nsubnets=1
            ),
            nn.Conv2d(
                in_channels=self.n_channels//2, 
                out_channels=self.n_channels, 
                kernel_size=1,
                stride=1,
                padding=0, # padding_mode="replicate",
                groups=1,
                bias=False
            )
        )


        self.ro00 = Parameter(
            torch.ones((n_graphs), dtype=torch.float32) * torch.log(ro_init[0]),
            requires_grad=True,
        )
        self.gamma00 = Parameter(
            torch.ones((n_graphs), dtype=torch.float32) * torch.log(gamma_init[0]),
            requires_grad=True,
        )
        self.GTVmodule00 = GTVFast(
            n_node_fts=self.n_node_fts,
            n_graphs=self.n_graphs,
            M_diag_init=1.0
        )

        self.muys00 = Parameter(
            torch.ones((n_graphs), dtype=torch.float32) * torch.log(muy_init[0]),
            requires_grad=True,
        )
        self.GLRmodule00 = GLRFast(
            n_node_fts=self.n_node_fts,
            n_graphs=self.n_graphs,
            M_diag_init=1.0
        )



    def apply_lightweight_transformer(self, patchs, graph_weightGTV, graph_weightGLR):

        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
        patchs = patchs.contiguous()
        graph_weights, graph_degree = graph_weightGLR[0]

        Lpatchs = self.GLRmodule00(patchs, graph_weights, graph_degree)
        Lpatchs = torch.einsum(
            "bHchw, H -> bHchw", Lpatchs, torch.exp(self.muys00)
        )

        graph_weights, graph_degree = graph_weightGTV[0]
        CtCpatchs = self.GTVmodule00(patchs, graph_weights, graph_degree)
        CtCpatchs = torch.einsum(
            "bHchw, H -> bHchw", CtCpatchs, torch.exp(self.ro00)
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
        batch_size, c_size, h_size, w_size = patchs.shape

        #####
        ## Graph low pass filter 00
        
        features_patchs_GTV00, features_patchs_GLR00 = patchs.chunk(2, dim=1)
        features_patchs_GLR = self.patchs_features_extractionGLR(features_patchs_GLR00)
        features_patchs_GTV = self.patchs_features_extractionGTV(features_patchs_GTV00)
        bz, nfts, h, w = features_patchs_GLR.shape
        
        graph_weightGLR00 = self.GLRmodule00.extract_edge_weights(
            features_patchs_GLR.view((bz, self.GLRmodule00.n_graphs, self.GLRmodule00.n_node_fts, h, w))
        )
        graph_weightGTV00 = self.GTVmodule00.extract_edge_weights(
            features_patchs_GTV.view((bz, self.GTVmodule00.n_graphs, self.GTVmodule00.n_node_fts, h, w))
        )

        y_tilde = patchs.view((bz, self.n_graphs, self.n_node_fts, h, w))
        #######################################################################################################################
        epsilonA00 = self.GTVmodule00.op_C(y_tilde, graph_weightGTV00[0], graph_weightGTV00[1])
        # Inital bias is zero
        left_hand_sizeA = y_tilde
        left_hand_sizeA = left_hand_sizeA + self.GTVmodule00.op_C_transpose(epsilonA00, graph_weightGTV00[0], graph_weightGTV00[1]) * torch.exp(self.ro00[None, :, None, None, None] )
        ############################################################
        output00          = left_hand_sizeA
        system_residual00 = left_hand_sizeA -  self.apply_lightweight_transformer(output00, [graph_weightGTV00], [graph_weightGLR00])
        output01          = output00 + self.alphaCGD[0, None, :, None, None, None] * system_residual00


        #######################################################################################################################
        tempB00 = self.GTVmodule00.op_C(output01, graph_weightGTV00[0], graph_weightGTV00[1])
        epsilonB00 = self.soft_threshold(
            tempB00,
            torch.exp(self.gamma00)
        )
        biasB00  = (tempB00 - epsilonB00)

        left_hand_sizeB = y_tilde
        left_hand_sizeB = left_hand_sizeB + self.GTVmodule00.op_C_transpose(epsilonB00 - biasB00, graph_weightGTV00[0], graph_weightGTV00[1]) * torch.exp(self.ro00[None, :, None, None, None])
        # ############################################################

        system_residual01 = left_hand_sizeB -  self.apply_lightweight_transformer(output01, [graph_weightGTV00], [graph_weightGLR00])
        update01 = system_residual01 + self.betaCGD[1, None, :, None, None, None] * system_residual00
        output02 = output01 + self.alphaCGD[1, None, :, None, None, None] * update01

        system_residual03 = left_hand_sizeB -  self.apply_lightweight_transformer(output02, [graph_weightGTV00], [graph_weightGLR00])
        update03 = system_residual03 + self.betaCGD[2, None, :, None, None, None] * update01
        output03 = output02 + self.alphaCGD[2, None, :, None, None, None] * update03


        output_final = output03.view((batch_size, c_size, h_size, w_size))

        return output_final





class OneGraphFilter(nn.Module):
    def __init__(self, 
        n_channels_in=3, 
        n_channels_hidden=96, 
        n_channels_out=3, 
    ):
        

        super(OneGraphFilter, self).__init__()

        self.ngraphs = 1
        self.n_channels_in = n_channels_in
        self.n_channels_hidden = n_channels_hidden
        self.localfilter = GTVGLR( 
            n_graphs=1,
            n_node_fts=n_channels_hidden,
            alpha_init=0.5,
            beta_init=0.1,
            muy_init=torch.tensor([[0.001]]),
            ro_init=torch.tensor([[0.000001]]),
            gamma_init=torch.tensor([[0.000001]]),
        )
        self.linear_combination = nn.Conv2d(
            in_channels=n_channels_in, 
            out_channels=n_channels_out, 
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )

    def forward(self, img):
        bz, _, h, w = img.shape

        img = img[:, None, :, :, :]
        img = img.repeat(1, self.n_channels_hidden // self.n_channels_in, 1, 1, 1).reshape(bz, self.n_channels_hidden, h, w)
        output = self.localfilter(img)
        output = self.linear_combination(output[:, :3, :, :])

        return output