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


        # kernel01 = torch.tensor([
        #     [0.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0],
        #     [0.0, 0.0, 0.0],
        # ])
        # kernel = []
        # for r in range(self.n_channels):
        #     kernel.append(kernel01[np.newaxis, np.newaxis, :, :])

        # kernel = torch.concat(kernel, axis=0)
        # self.stats_kernel_p01 = Parameter(
        #     torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 1.0,
        #     requires_grad=True,
        # )
        # self.stats_kernel01 = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

        # kernel02a = torch.tensor([
        #     [0.0, 0.0, 0.0],
        #     [0.0,-1.0, 1.0],
        #     [0.0, 0.0, 0.0],
        # ])
        # kernel = []
        # for r in range(self.n_channels):
        #     kernel.append(kernel02a[np.newaxis, np.newaxis, :, :])

        # kernel = torch.concat(kernel, axis=0)
        # self.stats_kernel_p02a = Parameter(
        #     torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 0.5,
        #     requires_grad=True,
        # )
        # self.stats_kernel02a = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

        # kernel02b = torch.tensor([
        #     [0.0, 0.0, 0.0],
        #     [0.0,-1.0, 0.0],
        #     [0.0, 1.0, 0.0],
        # ])
        # kernel = []
        # for r in range(self.n_channels):
        #     kernel.append(kernel02b[np.newaxis, np.newaxis, :, :])

        # kernel = torch.concat(kernel, axis=0)
        # self.stats_kernel_p02b = Parameter(
        #     torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 0.5,
        #     requires_grad=True,
        # )
        # self.stats_kernel02b = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

        # kernel03 = torch.tensor([
        #     [0.0, -1.0, 0.0],
        #     [-1.0, 4.0, -1.0],
        #     [0.0, -1.0, 0.0],
        # ])
        # kernel = []
        # for r in range(self.n_channels):
        #     kernel.append(kernel03[np.newaxis, np.newaxis, :, :])

        # kernel = torch.concat(kernel, axis=0)
        # self.stats_kernel_p03 = Parameter(
        #     torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 0.5,
        #     requires_grad=True,
        # )
        # self.stats_kernel03 = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

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
    
    # def stats_conv(self, patchs):
    #     stats_kernel = (
    #         self.stats_kernel_p01 * self.stats_kernel01
    #         + self.stats_kernel_p02a * self.stats_kernel02a
    #         + self.stats_kernel_p02b * self.stats_kernel02b
    #         + self.stats_kernel_p03 * self.stats_kernel03
    #     )
    #     batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
    #     temp_patch = patchs.view(batch_size, n_graphs*c_size, h_size, w_size)
    #     temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'replicate')
    #     temp_out_patch = nn.functional.conv2d(
    #         temp_patch,
    #         weight=stats_kernel,
    #         stride=1,
    #         padding=0,
    #         groups=self.n_channels,
    #     )
    #     out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
    #     return out_patch

    # def stats_conv_transpose(self, patchs):

    #     stats_kernel = (
    #         self.stats_kernel_p01 * self.stats_kernel01
    #         + self.stats_kernel_p02a * self.stats_kernel02a
    #         + self.stats_kernel_p02b * self.stats_kernel02b
    #         + self.stats_kernel_p03 * self.stats_kernel03
    #     )
    #     batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
    #     temp_patch = patchs.reshape(batch_size, n_graphs*c_size, h_size, w_size)
    #     temp_out_patch = nn.functional.conv_transpose2d(
    #         temp_patch,
    #         weight=stats_kernel,
    #         stride=1,
    #         padding=1,
    #         groups=self.n_channels,
    #     )
    #     out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
    #     return out_patch


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
        # patchs = self.stats_conv(patchs)
        output_patchs = self.op_L_norm(patchs, edge_weights, node_degree)
        # output_patchs = self.stats_conv_transpose(output_patchs)

        return output_patchs
    



# class GTVFast(nn.Module):
#     def __init__(self, 
#             n_node_fts, n_graphs, M_diag_init=0.4
#         ):
#         super(GTVFast, self).__init__()


#         # CONNECTION_FLAGS_5x5_small = np.array([
#         #     0,0,1,0,0,
#         #     0,1,1,1,0,
#         #     1,1,0,1,1,
#         #     0,1,1,1,0,
#         #     0,0,1,0,0,
#         # ]).reshape((5,5))

#         CONNECTION_FLAGS_3x3_small = np.array([
#             0,1,0,
#             1,0,1,
#             0,1,0,
#         ]).reshape((3,3))

#         self.n_channels        = n_node_fts * n_graphs
#         self.n_node_fts        = n_node_fts
#         self.n_graphs          = n_graphs

#         connection_window = CONNECTION_FLAGS_3x3_small
#         self.connection_window = connection_window
#         self.n_edges           = (connection_window == 1).sum()
#         self.buffer_size       = connection_window.sum()

#         # edges type from connection_window
#         window_size = connection_window.shape[0]
#         connection_window = connection_window.reshape((-1))
#         m = np.arange(window_size)-window_size//2
#         edge_delta = np.array(
#             list(itertools.product(m, m)),
#             dtype=np.int32
#         )
#         edge_delta = edge_delta[connection_window == 1]
#         pad_dim_hw = np.abs(edge_delta.min(axis=0))

#         self.edge_delta = torch.tensor(edge_delta, dtype=torch.int32)
#         self.pad_dim_hw = torch.tensor(pad_dim_hw, dtype=torch.int32)
#         # print(f"edge_delta={self.edge_delta}, pad_dim_hw={self.pad_dim_hw}")

#         kernel01 = torch.tensor([
#             [0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0],
#         ])
#         kernel = []
#         for r in range(self.n_channels):
#             kernel.append(kernel01[np.newaxis, np.newaxis, :, :])

#         kernel = torch.concat(kernel, axis=0)
#         self.stats_kernel_p01 = Parameter(
#             torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 1.0,
#             requires_grad=True,
#         )
#         self.stats_kernel01 = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

#         kernel02a = torch.tensor([
#             [0.0, 0.0, 0.0],
#             [0.0,-1.0, 1.0],
#             [0.0, 0.0, 0.0],
#         ])
#         kernel = []
#         for r in range(self.n_channels):
#             kernel.append(kernel02a[np.newaxis, np.newaxis, :, :])

#         kernel = torch.concat(kernel, axis=0)
#         self.stats_kernel_p02a = Parameter(
#             torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 0.5,
#             requires_grad=True,
#         )
#         self.stats_kernel02a = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

#         kernel02b = torch.tensor([
#             [0.0, 0.0, 0.0],
#             [0.0,-1.0, 0.0],
#             [0.0, 1.0, 0.0],
#         ])
#         kernel = []
#         for r in range(self.n_channels):
#             kernel.append(kernel02b[np.newaxis, np.newaxis, :, :])

#         kernel = torch.concat(kernel, axis=0)
#         self.stats_kernel_p02b = Parameter(
#             torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 0.5,
#             requires_grad=True,
#         )
#         self.stats_kernel02b = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

#         kernel03 = torch.tensor([
#             [0.0, -1.0, 0.0],
#             [-1.0, 4.0, -1.0],
#             [0.0, -1.0, 0.0],
#         ])
#         kernel = []
#         for r in range(self.n_channels):
#             kernel.append(kernel03[np.newaxis, np.newaxis, :, :])

#         kernel = torch.concat(kernel, axis=0)
#         self.stats_kernel_p03 = Parameter(
#             torch.ones((self.n_channels, 1, 1, 1), dtype=torch.float32) * 0.5,
#             requires_grad=True,
#         )
#         self.stats_kernel03 = torch.ones((self.n_channels, 1, 3, 3), dtype=torch.float32) * kernel

#         ### Trainable parameters
#         # features on nodes #self.n_node_fts
#         self.multiM = Parameter(
#             torch.ones((self.n_graphs, self.n_node_fts), dtype=torch.float32)*M_diag_init,
#             requires_grad=True,
#         )


#     def get_neighbors_pixels(self, img_features):
#         _, _, H, W = img_features.shape
#         padH, padW = self.pad_dim_hw
#         img_features_frame = nn.functional.pad(img_features, (padW, padW, padH, padH), "replicate")
#         neighbors_pixels_features = torch.stack([
#             img_features_frame[:, :, padH + self.edge_delta[0, 0]:padH + self.edge_delta[0, 0] + H, padW + self.edge_delta[0, 1]:padW + self.edge_delta[0, 1] + W],
#             img_features_frame[:, :, padH + self.edge_delta[1, 0]:padH + self.edge_delta[1, 0] + H, padW + self.edge_delta[1, 1]:padW + self.edge_delta[1, 1] + W],
#             img_features_frame[:, :, padH + self.edge_delta[2, 0]:padH + self.edge_delta[2, 0] + H, padW + self.edge_delta[2, 1]:padW + self.edge_delta[2, 1] + W],
#             img_features_frame[:, :, padH + self.edge_delta[3, 0]:padH + self.edge_delta[3, 0] + H, padW + self.edge_delta[3, 1]:padW + self.edge_delta[3, 1] + W],
#         ], axis=-3)
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

#         # print(f"img_features={img_features.shape}, img_features_neighbors={img_features_neighbors.shape}, ")
#         features_similarity = (img_features[:, :, None, :, :] * img_features_neighbors)
#         features_similarity = features_similarity.view(
#             batch_size, self.n_graphs, self.n_node_fts, self.n_edges, h_size, w_size
#         ).sum(axis=2)

#         edge_weights_norm = nn.functional.softmax(features_similarity, dim=2) 
#         node_degree = edge_weights_norm.sum(axis=2)

#         return edge_weights_norm, node_degree


#     def stats_conv(self, patchs):
#         stats_kernel = (
#             self.stats_kernel_p01 * self.stats_kernel01
#             + self.stats_kernel_p02a * self.stats_kernel02a
#             + self.stats_kernel_p02b * self.stats_kernel02b
#             + self.stats_kernel_p03 * self.stats_kernel03
#         )
#         batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
#         temp_patch = patchs.view(batch_size, n_graphs*c_size, h_size, w_size)
#         temp_patch = nn.functional.pad(temp_patch, (1,1,1,1), 'replicate')
#         temp_out_patch = nn.functional.conv2d(
#             temp_patch,
#             weight=stats_kernel,
#             stride=1,
#             padding=0,
#             groups=self.n_channels,
#         )
#         out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
#         return out_patch

#     def stats_conv_transpose(self, patchs):

#         stats_kernel = (
#             self.stats_kernel_p01 * self.stats_kernel01
#             + self.stats_kernel_p02a * self.stats_kernel02a
#             + self.stats_kernel_p02b * self.stats_kernel02b
#             + self.stats_kernel_p03 * self.stats_kernel03
#         )
        
#         batch_size, n_graphs, c_size, h_size, w_size = patchs.shape
#         temp_patch = patchs.reshape(batch_size, n_graphs*c_size, h_size, w_size)
#         temp_out_patch = nn.functional.conv_transpose2d(
#             temp_patch,
#             weight=stats_kernel,
#             stride=1,
#             padding=1,
#             groups=self.n_channels,
#         )
#         out_patch = temp_out_patch.view(batch_size, n_graphs, c_size, h_size, w_size)
#         return out_patch


#     def op_C(self, img_signals, edge_weights, node_degree):


#         batch_size, n_graphs, n_channels, h_size, w_size = img_signals.shape

#         img_signals = self.stats_conv(img_signals)

#         img_features_neighbors = self.get_neighbors_pixels(
#             img_signals.view(batch_size, n_graphs*n_channels, h_size, w_size)
#         ).view(batch_size, n_graphs, n_channels, self.n_edges, h_size, w_size)
#         Cx1 = img_signals[:, :, :, None, :, :] * edge_weights[:, :, None, :, :, :]
#         Cx2 = img_features_neighbors * edge_weights[:, :, None, :, :, :]
        
#         output = Cx1 - Cx2

#         return output
    
#     def op_C_transpose(self, edge_signals, edge_weights, node_degree):

#         batch_size, n_graphs, n_channels, n_edges, H, W = edge_signals.shape
#         edge_signals = edge_signals * edge_weights[:, :, None, :, :, :]

#         output = edge_signals.sum(axis=3)

#         padH, padW = self.pad_dim_hw
#         output = nn.functional.pad(
#             output.view(batch_size, n_graphs*n_channels, H, W),
#             (padW, padW, padH, padH), "replicate"
#         ).view(batch_size, n_graphs, n_channels, H + 2*padH, W + 2*padW)

#         i=0
#         for shift_h, shift_w in self.edge_delta:
#             fromh = padH + shift_h
#             toh = padH + shift_h + H
#             fromw = padW + shift_w
#             tow = padW + shift_w + W
            
#             output[:, :, :, fromh:toh, fromw:tow] = output[:, :, :, fromh:toh, fromw:tow] - edge_signals[:, :, :, i, :, :]
#             i+=1

#         output = output[:, :, :, padH:-padH, padW:-padW]

#         output = self.stats_conv_transpose(output)
#         return output

#     def forward(self, patchs, edge_weights, node_degree):
#         # C^T C
#         edges_signals = self.op_C(patchs, edge_weights, node_degree)
#         output_patchs = self.op_C_transpose(edges_signals, edge_weights, node_degree)

#         return output_patchs


# class MixtureGTVGLR(nn.Module):
#     def __init__(self, 
#             n_graphs, n_node_fts,
#             alpha_init, beta_init,
#             muy_init, ro_init, gamma_init,
#         ):
#         super(MixtureGTVGLR, self).__init__()
#         # MixtureGTVGLR( 
#         #     n_graphs, n_node_fts,
#         #     connection_window,
#         #     n_cgd_iters, alpha_init, beta_init,
#         #     muy_init, ro_init, gamma_init,
#         #     device
#         # )

#         self.n_graphs     = n_graphs
#         self.n_node_fts   = n_node_fts
#         self.n_channels   = n_graphs * n_node_fts
#         self.n_cgd_iters  = 4

#         self.alphaCGD =  Parameter(
#             torch.ones((self.n_cgd_iters, n_graphs), dtype=torch.float32) * alpha_init,
#             requires_grad=True
#         )

#         self.betaCGD =  Parameter(
#             torch.ones((self.n_cgd_iters, n_graphs), dtype=torch.float32) * beta_init,
#             requires_grad=True
#         )

#         self.patchs_features_extraction = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.n_channels, 
#                 out_channels=self.n_channels, 
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 padding_mode="replicate",
#                 groups=n_graphs,
#                 bias=False
#             )
#         )

#         self.ro00 = Parameter(
#             torch.ones((n_graphs), dtype=torch.float32) * ro_init[0],
#             requires_grad=True,
#         )
#         self.gamma00 = Parameter(
#             torch.ones((n_graphs), dtype=torch.float32) * torch.log(gamma_init[0]),
#             requires_grad=True,
#         )
#         self.GTVmodule00 = GTVFast(
#             n_node_fts=self.n_node_fts,
#             n_graphs=self.n_graphs,
#             M_diag_init=1.0
#         )

#         self.muys00 = Parameter(
#             torch.ones((n_graphs), dtype=torch.float32) * muy_init[0],
#             requires_grad=True,
#         )
#         self.GLRmodule00 = GLRFast(
#             n_node_fts=self.n_node_fts,
#             n_graphs=self.n_graphs,
#             M_diag_init=1.0
#         )

#     def apply_lightweight_transformer(self, patchs, graph_weightGTV, graph_weightGLR):

#         batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
#         patchs = patchs.contiguous()
#         graph_weights, graph_degree = graph_weightGLR

#         Lpatchs = self.GLRmodule00(patchs, graph_weights, graph_degree)
#         Lpatchs = torch.einsum(
#             "bHchw, H -> bHchw", Lpatchs, self.muys00
#         )

#         graph_weights, graph_degree = graph_weightGTV
#         CtCpatchs = self.GTVmodule00(patchs, graph_weights, graph_degree)
#         CtCpatchs = torch.einsum(
#             "bHchw, H -> bHchw", CtCpatchs, self.ro00
#         )

#         output = patchs + Lpatchs + CtCpatchs

#         return output
    
#     def soft_threshold(self, delta, gamma):
#         # batch_size, n_graphs, n_channels, n_edges, H, W = delta.shape
#         # n_graphs = gamma.shape

#         gamma = gamma[None, :, None, None, None, None]
#         # print(f"Gamma.shape={gamma.shape}")

#         condA = (delta < -gamma) 
#         outputA = torch.where(
#             condA,
#             delta+gamma,
#             0.0
#         )
#         condB = (delta > gamma) 
#         outputB = torch.where(
#             condB,
#             delta-gamma,
#             0.0
#         )
#         output = outputA + outputB
#         return output


#     def forward(self, patchs):
#         batch_size, c_size, h_size, w_size = patchs.shape

#         #####
#         ## Graph low pass filter
#         features_patchs = self.patchs_features_extraction(patchs)
#         bz, nfts, h, w = features_patchs.shape

#         graph_weightGTV = self.GTVmodule00.extract_edge_weights(
#             features_patchs.view((bz, self.GTVmodule00.n_graphs, self.GTVmodule00.n_node_fts, h, w))
#         )
#         graph_weightGLR = self.GLRmodule00.extract_edge_weights(
#             features_patchs.view((bz, self.GLRmodule00.n_graphs, self.GLRmodule00.n_node_fts, h, w))
#         )

#         y_tilde = patchs.view((bz, self.n_graphs, self.n_node_fts, h, w))
#         ###########################################################
#         epsilonA = self.GTVmodule00.op_C(y_tilde, graph_weightGTV[0], graph_weightGTV[1])
#         # Inital bias is zero
#         left_hand_sizeA = self.GTVmodule00.op_C_transpose(epsilonA, graph_weightGTV[0], graph_weightGTV[1]) * self.ro00[None, :, None, None, None] + y_tilde
#         ############################################################
#         output00          = left_hand_sizeA
#         system_residual00 = left_hand_sizeA -  self.apply_lightweight_transformer(output00, graph_weightGTV, graph_weightGLR)
#         output01          = output00 + self.alphaCGD[0, None, :, None, None, None] * system_residual00

#         # system_residual01 = left_hand_sizeA -  self.apply_lightweight_transformer(output01, graph_weightGTV, graph_weightGLR)
#         # update01 = system_residual01 + self.betaCGD[1, None, :, None, None, None] * system_residual00
#         # output02 = output01 + self.alphaCGD[1, None, :, None, None, None] * update01

#         # system_residual03 = left_hand_sizeA -  self.apply_lightweight_transformer(output02, graph_weightGTV, graph_weightGLR)
#         # update03 = system_residual03 + self.betaCGD[2, None, :, None, None, None] * update01
#         # output03 = output02 + self.alphaCGD[2, None, :, None, None, None] * update03

#         temp = self.GTVmodule00.op_C(output01, graph_weightGTV[0], graph_weightGTV[1])
#         epsilonB = self.soft_threshold(
#             temp,
#             torch.exp(self.gamma00)
#         )
#         biasB  = (temp - epsilonB)
#         left_hand_sizeB = self.GTVmodule00.op_C_transpose(epsilonB - biasB, graph_weightGTV[0], graph_weightGTV[1]) * self.ro00[None, :, None, None, None] + y_tilde
#         # ############################################################

#         system_residual01 = left_hand_sizeB -  self.apply_lightweight_transformer(output01, graph_weightGTV, graph_weightGLR)
#         update01 = system_residual01# + self.betaCGD[1, None, :, None, None, None] * system_residual00
#         output02 = output01 + self.alphaCGD[1, None, :, None, None, None] * update01

#         system_residual03 = left_hand_sizeB -  self.apply_lightweight_transformer(output02, graph_weightGTV, graph_weightGLR)
#         update03 = system_residual03 + self.betaCGD[2, None, :, None, None, None] * update01
#         output03 = output02 + self.alphaCGD[2, None, :, None, None, None] * update03


#         # output03 = left_hand_sizeB
#         # system_residual03 = left_hand_sizeB -  self.apply_lightweight_transformer(output03, graph_weightGTV, graph_weightGLR)
#         # update03 = system_residual03
#         # output04 = output03 + self.alphaCGD[2, None, :, None, None, None] * update03

#         # system_residual04 = left_hand_sizeB -  self.apply_lightweight_transformer(output04, graph_weightGTV, graph_weightGLR)
#         # update04 = system_residual04 + self.betaCGD[3, None, :, None, None, None] * update03
#         # output05 = output04 + self.alphaCGD[3, None, :, None, None, None] * update04

#         # system_residual05 = left_hand_sizeB -  self.apply_lightweight_transformer(output05, graph_weightGTV, graph_weightGLR)
#         # update05 = system_residual05 + self.betaCGD[4, None, :, None, None, None] * update04
#         # output06 = output05 + self.alphaCGD[4, None, :, None, None, None] * update05
        
#         # system_residual06 = left_hand_sizeB -  self.apply_lightweight_transformer(output06, graph_weightGTV, graph_weightGLR)
#         # update06 = system_residual06 + self.betaCGD[5, None, :, None, None, None] * update05
#         # output07 = output06 + self.alphaCGD[5, None, :, None, None, None] * update06
        
#         output_final = output03.view((batch_size, c_size, h_size, w_size))

#         return output_final


class MixtureGLR(nn.Module):
    def __init__(self, 
            n_graphs, n_node_fts,
            alpha_init, beta_init,
            muy_init
        ):
        super(MixtureGLR, self).__init__()
        # MixtureGLR( 
        #     n_graphs, n_node_fts,
        #     connection_window,
        #     n_cgd_iters, alpha_init, beta_init,
        #     muy_init
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

        self.patchs_features_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_channels, 
                out_channels=self.n_channels, 
                kernel_size=1,
                stride=1,
                padding=1, # padding_mode="replicate",
                groups=1,
                bias=False
            )
        )

        self.muys00 = Parameter(
            torch.ones((n_graphs), dtype=torch.float32) * muy_init[0],
            requires_grad=True,
        )
        self.GLRmodule00 = GLRFast(
            n_node_fts=self.n_node_fts,
            n_graphs=self.n_graphs,
            M_diag_init=1.0
        )

    def apply_lightweight_transformer(self, patchs, graph_weightGLR):

        batch_size, n_graphs, c_size, h_size, w_size = patchs.shape 
        patchs = patchs.contiguous()
        graph_weights, graph_degree = graph_weightGLR

        Lpatchs = self.GLRmodule00(patchs, graph_weights, graph_degree)
        Lpatchs = torch.einsum(
            "bHchw, H -> bHchw", Lpatchs, self.muys00
        )

        output = patchs + Lpatchs

        return output
    
    def forward(self, patchs):
        batch_size, c_size, h_size, w_size = patchs.shape

        #####
        ## Graph low pass filter
        features_patchs = self.patchs_features_extraction(patchs)
        bz, nfts, h, w = features_patchs.shape

        graph_weightGLR = self.GLRmodule00.extract_edge_weights(
            features_patchs.view((bz, self.GLRmodule00.n_graphs, self.GLRmodule00.n_node_fts, h, w))
        )

        left_hand_sizeA = patchs.view((bz, self.n_graphs, self.n_node_fts, h, w))
        ############################################################
        output00          = left_hand_sizeA
        system_residual00 = left_hand_sizeA -  self.apply_lightweight_transformer(output00, graph_weightGLR)
        output01          = output00 + self.alphaCGD[0, None, :, None, None, None] * system_residual00

        system_residual02 = left_hand_sizeA -  self.apply_lightweight_transformer(output01, graph_weightGLR)
        update01 = system_residual02 + self.betaCGD[1, None, :, None, None, None] * system_residual00
        output02 = output01 + self.alphaCGD[1, None, :, None, None, None] * update01

        system_residual03 = left_hand_sizeA -  self.apply_lightweight_transformer(output02, graph_weightGLR)
        update03 = system_residual03 + self.betaCGD[2, None, :, None, None, None] * update01
        output03 = output02 + self.alphaCGD[2, None, :, None, None, None] * update03

        output_final = output03.view((batch_size, c_size, h_size, w_size))

        return output_final

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
    def __init__(self, dim, nsubnets):
        super(LocalNonLinearBlock, self).__init__()

        # Linear Layer
        self.norm = CustomLayerNorm(dim, nsubnets)
        self.local_linear = LocalGatedLinearBlock(dim, dim*2, nsubnets)
        self.skip_weight= Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            requires_grad=True
        )
    def forward(self, x):
        x = self.skip_weight[0] * x + self.skip_weight[1] * self.local_linear(self.norm(x))
        return x

##########################################################################
class LocalLowpassFilteringBlock(nn.Module):
    def __init__(self, dim, nsubnets, ngraphs):
        super(LocalLowpassFilteringBlock, self).__init__()

        self.local_filter = MixtureGLR( 
            n_graphs=ngraphs,
            n_node_fts=dim//ngraphs,
            alpha_init=0.5,
            beta_init=0.1,
            muy_init=torch.tensor([[0.001], [0.0], [0.0], [0.0]]),
        )

    def forward(self, x):
        x = self.local_filter(x)
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

    def forward(self, x):
        x = self.channels_local_linear_op01(x)
        return x


##########################################################################
## Down/Up Sampling
class Downsampling(nn.Module):
    def __init__(self, dim_in, dim_out, nsubnets):
        super(Downsampling, self).__init__()
        self.local_linear = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2, padding=0, groups=nsubnets, bias=False)
    def forward(self, x):
        x = self.local_linear(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, dim_in, dim_out, nsubnets):
        super(Upsampling, self).__init__()
        self.local_linear = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2, padding=0, groups=nsubnets, bias=False)
    def forward(self, x):
        x = self.local_linear(x)
        return x



class AbtractMultiScaleGraphFilter(nn.Module):
    def __init__(self, 
        n_channels_in=3, 
        n_channels_out=3, 
        dims=[48, 64, 96, 128],
        nsubnets=[1, 1, 1, 1],
        ngraphs=[4, 4, 8, 8],
        num_blocks=[4, 6, 6, 8], 
        num_blocks_out=4
    ):

        super(AbtractMultiScaleGraphFilter, self).__init__()

        # MixtureGTVGLR( 
        #     n_graphs, n_node_fts,
        #     connection_window,
        #     n_cgd_iters, alpha_init, beta_init,
        #     muy_init, ro_init, gamma_init,
        #     device
        # )

        # ENCODING
        self.patch_3x3_embeding = ReginalPixelEmbeding(n_channels_in, dims[0])
        self.encoder_scale_00 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[0], nsubnets=nsubnets[0]) for i in range(num_blocks[0])
        ])
        
        self.down_sample_00_01 = Downsampling(dim_in=dims[0], dim_out=dims[1], nsubnets=nsubnets[0]) 
        self.encoder_scale_01 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[1], nsubnets=nsubnets[1]) for i in range(num_blocks[1])
        ])

        self.down_sample_01_02 = Downsampling(dim_in=dims[1], dim_out=dims[2], nsubnets=nsubnets[1]) 
        self.encoder_scale_02 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[2], nsubnets=nsubnets[2]) for i in range(num_blocks[2])
        ])

        self.down_sample_02_03 = Downsampling(dim_in=dims[2], dim_out=dims[3], nsubnets=nsubnets[2]) 
        self.encoder_scale_03 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[3], nsubnets=nsubnets[3]) for i in range(num_blocks[3])
        ])

        ## FILTER
        self.localfilter_scale_00 = LocalLowpassFilteringBlock(dim=dims[0], nsubnets=nsubnets[0], ngraphs=ngraphs[0])
        self.localfilter_scale_01 = LocalLowpassFilteringBlock(dim=dims[1], nsubnets=nsubnets[1], ngraphs=ngraphs[1])
        self.localfilter_scale_02 = LocalLowpassFilteringBlock(dim=dims[2], nsubnets=nsubnets[2], ngraphs=ngraphs[2])
        self.localfilter_scale_03 = LocalLowpassFilteringBlock(dim=dims[3], nsubnets=nsubnets[3], ngraphs=ngraphs[3])

        # DECODING
        self.up_sample_03_02 = Upsampling(dim_in=dims[3], dim_out=dims[2], nsubnets=nsubnets[3])
        self.combine_channels_02 = nn.Conv2d(dims[2]*2, dims[2], kernel_size=1, bias=False, groups=nsubnets[2])
        self.decoder_scale_02 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[2], nsubnets=nsubnets[2]) for i in range(num_blocks[2])
        ])

        self.up_sample_02_01 = Upsampling(dim_in=dims[2], dim_out=dims[1], nsubnets=nsubnets[2])
        self.combine_channels_01 = nn.Conv2d(dims[1]*2, dims[1], kernel_size=1, bias=False, groups=nsubnets[1])
        self.decoder_scale_01 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[1], nsubnets=nsubnets[1]) for i in range(num_blocks[1])
        ])

        self.up_sample_01_00 = Upsampling(dim_in=dims[1], dim_out=dims[0], nsubnets=nsubnets[1])
        self.combine_channels_00 = nn.Conv2d(dims[0]*2, dims[0], kernel_size=1, bias=False, groups=nsubnets[0])
        self.decoder_scale_00 = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[0], nsubnets=nsubnets[0]) for i in range(num_blocks[0])
        ])

        self.refining_block = nn.Sequential(*[
            LocalNonLinearBlock(dim=dims[0], nsubnets=nsubnets[0]) for i in range(num_blocks_out)
        ])
        self.linear_output = nn.Conv2d(dims[0], n_channels_out, kernel_size=1, bias=False)
    def forward(self, img):
        
        # Downward ENCODING
        inp_enc_scale_00 = self.patch_3x3_embeding(img)
        out_enc_scale_00 = self.encoder_scale_00(inp_enc_scale_00)
        
        inp_enc_scale_01 = self.down_sample_00_01(out_enc_scale_00)
        out_enc_scale_01 = self.encoder_scale_01(inp_enc_scale_01)

        inp_enc_scale_02 = self.down_sample_01_02(out_enc_scale_01)
        out_enc_scale_02 = self.encoder_scale_02(inp_enc_scale_02)

        inp_enc_scale_03 = self.down_sample_02_03(out_enc_scale_02)
        out_enc_scale_03 = self.encoder_scale_03(inp_enc_scale_03)

        # FILTERING
        out_enc_scale_00 = self.localfilter_scale_00(out_enc_scale_00)
        out_enc_scale_01 = self.localfilter_scale_01(out_enc_scale_01)
        out_enc_scale_02 = self.localfilter_scale_02(out_enc_scale_02)
        out_enc_scale_03 = self.localfilter_scale_03(out_enc_scale_03)


        # Upward DECODING
        inp_dec_scale_02 = self.up_sample_03_02(out_enc_scale_03)
        out_dec_scale_02 = torch.cat([inp_dec_scale_02, out_enc_scale_02], 1)
        out_dec_scale_02 = self.combine_channels_02(out_dec_scale_02)
        out_dec_scale_02 = self.decoder_scale_02(out_dec_scale_02)

        inp_dec_scale_01 = self.up_sample_02_01(out_dec_scale_02)
        out_dec_scale_01 = torch.cat([inp_dec_scale_01, out_enc_scale_01], 1)
        out_dec_scale_01 = self.combine_channels_01(out_dec_scale_01)
        out_dec_scale_01 = self.decoder_scale_01(out_dec_scale_01)

        inp_dec_scale_00 = self.up_sample_01_00(out_dec_scale_01)
        out_dec_scale_00 = torch.cat([inp_dec_scale_00, out_enc_scale_00], 1)
        out_dec_scale_00 = self.combine_channels_00(out_dec_scale_00)
        out_dec_scale_00 = self.decoder_scale_00(out_dec_scale_00)
 
        output = self.refining_block(out_dec_scale_00)
        output = self.linear_output(output) 
        return output