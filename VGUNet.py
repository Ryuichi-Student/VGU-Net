import math
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

import dgl
import dgl.function as fn

from time import perf_counter

from torch_geometric.utils import to_undirected, add_self_loops, degree
from torch_geometric.data import Data

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class MSpatialGCN(nn.Module):
    def __init__(self, plane, inter_plane=None, out_plane=None, num_heads=4):
        super(MSpatialGCN, self).__init__()
        if inter_plane is None:
            inter_plane = plane // 2  # Intermediary dimension per head
        if out_plane is None:
            out_plane = plane
        
        self.num_heads = num_heads
        self.inter_plane = inter_plane
        self.out_plane = out_plane

        # Define layers for keys, queries, and values for each head
        self.node_k = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)
        self.node_q = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)
        self.node_v = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)

        # Define output transformations per head
        self.conv_wgl = nn.Linear(self.inter_plane, self.out_plane)
        self.bn1 = nn.BatchNorm1d(self.out_plane * num_heads)
        self.conv_wgl2 = nn.Linear(self.out_plane, self.out_plane)
        self.bn2 = nn.BatchNorm1d(self.out_plane * num_heads)

        self.softmax = nn.Softmax(dim=2)
        self.final_projection = nn.Conv2d(self.out_plane * num_heads, out_plane, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Compute keys, queries, and values for all heads
        node_k = self.node_k(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)  # (b, num_heads, N, inter_plane)
        node_q = self.node_q(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 2, 3)  # (b, num_heads, inter_plane, N)
        node_v = self.node_v(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)  # (b, num_heads, N, inter_plane)

        # Compute adjacency matrices for each head
        Adj = torch.matmul(node_k, node_q)  # (b, num_heads, N, N)
        Adj = self.softmax(Adj)

        # Perform attention for each head
        AV = torch.matmul(Adj, node_v)  # (b, num_heads, N, inter_plane)
        AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(2, 3).reshape(b, -1, h * w)).reshape(b, self.num_heads, self.out_plane, h * w).transpose(2, 3))  # (b, num_heads, N, out_plane)
        AVW = F.dropout(AVW, training=self.training)

        # Second layer of attention
        AV = torch.matmul(Adj, AVW)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(2, 3).reshape(b, -1, h * w)).reshape(b, self.num_heads, self.out_plane, h * w).transpose(2, 3))
        AVW = F.dropout(AVW, training=self.training)

        # Combine heads and project back to original space
        AVW = AVW.permute(0, 1, 3, 2).contiguous().view(b, -1, h, w)  # (b, num_heads * out_plane, h, w)
        out = self.final_projection(AVW)  # (b, out_plane, h, w)

        return out
    

class HSpatialGCN(nn.Module):
    def __init__(self, plane, inter_plane=None, out_plane=None, num_heads=4):
        super(HSpatialGCN, self).__init__()
        if inter_plane is None:
            inter_plane = plane
        if out_plane is None:
            out_plane = plane

        self.num_heads = num_heads
        self.inter_plane = inter_plane
        self.out_plane = out_plane

        self.node_k = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)
        self.node_q = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)
        self.node_v = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)

        self.final_projection = nn.Conv2d(self.inter_plane * num_heads, out_plane, kernel_size=1)

        self.gcn_layer1 = nn.Conv2d(out_plane, out_plane, kernel_size=1)
        self.gcn_layer2 = nn.Conv2d(out_plane, out_plane, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)

        self.relu = nn.ReLU()

    def hydra_attention(self, x):
        """
        Implements Hydra Attention mechanism.
        """
        b, c, h, w = x.size()

        # Compute keys, queries, and values for all heads   # (b, num_heads, N, inter_plane)
        k = self.node_k(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)  
        q = self.node_q(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)
        v = self.node_v(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)

        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        # Global KV aggregation (avoids T x T matrix computation)
        kv = torch.einsum('bnhf,bnhf->bnf', k, v)
        kv = kv.unsqueeze(2)  # (b, num_heads, 1, inter_plane)

        # Apply gating using queries
        hydra_out = q * kv  # Element-wise gating: (b, num_heads, N, inter_plane)

        # Reshape and combine multi-head outputs  # (b, num_heads * inter_plane, h, w)
        hydra_out = hydra_out.permute(0, 1, 3, 2).reshape(b, -1, h, w)

        return self.final_projection(hydra_out)  # (b, out_plane, h, w)

    def forward(self, x):
        # Apply Hydra Attention
        x = self.hydra_attention(x)

        x = self.gcn_layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.gcn_layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    
@torch.jit.script
def _construct_hypergraph(x, top_k: int=5):
    """
    Constructs a hypergraph efficiently using sparse tensors, avoiding explicit dense incidence matrix construction.

    :param x: Input tensor of shape (b, c, N), where b is batch size, c is feature dimension, and N is the number of nodes.
    :param top_k: Number of neighbors to connect each node to form hyperedges.
    :return: Tuple of sparse adjacency matrices (one for each batch).
    """
    b, c, N = x.shape  # b: batch size, c: feature dimension, N: number of nodes (h * w)

    # t = perf_counter()
    # Normalize the features along the feature dimension
    x_normalized = F.normalize(x, dim=1)  # (b, c, N)

    # Compute pairwise similarity using the normalized dot product
    similarity = torch.matmul(x_normalized.transpose(1, 2), x_normalized)  # (b, N, N)

    # For each batch, compute the top-k neighbors for each node
    _, topk_indices = similarity.topk(k=top_k, dim=-1)  # (b, N, top_k)
    # print(f"matmul {perf_counter()-t}")
    sparse_adjacencies = []

    # t = perf_counter()

    for i in range(b):  # Process each batch independently
        edges_src = torch.arange(N, device=x.device).repeat_interleave(top_k)  # Source nodes
        edges_dst = topk_indices[i].flatten()  # Destination nodes (top-k neighbors)

        # Create adjacency matrix in sparse format
        edge_indices = torch.stack([edges_src, edges_dst], dim=0)  # Shape (2, num_edges)
        edge_values = torch.ones(edge_indices.size(1), device=x.device)  # All edges have weight 1

        adjacency_matrix = torch.sparse_coo_tensor(
            edge_indices, edge_values, size=(N, N), device=x.device
        )

        sparse_adjacencies.append(adjacency_matrix)

    # print(f"adjacency construction {perf_counter()-t}")
    return sparse_adjacencies

@torch.compiler.disable()
def construct_hypergraph(x, top_k: int=5):
    return _construct_hypergraph(x, top_k)

from typing import List

@torch.jit.script
def _apply_hypergraph_laplacian(sparse_adjacencies: List[torch.Tensor], tensor):
    """
    Applies the hypergraph Laplacian to the input tensor using sparse tensors.

    :param sparse_adjacencies: List of sparse adjacency matrices (one for each batch).
    :param tensor: Input tensor of shape (batches, num_heads, N, inter_plane).
    :return: Tensor after applying the Laplacian, same shape as input.
    """
    tensor = tensor.float()
    batches, num_heads, N, inter_plane = tensor.shape
    laplacian_applied = []

    # t0 = perf_counter()
    # t = 0

    for i, adjacency_matrix in enumerate(sparse_adjacencies):
        # Add self-loops to the adjacency matrix
        self_loop_indices = torch.arange(N, device=tensor.device).unsqueeze(0).repeat(2, 1)
        self_loop_values = torch.ones(N, device=tensor.device)
        self_loops = torch.sparse_coo_tensor(
            self_loop_indices, self_loop_values, size=(N, N), device=tensor.device
        )

        adjacency_with_self_loops = adjacency_matrix + self_loops

        # Compute degree matrix and its normalization
        # t1 = perf_counter()

        # Ensure the sparse adjacency matrix has self-loops
        adjacency_with_self_loops = adjacency_with_self_loops.coalesce()  # Ensure COO format

        # Compute the degree for each node directly in sparse format
        degrees = torch.sparse.sum(adjacency_with_self_loops, dim=(1,))  # Sparse sum along rows

        # Compute the normalization factor: D^(-1/2)
        norm = torch.pow(degrees.values(), -0.5).clamp(min=1e-9)  # Avoid divide-by-zero

        # Get indices and values of the sparse matrix
        indices = adjacency_with_self_loops.indices()
        row = indices[0]
        col = indices[1]
        values = adjacency_with_self_loops.values()

        # Normalize the adjacency matrix using the normalization factor
        normalized_values = norm[row] * values * norm[col]
        normalized_adjacency = torch.sparse_coo_tensor(
            adjacency_with_self_loops.indices(),
            normalized_values,
            adjacency_with_self_loops.size()
        )

        # t += perf_counter()-t1

        # Extract features for this batch (combine head and feature dimensions)
        # num head N inter_plane -> N inter_plane num_head
        features = tensor[i].permute(1, 2, 0).reshape(N, -1)

        # Apply normalized adjacency matrix to features
        updated_features = torch.sparse.mm(normalized_adjacency, features)

        # Reshape back to (num_heads, N, inter_plane)
        updated_features = updated_features.view(N, inter_plane, num_heads).permute(2, 0, 1)

        laplacian_applied.append(updated_features)

    # Stack results for all batches
    out = torch.stack(laplacian_applied, dim=0)

    # print(f"dense: {t}")
    # print(f"laplacian application: {perf_counter()-t0}")

    return out  # Shape (batches, num_heads, N, inter_plane)

@torch.compiler.disable()
def apply_hypergraph_laplacian(sparse_adjacencies: List[torch.Tensor], tensor):
    return _apply_hypergraph_laplacian(sparse_adjacencies, tensor)
    
    
class HSpatialHyperGCN(nn.Module):
    def __init__(self, plane, inter_plane=None, out_plane=None, num_heads=4):
        super(HSpatialHyperGCN, self).__init__()
        if inter_plane is None:
            inter_plane = plane
        if out_plane is None:
            out_plane = plane

        self.num_heads = num_heads
        self.inter_plane = inter_plane
        self.out_plane = out_plane

        self.node_k = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)
        self.node_q = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)
        self.node_v = nn.Conv2d(plane, self.inter_plane * num_heads, kernel_size=1)

        self.final_projection = nn.Conv2d(self.inter_plane * num_heads, out_plane, kernel_size=1)

        self.gcn_layer1 = nn.Conv2d(out_plane, out_plane, kernel_size=1)
        self.gcn_layer2 = nn.Conv2d(out_plane, out_plane, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)

        self.relu = nn.ReLU()

    def hydra_attention(self, x):
        b, c, h, w = x.size()

        # Compute keys, queries, and values for all heads   # (b, num_heads, N, inter_plane)
        k = self.node_k(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)  
        q = self.node_q(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)
        v = self.node_v(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)

        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)
        
        # Sparse tensors can't be used with amp
        
        # Construct and compute hypergraph Laplacian
        # L = self.compute_hypergraph_laplacian(H)  # Laplacian (b, inter_plane, N, N)
        
        with torch.cuda.amp.autocast(enabled=False):
            H = construct_hypergraph(x.view(b, c, h*w).float())  # Incidence matrix (b, inter_plane, N, N)
            k = apply_hypergraph_laplacian(H, k)
            v = apply_hypergraph_laplacian(H, v)
                
        # Global KV aggregation (avoids T x T matrix computation)
        kv = torch.einsum('bnhf,bnhf->bnf', k, v)
        kv = kv.unsqueeze(2)  # (b, num_heads, 1, inter_plane)

        # Apply gating using queries
        hydra_out = q * kv  # Element-wise gating: (b, num_heads, N, inter_plane)

        # Reshape and combine multi-head outputs  # (b, num_heads * inter_plane, h, w)
        hydra_out = hydra_out.permute(0, 1, 3, 2).reshape(b, -1, h, w)

        return self.final_projection(hydra_out)  # (b, out_plane, h, w)

    def forward(self, x):
        # Apply Hydra Attention
        x = self.hydra_attention(x)

        x = self.gcn_layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.gcn_layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SpatialGCN(nn.Module):
    def __init__(self, plane,inter_plane=None,out_plane=None):
        super(SpatialGCN, self).__init__()
        if inter_plane==None:
            inter_plane = plane #// 2
        if out_plane==None:
            out_plane = plane
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.conv_wgl = nn.Linear(inter_plane,out_plane)
        self.bn1 = nn.BatchNorm1d(out_plane)
        self.conv_wgl2 = nn.Linear(out_plane, out_plane)
        self.bn2 = nn.BatchNorm1d(out_plane)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        node_k = self.node_k(x)
        node_q = self.node_q(x)
        node_v = self.node_v(x)
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1).float()  ##b N C
        node_q = node_q.view(b, c, -1).float()  ###b c N
        node_v = node_v.view(b, c, -1).permute(0, 2, 1).float()  ##b N C
        Adj = torch.bmm(node_k, node_q)  ###Q*K^T

        Adj = self.softmax(Adj)

        AV = torch.bmm(Adj,node_v)###AX
        AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2))###AXW b n C
        AVW = F.dropout(AVW)
        # add one more layer
        AV = torch.bmm(Adj,AVW)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW = F.dropout(AVW)
        # end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        AVW = AVW.view(b, c, h, -1)
        return AVW

class VGUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3,base_nc=64,fix_grad=True):
        super(VGUNet, self).__init__()
        self.fix_grad = fix_grad
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv2 = DoubleConv(base_nc, 2 * base_nc)
        self.pool2 = nn.Conv2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv3 = DoubleConv(2 * base_nc, 4 * base_nc)
        self.pool3 = nn.Conv2d(4 * base_nc, 4 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        
        self.sgcn3 = HSpatialGCN(2 * base_nc)
        self.sgcn2 = HSpatialGCN(4 * base_nc)
        self.sgcn1 = HSpatialGCN(4 * base_nc)
        
        self.up6 = nn.ConvTranspose2d(4 * base_nc, 4 * base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv6 = DoubleConv(8 * base_nc, 4 * base_nc)
        self.up7 = nn.ConvTranspose2d(4 * base_nc, 2 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv7 = DoubleConv(4 * base_nc, 2 * base_nc)
        self.up8 = nn.ConvTranspose2d(2 * base_nc, base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv8 = DoubleConv(2 * base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1, padding=0)
        
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)
        
        c4=self.sgcn1(p3)   ###spatial gcn 4nc
        up_6= self.up6(c4)
        merge6 = torch.cat([up_6, self.sgcn2(c3)], dim=1)##gcn
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, self.sgcn3(c2)], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)
        
        return c9
    


#     def construct_hypergraph(self, x):
#         b, c, N = x.shape  # b: batch size, c: features (inter_plane), N: nodes (h * w)

#         # Compute pairwise similarity using normalized dot product
#         x_normalized = F.normalize(x, dim=1)  # Normalize along feature dimension (c)
#         similarity = torch.matmul(x_normalized.transpose(1, 2), x_normalized)  # (b, N, N)

#         # Create hyperedges: Top-k neighbors
#         top_k = 5  # Number of neighbors per node
#         _, hyperedges = similarity.topk(k=top_k, dim=-1)  # Indices of top-k neighbors (b, N, top_k)

#         # Initialize incidence matrix
#         incidence_matrix = torch.zeros((b, c, N, N), device=x.device)

#         # Populate incidence matrix
#         batch_indices = torch.arange(b, device=x.device).view(-1, 1, 1).expand(-1, N, top_k)
#         node_indices = torch.arange(N, device=x.device).view(1, -1, 1).expand(b, -1, top_k)
#         incidence_matrix[batch_indices, :, node_indices, hyperedges] = 1
#         return incidence_matrix

#     def compute_hypergraph_laplacian(self, H):
#         """
#         Compute the hypergraph Laplacian without dense memory-expensive operations.

#         Args:
#             H (torch.Tensor): Incidence matrix of shape (b, inter_planes, N, N).

#         Returns:
#             torch.Tensor: Laplacian matrix of shape (b, inter_planes, N, N).
#         """
#         b, inter_planes, N, _ = H.shape

#         # Node degrees (D_v): Sum of incidences across hyperedges
#         D_v = H.sum(dim=-1)  # Shape: (b, inter_planes, N)

#         # Hyperedge degrees (D_e): Sum of incidences across nodes
#         D_e = H.sum(dim=-2)  # Shape: (b, inter_planes, N)

#         # Avoid divide-by-zero by adding a small epsilon
#         D_e_inv = 1.0 / (D_e + 1e-4)  # Shape: (b, inter_planes, N)

#         # First term: H * D_e^-1 * H^T
#         # Efficient scaling using einsum
#         H_scaled = torch.einsum('bfmn,bfn->bfmn', H, D_e_inv)  # Shape: (b, inter_planes, N, N)
#         H_T = H.transpose(-1, -2)  # Transpose across last two dimensions
#         first_term = torch.matmul(H_scaled, H_T)  # Shape: (b, inter_planes, N, N)

#         # Second term: Subtract node degree matrix (diagonal)
#         D_v_diag = torch.diag_embed(D_v)  # Diagonalize D_v (b, inter_planes, N, N)

#         # Laplacian: L = first_term - D_v_diag
#         L = first_term - D_v_diag
#         return L



#     @torch.compiler.disable()
#     def construct_hypergraph(self, x, top_k=5):
#         """
#         Constructs a hypergraph efficiently using DGL, avoiding explicit dense incidence matrix construction.

#         :param x: Input tensor of shape (b, c, N), where b is batch size, c is feature dimension, and N is the number of nodes.
#         :param top_k: Number of neighbors to connect each node to form hyperedges.
#         :return: List of DGL graphs representing the hypergraphs for each batch.
#         """
#         b, c, N = x.shape  # b: batch size, c: feature dimension, N: number of nodes (h * w)

#         # Normalize the features along the feature dimension
#         x_normalized = F.normalize(x, dim=1)  # (b, c, N)

#         # Compute pairwise similarity using the normalized dot product
#         similarity = torch.matmul(x_normalized.transpose(1, 2), x_normalized)  # (b, N, N)

#         # For each batch, compute the top-k neighbors for each node
#         _, topk_indices = similarity.topk(k=top_k, dim=-1)  # (b, N, top_k)

#         hypergraphs = []

#         for i in range(b):  # Process each batch independently
#             edges_src = torch.arange(N, device=x.device).repeat_interleave(top_k)  # Source nodes
#             edges_dst = topk_indices[i].flatten()  # Destination nodes (top-k neighbors)
            
#             # Create a DGL graph
#             graph = dgl.graph((edges_src, edges_dst), num_nodes=N)

#             graph.edata['edge_id'] = torch.arange(graph.num_edges(), device=x.device)

#             # Add features to the graph nodes if needed
#             graph.ndata['feat'] = x[i].transpose(0, 1)  # Transpose to (N, c)

#             hypergraphs.append(graph)

#         return hypergraphs
    
#     @torch.compiler.disable()
#     def apply_hypergraph_laplacian(self, hypergraphs, tensor):
#         """
#         Applies the hypergraph Laplacian to the input tensor.

#         :param hypergraphs: List of DGL graphs (one for each batch).
#         :param tensor: Input tensor of shape (batches, num_heads, N, inter_plane).
#         :return: Tensor after applying the Laplacian, same shape as input.
#         """
#         batches, num_heads, N, inter_plane = tensor.shape
#         laplacian_applied = []

#         for i, graph in enumerate(hypergraphs):
#             # Add self-loops and compute degree normalization
#             graph = dgl.add_self_loop(graph)
#             degs = graph.in_degrees().float().clamp(min=1).to(tensor.device)  # Avoid divide-by-zero
#             norm = torch.pow(degs, -0.5).unsqueeze(-1)  # Shape (N, 1)

#             # Extract features for this batch (combine head and feature dimensions)
#             features = tensor[i].permute(1, 2, 0).reshape(N, -1)  # Dynamic reshape to avoid size issues

#             # Apply normalization and assign to graph
#             graph.ndata['h'] = features * norm

#             # Perform message passing
#             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

#             # Normalize results
#             h = graph.ndata.pop('h') * norm

#             # Reshape back to (num_heads, N, inter_plane)
#             h = h.view(N, num_heads, inter_plane).permute(1, 0, 2)  # Shape (num_heads, N, inter_plane)

#             laplacian_applied.append(h)

#         # Stack results for all batches
        
#         out = torch.stack(laplacian_applied, dim=0)
        
#         return out  # Shape (batches, num_heads, N, inter_plane)