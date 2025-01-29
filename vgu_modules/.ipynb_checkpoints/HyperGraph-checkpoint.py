import torch
import torch.nn as nn
import torch.nn.functional as F
from .DoubleConv import DoubleConv


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

    
class HGNN(nn.Module):
    def __init__(self, in_ch, node=None, K_neigs=[1], kernel_size=3, stride=1):
        super(HGNN, self).__init__()
        self.conv = nn.Linear(in_ch, in_ch)
        self.bn = nn.BatchNorm1d(in_ch)

        self.K_neigs = K_neigs  # neighbourhood sizes for KNN
        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        b,c,w,h = x.shape
        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        
        B, N, C = x.shape

        # 1) Build hypergraph incidence matrix H via KNN
        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(x, k=self.K_neigs[0])
        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)

        # 2) Compute Dv and adjust
        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.0
        Dv = Dv * alpha
        max_k = int(Dv.max())

        # 3) For nodes with degree < max_k, expand the adjacency
        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(x, k=max_k - 1)
        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(x.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(x.device)
        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()

        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)

        # 4) Local hypergraph kernel
        local_H = self.local_H.repeat(B, 1, 1).to(new_H.device)

        # 5) Combine the expanded and local incidence matrices, and compute G
        _H = torch.cat([new_H, local_H], dim=2)
        _G = self._generate_G_from_H_b(_H)

        # 6) linear -> matmul -> ReLU + BN + residual
        residual = x
        
        x = self.conv(x)
        x = _G.matmul(x)
        x = F.relu(self.bn(x.permute(0, 2, 1).contiguous())).permute(0, 2, 1).contiguous() + residual
        
        x = x.permute(0,2,1).contiguous().view(b,c,w,h)
        return x

    @torch.compiler.disable()
    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        Calculate hypergraph Laplacian-like matrix G from the incidence matrix H.
        """
        bs, n_node, n_hyperedge = H.shape

        # Hyperedge weights
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)
        # Node degrees
        DV = torch.sum(H, dim=2)
        # Hyperedge degrees
        DE = torch.sum(H, dim=1)

        invDE = torch.diag_embed(torch.pow(DE, -1))
        DV2 = torch.diag_embed(torch.pow(DV, -0.5))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)

        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 @ H @ W @ invDE @ HT @ DV2
            return G

    @torch.compiler.disable()
    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        x: (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)
    
    @torch.compiler.disable()
    @torch.no_grad()
    def batched_knn(self, x, k=1):
        """
        Perform KNN in a batched manner.
        """
        ori_dists = self.pairwise_distance(x)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)
        return topk_dists, topk_inds, ori_dists, avg_dists
    
    @torch.compiler.disable()
    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        """
        Create an incidence matrix from the top distances and their indices.
        """
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        # Return (B, N, N)
        return incidence_matrix.permute(0, 2, 1).contiguous()

    @torch.compiler.disable()
    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        """
        Compute the weights for the incidence matrix edges.
        """
        if prob:
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights
    
    @torch.compiler.disable()
    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        """
        Create a local hypergraph kernel matrix.
        """
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]
        inp_unf = F.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride)
        inp_unf = inp_unf.squeeze(0).transpose(0, 1).long()

        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()

        H_local = torch.zeros((size * size, edge))
        H_local[inp_unf, matrix] = 1.0

        return H_local


class HyperNet(nn.Module):
    def __init__(self, in_ch, image_height=10):
        super(HyperNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, in_ch)
        self.pool1 = nn.Conv2d(in_ch, in_ch, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv2 = DoubleConv(in_ch, in_ch)
        
        self.hgnn1 = HGNN(in_ch, node=image_height*2)
        self.hgnn2 = HGNN(in_ch, node=image_height)
        
        self.channels = [in_ch*4, in_ch*2]
        self.decoder1 = Decoder(in_ch*4, in_ch)
        
        
    def forward(self, x):
        features = []
        x = self.conv1(x)

        features.append(torch.concat([x, self.hgnn1(x)], dim=1))
        x = self.pool1(x)
        x = self.conv2(x)
        features.append(torch.concat([x, self.hgnn2(x)], dim=1))
        
        x = self.decoder1(features[-1], features[-2])
        
        return F.interpolate(x, scale_factor=2, mode='bilinear')