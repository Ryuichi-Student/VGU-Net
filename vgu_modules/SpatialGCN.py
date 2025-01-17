import torch
from torch import nn
import torch.nn.functional as F


class SpatialGCN(nn.Module):
    def __init__(self, plane,inter_plane=None,out_plane=None):
        super(SpatialGCN, self).__init__()
        if inter_plane==None:
            inter_plane = plane
        if out_plane==None:
            out_plane = plane
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.conv_wgl = nn.Linear(inter_plane, out_plane)
        self.bn1 = nn.BatchNorm1d(out_plane)
        self.conv_wgl2 = nn.Linear(out_plane, out_plane)
        self.bn2 = nn.BatchNorm1d(out_plane)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        node_k = self.node_k(x)  #####nosym better, softmax better,only one gcn better
        node_q = self.node_q(x)
        node_v = self.node_v(x)
        
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)  ##b N C
        node_q = node_q.view(b, c, -1)  ###b C N
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)  ##b N C

        Adj = torch.bmm(node_k, node_q)  ###Q*K^T
        Adj = self.softmax(Adj)  ###adjacency matrix of size b N N

        AV = torch.bmm(Adj,node_v)###AX
        AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2))###AXW b n C
        AVW = F.dropout(AVW)

        # add one more layer
        AV = torch.bmm(Adj, AVW)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW = F.dropout(AVW)

        # end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        AVW = AVW.view(b, c, h, -1)
        return AVW


class HydraGCN(nn.Module):
    def __init__(self, plane, inter_plane=None, out_plane=None, num_heads=4):
        super(HydraGCN, self).__init__()
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
        
        self.conv_wgl1 = nn.Linear(self.inter_plane * num_heads, out_plane)
        self.conv_wgl2 = nn.Linear(inter_plane, out_plane)

        self.bn1 = nn.BatchNorm1d(out_plane)
        self.bn2 = nn.BatchNorm1d(out_plane)
        
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        b, c, h, w = x.size()

        # Compute keys, queries, and values for all heads   # (b, num_heads, N, inter_plane)
        k = self.node_k(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)  
        q = self.node_q(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)
        v = self.node_v(x).view(b, self.num_heads, self.inter_plane, h * w).permute(0, 1, 3, 2)

        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        # Global KV aggregation
        kv = torch.einsum('bnhf,bnhf->bnf', k, v)
        kv = kv.unsqueeze(2)  # (b, num_heads, 1, inter_plane)

        hydra_out = q * kv  # Element-wise query gating: (b, num_heads, N, inter_plane)
        hydra_out = hydra_out.transpose(1, 2).view(b, -1, self.num_heads * self.inter_plane) # (b, N, num_heads * inter_plane)
        hydra_out = F.relu(self.bn1(self.conv_wgl1(hydra_out).transpose(1,2)).transpose(1,2))  # b, N, out
        
        # Compute mean of keys and queries across heads
        k_mean = k.mean(dim=1)  # (b, N, inter_plane)
        q_mean = q.mean(dim=1).permute(0, 2, 1)  # (b, inter_plane, N)
        
        Adj = torch.bmm(k_mean, q_mean)  # (b, N, N)
        Adj = self.softmax(Adj)  # Normalize adjacency matrix
        
        # add one more layer
        AV = torch.bmm(Adj, hydra_out)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW = F.dropout(AVW)

        # end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        AVW = AVW.view(b, c, h, -1)
        return AVW
