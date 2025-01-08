import torch
from torch import nn
import torch.nn.functional as F


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
        AV = torch.bmm(Adj,AVW)
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

        # Global KV aggregation (avoids T x T matrix computation)
        kv = torch.einsum('bnhf,bnhf->bnf', k, v)
        kv = kv.unsqueeze(2)  # (b, num_heads, 1, inter_plane)

        # Apply gating using queries
        hydra_out = q * kv  # Element-wise gating: (b, num_heads, N, inter_plane)

        # Reshape and combine multi-head outputs  # (b, num_heads * inter_plane, h, w)
        hydra_out = hydra_out.permute(0, 1, 3, 2).reshape(b, -1, h, w)

        return self.final_projection(hydra_out)  # (b, out_plane, h, w)

    def forward(self, x):
        x = self.hydra_attention(x)

        x = self.gcn_layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.gcn_layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
