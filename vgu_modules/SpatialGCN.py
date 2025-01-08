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
        node_k = self.node_k(
            x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)   #####nosym better, softmax better,only one gcn better
        node_q = self.node_q(x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)#
        node_v = self.node_v(x)  # x#
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)  ##b N C
        node_q = node_q.view(b, c, -1)  ###b c N
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
