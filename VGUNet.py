import math
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from vgu_modules.DoubleConv import DoubleConv
from vgu_modules.SpatialGCN import SpatialGCN, HydraGCN


class VGUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_nc=64):
        super(VGUNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv2 = DoubleConv(base_nc, 2 * base_nc)
        self.pool2 = nn.Conv2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv3 = DoubleConv(2 * base_nc, 4 * base_nc)
        self.pool3 = nn.Conv2d(4 * base_nc, 4 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        
        self.sgcn3 = HydraGCN(2 * base_nc)
        self.sgcn2 = HydraGCN(4 * base_nc)
        self.sgcn1 = HydraGCN(4 * base_nc)  ###changed with spatialGCN
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
    
    @staticmethod
    def load(path='./models/vgunet/vgunet_normal.pth', in_ch=2, out_ch=2, base_nc=64):
        print("usig pretrained model!!!")
        pretrained_model_dict = torch.load(path)
        # Remove torch.compiled model prefix
        for key in list(pretrained_model_dict.keys()):
            pretrained_model_dict[key.replace("_orig_mod.", "")] = pretrained_model_dict.pop(key)
        model = VGUNet(in_ch, out_ch, base_nc)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
