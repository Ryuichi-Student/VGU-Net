import math
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from vgu_modules.DoubleConv import DoubleConv
from vgu_modules.SpatialGCN import SpatialGCN, HydraGCN
from vgu_modules.HyperGraph import HyperNet
from dataclasses import dataclass
from collections.abc import Mapping


class Backbone(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_nc=64, use_hypergraph=True, image_height=160 , **kwargs):
        super(Backbone, self).__init__()
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv2 = DoubleConv(base_nc, 2 * base_nc)
        self.pool2 = nn.Conv2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv3 = DoubleConv(2 * base_nc, 4 * base_nc)
        self.pool3 = nn.Conv2d(4 * base_nc, 4 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        
        self.sgcn1 = HydraGCN(4 * base_nc)
        self.up6 = None if use_hypergraph else nn.ConvTranspose2d(4 * base_nc, 4 * base_nc, 2, stride=2,padding=0)  ##upsampling
        
        self.use_hypergraph = use_hypergraph
        self.hypernet = None if not use_hypergraph else HyperNet(4 * base_nc, image_height // 16)
        
    def forward(self, x):
        c1 = self.conv1(x)  ## 2 nc
        p1 = self.pool1(c1)  ##
        c2 = self.conv2(p1) ##nc 2nc
        p2 = self.pool2(c2)
        c3 = self.conv3(p2) ##2nc 2nc
        p3 = self.pool3(c3)
        
        c4 = self.sgcn1(p3)   ###spatial gcn 4nc
        
        if self.use_hypergraph:
            up_6 = self.hypernet(c4)
            
        else:
            up_6 = self.up6(c4)
            
        return c1, c2, c3, up_6
            
        
class Head(nn.Module):
    def __init__(self, **kwargs):
        super(Head, self).__init__()

    def forward(self, c1, c2, c3, up_6):
        raise NotImplementedError("Subclasses must implement the forward method.")
            
class SegmentationHead(Head):
    def __init__(self, base_nc=64, out_ch=2, **kwargs):
        super(SegmentationHead, self).__init__()
        self.sgcn3 = SpatialGCN(2 * base_nc)
        self.sgcn2 = SpatialGCN(4 * base_nc)

        self.conv6 = DoubleConv(8 * base_nc, 4 * base_nc)
        self.up7 = nn.ConvTranspose2d(4 * base_nc, 2 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv7 = DoubleConv(4 * base_nc, 2 * base_nc)
        self.up8 = nn.ConvTranspose2d(2 * base_nc, base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv8 = DoubleConv(2 * base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1, padding=0)
        
    def forward(self, c1, c2, c3, up_6):
        c5 = self.sgcn2(c3)
        merge6 = torch.cat([up_6, c5], dim=1)
        c6 = self.conv6(merge6)
        
        up_7 = self.up7(c6)
        c7 = self.sgcn3(c2)
        merge7 = torch.cat([up_7, c7], dim=1)
        c8 = self.conv7(merge7)
        up_8 = self.up8(c8)
        
        merge8 = torch.cat([up_8, c1], dim=1)
        c9 = self.conv8(merge8)

        return self.conv9(c9)
    
class ClassificationHead(Head):
    def __init__(self, base_nc=64, num_classes=4, **kwargs):
        super(SegmentationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(8*base_nc, 4*base_nc),
            nn.BatchNorm1d(4*base_nc),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4*base_nc, num_classes)
        )
        
    def forward(self, c1, c2, c3, up_6):
        return self.classifier(up_6)
    
    
@dataclass
class VGUNetParams(Mapping):
    in_ch: int = 2
    out_ch: int = 2
    base_nc: int = 64
    num_classes: int = 4
    use_hypergraph: bool = True
    image_height: int = 160
    mode: str = "segmentation"
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dataclass_fields__)

    def __len__(self):
        return len(self.__dataclass_fields__)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
    
VGUNET_PARAMS = VGUNetParams()
    

class VGUNet(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, base_nc=None, use_hypergraph=None, mode=None, num_classes=None, image_height=None):
        super(VGUNet, self).__init__()
        
        VGUNET_PARAMS.update(in_ch=in_ch, out_ch=out_ch, base_nc=base_nc, use_hypergraph=use_hypergraph, mode=mode, num_classes=num_classes, image_height=image_height)
        
        print(dict(**VGUNET_PARAMS))
        self.backbone = Backbone(**VGUNET_PARAMS)
        
        mode = VGUNET_PARAMS['mode']
        if mode == "segmentation":
            self.head = SegmentationHead(**VGUNET_PARAMS)
        elif mode == "classification":
            self.head = ClassificationHead(**VGUNET_PARAMS)
        else:
            print(f"{mode} is not implemented yet")
            exit(1)
    
    def forward(self, x):
        c1, c2, c3, up_6 = self.backbone(x)
        
        out = self.head(c1, c2, c3, up_6)
        
        return out
    
    @staticmethod
    def load(path='./models/vgunet/vgunet_normal.pth', in_ch=None, out_ch=None, base_nc=None, use_hypergraph=None, mode=None, num_classes=None, image_height=None):
        ignore_containing = []
        print("usig pretrained model!!!")
        
        pretrained_model_dict = torch.load(path)
        # Remove torch.compiled model prefix
        for key in list(pretrained_model_dict.keys()):
            value = pretrained_model_dict.pop(key)
            if not ignore_containing or not any(c in key for c in ignore_containing):
                pretrained_model_dict[key.replace("_orig_mod.", "")] = value

        model = VGUNet(in_ch=in_ch, out_ch=out_ch, base_nc=base_nc, use_hypergraph=use_hypergraph, mode=mode, num_classes=num_classes, image_height=image_height)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
    