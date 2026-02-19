import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import datasets, transforms, models
    
def build_models(emb_dim=2048):
    EMA=ResNet18Projv3(emb_dim=emb_dim)
    model=ResNet18Projv3(emb_dim=emb_dim)
    p_model=ResNet18Projv3(emb_dim=emb_dim)
    predictor = nn.Sequential(
            nn.Linear(emb_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, emb_dim),
            #nn.BatchNorm1d(emb_dim, affine=False)
            )
    p_predictor=nn.Sequential(
            nn.Linear(emb_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, emb_dim),
            #nn.BatchNorm1d(emb_dim, affine=False)
            )
    return model, EMA, predictor, p_model, p_predictor

class CIFARResNet18Backbone(ResNet):
    def __init__(self):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.fc = nn.Identity()

    def forward(self, x):
        # Standard ResNet forward, but returns 512-d features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # [B, 512]

class ResNet18Projv3(nn.Module):
    def __init__(self, emb_dim=2048):
        super().__init__()
        self.backbone = CIFARResNet18Backbone()

        # Paper: "replace last linear with a two-layer MLP, same as predictor"
        #self.proj = nn.Sequential(
        #    nn.Linear(512, 4096, bias=True), #Should possibly be True
        #    nn.BatchNorm1d(4096),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(4096, emb_dim, bias=False),
        #)
        self.proj = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, emb_dim),#Maybe False
            #nn.BatchNorm1d(emb_dim, affine=False)
        )

    def forward(self, x, normalize=False):
        h = self.backbone(x)          # [B,512]
        z = self.proj(h)              # [B,2048]
        return F.normalize(z, dim=1) if normalize else z

    @torch.no_grad()
    def encode_backbone(self, x, normalize=False):
        h = self.backbone(x)
        return F.normalize(h, dim=1) if normalize else h
