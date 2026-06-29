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
from attacks import LokiModule

def build_models(emb_dim=2048, loki=False, loki_image_shape=(3, 32, 32),
                 loki_fc_size=1024, loki_num_kernels=3):
    """
    Build the BYOL encoders/predictors. When `loki=True` the three encoders are
    wrapped so a LOKI attack module (conv identity-map + 2 FC layers) is prepended
    at the network input — this is what gives the malicious server a fully-
    connected layer that sees the raw image, enabling linear-layer leakage. The
    module is shape-preserving (residual passthrough), so with an all-zero / inert
    module the encoder behaves exactly like the benign ResNet18. When `loki=False`
    the function returns the original benign architecture unchanged.
    """
    def make_encoder():
        if loki:
            return LokiEncoder(emb_dim, loki_image_shape, loki_fc_size, loki_num_kernels)
        return ResNet18Projv3(emb_dim=emb_dim)

    EMA = make_encoder()
    model = make_encoder()
    predictor = nn.Sequential(
            nn.Linear(emb_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, emb_dim),
            #nn.BatchNorm1d(emb_dim, affine=False)
            )
    # p_model / p_predictor dropped: the trainer never read them. Building the
    # 3rd encoder wasted ~1 GB/client (a full LokiEncoder). Previous return:
    # return model, EMA, predictor, p_model, p_predictor
    return model, EMA, predictor


class CIFARResNet18Backbone(ResNet):
    def __init__(self):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        # Adaptive pooling so the backbone accepts sub-32px inputs too. On 32px
        # inputs (4x4 feature map) this equals AvgPool2d(4, 4) exactly.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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


class LokiEncoder(nn.Module):
    """
    A benign ResNet18 encoder with a LOKI attack module prepended at the input.

    The submodule order matters: `loki` is declared first so its parameters lead
    the state-dict (the malicious server arms/reads exactly the first four float
    arrays — conv.weight, fc1.weight, fc1.bias, fc2.weight). The module is a
    residual passthrough; when its weights are all zero it is fully inert (output
    == input, zero gradient), so a non-target client trains the benign network
    untouched. When armed with trap weights it leaks the target's images through
    fc1's weight gradient while still passing the (near-unchanged) image to the
    backbone for normal training. Mirrors ResNet18Projv3's interface so SSL,
    FedEMA and the kNN eval use it unchanged.
    """

    def __init__(self, emb_dim=2048, image_shape=(3, 32, 32), fc_size=1024, num_kernels=3):
        super().__init__()
        self.loki = LokiModule(image_shape, fc_size, num_kernels, fedavg=True)
        self.net = ResNet18Projv3(emb_dim=emb_dim)

    def forward(self, x, normalize=False):
        return self.net(self.loki(x, residual=True), normalize=normalize)

    @torch.no_grad()
    def encode_backbone(self, x, normalize=False):
        return self.net.encode_backbone(self.loki(x, residual=True), normalize=normalize)
