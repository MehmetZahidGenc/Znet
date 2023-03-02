import torch
import torch.nn as nn
from Znet_parts import DoubleConv, MaxPoolLayer, Zpoint, BatchNormalization, classifierPart

"""
    Z-Net Model
"""

class ZNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(ZNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.maxP = MaxPoolLayer()

        self.batchNorm = BatchNormalization(num_of_features=640)

        self.block_1 = DoubleConv(in_channels=self.n_channels, mid_channels=32, out_channels=64)
        self.block_2 = DoubleConv(in_channels=64, mid_channels=64, out_channels=128)
        self.block_3 = DoubleConv(in_channels=128, mid_channels=128, out_channels=128)
        self.Zpoint_1 = Zpoint()
        self.block_4 = DoubleConv(in_channels=131, mid_channels=195, out_channels=256)
        self.block_5 = DoubleConv(in_channels=256, mid_channels=512, out_channels=512)
        self.Zpoint_2 = Zpoint()

        self.classifier_part = classifierPart(n_classes=self.n_classes)



    def forward(self, x):
        identify_1 = x

        x1 = self.block_1(x)
        x1 = self.maxP(x1)

        x2 = self.block_2(x1)
        x2 = self.maxP(x2)

        identify_2 = x2

        x3 = self.block_3(x2)

        x4 = self.Zpoint_1(x3, identify_1, is_MaxPool=True)

        x5 = self.block_4(x4)
        x5 = self.maxP(x5)

        x6 = self.block_5(x5)

        x7 = self.Zpoint_2(x6, identify_2, is_MaxPool=True)

        bn = self.batchNorm(x7)

        x8 = self.maxP(bn)

        flatten_x = torch.flatten(x8, 1)

        output = self.classifier_part(flatten_x)

        return output