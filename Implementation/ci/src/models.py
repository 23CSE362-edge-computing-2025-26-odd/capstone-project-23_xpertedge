import torch.nn as nn
import torch.nn.functional as F
from .genetic import Hyperparameters

class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dropout_rate=0.0):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return self.dropout(x)

class CNN(nn.Module):
    def __init__(self, num_classes=62, hyperparams=None):
        super().__init__()

        if hyperparams is None:
            hyperparams = Hyperparameters(
                width_mult=1.0, learning_rate=0.01, batch_size=128,
                dropout_rate=0.2, weight_decay=2e-4, momentum=0.9,
                conv_channels=[16, 32, 64, 128]
            )

        self.hyperparams = hyperparams
        def c(ch): return max(8, int(ch * hyperparams.width_mult))

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c(hyperparams.conv_channels[0]), 3, 1, 1, bias=False),
            nn.BatchNorm2d(c(hyperparams.conv_channels[0])),
            nn.ReLU(inplace=True),
            nn.Dropout2d(hyperparams.dropout_rate * 0.5)
        )

        self.sep1 = SeparableConv(
            c(hyperparams.conv_channels[0]), c(hyperparams.conv_channels[1]), 
            stride=2, dropout_rate=hyperparams.dropout_rate * 0.7
        )
        self.sep2 = SeparableConv(
            c(hyperparams.conv_channels[1]), c(hyperparams.conv_channels[2]), 
            stride=2, dropout_rate=hyperparams.dropout_rate * 0.8
        )
        self.sep3 = SeparableConv(
            c(hyperparams.conv_channels[2]), c(hyperparams.conv_channels[3]), 
            stride=2, dropout_rate=hyperparams.dropout_rate
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout_fc = nn.Dropout(hyperparams.dropout_rate)
        self.fc = nn.Linear(c(hyperparams.conv_channels[3]), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sep1(x)
        x = self.sep2(x)
        x = self.sep3(x)
        x = self.pool(x)
        x = self.dropout_fc(x.view(x.size(0), -1))
        return self.fc(x)
