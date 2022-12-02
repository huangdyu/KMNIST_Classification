
"""

For the link to the paper of densenet, please refer to
"https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf"

"""

from torch.nn import functional as F
import torch
from torch import nn

''''

The architecture of different DenseNet, the former number of each tuple 
is number of basic convolution block in each dense block, the latter one is the growth rate in each dense block

'''

arch = {121: [(6, 32), (12,32), (24, 32), (16, 32)],
        169: [(6, 32), (12, 32), (32, 32), (32, 32)],
        201: [(6, 32), (12, 32), (48, 32), (32, 32)],
        264: [(6, 32), (12, 32), (64, 32), (48, 32)]
}


def conv_blk(in_channels, out_channels):
    return(nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.Conv2d(in_channels, in_channels, kernel_size = 1), nn.ReLU(),
                        nn.BatchNorm2d(in_channels),
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), nn.ReLU()))


class Dense_blk(nn.Module):
    def __init__(self, num_convs, in_channels, k):
        super(Dense_blk, self).__init__()
        blk = []
        for i in range(num_convs):
            blk.append(conv_blk(in_channels + k * i, k))
        self.net = nn.Sequential(*blk)
    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            X = torch.cat((X, Y), dim = 1)
        return X

def transition_blk(in_channels, out_channels):
    return nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                        nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, arch):
        super(DenseNet, self).__init__()
        part1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64), nn.ReLU(),
                            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)) 
        in_channels = 64

        part2 = []

        for i, (num_convs, k) in enumerate(arch):
            if i == 0:
                part2.append(Dense_blk(num_convs, in_channels, k))
                in_channels = num_convs * k + in_channels
                part2.append(transition_blk(in_channels, in_channels//2))
                in_channels = in_channels//2
            else:
                if i == 3:
                    part2.append(Dense_blk(num_convs, in_channels, k))
                    in_channels = num_convs * k + in_channels
                    part2.append(nn.AdaptiveAvgPool2d((1, 1)))
                    part2.append(nn.Sequential(nn.Flatten(), nn.Linear(in_channels, 10)))
                else:
                    part2.append(Dense_blk(num_convs, in_channels, k))
                    in_channels = num_convs * k + in_channels
                    part2.append(transition_blk(in_channels, in_channels//2))
                    in_channels = in_channels//2
        
        self.net = nn.Sequential(part1, nn.Sequential(*part2))

    def forward(self, X):
        return self.net(X)                
        

def get_densenet121(arch = arch):
    return DenseNet(arch[121])

def get_densenet169(arch = arch):
    return DenseNet(arch[169])

def get_densenet201(arch = arch):
    return DenseNet(arch[201])

def get_densenet264(arch = arch):
    return DenseNet(arch[264])

