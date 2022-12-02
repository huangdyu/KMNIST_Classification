from torch import nn
import torch 
"""

This is to define GoogleNet, we add batch normalization after each convolution and before Relu
and we don't use dropout and LRN as the paper mentioned
for the link of the googlenet's paper, please refer to
"https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf"

"""


'''
basic inception block
'''

class Inception(nn.Module):

    def __init__(self, in_channel, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv2d(in_channels = in_channel, out_channels = c1, kernel_size = 1),
                                         nn.BatchNorm2d(c1), nn.ReLU())
        self.p2_1 = nn.Sequential(nn.Conv2d(in_channels = in_channel, out_channels = c2[0], kernel_size = 1),
                                            nn.BatchNorm2d(c2[0]), nn.ReLU())
        self.p2_2 = nn.Sequential(nn.Conv2d(in_channels = c2[0], out_channels = c2[1], kernel_size = 3, padding = 1),
                                            nn.BatchNorm2d(c2[1]), nn.ReLU())
        self.p3_1 = nn.Sequential(nn.Conv2d(in_channels = in_channel, out_channels = c3[0], kernel_size = 1),
                                            nn.BatchNorm2d(c3[0]), nn.ReLU())
        self.p3_2 = nn.Sequential(nn.Conv2d(in_channels = c3[0], out_channels = c3[1], kernel_size = 5, padding = 2),
                                            nn.BatchNorm2d(c3[1]), nn.ReLU())
        self.p4 = nn.Sequential(nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 1),
                                nn.Conv2d(in_channels = in_channel, out_channels = c4, kernel_size = 1),
                                nn.BatchNorm2d(c4), nn.ReLU())
    def forward(self, X):
        X1 = self.p1(X)
        X2 = self.p2_2(self.p2_1(X))
        X3 = self.p3_2(self.p3_1(X))
        X4 = self.p4(X)
        return torch.cat((X1, X2, X3, X4), dim = 1)

class googlenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, padding = 3, stride = 2),  
                                            nn.BatchNorm2d(64), nn.ReLU(), 
                                            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.net2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1),  
                                            nn.BatchNorm2d(64), nn.ReLU(), 
                                   nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, padding = 1),
                                            nn.BatchNorm2d(192), nn.ReLU(),
                                            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.net3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                  Inception(256, 128, (128, 192), (32, 96), 64),
                                  nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.net4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                  Inception(512, 160, (112, 224), (24, 64), 64),
                                  Inception(512, 128, (128, 256), (24, 64), 64),
                                  Inception(512, 112, (144, 288), (32, 64), 64),
                                  Inception(528, 256, (160, 320), (32, 128), 128),
                                  nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.net5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                  Inception(832, 384, (192, 384), (48, 128), 128),
                                  nn.AdaptiveAvgPool2d((1,1)))
        self.net6 = nn.Sequential(nn.Flatten(), nn.Linear(1024, 10))

    def forward(self, X):

        X = self.net1(X)

        X = self.net2(X)

        X = self.net3(X)

        X = self.net4(X)

        X = self.net5(X)

        return self.net6(X)
    

def get_googlenet():
    return googlenet()