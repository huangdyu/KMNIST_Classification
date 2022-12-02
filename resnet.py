# resnet 18 and resnet 34 

from torch import nn
from torch.nn import functional as F

'''

For KMNISZ, we only define resnet 18 and 34, no need to use deeper network
for the paer of resnet, please refer to 
"https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf"

'''


arch = {
        "res18": [(64, 2),(128, 2),(256, 2),(512, 2)],
        "res34": [(64, 3), (128, 4), (256, 6), (512, 3)]
}               


#### basic residual block for resnet 18 and 34


class res_blk(nn.Module):
    def __init__(self, in_channel, out_channel, channel_change = False,stride = 2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = in_channel, out_channels = out_channel, 
                                    kernel_size = 3, padding = 1, stride = stride),
                                    nn.BatchNorm2d(out_channel), nn.ReLU(),
                                    nn.Conv2d(in_channels = out_channel, out_channels = out_channel, 
                                    kernel_size = 3, padding= 1),
                                    nn.BatchNorm2d(out_channel))
        if channel_change:
            self.conv2 = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, 
                                    kernel_size = 1, stride = stride) 
        else:
            self.conv2 = None
    def forward(self, X):
        if self.conv2 == None:
            return F.relu(X + self.conv1(X))
        else:
            return F.relu(self.conv1(X) + self.conv2(X))

def make_layer(arch):
    net = []
    in_channel = 64
    for out_channel, num_blk in arch:
        for i in range(num_blk):
            if i == 0:
                net.append(res_blk(in_channel, out_channel, channel_change = True))
            else:
                net.append(res_blk(out_channel, out_channel, stride = 1))
        in_channel = out_channel
    return nn.Sequential(*net)

class resnet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        #architecture for resnet18 and 34 respectively

        self.net1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, 
                                    kernel_size = 7, padding = 3, stride = 2),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2))

        self.net2 = make_layer(arch)

        self.net3 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                nn.Flatten(),
                                nn.Linear(512,10))

    def forward(self, X):
        X = self.net1(X)
        X = self.net2(X)
        return self.net3(X)
    


def get_res18(arch = arch):
    return resnet(arch['res18'])

def get_res34(arch = arch):
    return resnet(arch['res34'])





