from torch import nn

'''

this is for construct vgg net, including vgg11, 13, 16, 19,
not totally same as the paper, we use batch normalization here.
and we do not use conv1 here
we only define ConvNet Configuration A, B, D, E. 
For paper of vgg, please refer to "https://arxiv.org/pdf/1409.1556.pdf"
Because of large parameters in vgg, you may switch your batch size to 128 or even smaller.

'''


arch = {
    "A": [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
    "B": [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
    "D": [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
    "E": [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)]
}



def vgg_blk(in_channel, out_channel, num_conv):
    blk = []
    for i in range(num_conv):
        blk.append(nn.Sequential(nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, padding = 1),
                                 nn.BatchNorm2d(out_channel), nn.ReLU()))
        in_channel = out_channel
        if i == num_conv - 1:
            blk.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    return nn.Sequential(*blk)
    
def make_layer(arch):
    feature = []
    in_channel = 1
    for out_channel, num_conv in arch:
        feature.append(vgg_blk(in_channel, out_channel, num_conv))
        in_channel = out_channel
    return nn.Sequential(*feature)

class vgg(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.feature = make_layer(arch)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(7 * 7 * 512, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p = 0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p = 0.5),
                                        nn.Linear(4096, 10))

    def forward(self, X):
        X = self.feature(X)
        return (self.classifier(X))

def get_vgg11(arch = arch):
    return vgg(arch['A'])

def get_vgg13(arch = arch):
    return vgg(arch['B'])

def get_vgg16(arch = arch):
    return vgg(arch['D'])

def get_vgg19(arch = arch):
    return vgg(arch['E'])

