'''

This is for some basic and simple models, including MLP, Lenet and ALexNet

'''

from torch import nn

'''

    a small mlp, only one hidden layer is used.
    dropout with p = 0.5 here
    if you are using MLP model, please set weight_decay(wd) = 0
    for this model, no need to resize the image to 224 x 224

'''
class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p = 0.5),
                                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p = 0.5),
                                nn.Linear(64,10))

    def forward(self, X):
        return(self.net(X))


'''

    lenet, batch normalization is used here after each convolution and before activation,
    we use relu for activation instead of sigmoid (original structure use sigmoid)
    dropout is also used with p = 0.5
    for this model, no need to resize the image to 224 x 224

'''

class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 6, kernel_size = 5, padding = 2)
                                ,nn.BatchNorm2d(6), nn.ReLU(), nn.AvgPool2d(kernel_size = 2, stride = 2),
                                nn.Conv2d(6, 16, kernel_size = 5),
                                nn.BatchNorm2d(16), nn.ReLU(), nn.AvgPool2d(kernel_size = 2, stride = 2),
                                nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.Sigmoid(),
                                nn.Dropout(p=0.5),
                                nn.Linear(120, 84),
                                nn.Sigmoid(),
                                nn.Dropout(p=0.5),   
                                nn.Linear(84,10),                                             
                                )

    def forward(self, X):
        return self.net(X)



'''

    ALexNet. for this model, you have to resize the image to 224 x 224, 
    same manipulation should be done in other models like resnet, vgg and so on.
    add batch normalization

'''

class aLexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding = 2), 
                                nn.BatchNorm2d(96), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2),
                                nn.Conv2d(96, 256, kernel_size=5, padding=2), 
                                nn.BatchNorm2d(256), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2),
                                nn.Conv2d(256, 384, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(384), nn.ReLU(),
                                nn.Conv2d(384, 384, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(384), nn.ReLU(),
                                nn.Conv2d(384, 256, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(256), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=5, stride=2),
                                nn.Flatten(),
                                nn.Linear(6400, 4096), nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, 4096), nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, 10))                                                                        
    def forward(self, X):
        return self.net(X)



def get_mlp():
    return mlp()

def get_lenet():
    return lenet()

def get_alexnet():
    return aLexnet()
