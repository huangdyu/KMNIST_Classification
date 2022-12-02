import torchvision
from torch.utils import data
from torchvision import transforms
import torch
from torch import nn
from prettytable import PrettyTable
from matplotlib  import pyplot as plt
import pandas as pd
import numpy as np
import japanize_matplotlib #this is to show Japanese in confusion matrix(only for matplotlib)

##initial weigths of networks via kaiming normal
def kaiming_init(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'relu')

## select model

def get_model(args):
    if args.net == 'mlp':
        from models.basic import get_mlp
        args.resize = None
        net = get_mlp()

    elif args.net == 'lenet':
        from models.basic import get_lenet
        args.resize = None
        net = get_lenet()

    elif args.net == 'alexnet':
        from models.basic import get_alexnet
        net = get_alexnet()

    elif args.net == 'vgg11':
        from models.vgg import get_vgg11
        net = get_vgg11()
    
    elif args.net == 'vgg13':
        from models.vgg import get_vgg13
        net = get_vgg13()
    
    elif args.net == 'vgg16':
        from models.vgg import get_vgg16
        net = get_vgg16()
    
    elif args.net == 'vgg19':
        from models.vgg import get_vgg19
        net = get_vgg19()
    
    elif args.net == 'res18':
        from models.resnet import get_res18
        net = get_res18()
    
    elif args.net == 'res34':
        from models.resnet import get_res34
        net = get_res34()
    
    elif args.net == 'googlenet':
        from models.googlenet import get_googlenet
        net = get_googlenet()

    elif args.net == 'densenet121':
        from models.densenet import get_densenet121
        net = get_densenet121()
    
    elif args.net == 'densenet169':
        from models.densenet import get_densenet169
        net = get_densenet169()
    
    elif args.net == 'densenet201':
        from models.densenet import get_densenet201
        net = get_densenet201()
    
    elif args.net == 'densenet264':
        from models.densenet import get_densenet264
        net = get_densenet264()    
    
    else:
        print(f'No network named {args.net} exists here')


    net.apply(kaiming_init)

    if args.gpu:
        net = net.cuda()

    return net

#    load data for train and evaluate

def load_KMNIST_train(args):
    trans_train = [transforms.ToTensor()]
    '''

    using image augmentation, note for this dataset, it's not required to do horizontal flip or vertical flip
    since we may get a different word and it has no correlation with the original word
    but we can rotate it at a small degree

    '''
    
    if args.aug:

        trans_train.insert(0, transforms.RandomApply([transforms.RandomPerspective(distortion_scale = 0.2, p = 1)], p = 0.2))
        trans_train.insert(0, transforms.RandomApply([transforms.RandomRotation(degrees = 15)], p = 0.2))
        trans_train.insert(0, transforms.RandomApply([transforms.RandomEqualize()], p = 0.2))
   
    '''resize the image to the given size'''

    if args.resize:
        trans_train.insert(0, transforms.Resize(args.resize))   
    
    trans_train = transforms.Compose(trans_train)

    kmnist_train = torchvision.datasets.KMNIST(
                  root='./data', train=True, transform = trans_train, download = True)

    return data.DataLoader(kmnist_train, batch_size = args.bs, shuffle=True,
                            num_workers = args.num_workers)
  

def load_KMNIST_test(args):
    trans_test = [transforms.ToTensor()]
    if args.resize:
        trans_test.insert(0, transforms.Resize(args.resize))
    trans_test = transforms.Compose(trans_test)

    kmnist_test = torchvision.datasets.KMNIST(
                    root="./data", train = False, transform = trans_test, download = True)
    
    return data.DataLoader(kmnist_test, batch_size = args.bs, shuffle=False,
                            num_workers = args.num_workers)

## transfomr the numeric label into Japanese
'''

    the rules for numeric label correponds to its codepoint char are listed in 'kmnist_classmap.csv'

'''
def get_label(id):
    text = [u'\u304a', u'\u304d', u'\u3059', 
            u'\u3064', u'\u306a', u'\u306f', 
            u'\u307e', u'\u3084', u'\u308c', u'\u3092']
    return text[id]


## def an accumulator to calculate loss and accuracy in each epoch
class Accumulator():
    def __init__(self, n):
        self.data = [0.0] * n      #create a list with length n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, id):
        return self.data[id]

## this is to get the average accuracy for all of the classes classificatioj
def accuracy(y_hat, y, cm = None,  cm_cal = False):
    y_hat = y_hat.argmax(axis=1).type(y.dtype)
    #make sure y_hat has the same datatype with y, and output a bool array with length len(y)
    y_pre = (y_hat == y)
    """
    
    Only do testing, we report confusion matrix and class-wise accuracy

    """
    if cm_cal:
        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1
    return float(y_pre.type(y.dtype).sum())

def class_acc(cm):
    class_wise_acc = torch.zeros(size = (10,))
    cm_acc = torch.zeros_like(cm)
    for i in range(cm.shape[0]):
        class_wise_acc[i] = cm[i,i] / cm[i].sum()
        for j in range(cm.shape[0]):
            cm_acc[i,j] = cm[i,j]/cm[i].sum()
    return class_wise_acc, cm_acc

## show class - wise accuracy table
def show_table(class_wise_acc):
    table = PrettyTable(['label', 'word', 'accuracy'])
    for i in range(len(class_wise_acc)):
        table.add_row([i, get_label(i), f'{class_wise_acc[i]*100:.3f}%'])
    return table


## save the results
def make_df(df, train_l, train_acc, test_l, test_acc):
    df[0].append(format(train_l, '.4f'))
    df[1].append(format(train_acc, '.4f'))
    df[2].append(format(test_l, '.4f'))
    df[3].append(format(test_acc, '.4f'))


def make_plot(args):
    df = pd.read_csv(f'{args.path}/{args.net}/train.csv')
    plot = f'{args.path}/{args.net}/{args.name}'
    plt.plot(df['train loss'].to_numpy(), label = 'train loss', color = 'b', linestyle = '--', alpha = 0.5)
    plt.plot(df['train accuracy'].to_numpy(), label = 'train accuracy', color = 'r', linestyle = '--', alpha = 0.5)
    plt.plot(df['test loss'].to_numpy(), label = 'test loss', color = 'b')
    plt.plot(df['test accuracy'].to_numpy(), label = 'test accuracy', color = 'r')
    plt.xlabel('epoch')
    plt.xlim(0, df.shape[0])
    plt.legend()
    plt.savefig(plot)

def make_cm_plot(args, cm):
    table = PrettyTable(['word', u'\u304a', u'\u304d', u'\u3059', 
            u'\u3064', u'\u306a', u'\u306f', 
            u'\u307e', u'\u3084', u'\u308c', u'\u3092'])
    for i in range(cm.shape[0]):
        cm_i = []
        cm_i.append(get_label(i))
        for j in range(cm.shape[0]):    
            cm_i.append(f'{cm[i, j].numpy()*100:.2f}%')  
        table.add_row(cm_i)
    plt.imshow(cm, vmin = 0, vmax = 1, cmap = 'Blues')
    plt.xticks(np.arange(cm.shape[0]), labels = [get_label(i) for i in range(10)])
    plt.yticks(np.arange(cm.shape[0]), labels = [get_label(i) for i in range(10)])
    plt.colorbar()
    plt.savefig(f'{args.path}/{args.net}/{args.name}')
    print(table)