from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils import data
import argparse
from utils import get_model, show_table, make_cm_plot, accuracy, class_acc, Accumulator, get_label
import cv2
from rich.progress import Progress
import numpy as np 



def load_Mydataset(args):
    transform = transforms.Compose([transforms.Resize(size = (args.resize, args.resize)),
                                transforms.Grayscale(num_output_channels = 1),
                                transforms.ToTensor()])
    MyDataset = ImageFolder(args.path_dataset, transform = transform)
    return data.DataLoader(MyDataset, batch_size = args.bs, shuffle = False, num_workers = args.num_workers)

def trans_image(image):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size = (args.resize, args.resize)),
                                transforms.Grayscale(num_output_channels = 1)])
    return(transform(image))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type = str, required = True, help = 'model option')
    parser.add_argument('-gpu', action = 'store_true', default = False, help = 'use gpu or not')
    parser.add_argument('-image', type = str, help = 'the image you want to predict')
    parser.add_argument('-all', action = 'store_true', default = False, help = 'Do prediction in whole datasets or not')
    parser.add_argument('-bs', type = int, default = 100, help = 'batch size test set')
    parser.add_argument('-resize' ,type = int, default = 224, help = 'resize the image to the given size')
    parser.add_argument('-num_workers', type = int, default = 4, help = 'number of workers for dataloader')
    parser.add_argument('-path', type = str, default = './runs', help = 'the path to save the results')
    parser.add_argument('-path_dataset', type = str, default = './MyDataset', help = 'the path to your datasets')
    parser.add_argument('-name', type = str ,default = 'predict.jpg', help = 'the file name to sava the fig result')
    args = parser.parse_args()

    net = get_model(args)

    net.load_state_dict(torch.load(f'{args.path}/{args.net}/last.pt'))

    net.eval()

    if args.net == 'mlp' or args.net == 'lenet':
        
        args.resize = 28
    
    if args.all:

        dataset = load_Mydataset(args)

        cm = torch.zeros(size = (10, 10))

        metric = Accumulator(2)

        with Progress() as progress:
            task = progress.add_task(f"[red]predicting", total = len(dataset))
            while not progress.finished:
                for X, y in dataset:
                    if args.gpu:
                        X, y = X.cuda(), y.cuda()
                    with torch.no_grad():
                        y_hat = net(X)
                        metric.add(accuracy(y_hat, y, cm, cm_cal = True), X.shape[0])
                    progress.update(task, advance = 1)

        class_wise_acc, cm_acc = class_acc(cm)

        print(show_table(class_wise_acc))

        print(f'\nAvg accuracy: {metric[0]/metric[1]*100:.2f}%\n')

        print('Confusion matrix:')

        make_cm_plot(args, cm_acc)

    else:

        image = cv2.imread(args.image)
        
        image = trans_image(image)

        image_show = image[0].numpy()

        if args.gpu:
            image.cuda()

        with torch.no_grad():
            predict = get_label(net(image).argmax())
        
        print(net(image).argmax())
        '''
        plt.imshow(image_show , vmin = 0, vmax = 1, cmap = 'gray')
        plt.xlabel(f'predict: {predict}')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    '''