import torch
from torch import nn
import argparse
from utils import (load_KMNIST_train, load_KMNIST_test, get_model, Accumulator, accuracy,
                    class_acc,show_table, make_df, make_plot)
from rich.progress import Progress
import pandas as pd
import os



def train_epoch(net, optimizer, train_data, epoch):
    net.train()
    metric = Accumulator(3)
    loss = nn.CrossEntropyLoss()
    num_iter = len(train_data)     #number of iterations in one epoch
    with Progress() as progress:
        task = progress.add_task(f"[red]epoch{epoch + 1}", total = num_iter)
        while not progress.finished:
            for X, y in train_data:
                if args.gpu:
                    X, y = X.cuda(), y.cuda()
                y_hat = net(X)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                progress.update(task, advance = 1)
    return metric[0] / metric[2], metric[1] / metric[2]         # avg loss and avg accuracy


## evaluate the model (average loss and average accuracy)
def eval_epoch(net, test_data):
    net.eval()
    metric = Accumulator(3)
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in test_data:
            if args.gpu:
                X, y = X.cuda(), y.cuda()
            y_hat = net(X)
            l = loss(y_hat, y) 
            metric.add(l * X.shape[0], accuracy(y_hat, y, cm, cm_cal = True), X.shape[0])
    return metric[0]/metric[2], metric[1]/metric[2]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type = str, required = True, help = 'model option')
    parser.add_argument('-gpu', action = 'store_true', default = False ,help = 'use gpu or not')
    parser.add_argument('-bs', type = int, default = 256, help = 'batch size for train and validation set')
    parser.add_argument('-Epoch', type = int, default = 50, help = 'number of epochs for training')
    parser.add_argument('-lr', type = float, default = 0.01, help = 'initial learning rate')
    parser.add_argument('-wd', type = float, default = 5e-4, help = 'weight_decay')
    parser.add_argument('-momentum', type = float, default = 0.9, help = 'momentum')
    parser.add_argument('-resize' ,type = int, default = 224, help = 'resize the image to the given size')
    parser.add_argument('-num_workers', type = int, default = 4, help = 'number of workers for dataloader')
    parser.add_argument('-aug', action = 'store_true', default = True, help = 'use image augmentation or not')
    parser.add_argument('-path', type = str, default = './runs', help = 'the path to save the results')
    parser.add_argument('-name', type = str, default = 'train.jpg', help = 'the file name to save the fig result')
    args = parser.parse_args()



    net = get_model(args)

    train_data =load_KMNIST_train(args)
    test_data = load_KMNIST_test(args)

    optimizer = torch.optim.SGD(net.parameters(), weight_decay=args.wd, lr = args.lr, momentum = args.momentum)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10)

    df = [[] for i in range(4)]
    
    if not os.path.exists(f'{args.path}/{args.net}'):
        os.makedirs(f'{args.path}/{args.net}')

    ## training 
    if args.gpu:
        print(f'training on {torch.cuda.get_device_name()}')
    

    for epoch in range(args.Epoch):
        cm = torch.zeros(size = (10, 10))
        train_l, train_acc = train_epoch(net, optimizer, train_data, epoch)
        test_l, test_acc = eval_epoch(net, test_data)
        class_wise_acc = class_acc(cm)[0]
        print(f'------------------------')
        print('Train error:')
        print(f'Avg loss: {train_l:.5f}     Accuracy: {train_acc*100:.3f}%')
        print('Test error:')
        print(f'Avg loss: {test_l:.5f}      Accuracy: {100*test_acc:.3f}%')
        print(f'class-wise accuracy:')
        print(show_table(class_wise_acc))
        scheduler.step(test_acc)
        
        make_df(df, train_l, train_acc, test_l, test_acc)

    df = pd.DataFrame({'train loss': df[0], 'train accuracy': df[1], 
                'test loss': df[2], 'test accuracy': df[3]})

    df.to_csv(f'{args.path}/{args.net}/train.csv', index = False)

    make_plot(args)

    torch.save(net.state_dict(), f'{args.path}/{args.net}/last.pt')

    

