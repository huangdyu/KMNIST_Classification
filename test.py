'''

This is for testing the model 

'''

import torch

import argparse

from utils import load_KMNIST_test, get_model, show_table, make_cm_plot, accuracy, class_acc, Accumulator



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type = str, required = True, help = 'model option')
    parser.add_argument('-gpu', action = 'store_true', default = False, help = 'use gpu or not')
    parser.add_argument('-bs', type = int, default = 200, help = 'batch size test set')
    parser.add_argument('-resize' ,type = int, default = 224, help = 'resize the image to the given size')
    parser.add_argument('-num_workers', type = int, default = 4, help = 'number of workers for dataloader')
    parser.add_argument('-path', type = str, default = './runs', help = 'the path to save the results')
    parser.add_argument('-name', type = str, default = 'test.jpg', help = 'the file name to save the fig results')
    args = parser.parse_args()

    net = get_model(args)

    net.load_state_dict(torch.load(f'{args.path}/{args.net}/last.pt'))

    test_data = load_KMNIST_test(args)

    cm = torch.zeros(size = (10, 10))

    metric = Accumulator(2)

    print('--------begin testing--------\n')

    net.eval()

    with torch.no_grad():
        for iter, (X, y) in zip(range(len(test_data)), test_data):
            if args.gpu:
                X, y = X.cuda(), y.cuda()
            y_hat = net(X)
            metric.add(accuracy(y_hat, y, cm, cm_cal = True), X.shape[0])
            if not (iter+1) % 5 and iter != len(test_data) - 1:
                print(f'iter{iter + 1}:   {(iter+1) * args.bs}/10000')

    print('\n--------end testing----------\n')

    class_wise_acc, cm_acc = class_acc(cm)

    print(show_table(class_wise_acc))

    print(f'\nAvg accuracy: {metric[0]/metric[1]*100:.2f}%\n')

    print('Confusion matrix:')

    make_cm_plot(args, cm_acc)

    

    