# coding: utf-8

import os
import torch
import torch.nn as nn
import argparse
import importlib
from tensorboardX import SummaryWriter
import numpy as np
import pickle
import utils

parser = argparse.ArgumentParser(description='Weight Decay Experiments')
parser.add_argument('--dataset',
                    dest='dataset',
                    help='training dataset',
                    default='cifar10',
                    type=str)
parser.add_argument('--net',
                    dest='net',
                    help='training network',
                    default='resnet110',
                    type=str)
parser.add_argument('--save_dir',
                    dest='save_dir',
                    help='saving data dir',
                    default='tmp',
                    type=str)
parser.add_argument('--save_best',
                    dest='save_best',
                    help='whether only save best model',
                    default=False,
                    type=bool)
parser.add_argument('--save_interval',
                    dest='save_interval',
                    help='save interval',
                    default=10,
                    type=int)
parser.add_argument('--train_batch_size',
                    dest='train_batch_size',
                    help='training batch size',
                    default=128,
                    type=int)
parser.add_argument('--test_batch_size',
                    dest='test_batch_size',
                    help='test batch size',
                    default=128,
                    type=int)
parser.add_argument('--optimizer',
                    dest='optimizer',
                    help='optimizer',
                    default='sgd',
                    type=str)
parser.add_argument('--learning_rate',
                    dest='learning_rate',
                    help='learning rate',
                    default=0.1,
                    type=float)
parser.add_argument('--momentum',
                    dest='momentum',
                    help='momentum',
                    default=0.9,
                    type=float)
parser.add_argument('--weight_decay',
                    dest='weight_decay',
                    help='weight decay',
                    default=1e-4,
                    type=float)
parser.add_argument('--dampening',
                    dest='dampening',
                    help='inhibitory factor of momentum',
                    default=0.,
                    type=float)
parser.add_argument('--nesterov',
                    dest='nesterov',
                    help='whether use Nesterov momentum',
                    default=True,
                    type=bool)
parser.add_argument('--epochs',
                    dest='epochs',
                    help='epochs',
                    default=500,
                    type=int)
parser.add_argument('--schedule',
                    dest='schedule',
                    help='Decrease learning rate',
                    default=[250, 375],
                    type=int,
                    nargs='+')
parser.add_argument('--gamma',
                    dest='gamma',
                    help='gamma',
                    default=0.1,
                    type=float)

args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    model = importlib.import_module('models.__init__').__dict__[args.net](
        False, None)

    train_loader, val_loader, test_loader = importlib.import_module(
        'dataset.' + args.dataset).__dict__['load_data']('/148Dataset/data-zhai.pucheng', args.train_batch_size,
                                                         args.test_batch_size)

    writer = SummaryWriter(args.save_dir)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        print("Cuda is available:", torch.cuda.is_available())
        model = model.cuda()
        criterion = criterion.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad,
                                           model.parameters()),
                                    args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    dampening=args.dampening,
                                    nesterov=args.nesterov)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule, gamma=args.gamma)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda i: i.requires_grad,
                                            model.parameters()),
                                     args.learning_rate,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = None

    best_acc = 0
    for i in range(args.epochs):
        utils.train(train_loader, model, criterion, optimizer, i, writer)

        top1_acc, top5_acc = utils.validate(val_loader, model, criterion)

        if args.optimizer == 'sgd':
            lr_scheduler.step()

        if best_acc < top1_acc:
            torch.save(model.state_dict(),
                        args.save_dir + '/model_best.pth')
            best_acc = top1_acc
        if not args.save_best:
            if (i + 1) % args.save_interval == 0 and i != 0:
                torch.save(model.state_dict(),
                           args.save_dir + '/model.pth')

        writer.add_scalar('val-acc', top1_acc, i)
        writer.add_scalar('val-top5-acc', top5_acc, i)
        writer.add_scalar('lr', lr_scheduler.get_lr()[0], i)
    print('best acc: {:.2f}'.format(best_acc))

    model.load_state_dict(torch.load(args.save_dir + '/model_best.pth'))
    top1_acc, top5_acc = utils.validate(test_loader, model, criterion)
    with open('test_acc.txt', 'w') as f:
        f.write('test_acc:'+str(top1_acc))
