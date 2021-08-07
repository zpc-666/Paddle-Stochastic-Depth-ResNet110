# coding: utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import time
import numpy as np
import os
import random
from paddle.distribution import Distribution

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = paddle.equal(pred, paddle.reshape(target, (1, -1)).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct.numpy()[:k].reshape(-1).astype('float32').sum(0, keepdims=True)
        res.append(correct_k*100.0 / batch_size)
    return res

def seed_paddle(seed=1024):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def train_per_epoch(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_interval = len(train_loader)//3
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input
        target = target
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.clear_grad()
        loss.backward()

        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.shape[0])
        top1.update(prec1.item(), input.shape[0])
        top5.update(prec5.item(), input.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_interval == 0:
            writer.add_scalar('loss', losses.val,
                              i + epoch * len(train_loader))
            writer.add_scalar('acc', top1.val, i + epoch * len(train_loader))
            writer.add_scalar('top5-acc', top5.val,
                              i + epoch * len(train_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_interval = len(val_loader)//3
    # switch to evaluate mode
    model.eval()

    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input
            target = target
            # compute output
            end = time.time()
            output = model(input)
            batch_time.update(time.time() - end)

            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.shape[0])
            top1.update(prec1.item(), input.shape[0])
            top5.update(prec5.item(), input.shape[0])

            if i % print_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

        print(
            ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {batch_time.avg:.3f}'
            .format(top1=top1, top5=top5, batch_time=batch_time))

    return top1.avg, top5.avg

class Bernoulli(Distribution):
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`.

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::
        >>> m = Bernoulli(paddle.to_tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (float, Tensor): the probability of sampling `1`
    """
    def __init__(self, probs=None):
        if not isinstance(probs, (float, paddle.Tensor)):
            assert TypeError("type of probs must be float or paddle.Tensor")


        if isinstance(probs, float):
            probs = paddle.to_tensor(probs)

        if str(probs.dtype) not in ['paddle.float32', 'paddle.float64']:
            assert TypeError('dtype of probs must be paddle.float32 or paddle.float64')

        assert paddle.all(paddle.greater_equal(probs, paddle.to_tensor(0.)) and paddle.less_equal(probs, paddle.to_tensor(1.))), \
                'Error, probs must be  greater than or equal to 0 and equal to or less than 1.'
        
        self.probs = probs

    def sample(self, sample_shape=None):
        if sample_shape is not None:
            shape = sample_shape+self.probs.shape
            new_probs = paddle.expand(self.probs, shape)
        else:
            new_probs = self.probs

        with paddle.no_grad():
            return paddle.bernoulli(new_probs)