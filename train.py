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

import os
import paddle
import paddle.nn as nn
import importlib
from visualdl import LogWriter
import numpy as np
import pickle
from models import utils
from config import parser_args

def train_model(args):
    if args.dataset=='cifar10':
        root = os.path.join(args.data_dir, args.dataset, 'cifar-10-python.tar.gz')
    print(args)
    model = importlib.import_module('models.__init__').__dict__[args.net](
        None, drop_path_rate=args.drop_path_rate, use_drop_path=args.use_drop_path, use_official_implement=args.use_official_implement)

    train_loader, val_loader, test_loader = importlib.import_module(
        'dataset.' + args.dataset).__dict__['load_data'](root, args.train_batch_size,
                                                         args.test_batch_size, has_val_dataset=args.has_val_dataset)

    writer = LogWriter(logdir=args.save_dir)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=args.learning_rate, milestones=args.milestones, gamma=args.gamma)
        optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                            learning_rate=lr_scheduler,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay,
                                            use_nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
                                        learning_rate=args.learning_rate,
                                        weight_decay=args.weight_decay)
    else:
        raise ValueError("optimizer must be sgd or adam.")

    best_acc = 0
    for i in range(args.epochs):
        utils.train_per_epoch(train_loader, model, criterion, optimizer, i, writer)

        top1_acc, top5_acc = utils.validate(val_loader, model, criterion)

        if args.optimizer == 'sgd':
            lr_scheduler.step()

        if best_acc < top1_acc:
            paddle.save(model.state_dict(),
                        args.save_dir + '/model_best.pdparams')
            best_acc = top1_acc
        if not args.save_best:
            if (i + 1) % args.save_interval == 0 and i != 0:
                paddle.save(model.state_dict(),
                           args.save_dir + '/model.pdparams')

        writer.add_scalar('val-acc', top1_acc, i)
        writer.add_scalar('val-top5-acc', top5_acc, i)
        writer.add_scalar('lr', optimizer.get_lr(), i)
    print('best acc: {:.2f}'.format(best_acc))
    
    model.set_state_dict(paddle.load(args.save_dir + '/model_best.pdparams'))
    top1_acc, top5_acc = utils.validate(test_loader, model, criterion)
    with open(os.path.join(args.save_dir, 'test_acc.txt'), 'w') as f:
        f.write('test_acc:'+str(top1_acc))

def train_hl_api(args):
    if args.dataset=='cifar10':
        root = os.path.join(args.data_dir, args.dataset, 'cifar-10-python.tar.gz')
    print(args)
    model = importlib.import_module('models.__init__').__dict__[args.net](
        None, drop_path_rate=args.drop_path_rate, use_drop_path=args.use_drop_path, use_official_implement=args.use_official_implement)

    train_loader, val_loader, test_loader = importlib.import_module(
        'dataset.' + args.dataset).__dict__['load_data'](root, args.train_batch_size,
                                                         args.test_batch_size, has_val_dataset=args.has_val_dataset)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        # 因为高层API是每个iter就执行lr_scheduler.step()，故这里把间隔调成m*len(train_loader)才合适
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=args.learning_rate, milestones=[m*len(train_loader) for m in args.milestones], gamma=args.gamma)
        optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                            learning_rate=lr_scheduler,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay,
                                            use_nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
                                        learning_rate=args.learning_rate,
                                        weight_decay=args.weight_decay)
    else:
        raise ValueError("optimizer must be sgd or adam.")
    
    model = paddle.Model(model)
    model.prepare(optimizer=optimizer, #指定优化器
                loss=criterion, #指定损失函数
                metrics=paddle.metric.Accuracy()) #指定评估方法
    #用于visualdl可视化
    visualdl = paddle.callbacks.VisualDL(log_dir=args.save_dir)
    #早停机制，这里使用只是为了在训练过程中保存验证集上的最佳模型，最后用于测试集验证
    early_stop = paddle.callbacks.EarlyStopping('acc', mode='max', patience=args.epochs, verbose=1,
                                            min_delta=0, baseline=None, save_best_model=True)

    model.fit(train_data=train_loader,            #训练数据集
            eval_data=val_loader,                 #验证数据集
            epochs=args.epochs,                 #迭代轮次
            save_dir=args.save_dir,             #把模型参数、优化器参数保存至自定义的文件夹
            save_freq=args.save_interval,       #设定每隔多少个epoch保存模型参数及优化器参数
            verbose=1,
            log_freq=20,
            eval_freq=args.eval_freq,
            callbacks=[visualdl, early_stop])

    #用验证集上最好模型在测试集上验证精度
    model.load(os.path.join(args.save_dir, 'best_model.pdparams'))
    result = model.evaluate(eval_data=test_loader, verbose=1)
    print('test acc:', result['acc'], 'test error:', 1-result['acc'])
if __name__ == '__main__':
    args = parser_args()
    utils.seed_paddle(args.seed)
    if not args.high_level_api:
        train_model(args)
    else:
        train_hl_api(args)