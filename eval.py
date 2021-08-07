#encoding=utf8
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
from models import utils
from config import parser_args

def evaluate(args):
    if args.dataset=='cifar10':
        root = os.path.join(args.data_dir, args.dataset, 'cifar-10-python.tar.gz')
    print(args)
    model = importlib.import_module('models.__init__').__dict__[args.net](
        args.checkpoint, drop_path_rate=args.drop_path_rate)

    train_loader, val_loader, test_loader = importlib.import_module(
        'dataset.' + args.dataset).__dict__['load_data'](root, args.train_batch_size,
                                                         args.test_batch_size)
    criterion = nn.CrossEntropyLoss()
    if not args.high_level_api:
        top1_acc, top5_acc = utils.validate(test_loader, model, criterion)
    else:
        model = paddle.Model(model)
        model.prepare(optimizer=None, #指定优化器
                    loss=criterion, #指定损失函数
                    metrics=paddle.metric.Accuracy(topk=(1, 5))) #指定评估方法
        result = model.evaluate(eval_data=test_loader, verbose=1)
        print('test acc:', result['acc_top1'], 'test error:', 1-result['acc_top1'])

if __name__ == '__main__':
    args = parser_args()
    evaluate(args)