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

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parser_args():
    parser = argparse.ArgumentParser(description='Weight Decay Experiments')
    parser.add_argument('--dataset',
                        dest='dataset',
                        help='training dataset',
                        default='cifar10',
                        type=str)
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        help='loading dataset dir',
                        default='data',
                        type=str)
    parser.add_argument('--net',
                        dest='net',
                        help='training network',
                        default='resnet110',
                        type=str)
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        help='saving data dir',
                        default='output',
                        type=str)
    parser.add_argument('--save_best',
                        dest='save_best',
                        help='whether only save best model',
                        default=False,
                        type=str2bool)
    parser.add_argument('--save_interval',
                        dest='save_interval',
                        help='save interval',
                        default=10,
                        type=int)
    parser.add_argument('--eval_freq',
                        dest='eval_freq',
                        help='eval freq(high level api valid!)',
                        default=1,
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
    parser.add_argument('--nesterov',
                        dest='nesterov',
                        help='whether use Nesterov momentum',
                        default=True,
                        type=str2bool)
    parser.add_argument('--epochs',
                        dest='epochs',
                        help='epochs',
                        default=500,
                        type=int)
    parser.add_argument('--milestones',
                        dest='milestones',
                        help='Decrease learning rate',
                        default=[250, 375],
                        type=int,
                        nargs='+')
    parser.add_argument('--gamma',
                        dest='gamma',
                        help='gamma',
                        default=0.1,
                        type=float)
    parser.add_argument('--drop_path_rate',
                        dest='drop_path_rate',
                        help='drop path rate = 1-survival rate',
                        default=0.5,
                        type=float)
    parser.add_argument('--mode',
                        dest='mode',
                        help="train model or only eval model, mode in ['train', 'eval']",
                        default='train',
                        type=str)
    parser.add_argument('--high_level_api',
                        dest='high_level_api',
                        help='whether use high level api to train or eval model.',
                        default=False,
                        type=str2bool)
    parser.add_argument('--checkpoint',
                        dest='checkpoint',
                        help='whether use checkpoint.',
                        default=None,
                        type=str)
    parser.add_argument('--has_val_dataset',
                        dest='has_val_dataset',
                        help='whether dataset is split to train 45000 , val 5000 and test 10000.',
                        default=True,
                        type=str2bool)
    parser.add_argument('--use_drop_path',
                        dest='use_drop_path',
                        help='whether use Stochastic Depth.',
                        default=True,
                        type=str2bool)
    parser.add_argument('--use_official_implement',
                        dest='use_official_implement',
                        help='whether use official implement.',
                        default=True,
                        type=str2bool)
    parser.add_argument('--seed',
                        dest='seed',
                        help='seed',
                        default=2021,
                        type=int)
    
    args = parser.parse_args()

    return args