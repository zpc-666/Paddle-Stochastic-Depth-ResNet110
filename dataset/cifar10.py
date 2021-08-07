#coding: utf-8
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
import paddle.vision.transforms as transforms
from paddle.io import Dataset

class MyDataset(Dataset):
    def __init__(self, datasets, data_transforms=None):
        self.datasets = datasets
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        img, label = self.datasets[idx]
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label

    def __len__(self):
        return len(self.datasets)

def load_data(root, train_batch_size, test_batch_size, train_size=45000, val_size=5000):
    print('Loading data...')
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_trainsform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = paddle.vision.datasets.Cifar10(data_file=root, mode='train', download=True, transform=None, backend='pil')
    # 这里还要进一步划分为训练集45000，验证集5000（论文设置），所以之后再用MyDataset封装和数据增强
    train_set, val_set = paddle.io.random_split(train_set, [train_size, val_size])

    train_set = MyDataset(train_set, data_transforms=train_transform)
    val_set = MyDataset(val_set, data_transforms=test_trainsform)

    # 不设置places=paddle.CPUPlace()有时出现莫名的Segmentation fault错误
    # 设置num_workers>0，后期共享内存会爆，就干脆这样了，稍慢些
    train_loader = paddle.io.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, num_workers=0, places=paddle.CPUPlace())

    val_loader = paddle.io.DataLoader(val_set, batch_size=test_batch_size,
                                              shuffle=False, num_workers=0, places=paddle.CPUPlace())

    test_set = paddle.vision.datasets.Cifar10(data_file=root, mode='test', download=True, transform=test_trainsform, backend='cv2')

    test_loader = paddle.io.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=0, places=paddle.CPUPlace())

    print('Finish loading! tran data length:{}, val data length:{}, test data length:{}'.format(len(train_set), len(val_set), len(test_set)))
    
    return train_loader, val_loader, test_loader
