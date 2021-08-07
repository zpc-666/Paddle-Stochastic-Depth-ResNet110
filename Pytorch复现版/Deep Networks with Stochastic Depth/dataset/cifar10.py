#coding: utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
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

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=None)

    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_set = MyDataset(train_set, data_transforms=train_transform)
    val_set = MyDataset(val_set, data_transforms=test_trainsform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=test_batch_size,
                                              shuffle=False, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_trainsform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)

    print('Finish loading! tran data length:{}, val data length:{}, test data length:{}'.format(len(train_set), len(val_set), len(test_set)))
    
    return train_loader, val_loader, test_loader
