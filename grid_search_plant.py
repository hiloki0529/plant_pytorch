from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from models import ProposeCNN
from models.xception import Xception
from models.multi_gpu_propose_cnn import MultiGpuProposeCNN
from trainer import Trainer
from sklearn.model_selection import StratifiedKFold
import extensions
from datasets.sub_dataset import SubDataset
from sklearn.model_selection import ParameterGrid

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='Momentum',
                    help='momentume (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
    
def optimize_params():
    params = {
            "lr": [2**-i for i in range(1, 4)],
            "momentum": [2**-i for i in range(1, 5)]
            }

    grid_params = ParameterGrid(params)
    errors = []
    for params in grid_params:
        error = train(params)
        print(f"{params}\n{error}")
        errors.append([params, error])
    errors = sorted(errors, key=lambda x: x[1])
    print(errors)
    print(errors[0])

def train(params):
    kwargs = {'num_workers': 32, 'pin_memory': True}

    train_dataset = datasets.ImageFolder('/home/agri/datasets_sai/leaves_datasets/12_class/')
    test_dataset = datasets.ImageFolder('/home/agri/datasets_sai/leaves_datasets/12_class/')
    y = list(map(lambda x: x[1], train_dataset.imgs))

    train_dataset.transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomRotation(i, resample=2, expand=False, center=None) for i in range(0, 360, 10)
        ]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    test_dataset.transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    for train_indecas, test_indecas in k_fold.split(train_dataset.imgs, y):
        train_X = SubDataset(train_dataset, train_indecas)
        test_X = SubDataset(test_dataset, test_indecas)

        #model = nn.DataParallel(Xception())
        #model = Xception()
    
        #model = MultiGpuProposeCNN()
        #model.features = nn.DataParallel(model.features)
    
        model = ProposeCNN(out=12, fc=512*4)
        #model = nn.DataParallel(model)
        
        torch.backends.cudnn.benchmark = True
        optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])
    
        trainer = Trainer(
                model=model,
                max_epochs=args.epochs,
                optimizer=optimizer,
                batch_size=args.batch_size,
                out=f"./results_plant/lr-{params['lr']}_momentum-{params['momentum']}")

        trainer.extend(scheduler(optimizer))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport([
            "epoch", 
            "iteration", 
            "train/loss", 
            "train/accuracy", 
            "validation/loss", 
            "validation/accuracy", 
            "elapsed_time", 
            "optim_lr", 
            "optim_momentum"]))

        trainer.fit(train_X, val_X=test_X, **kwargs)
        return 1 - trainer.acc(test_X, **kwargs)

def scheduler(optimizer):
    _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    def step(trainer):
        _scheduler.step(1 - trainer.observation["validation/accuracy"])
    return step

def save_model(path, model):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    optimize_params()
