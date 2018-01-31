from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import ProposeCNN
from models.xception import Xception
from datasets import PlantDatasetLoader
import time
from trainer import Trainer

def main():
    
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=False), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    kwargs = {'num_workers': 2, 'pin_memory': True}
    plant_dataset_loader = PlantDatasetLoader("bad condition", is_multilabel=False)
    for dataset in plant_dataset_loader.four_fold_cross_val():
        train_dataset, test_dataset = dataset
        break
    train_dataset.transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset.transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #model = nn.DataParallel(Xception())
    model = ProposeCNN()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            max_epochs=args.epochs,
            optimizer=optim.Adam(model.parameters(), lr=args.lr))

    trainer.run()

def save_model(path, model):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    main()
