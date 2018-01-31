import torch
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
from pathlib import Path
import json

class Trainer:
    def __init__(self, model, max_epochs, optimizer, train_loader, test_loader, out="./result", log_interval=10, seed=1):
        self.model = model
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.out = out
        self.log_interval = log_interval
        self.logs = []
        self.log = {}
        if torch.cuda.is_available():
            self.model.cuda()
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)


    def train(self, epoch):
        self.model.train()
        epoch_start_time = time()
        len_data = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            len_data += len(data)
            if batch_idx % self.log_interval == 0:
                process_speed = len_data / (time() - epoch_start_time)
                len_data = 0
                epoch_start_time = time()
                self.print_log(epoch, batch_idx, output, target, loss, process_speed)
        self.log.update({"train/loss": loss.data[0], "train/accuracy": accuracy(output, target)})
        print()

    def print_log(self, epoch, batch_idx, output, target, loss, process_speed):
        done_data = batch_idx * len(target)
        estimated_end_sec = (len(self.train_loader.dataset) - done_data) / process_speed
        print("Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}  Accuracy: {:.6f}  Process speed: {:.6f} data/sec.  Estimated end time: {}".format(
            epoch,
            done_data,
            len(self.train_loader.dataset),
            100. * batch_idx / len(self.train_loader),
            loss.data[0],
            accuracy(output, target),
            process_speed,
            seconds_to_dateformat(estimated_end_sec)), end="\r")

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = self.to_cuda_if_available(data, target)
            data, target = self.to_variable(data, target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print("Test set: Average loss: {:.4f}, Accuracy {}/{} ({:.6f})".format(
            test_loss,
            correct,
            len(self.test_loader.dataset),
            accuracy(output, target)))
        self.log.update({"validation/loss": test_loss, "validation/accuracy": accuracy(output, target)})

    def run(self):
        for epoch in range(1, self.max_epochs + 1):
            start_time = time()
            self.train(epoch)
            self.save(self.out)
            self.test()
            elapsed_time = time() - start_time
            print("Elapsed time: {}\n".format(elapsed_time))
            self.log.update({"elapsed time": elapsed_time})
            self.logs.append(self.log)
            self.logging()

    def to_cuda_if_available(self, *xs):
        if torch.cuda.is_available():
            cudas = []
            for x in xs:
                cudas.append(x.cuda())
            return cudas
        else:
            return xs

    def to_variable(self, *xs):
        variables = []
        for x in xs:
            variables.append(Variable(x))
        return variables

    def save(self, path):
        path = Path(path)
        if not path.exists():
            path.mkdir()
        objects = {"model.save": self.model, "opt.sve": self.optimizer}
        for key, obj in objects.items():
            torch.save(obj.state_dict(), str(path/key))

    def logging(self):
        out_path = Path(self.out)
        with open(out_path/"log", "w") as f:
            json.dump(self.logs, f, indent=4)

def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    correct = pred.eq(t.data.view_as(pred)).cpu().sum()
    return correct / len(t.data)

def seconds_to_dateformat(seconds):
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24

    dateformat = {}
    if days > 0:
        dateformat["days"] = "{:.0f} days ".format(days)
    else:
        dateformat["days"] = ""
    dateformat["hours"] = hours - 24 * days
    dateformat["minutes"] = minutes - 60 * hours
    dateformat["seconds"] = seconds - 60 * minutes
    return "{days}{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}".format(**dateformat)
