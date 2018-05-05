import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from time import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path

class Trainer():
    def __init__(self, model, max_epochs, optimizer, batch_size, out="./result", seed=1, **args):
        gpu = 0
        log_interval = 10
        self.seed = 1
        self.batch_size = batch_size
        self.device = torch.device(f"cuda" if gpu >= 0 or type(gpu) is list else "cpu")
        self.model = model.to(self.device)
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.extensions = {}
        self.out = out
        out_path = Path(out)
        if not out_path.exists():
            out_path.mkdir(parents=True)
        self.log_interval = log_interval
        if torch.cuda.is_available():
            self.model.cuda()
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        self.observation = {}

    def extend(self, extension, name=None, trigger=1):
        if name is None:
            if "__class__" in dir(extension):
                name = extension.__class__.__name__
            elif "__name__" in dir(extension):
                name = extension.__name__
            else:
                raise ValueError("{} is invailed".format(extension))
        self.extensions[name] = (extension, trigger)

    def get_extension(self, name):
        return self.extensions[name][0]

    def _train(self, epoch, train_loader):
        self.model.train()
        epoch_start_time = time()
        len_data = 0
        total_acc = 0
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            len_data += len(data)
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            total_acc += accuracy(output, target) * len(data)

            loss = F.nll_loss(output, target)
            total_loss += loss.item() * len(data)

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                process_speed = len_data / (time() - epoch_start_time)
                #self.print_log(epoch, batch_idx, output, target, total_loss/len_data, process_speed, total_acc/len_data)
        self.observation.update({"train/loss": total_loss/len(train_loader.sampler), 
                                 "train/accuracy": total_acc/len(train_loader.sampler),
                                 "iteration": epoch * batch_idx})

    def _test(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        correct = 0
        #total_pred = np.zeros((len(test_loader.sampler), ))
        with torch.no_grad():
            for iteration, (data, target) in enumerate(test_loader):
                data, target = (data.to(self.device), target.to(self.device))
                output = self.model(data)
                loss = F.nll_loss(output, target, size_average=False)
                test_loss += loss.item()
                test_acc += accuracy(output, target) * len(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                #total_pred[iteration*len(data): iteration*len(data)+len(data)] = output.max(1, keepdim=True)[1].cpu().numpy().ravel()

        test_loss /= len(test_loader.sampler)
        test_acc /= len(test_loader.sampler)
#        print("Test set: Average loss: {:.4f}, Accuracy {}/{} ({:.6f})".format(
#            test_loss,
#            correct,
#            len(self.test_loader.sampler),
#            test_acc))
        self.observation.update({"validation/loss": test_loss, "validation/accuracy": test_acc})
        return test_acc

    def optim_params(self):
        param_groups = self.optimizer.state_dict()["param_groups"][-1]
        return {f"optim_{key}": param_groups[key] for key in filter(lambda x: x != "params", param_groups.keys())}

    def fit(self, X, y=None, val_X=None, val_y=None, **kwargs):
        if isinstance(X, data.Dataset):
            train_loader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=True, **kwargs)
        if isinstance(val_X, data.Dataset):
            test_loader = data.DataLoader(val_X, batch_size=self.batch_size, shuffle=False, **kwargs)

        for epoch in range(1, self.max_epochs + 1):
            start_time = time()
            self.observation["epoch"] = epoch
            self.observation.update(self.optim_params())
            self._train(epoch, train_loader)
            if isinstance(val_X, data.Dataset):
                self._test(test_loader)
            elapsed_time = time() - start_time
            self.observation["elapsed_time"] = elapsed_time
            for (extension, trigger) in self.extensions.values():
                if epoch % trigger == 0:
                    extension(self)
            self.observation = {}
        self.save(self.out)
        return self

    def predict(self, X, **kwargs):
        test_loader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False, **kwargs)
        return self._test(test_loader)

    def acc(self, X, **kwargs):
        acc = self.predict(X, **kwargs)
        return acc

    def score(self, X, y, **args):
        pred = self.predict(X)
        return accuracy_score(y, pred)

    def get_params(self, deep=True):
        params = {"optimizer": self.optimizer,
                  "max_epochs": self.max_epochs,
                  "model": self.model,
                  "out": self.out,
                  "seed": self.seed,
                  "extensions": self.extensions}
        #params.update(self.optim_params())
        return params

    def save(self, path):
        path = Path(path)
        if not path.exists():
            path.mkdir()
        objects = {"model.save": self.model, "opt.sve": self.optimizer}
        for key, obj in objects.items():
            torch.save(obj.state_dict(), str(path/key))


def accuracy(y, t):
    pred = y.max(1, keepdim=True)[1]
    t = t.view_as(pred)
    correct = pred.eq(t.view_as(pred)).sum().item()
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
