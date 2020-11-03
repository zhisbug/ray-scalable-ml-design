from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as torchmodels
from torchvision import datasets, transforms
from filelock import FileLock


class Worker(object):
    def __init__(self, model='resnet50'):
        self.model_type = model
        print("=> creating model '{}'".format(model))
        self.model = torchmodels.__dict__[model]().cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.train_loader = self.get_data_loader()

    def num_params(self):
        return len(self.get_weights())

    def params_distribution(self):
        distribution = []
        weights = self.get_weights()
        for k, v in weights.items():
            distribution.append(v.numel())
        return distribution

    def get_data_loader(self):
        """Safely downloads data. Returns training/validation set dataloader."""
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        with FileLock(os.path.expanduser("/tmp/data.lock")):
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            trainset = torchvision.datasets.CIFAR10(root='/tmp/', train=True,
                                                    download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                      shuffle=True, num_workers=2)

        return trainloader

    def compute_gradients(self, weights, batch_size=64):

        x, y = next(iter(self.train_loader))
        x = x.cuda()
        y = y.cuda()
        self.set_weights(weights)
        self.model.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        return self.get_gradients()

    def split_gradients(self, assignments):
        # assuming messages are gradients or parameters
        # this grad is ready to be called by apply_gradients in ParameterServer
        grad = self.get_gradients()
        num_shards = np.unique(np.array(assignments)).size
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(grad.items()):
            shards[assignments[i]][k] = v
        return shards

    def split_parameters(self, assignments):
        params = self.get_weights()
        num_shards = np.unique(np.array(assignments)).size
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(params.items()):
            shards[assignments[i]][k] = v
        return shards

    def stitch_parameters(self, split_params):
        # need to construct a weight dict
        stitch_param = dict()
        for i, param in enumerate(split_params):
            for k, v in param.items():
                stitch_param[k] = v
        return stitch_param

    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(weights[name])

    def get_weights(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def get_gradients(self):
        grad_dict = {}
        for name, p in self.model.named_parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grad_dict[name] = grad
        return grad_dict


class PS(object):
    def __init__(self, params):
        self.params = params
        self.optimizer = torch.optim.SGD(self.params.values(), lr=0.01)

    def get_params(self):
        return self.params

    def apply_updates(self, list_of_gradients):
        assert(len(list_of_gradients) >= 1)

        summed_gradient_dict = dict()
        for name in self.params:
            summed_gradient_dict[name] = \
                np.stack([grads[name] for grads in list_of_gradients]).sum(axis=0)

        self.optimizer.zero_grad()
        self.set_gradients(summed_gradient_dict)
        self.optimizer.step()

    def set_gradients(self, gradients):
        # gradients should be a stitched dict
        for name, p in self.get_params().items():
            if gradients[name] is not None:
                if p.grad is not None:
                    p.grad = torch.from_numpy(gradients[name]).to(p.grad.device)
                else:
                    p.grad = torch.from_numpy(gradients[name])

# test_worker = Worker()
# print(test_worker.num_params())
# # print(test_worker.params_distribution())
# weights = test_worker.get_weights()
# # print(weights)
# # test_worker.set_weights(weights)
# test_worker.compute_gradients(weights)
# gradients = test_worker.get_gradients()
# # print(gradients)
# server = PS(weights)
# print('reach here..')
# # print(server.get_params())
# # print(server.optimizer)
# # server.set_gradients(gradients)
#
# index = 0
# theta_1 = server.get_params()['conv1.weight'][0, 0, 0, index].data.cpu().numpy()
# grad = gradients['conv1.weight'][0,0,0,0]
# theta_2 = theta_1 - 0.01 * grad
#
# server.apply_updates([gradients])
# # print(server.get_params()['conv1.weight'][0, 0, 0, index].data.cpu().numpy())
# true = server.get_params()['conv1.weight'][0, 0, 0, index].data.cpu().numpy()
#
# print("{} vs. {}".format(true, theta_2))
# assert(np.allclose(true, theta_2))

def round_robin_sharding(worker):
    """Generate the assignment of variable to servers."""
    parameter_distribution = worker.params_distribution()

    assignments = len(parameter_distribution) * [0]
    loads = 2 * [0]
    for i, var_size in enumerate(parameter_distribution):
        min_ps_index = loads.index(min(loads))
        loads[min_ps_index] += var_size
        assignments[i] = min_ps_index

    print("Load of each ps {}".format(loads))
    return assignments
test_worker = Worker()
assignments = round_robin_sharding(test_worker)

shards = test_worker.split_parameters(assignments)
parameters = test_worker.stitch_parameters(shards)

# weights = test_worker.get_weights()
# test_worker.set_weights(weights)
# test_worker.compute_gradients(weights)
# grad_shards = test_worker.split_gradients(assignments)
print(123)

# server = PS()
