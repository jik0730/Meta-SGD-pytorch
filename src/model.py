import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_FILTERS = 64  # number of filters used in conv_block
K_SIZE = 3  # size of kernel
MP_SIZE = 2  # size of max pooling
EPS = 1e-8  # epsilon for numerical stability


class MetaLearner(nn.Module):
    """
    The class defines meta-learner for Meta-SGD algorithm.
    Training details will be written in train.py.
    TODO base-model invariant MetaLearner class
    """

    def __init__(self, params):
        super(MetaLearner, self).__init__()
        self.params = params
        self.meta_learner = Net(
            params.in_channels, params.num_classes, dataset=params.dataset)
        # Defined for Meta-SGD
        # TODO do we need strictly positive task_lr?
        self.task_lr = OrderedDict()

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X)
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def cloned_state_dict(self):
        """
        Only returns state_dict of meta_learner (not task_lr)
        """
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

    def define_task_lr_params(self):
        for key, val in self.named_parameters():
            # self.task_lr[key] = 1e-3 * torch.ones_like(val, requires_grad=True)
            self.task_lr[key] = nn.Parameter(
                1e-3 * torch.ones_like(val, requires_grad=True))


class Net(nn.Module):
    """
    The base CNN model for MAML (Meta-SGD) for few-shot learning.
    The architecture is same as of the embedding in MatchingNet.
    """

    def __init__(self, in_channels, num_classes, dataset='Omniglot'):
        """
        self.net returns:
            [N, 64, 1, 1] for Omniglot (28x28)
            [N, 64, 5, 5] for miniImageNet (84x84)
        self.fc returns:
            [N, num_classes]
        
        Args:
            in_channels: number of input channels feeding into first conv_block
            num_classes: number of classes for the task
            dataset: for the measure of input units for self.fc, caused by 
                     difference of input size of 'Omniglot' and 'ImageNet'
        """
        super(Net, self).__init__()
        self.features = nn.Sequential(
            conv_block(0, in_channels, padding=1, pooling=True),
            conv_block(1, N_FILTERS, padding=1, pooling=True),
            conv_block(2, N_FILTERS, padding=1, pooling=True),
            conv_block(3, N_FILTERS, padding=1, pooling=True))
        if dataset == 'Omniglot':
            self.add_module('fc', nn.Linear(64, num_classes))
        elif dataset == 'ImageNet':
            self.add_module('fc', nn.Linear(64 * 5 * 5, num_classes))
        else:
            raise Exception("I don't know your dataset")

    def forward(self, X, params=None):
        """
        Args:
            X: [N, in_channels, W, H]
            params: a state_dict()
        Returns:
            out: [N, num_classes] unnormalized score for each class
        """
        if params == None:
            out = self.features(X)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            """
            The architecure of functionals is the same as `self`.
            """
            out = F.conv2d(
                X,
                params['meta_learner.features.0.conv0.weight'],
                params['meta_learner.features.0.conv0.bias'],
                padding=1)
            # NOTE we do not need to care about running_mean anv var since
            # momentum=1.
            out = F.batch_norm(
                out,
                params['meta_learner.features.0.bn0.running_mean'],
                params['meta_learner.features.0.bn0.running_var'],
                params['meta_learner.features.0.bn0.weight'],
                params['meta_learner.features.0.bn0.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.1.conv1.weight'],
                params['meta_learner.features.1.conv1.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.1.bn1.running_mean'],
                params['meta_learner.features.1.bn1.running_var'],
                params['meta_learner.features.1.bn1.weight'],
                params['meta_learner.features.1.bn1.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.2.conv2.weight'],
                params['meta_learner.features.2.conv2.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.2.bn2.running_mean'],
                params['meta_learner.features.2.bn2.running_var'],
                params['meta_learner.features.2.bn2.weight'],
                params['meta_learner.features.2.bn2.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.3.conv3.weight'],
                params['meta_learner.features.3.conv3.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.3.bn3.running_mean'],
                params['meta_learner.features.3.bn3.running_var'],
                params['meta_learner.features.3.bn3.weight'],
                params['meta_learner.features.3.bn3.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['meta_learner.fc.weight'],
                           params['meta_learner.fc.bias'])

        out = F.log_softmax(out, dim=1)
        return out


def conv_block(index,
               in_channels,
               out_channels=N_FILTERS,
               padding=0,
               pooling=True):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True)),
                ('pool'+str(index), nn.MaxPool2d(MP_SIZE))
            ]))
    else:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True))
            ]))
    return conv


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# Maintain all metrics required in this dictionary.
# These are used in the training and evaluation loops.
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}