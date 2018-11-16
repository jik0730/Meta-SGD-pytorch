import unittest
import os
import torch
import torch.nn as nn
from src.model import MetaLearner
import utils


class test_meta_learner(unittest.TestCase):
    def setUp(self):
        # Configurations 3-way 3-shot with 3 query set
        model_dir = 'experiments/base_model'
        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(
            json_path), "No json configuration file found at {}".format(
                json_path)
        params = utils.Params(json_path)
        params.in_channels = 3
        params.num_classes = 5
        params.dataset = 'ImageNet'
        params.cuda = True

        # Data setting
        N = 5
        self.X = torch.ones([N, params.in_channels, 84, 84])
        self.Y = torch.randint(params.num_classes, (N, ), dtype=torch.long)

        # Optim & loss setting
        if params.cuda:
            self.model = MetaLearner(params).cuda()
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
        else:
            self.model = MetaLearner(params)

        self.model.define_task_lr_params()
        model_params = list(self.model.parameters()) + list(
            self.model.task_lr.values())
        self.optim = torch.optim.SGD(model_params, lr=1e-3)
        self.loss_fn = nn.NLLLoss()

    def test_params(self):
        for key, val in self.model.state_dict().items():
            print(key)
        for key, val in self.model.task_lr.items():
            print(key, val.requires_grad)

    def test_grad_check(self):
        # Update the model once with data
        stored_params = {
            key: val.clone()
            for key, val in self.model.named_parameters()
        }
        task_lr_params = {
            key: val.clone()
            for key, val in self.model.task_lr.items()
        }
        stored_params.update(task_lr_params)

        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Test grad check
        for key, val in self.model.named_parameters():
            self.assertTrue((val != stored_params[key]).any())
        # TODO
        # for key, val in self.model.task_lr.items():
        #     self.assertTrue((val != stored_params[key]).any())


if __name__ == '__main__':
    unittest.main()