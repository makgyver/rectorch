"""Unit tests for the rectorch.utils module
"""
import os
import sys
import pytest
sys.path.insert(0, os.path.abspath('..'))

import torch
from torch.optim import Adam, SGD, Adagrad, Adadelta, Adamax, AdamW
from torch import nn
from rectorch.utils import init_optimizer, get_data_cfg

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        return x

def test_init_opt():
    """Tests the 'init_optimizer' function
    """
    net = Net()
    cfg = {
        "lr" : 0.001,
        "weight_decay" : 0.001,
    }
    cfg["name"] = "adam"

    params = net.parameters()
    opt = init_optimizer(params, cfg)
    assert isinstance(opt, Adam)

    params = net.parameters()
    cfg["name"] = "adamax"
    opt = init_optimizer(params, cfg)
    assert isinstance(opt, Adamax)

    params = net.parameters()
    cfg["name"] = "adamw"
    opt = init_optimizer(params, cfg)
    assert isinstance(opt, AdamW)

    params = net.parameters()
    cfg["name"] = "adagrad"
    opt = init_optimizer(params, cfg)
    assert isinstance(opt, Adagrad)

    params = net.parameters()
    cfg["name"] = "adadelta"
    opt = init_optimizer(params, cfg)
    assert isinstance(opt, Adadelta)

    params = net.parameters()
    cfg["name"] = "sgd"
    opt = init_optimizer(params, cfg)
    assert isinstance(opt, SGD)


def test_get_data_cfg():
    """Tests the 'get_data_cfg' function
    """
    cfg = get_data_cfg("ml100k")
    cfg["processing"]["data_path"] = "../rectorch/" + cfg["processing"]["data_path"]
    assert set(cfg.keys()) == set(["splitting", "processing"])
    assert "ml-100k" in cfg["processing"]["data_path"]
    assert cfg["splitting"]["valid_size"] == 100

    cfg = get_data_cfg("ml1m")
    cfg["processing"]["data_path"] = "../rectorch/" + cfg["processing"]["data_path"]
    assert set(cfg.keys()) == set(["splitting", "processing"])
    assert "ml-1m" in cfg["processing"]["data_path"]
    assert cfg["splitting"]["valid_size"] == 750

    cfg = get_data_cfg("ml20m")
    cfg["processing"]["data_path"] = "../rectorch/" + cfg["processing"]["data_path"]
    assert set(cfg.keys()) == set(["splitting", "processing"])
    assert "ml-20m" in cfg["processing"]["data_path"]
    assert cfg["splitting"]["valid_size"] == 10000

    cfg = get_data_cfg("msd")
    cfg["processing"]["data_path"] = "../rectorch/" + cfg["processing"]["data_path"]
    assert set(cfg.keys()) == set(["splitting", "processing"])
    assert "msd" in cfg["processing"]["data_path"]
    assert cfg["splitting"]["valid_size"] == 50000

    cfg = get_data_cfg("netflix")
    cfg["processing"]["data_path"] = "../rectorch/" + cfg["processing"]["data_path"]
    assert set(cfg.keys()) == set(["splitting", "processing"])
    assert "netflix" in cfg["processing"]["data_path"]
    assert cfg["splitting"]["valid_size"] == 40000
