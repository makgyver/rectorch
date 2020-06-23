"""Unit tests for the rectorch module
"""
import sys
import os
import logging
import torch
sys.path.insert(0, os.path.abspath('..'))

import rectorch
from rectorch import Environment

def test_init():
    assert rectorch.env._default_env
    assert isinstance(rectorch.env, Environment)
    assert isinstance(rectorch.env.logger, logging.Logger)
    assert isinstance(rectorch.env.device, torch.device)
    rectorch.env.init()
    assert not rectorch.env._default_env
    assert rectorch.env.device.type == "cpu"
    rectorch.env.device = "cuda"
    assert rectorch.env.device.type == "cuda"
    rectorch.env.device = "pippo"
    assert rectorch.env.device.type == "cuda"
    rectorch.env.device = "cpu"
    assert rectorch.env.device.type == "cpu"
    rectorch.env.reset()
    assert rectorch.env._default_env
