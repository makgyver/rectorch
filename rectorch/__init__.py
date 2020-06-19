"""rectorch: State-of-the-art recsys approaches implemented in pytorch.
"""
__all__ = ['configuration', 'data', 'evaluation', 'metrics', 'models', 'nets', 'samplers']


import logging
import torch

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]  %(message)s",
                    datefmt='%H:%M:%S-%d%m%y')

_device = torch.device("cpu")
_default_env = True

def _check_default():
    if _default_env:
        LOGGER.warning("You are using the default rectorch environment. If you are aware of this\
                       then it is ok, however to remove this warning call'init_environment()'.\
                       Otherwise, please check the documentation for the proper rectorch\
                       environment configuration")

def _check_for_cuda():
    if torch.cuda.is_available() and _device.type == "cpu":
        LOGGER.warning("You have a CUDA device, so you should probably run with --cuda")

def _get_logger():
    return logging.getLogger(__name__)

def reset_environment():
    _default_env = True
    _device = torch.device("cpu")

def init_environment(device="cpu"):
    _check_for_cuda()
    _default_env = False
    _set_device(device)

def _get_device():
    _check_default()
    return _device

def _set_device(device):
    _device = torch.device(device)

def _del_device():
    _device = torch.device("cpu")


DEVICE = property(fget=_get_device, fset=_set_device, fdel=_del_device)
LOGGER = property(_get_logger)
