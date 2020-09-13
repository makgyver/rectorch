"""rectorch: state-of-the-art recsys approaches implemented in pytorch.
"""
import logging
import random
import torch
import numpy as np

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['StatefulObject', 'configuration', 'data', 'evaluation', 'metrics', 'models', 'utils',\
           'validation', 'set_seed']


class StatefulObject():
    """Stateful object.

    Any classes that inherited from this superclass must implements the :meth:`get_state` and
    :meth:`from_state`.
    """

    def get_state(self):
        r"""Return the state of the object as a dictionary.

        The state contains all useful information to construct a new object from scratch
        that is identical to the current object.
        """
        return {}

    @classmethod
    def from_state(cls, state):
        r"""Create a new object from the given state (i.e., dictionary).

        The state is a dictionary containing all the necesssary information to build a
        ``StatefulObject`` equivalent to the one that generated the state (thorugh the method
        :meth:`get_state`).

        Parameters
        ----------
        state : :obj:`dict`
            The object's state dictionary useful to replicate the saved stateful object.
        """
        return state


class Environment():
    r"""Rectorch environment class.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s]  %(message)s",
                            datefmt='%H:%M:%S-%d%m%y')

        self._logger = logging.getLogger(__name__)
        self._device = torch.device("cpu")
        self._default_env = True

    def init(self, device="cpu"):
        r"""Initialize the environment.

        Parameters
        ----------
        device : :obj:`str`, optional
            The pytorch device, by default 'cpu'.
        """
        self._set_device(device)
        self._default_env = False

    def _check_default(self):
        if self._default_env:
            out_str = ("You are using the default rectorch environment. "
                       "If you are aware of this then it is ok, however to remove this "
                       "warning call 'init()'. Otherwise, please check the "
                       "documentation for properly configure the rectorch environment.")
            self._logger.warning(out_str)

    def _check_for_cuda(self):
        if torch.cuda.is_available() and self._device.type == "cpu":
            out_str = "You have a CUDA device, so you should probably set the device to 'cuda'."
            self._logger.warning(out_str)

    def reset(self):
        r"""Reset the rectorch environment to the standard configuration.
        """
        self._default_env = True
        self._device = torch.device("cpu")

    def _get_device(self):
        self._check_default()
        return self._device

    def _set_device(self, device):
        old = self._device.type
        try:
            self._device = torch.device(device)
            self._check_for_cuda()
        except RuntimeError:
            self._logger.warning("Unsupported device type '%s'", device)
            self._device = torch.device(old)

    def _get_logger(self):
        return self._logger

    def is_default(self):
        r"""Get if the environment is the default one.
        """
        return self._default_env

    device = property(fget=_get_device, fset=_set_device)
    logger = property(fget=_get_logger)


env = Environment()

def set_seed(seed):
    """Set the ``rectorch`` random seed.

    Parameters
    ----------
    seed : :obj:`int`
        An integer random seed. If set to :obj:`None` the seed wont be changed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
