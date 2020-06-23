"""rectorch: state-of-the-art recsys approaches implemented in pytorch.
"""
__all__ = ['configuration', 'data', 'evaluation', 'metrics', 'models', 'nets', 'samplers']

import logging
import torch

class Environment():
    """Rectorch environment class
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s]  %(message)s",
                            datefmt='%H:%M:%S-%d%m%y')

        self._logger = logging.getLogger(__name__)
        self._device = torch.device("cpu")
        self._default_env = True

    def init(self, device="cpu"):
        """Initialize the environment.

        Parameters
        ----------
        device : :obj:`str`, optional
            The pytorch device, by default 'cpu'.
        """
        self._set_device(device)
        self._default_env = False

    def _check_default(self):
        if self._default_env:
            self._logger.warning("You are using the default rectorch environment.\
                                  If you are aware of this then it is ok, however to remove this\
                                  warning call 'init()'. Otherwise, please check the\
                                  documentation for properly configure the rectorch environment.")

    def _check_for_cuda(self):
        if torch.cuda.is_available() and self._device.type == "cpu":
            self._logger.warning("You have a CUDA device,\
                                  so you should probably set the device to 'cuda'.")

    def reset(self):
        """Reset the rectorch environment to the standard configuration.
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

    device = property(fget=_get_device, fset=_set_device)
    logger = property(fget=_get_logger)


env = Environment()
