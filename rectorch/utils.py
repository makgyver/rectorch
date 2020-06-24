"""Module containing utility functions.
"""
from torch.optim import Adam, SGD, Adagrad, Adadelta, Adamax, AdamW

__all__ = ['init_optimizer']

def init_optimizer(params, opt_cfg=None):
    """Get a new optimizer initialize according to the given configurations.

    Parameters
    ----------
    params: iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    opt_cfg : :obj:`dict` or :obj:`None`, optional
        Dictionary containing the configuration for the optimizer, by default :obj:`None`.
        If :obj:`None` a default optimizer is returned, i.e., ``torch.optim.Adam(params)``.
    """
    if opt_cfg is None:
        return Adam(params)

    cfg = opt_cfg.copy()
    opt_name = cfg['name']
    del cfg['name']

    if opt_name == "adam":
        opt_cls = Adam
    elif opt_name == "adadelta":
        opt_cls = Adadelta
    elif opt_name == "adagrad":
        opt_cls = Adagrad
    elif opt_name == "adamw":
        opt_cls = AdamW
    elif opt_name == "adamax":
        opt_cls = Adamax
    elif opt_name == "sgd":
        opt_cls = SGD

    return opt_cls(params, **cfg)
