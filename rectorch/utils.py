r"""Module containing utility functions.
"""
import os
import json
import rectorch
from torch.optim import Adam, SGD, Adagrad, Adadelta, Adamax, AdamW

__all__ = ['init_optimizer', 'get_data_cfg']

def init_optimizer(params, opt_cfg=None):
    r"""Get a new optimizer initialize according to the given configurations.

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

def get_data_cfg(ds_name=None):
    r"""Return standard data processing/splitting configuration.

    Parameters
    ----------
    ds_name : :obj:`str` [optional]
        The name of the dataset. Possible values are:
        * 'ml1m' : Movielens 100k;
        * 'ml1m' : Movielens 1 million;
        * 'ml20m': Movielens 20 million;
        * 'msd' : Million Song Dataset;
        * 'netflix' : Netflix Challenge Dataset.

        .. warning:: The Netflix dataset is assumed of being merged into a single CSV file named
           'ratings.csv'. In the `github homepage <https://github.com/makgyver/rectorch>`_
           of the framework it is provided a python notebook (``process_netflix.ipynb``) that
           performs such merging.

        If :obj:`None` the function returns a generic configuration with no thresholding and
        horizontal leave-one-out splitting. The name of the raw file is empty and must be set.
    Returns
    -------
    :obj:`dict`
        Dictionary containing the configurations.
    """
    p = os.path.dirname(rectorch.__file__) + "/"
    if ds_name == "ml100k":
        with open(p+'config/config_data_ml100k.json') as f:
            cfg = json.load(f)
    elif ds_name == "ml1m":
        with open(p+'config/config_data_ml1m.json') as f:
            cfg = json.load(f)
    elif ds_name == "ml20m":
        with open(p+'config/config_data_ml20m.json') as f:
            cfg = json.load(f)
    elif ds_name == "netflix":
        with open(p+'config/config_data_netflix.json') as f:
            cfg = json.load(f)
    elif ds_name == "msd":
        with open(p+'config/config_data_msd.json') as f:
            cfg = json.load(f)
    else:
        cfg = {
            "processing": {
                "data_path": None,
                "threshold": 0,
                "separator": ",",
                "header": None,
                "u_min": 0,
                "i_min": 0
            },
            "splitting": {
                "split_type": "horizontal",
                "sort_by": None,
                "seed": 42,
                "shuffle": False,
                "valid_size": 1,
                "test_size": 1,
                "test_prop": .5
            }
        }

    return cfg
