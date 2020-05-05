"""The module contains useful classes to manage the configuration files.
"""
import json
from os.path import exists
from munch import DefaultMunch


class Singleton(type):
    """Define an Instance operation that lets clients access its unique instance.
    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class DataConfig(DefaultMunch):
    """Class containing the configurations for reading/writing the data set.

    Parameters
    ----------
    file_path : :obj:`str`
        The path to the data configuration ``.json`` file.

    Notes
    -----
    The data configuration file **must** have the following structure:
    ::

        {
            "data_path": path to the data set csv,
            "proc_path": path to the folder to save the pre-processed files,
            "seed": random seed,
            "threshold": cut-off value for implicit feedback,
            "separator": delimiter used in the csv,
            "u_min": min number of items for a user,
            "i_min": min number of users for an item,
            "heldout": heldout size (number of users) for validation and test set,
            "test_prop": test size proportion between 0 and 1,
            "topn": 1 iff the dataset should be pre-processed for implicit feedback
        }

    """

    def __init__(self, file_path):
        super(DataConfig, self).__init__(None, json.load(open(file_path, "r")))

    def __str__(self):
        return "DataConfig(" + ", ".join(["%s=%s" %(k, self[k]) for k in self]) + ")"

    def __repr__(self):
        return str(self)


class ModelConfig():
    """Class containing the configurations for creating, training and testing
    the model.

    Parameters
    ----------
    file_path : :obj:`str`
        The path to the model configuration ``.json`` file.

    Attributes
    ----------
    model : `DefaultMunch <lhttps://github.com/Infinidat/munch>`_
        Munch object containing the model's configurations according to the
        ``model`` key in the ``file_path`` json file.
    train : `DefaultMunch <lhttps://github.com/Infinidat/munch>`_
        Munch object containing the model training's configurations according
        to the ``train`` key in the ``file_path`` json file.
    test : `DefaultMunch <lhttps://github.com/Infinidat/munch>`_
        Munch object containing the test configurations according to the
        ``test`` key in the ``file_path`` json file.
    sampler : `DefaultMunch <lhttps://github.com/Infinidat/munch>`_
        Munch object containing the sampler configurations according to the
        ``model`` key in the ``file_path`` json file.

    Notes
    -----
    The model configuration file **must** have the following structure:
    ::

        {
            "model": {
                "model_param1" : value_m1,
                ...
            },
            "train": {
                "train_param1" : value_tr1,
                ...
            },
            "test":{
                "test_param1" : value_te1,
                ...
            },
            "sampler": {
                "sampler_param1" : value_s1,
                ...
            }
        }

    """

    def __init__(self, file_path):
        json_cfg = json.load(open(file_path, "r"))
        self.model = DefaultMunch(None, json_cfg["model"])
        self.train = DefaultMunch(None, json_cfg["train"])
        self.test = DefaultMunch(None, json_cfg["test"])
        self.sampler = DefaultMunch(None, json_cfg["sampler"])

    def __str__(self):
        return "ModelConfig(model={}, train={}, test={}, sampler={}".format(
            self.model, self.train, self.test, self.sampler)

    def __repr__(self):
        return str(self)


class ConfigManager(metaclass=Singleton):
    """Wrapper class for both the data and model configurations.

    Parameters
    ----------
    data_config_path : :obj:`str`
        The path to the data configuration ``.json`` file.
    model_config_path : :obj:`str`
        The path to the model configuration ``.json`` file.

    Attributes
    ----------
    data_config : :class:`DataConfig`
        Object containing the configurations for reading/writing the data set.
    model_config : :class:`ModelConfig`
        Object containing the configurations for creating, training and testing
        the model.

    Examples
    --------
    Initializing the :class:`ConfigManager` singleton:

    >>> from rectorch.configuration import ConfigManager
    >>> ConfigManager("path/to/the/dataconfig/file", "path/to/the/modelconfig/file")
    """

    @classmethod
    def get(cls):
        """Return the singleton ConfigManager instance.

        Returns
        -------
        :class:`ConfigManager`
            The singletion :class:`ConfigManager` instance.

        Raises
        ------
        Exception
            Raised when the singleton :class:`ConfigManager` object has not been
            previously created. To initialize the :class:`ConfigManager` simply call
            its constructor. Please, see **Examples**.
        """
        if cls._instance:
            return cls._instance
        else:
            raise Exception("Singleton object not instantiated!")

    def __init__(self, data_config_path, model_config_path):
        assert exists(data_config_path), "Data config file does not exist."
        assert exists(model_config_path), "Model config file does not exist."
        self.data_config = DataConfig(data_config_path)
        self.model_config = ModelConfig(model_config_path)
