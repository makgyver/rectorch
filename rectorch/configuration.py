r"""The module contains useful classes to manage the configuration files.

Configuration files are useful to correctly initialize the data processing and the
recommendation engines. Configuration files must be `.json <https://www.json.org/json-en.html>`_
files with a specific format. Details about the file formats are described in :ref:`config-format`.
"""
import json
from os.path import exists
from munch import DefaultMunch

__all__ = ['DataConfig', 'ModelConfig', 'ConfigManager']

class Singleton(type):
    r"""Define an Instance operation that lets clients access its unique instance.
    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class DataConfig(DefaultMunch):
    r"""Class containing the configurations for reading/writing the data set.

    Parameters
    ----------
    file_path : :obj:`str`
        The path to the data configuration `.json <https://www.json.org/json-en.html>`_ file.

    Notes
    -----
    The data configuration file **must** have the structure described in :ref:`config-format`.
    """

    def __init__(self, file_path):
        super(DataConfig, self).__init__(None, json.load(open(file_path, "r")))

    def __str__(self):
        return "DataConfig(" + ", ".join(["%s=%s" %(k, self[k]) for k in self]) + ")"

    def __repr__(self):
        return str(self)


class ModelConfig():
    r"""Class containing the configurations for creating, training and testing
    the model.

    Parameters
    ----------
    file_path : :obj:`str`
        The path to the model configuration `.json <https://www.json.org/json-en.html>`_ file.

    Attributes
    ----------
    model : `DefaultMunch <https://github.com/Infinidat/munch>`_
        Munch object containing the model's configurations according to the
        ``model`` key in the ``file_path`` json file.
    train : `DefaultMunch <https://github.com/Infinidat/munch>`_
        Munch object containing the model training's configurations according
        to the ``train`` key in the ``file_path`` json file.
    test : `DefaultMunch <https://github.com/Infinidat/munch>`_
        Munch object containing the test configurations according to the
        ``test`` key in the ``file_path`` json file.
    sampler : `DefaultMunch <https://github.com/Infinidat/munch>`_
        Munch object containing the sampler configurations according to the
        ``model`` key in the ``file_path`` json file.

    Notes
    -----
    The data configuration file **must** have the structure described in :ref:`config-format`.
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
    r"""Wrapper class for both the data and model configurations.

    Parameters
    ----------
    data_config_path : :obj:`str`
        The path to the data configuration `.json <https://www.json.org/json-en.html>`_ file.
    model_config_path : :obj:`str`
        The path to the model configuration `.json <https://www.json.org/json-en.html>`_ file.

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
    ConfigManager(data_config=DataConfig(...), model_config=ModelConfig(...))
    """

    @classmethod
    def get(cls):
        r"""Return the singleton ConfigManager instance.

        Returns
        -------
        :class:`ConfigManager`
            The singletion :class:`ConfigManager` instance.

        Raises
        ------
        :class:`Exception`
            Raised when the singleton :class:`ConfigManager` object has not been
            previously created. To initialize the :class:`ConfigManager` simply call
            its constructor. Please, see **Examples**.

        Examples
        --------
        >>> from rectorch.configuration import ConfigManager
        >>> man = ConfigManager.get()
        Exception: Singleton object not instantiated!

        The :class:`ConfigManager` singleton object must be initialized to get it.

        >>> ConfigManager("path/to/the/dataconfig/file", "path/to/the/modelconfig/file")
        ConfigManager(data_config=DataConfig(...), model_config=ModelConfig(...))
        >>> man = ConfigManager.get()
        >>> man
        ConfigManager(data_config=DataConfig(...), model_config=ModelConfig(...))

        Getting the configuration objects is as easy as getting an attribute

        >>> man.data_config
        DataConfig(...)
        >>> man.model_config
        ModelConfig(...)
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

    def __str__(self):
        return "ConfigManager(data_config=%s, model_config=%s"%(self.data_config, self.model_config)

    def __repr__(self):
        return str(self)
