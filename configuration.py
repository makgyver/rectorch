import json
from munch import DefaultMunch
from os.path import exists

class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique instance.
    """
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class DataConfig(DefaultMunch):
    def __init__(self, file_path):
        super(DataConfiguration, self).__init__(None, json.load(open(file_path, "r")))


class ModelConfig():
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
    @classmethod
    def get(cls):
        if cls._instance:
            return cls._instance
        else:
            raise Exception("Singleton object not instantiated!")

    def __init__(self, data_config_path, model_config_path):
        assert exists(data_config_path), "Data config file does not exist."
        assert exists(model_config_path), "Model config file does not exist."
        self.data_config = DataConfig(data_config_path)
        self.model_config = ModelConfig(model_config_path)
