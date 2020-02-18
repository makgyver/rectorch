import json
from munch import DefaultMunch
from os.path import exists

class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class Configuration(DefaultMunch):
    def __init__(self, file_path):
        super(Configuration, self).__init__(None, json.load(open(file_path, "r")))

class DataConfiguration(Configuration):
    pass

class ModelConfiguration(Configuration):
    pass

class ConfigurationManager(metaclass=Singleton):
    @classmethod
    def get_instance(cls):
        if cls._instance:
            return cls._instance
        else:
            raise Exception("Singleton object not instantiated!")

    def __init__(self, data_config_path, model_config_path):
        assert exists(data_config_path), "Data config file does not exist."
        assert exists(model_config_path), "Model config file does not exist."
        self.data_config = DataConfiguration(data_config_path)
        self.model_config = ModelConfiguration(model_config_path)
