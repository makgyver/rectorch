import json
from munch import DefaultMunch

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

class DataConfiguration(Configuration, metaclass=Singleton):
    pass

class ModelConfiguration(Configuration, metaclass=Singleton):
    pass
