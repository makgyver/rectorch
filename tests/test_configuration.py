"""Unit tests for the rectorch.configuration module
"""
import os
import sys
import json
import pytest
import tempfile
sys.path.insert(0, os.path.abspath('..'))

from rectorch.configuration import DataConfig, ModelConfig, ConfigManager

def test_DataConfig():
    """Test the :class:`DataConfig` initialization.
    """
    tmp = tempfile.NamedTemporaryFile()

    cfg = {
        "processing": {
            "data_path": "path/to/data",
            "threshold": 1,
            "separator": ",",
            "header": None,
            "u_min": 5,
            "i_min": 2
        },
        "splitting": {
            "split_type": "vertical",
            "sort_by": None,
            "seed": 42,
            "shuffle": False,
            "valid_size": 100,
            "test_size": 100,
            "test_prop": .5
        }
    }

    with pytest.raises(ValueError):
        DataConfig(1.2)

    json.dump(cfg, open(tmp.name, "w"))
    data_config = DataConfig(tmp.name)

    assert str(data_config) == data_config.__repr__(), "__str__ and __repr__ should be the same"
    assert hasattr(data_config, "processing")
    assert hasattr(data_config, "splitting")

    for k in cfg["processing"]:
        assert hasattr(data_config.processing, k)
        assert data_config.processing[k] == cfg["processing"][k],\
            "%s: got %s expected %s" %(k, data_config.processing[k], cfg["processing"][k])

    for k in cfg["splitting"]:
        assert hasattr(data_config.splitting, k)
        assert data_config.splitting[k] == cfg["splitting"][k],\
            "%s: got %s expected %s" %(k, data_config.splitting[k], cfg["splitting"][k])


def test_ModelConfig():
    """Test the :class:`ModelConfig` initialization.
    """
    tmp = tempfile.NamedTemporaryFile()

    cfg = {
        "model": {
            "param1": 1,
            "param2": 2,
            "param3": 3.0,
            "param4": "four"
        },
        "train": {
            "train_param1" : 1
        },
        "test":{
            "test_param2" : 2.0
        },
        "sampler": {
            "sampler_param3" : 3
        }
    }

    json.dump(cfg, open(tmp.name, "w"))
    model_config = ModelConfig(tmp.name)

    assert str(model_config) == model_config.__repr__(), "__str__ and __repr__ should be the same"

    assert hasattr(model_config, "model")
    for k in cfg["model"]:
        assert model_config.model[k] == cfg["model"][k], \
            "model-%s: got %s expected %s" %(k, model_config.model[k], cfg["model"][k])

    assert hasattr(model_config, "train")
    for k in cfg["train"]:
        assert model_config.train[k] == cfg["train"][k], \
            "model-%s: got %s expected %s" %(k, model_config.train[k], cfg["train"][k])

    assert hasattr(model_config, "test")
    for k in cfg["test"]:
        assert model_config.test[k] == cfg["test"][k], \
            "model-%s: got %s expected %s" %(k, model_config.test[k], cfg["test"][k])

    assert hasattr(model_config, "sampler")
    for k in cfg["sampler"]:
        assert model_config.sampler[k] == cfg["sampler"][k], \
            "model-%s: got %s expected %s" %(k, model_config.sampler[k], cfg["sampler"][k])


def test_ConfigManager():
    """Test the :class:`ConfigManager` class
    """
    with pytest.raises(Exception):
        ConfigManager.get()

    tmp_d = tempfile.NamedTemporaryFile()
    cfg_d = {
        "processing": {
            "data_path": "path/to/data",
            "threshold": 1,
            "separator": ",",
            "header": None,
            "u_min": 5,
            "i_min": 2
        },
        "splitting": {
            "split_type": "vertical",
            "sort_by": None,
            "seed": 42,
            "shuffle": False,
            "valid_size": 100,
            "test_size": 100,
            "test_prop": .5
        }
    }
    json.dump(cfg_d, open(tmp_d.name, "w"))

    tmp_m = tempfile.NamedTemporaryFile()
    cfg_m = {
        "model": {
            "param1": 1,
            "param2": 2,
            "param3": 3.0,
            "param4": "four"
        },
        "train": {
            "train_param1" : 1
        },
        "test":{
            "test_param2" : 2.0
        },
        "sampler": {
            "sampler_param3" : 3
        }
    }
    json.dump(cfg_m, open(tmp_m.name, "w"))

    ConfigManager(tmp_d.name, tmp_m.name)
    man = ConfigManager.get()

    assert hasattr(man, "data_config")
    assert hasattr(man, "model_config")
    assert isinstance(man.data_config, DataConfig), "Should be a DataConfig object"
    assert isinstance(man.model_config, ModelConfig), "Should be a ModelConfig object"
    assert str(man).startswith("ConfigManager")
    assert repr(man).startswith("ConfigManager")
