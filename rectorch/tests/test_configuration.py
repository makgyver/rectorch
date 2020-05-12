"""Unit tests for the rectorch.configuration module
"""
import os
import sys
import json
import pytest
import tempfile
sys.path.insert(0, os.path.abspath('..'))

from configuration import DataConfig, ModelConfig, ConfigManager

def test_DataConfig():
    """Test the :class:`DataConfig` initialization.
    """
    tmp = tempfile.NamedTemporaryFile()

    cfg = {
        "data_path": "path/to/data",
        "proc_path": "path/to/preproc/folder",
        "seed": 42,
        "threshold": 1,
        "separator": ",",
        "u_min": 5,
        "i_min": 2,
        "heldout": 100,
        "test_prop": 0.2,
        "topn": 1
    }

    json.dump(cfg, open(tmp.name, "w"))
    data_config = DataConfig(tmp.name)

    assert str(data_config) == data_config.__repr__(), "__str__ and __repr__ should be the same"

    for k in cfg:
        assert hasattr(data_config, k)
        assert data_config[k] == cfg[k], "%s: got %s expected %s" %(k, data_config[k], cfg[k])


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
        "data_path": "path/to/data",
        "proc_path": "path/to/preproc/folder",
        "seed": 42,
        "threshold": 1,
        "separator": ",",
        "u_min": 5,
        "i_min": 2,
        "heldout": 100,
        "test_prop": 0.2,
        "topn": 1
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
