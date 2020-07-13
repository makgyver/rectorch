"""Unit tests for the rectorch.data module
"""
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import scipy
sys.path.insert(0, os.path.abspath('..'))

from rectorch.data import DataProcessing, Dataset
from rectorch.configuration import DataConfig

def test_DataProcessing():
    """Test for the DataProcessing class
    """
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write("1 1 4\n1 2 5\n1 3 2\n1 5 4\n")
        f.write("2 2 3\n2 3 1\n2 5 4\n")
        f.write("3 1 5\n3 2 5\n3 4 3\n3 5 4\n")
        f.write("4 1 3\n4 3 4\n4 4 2\n4 5 4\n")

    #VERTICAL
    with tempfile.TemporaryDirectory():
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "processing": {
                "data_path": tmp.name,
                "threshold": 2.5,
                "separator": " ",
                "header": None,
                "u_min": 1,
                "i_min": 1
            },
            "splitting": {
                "split_type": "vertical",
                "sort_by": None,
                "seed": 42,
                "shuffle": False,
                "valid_size": 1,
                "test_size": 1,
                "test_prop": .5
            }
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        with pytest.raises(TypeError):
            DataProcessing(1)

        dp = DataProcessing(tmp_d.name)
        data = dp.process()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 12

        dataset = dp.split(data)

        assert dataset.n_users == 4
        assert dataset.n_items == 3
        assert dataset.n_ratings == 10
        assert set(dataset.i2id.keys()) == set([1, 2, 5])
        assert set(dataset.u2id.keys()) == set([1, 2, 3, 4])
        assert list(dataset.unique_iid) == [1, 2, 5]
        assert list(dataset.unique_uid) == [1, 2, 3, 4]
        assert len(dataset.train_set) == 5
        assert len(dataset.valid_set) == 2
        assert len(dataset.valid_set[0]) == 2
        assert len(dataset.test_set) == 2
        assert len(dataset.test_set[0]) == 1
        assert len(dataset.valid_set[1]) == 1
        assert len(dataset.test_set[1]) == 1

    #HORIZONTAL
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write("1 1 4\n1 2 5\n1 3 2\n1 5 4\n")
        f.write("2 5 4\n2 2 3\n2 3 1\n")
        f.write("3 4 3\n3 5 4\n3 1 5\n3 2 5\n")
        f.write("4 4 2\n4 3 4\n4 5 4\n4 1 3\n")

    with tempfile.TemporaryDirectory():
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "processing": {
                "data_path": tmp.name,
                "threshold": 0,
                "separator": " ",
                "header": None,
                "u_min": 1,
                "i_min": 1
            },
            "splitting": {
                "split_type": "horizontal",
                "sort_by": None,
                "seed": 42,
                "shuffle": False,
                "valid_size": 1,
                "test_size": 1,
                "test_prop": .5 #ignored
            }
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        dc = DataConfig(tmp_d.name)
        dp = DataProcessing(dc)
        dataset0 = dp.split(dp.process())

        assert dataset0.n_users == 4
        assert dataset0.n_items == 5
        assert dataset0.n_ratings == 15

        assert set(dataset0.i2id.keys()) == set([1, 2, 3, 4, 5])
        assert set(dataset0.u2id.keys()) == set([1, 2, 3, 4])
        assert set(dataset0.unique_iid) == set([1, 2, 3, 4, 5])
        assert set(dataset0.unique_uid) == set([1, 2, 3, 4])
        assert len(dataset0.train_set) == 7
        assert len(dataset0.valid_set) == 4
        assert len(dataset0.test_set) == 4

    with pytest.raises(ValueError):
        cfg_d["splitting"]["split_type"] = "pippo"
        DataProcessing(cfg_d).process_and_split()

    cfg_d["splitting"]["split_type"] = "horizontal"
    cfg_d["splitting"]["shuffle"] = True
    dp = DataProcessing(cfg_d)
    dataset2 = dp.split(dp.process())
    assert list(dataset2.train_set.index) != list(dataset0.train_set.index)

    cfg_d["splitting"]["split_type"] = "vertical"
    dp = DataProcessing(cfg_d)
    dataset3 = dp.split(dp.process())
    assert list(dataset3.train_set.index) != list(dataset.train_set.index)

    cfg_d["splitting"]["valid_size"] = 0
    dp = DataProcessing(cfg_d)
    dataset = dp.split(dp.process())
    assert dataset.valid_set is None


def test_Dataset():
    """Test for the Dataset class
    """
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write("1 1 4\n1 2 5\n1 3 2\n1 5 4\n")
        f.write("2 2 3\n2 3 1\n2 5 4\n")
        f.write("3 1 5\n3 2 5\n3 4 3\n3 5 4\n")
        f.write("4 1 3\n4 3 4\n4 4 2\n4 5 4\n")

    cfg_d = {
        "processing": {
            "data_path": tmp.name,
            "threshold": 2.5,
            "separator": " ",
            "header": None,
            "u_min": 1,
            "i_min": 1
        },
        "splitting": {
            "split_type": "vertical",
            "sort_by": None,
            "seed": 42,
            "shuffle": False,
            "valid_size": 1,
            "test_size": 1,
            "test_prop": .5
        }
    }
    dp = DataProcessing(cfg_d)
    dataset = dp.split(dp.process())

    with tempfile.TemporaryDirectory() as tmp_folder:
        dataset.save(tmp_folder)

        train_file = ""
        with open(tmp_folder + "/" + "train.csv", "r") as t:
            train_file = "".join(t.readlines())

        iid_file = ""
        with open(tmp_folder + "/" + "unique_iid.txt", "r") as t:
            iid_file = "".join(t.readlines())

        uid_file = ""
        with open(tmp_folder + "/" + "unique_uid.txt", "r") as t:
            uid_file = "".join(t.readlines())
        vtr_file = ""
        with open(tmp_folder + "/" + "validation_tr.csv", "r") as t:
            vtr_file = "".join(t.readlines())

        vte_file = ""
        with open(tmp_folder + "/" + "validation_te.csv", "r") as t:
            vte_file = "".join(t.readlines())

        ttr_file = ""
        with open(tmp_folder + "/" + "test_tr.csv", "r") as t:
            ttr_file = "".join(t.readlines())

        tte_file = ""
        with open(tmp_folder + "/" + "test_te.csv", "r") as t:
            tte_file = "".join(t.readlines())

        train_gt = 'uid,iid,rating\n0,0,4\n0,1,5\n0,2,4\n1,1,3\n1,2,4\n'
        iid_gt = '1\n2\n5\n'
        uid_gt = '1\n2\n3\n4\n'
        vtr_gt = 'uid,iid,rating\n2,0,5\n2,1,5\n'
        vte_gt = 'uid,iid,rating\n2,2,4\n'
        ttr_gt = 'uid,iid,rating\n3,0,3\n'
        tte_gt = 'uid,iid,rating\n3,2,4\n'

        assert train_gt == train_file, "the content of train.csv should be %s" %train_gt
        assert iid_gt == iid_file, "the content of unique_iid.txt should be %s" %iid_gt
        assert uid_gt == uid_file, "the content of unique_uid.txt should be %s" %uid_gt
        assert vtr_gt == vtr_file, "the content of validation_tr.csv should be %s" %vtr_gt
        assert vte_gt == vte_file, "the content of validation_te.csv should be %s" %vte_gt
        assert ttr_gt == ttr_file, "the content of test_tr.csv should be %s" %ttr_gt
        assert tte_gt == tte_file, "the content of test_te.csv should be %s" %tte_gt


        dataset2 = Dataset.load(tmp_folder)
        assert str(dataset) == str(dataset2)

        sp = dataset.to_sparse()
        assert isinstance(sp[0], scipy.sparse.csr_matrix)
        assert len(sp) == 3
        assert len(sp[0].nonzero()[0]) == 5
        assert len(sp[1][0].nonzero()[0]) == 2
        assert len(sp[1][1].nonzero()[0]) == 1
        assert len(sp[2][0].nonzero()[0]) == 1
        assert len(sp[2][1].nonzero()[0]) == 1

        ar = dataset.to_array()
        assert isinstance(ar[0], np.ndarray)
        assert len(ar) == 3
        assert np.sum(ar[0] > 0) == 5
        assert np.sum(ar[1][0] > 0) == 2
        assert np.sum(ar[1][1] > 0) == 1
        assert np.sum(ar[2][0] > 0) == 1
        assert np.sum(ar[2][1] > 0) == 1

        dd = dataset.to_dict()
        assert isinstance(dd[0], dict)
        assert len(dd) == 3
        assert dd[0][0] == [0, 1, 2]
        assert dd[0][1] == [1, 2]
        assert dd[1][0][2] == [0, 1]
        assert dd[1][1][2] == [2]
        assert dd[2][0][3] == [0]
        assert dd[2][1][3] == [2]

        tn = dataset.to_tensor()
        assert isinstance(tn[0], torch.FloatTensor)
        assert len(tn) == 3
        assert torch.nonzero(tn[0], as_tuple=True)[0].shape[0] == 5
        assert torch.nonzero(tn[1][0], as_tuple=True)[0].shape[0] == 2
        assert torch.nonzero(tn[1][1], as_tuple=True)[0].shape[0] == 1
        assert torch.nonzero(tn[2][0], as_tuple=True)[0].shape[0] == 1
        assert torch.nonzero(tn[2][1], as_tuple=True)[0].shape[0] == 1

        tn = dataset.to_tensor(cold_users=False)
        assert isinstance(tn[0], torch.FloatTensor)
        assert len(tn) == 3
        assert tn[0].shape == torch.Size([4, 3])
        assert tn[1].shape == torch.Size([1, 3])
        assert tn[2].shape == torch.Size([1, 3])
        assert torch.nonzero(tn[0], as_tuple=True)[0].shape[0] == 8
        assert torch.nonzero(tn[1], as_tuple=True)[0].shape[0] == 1
        assert torch.nonzero(tn[2], as_tuple=True)[0].shape[0] == 1

        dd = dataset.to_dict(cold_users=False)
        assert isinstance(dd[0], dict)
        assert len(dd) == 3
        assert dd[0][0] == [0, 1, 2]
        assert dd[0][1] == [1, 2]
        assert dd[0][2] == [0, 1]
        assert dd[0][3] == [0]
        assert dd[1][2] == [2]
        assert dd[2][3] == [2]

        ar = dataset.to_array(cold_users=False)
        assert isinstance(ar[0], np.ndarray)
        assert len(ar) == 3
        assert np.sum(ar[0] > 0) == 8
        assert np.sum(ar[1] > 0) == 1
        assert np.sum(ar[2] > 0) == 1

        sp = dataset.to_sparse(cold_users=False)
        assert isinstance(sp[0], scipy.sparse.csr_matrix)
        assert len(sp) == 3
        assert len(sp[0].nonzero()[0]) == 8
        assert len(sp[1].nonzero()[0]) == 1
        assert len(sp[2].nonzero()[0]) == 1