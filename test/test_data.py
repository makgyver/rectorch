"""Unit tests for the rectorch.data module
"""
import os
import sys
import json
import tempfile
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath('..'))

from data import DataProcessing, DataReader, DatasetManager
from configuration import DataConfig

def test_DataProcessing():
    """Test for the DataProcessing class
    """
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write("1 1 4\n1 2 5\n1 3 2\n1 5 4\n")
        f.write("2 2 3\n2 3 1\n2 5 4\n")
        f.write("3 1 5\n3 2 5\n3 4 3\n3 5 4\n")
        f.write("4 1 1\n4 3 4\n4 4 2\n4 5 4\n")

    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "data_path": tmp.name,
            "proc_path": tmp_folder,
            "seed": 42,
            "threshold": 2.5,
            "separator": " ",
            "u_min": 1,
            "i_min": 1,
            "heldout": 1,
            "test_prop": 0.5,
            "topn": 1
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        with pytest.raises(TypeError):
            DataProcessing(1)

        dp = DataProcessing(tmp_d.name)
        dp.process()
        files = set(os.listdir(tmp_folder))
        f_list = ['validation_te.csv',
                  'validation_tr.csv',
                  'unique_iid.txt',
                  'unique_uid.txt',
                  'test_tr.csv',
                  'test_te.csv',
                  'train.csv']

        for f in f_list:
            assert f in files, "%s should have been created" %f

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

        train_gt = 'uid,iid\n0,0\n0,1\n1,2\n1,1\n'
        iid_gt = '2\n5\n3\n'
        uid_gt = '2\n4\n1\n3\n'
        vtr_gt = 'uid,iid\n2,0\n'
        vte_gt = 'uid,iid\n2,1\n'
        ttr_gt = 'uid,iid\n3,0\n'
        tte_gt = 'uid,iid\n3,1\n'

        assert train_gt == train_file, "the content of train.csv should be %s" %train_gt
        assert iid_gt == iid_file, "the content of unique_iid.txt should be %s" %iid_gt
        assert uid_gt == uid_file, "the content of unique_uid.txt should be %s" %uid_gt
        assert vtr_gt == vtr_file, "the content of validation_tr.csv should be %s" %vtr_gt
        assert vte_gt == vte_file, "the content of validation_te.csv should be %s" %vte_gt
        assert ttr_gt == ttr_file, "the content of test_tr.csv should be %s" %ttr_gt
        assert tte_gt == tte_file, "the content of test_te.csv should be %s" %tte_gt

        dp2 = DataProcessing(DataConfig(tmp_d.name))
        assert dp2.cfg == dp.cfg, "dp and dp2 should be equal"


def test_DataReader():
    """Test for the DataReader class
    """
    names = ['train.csv',
             'unique_iid.txt',
             'unique_uid.txt',
             'validation_tr.csv',
             'validation_te.csv',
             'test_tr.csv',
             'test_te.csv']
    files = ['uid,iid\n0,0\n0,1\n1,2\n1,1\n',
             '2\n5\n3\n',
             '2\n4\n1\n3\n',
             'uid,iid\n2,0\n',
             'uid,iid\n2,1\n',
             'uid,iid\n3,0\n',
             'uid,iid\n3,1\n']

    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "data_path": "NOT USED",
            "proc_path": tmp_folder,
            "seed": 42,
            "threshold": 2.5,
            "separator": " ",
            "u_min": 1,
            "i_min": 1,
            "heldout": 1,
            "test_prop": 0.5,
            "topn": 1
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        for i, f in enumerate(files):
            with open(tmp_folder + "/" + names[i], "w") as tf:
                tf.write(f)

        with pytest.raises(TypeError):
            DataReader(1)

        reader = DataReader(tmp_d.name)
        reader2 = DataReader(DataConfig(tmp_d.name))
        assert reader.n_items == 3, "number of items should be 3"
        assert reader.cfg == reader2.cfg, "reader and reader2 should be equal"

        with pytest.raises(ValueError):
            reader.load_data("training")

        sp_data = reader.load_data("full")
        sp_train = reader.load_data("train")
        sp_vtr, sp_vte = reader.load_data("validation")
        sp_ttr, sp_tte = reader.load_data("test")

        assert np.all(sp_data.data == np.array([1., 1., 1., 1., 1., 1., 1., 1.])),\
            "sp_data should be 8 one"
        r, c = sp_data.nonzero()
        assert np.all(r == np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        assert np.all(c == np.array([0, 1, 1, 2, 0, 1, 0, 1]))

        assert np.all(sp_train.data == np.array([1., 1., 1., 1.])),\
            "sp_train should be 4 one"
        r, c = sp_train.nonzero()
        assert np.all(r == np.array([0, 0, 1, 1]))
        assert np.all(c == np.array([0, 1, 1, 2]))

        assert np.all(sp_vtr.data == np.array([1.])), "sp_vtr should be a one"
        assert np.all(sp_vte.data == np.array([1.])), "sp_vtt should be a one"
        assert np.all(sp_ttr.data == np.array([1.])), "sp_ttr should be a one"
        assert np.all(sp_tte.data == np.array([1.])), "sp_tte should be a one"

        r, c = sp_vtr.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([0]))

        r, c = sp_vte.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([1]))

        r, c = sp_ttr.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([0]))

        r, c = sp_tte.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([1]))


def test_DatasetManager():
    """Test for the DatasetManager class
    """
    names = ['train.csv',
             'unique_iid.txt',
             'unique_uid.txt',
             'validation_tr.csv',
             'validation_te.csv',
             'test_tr.csv',
             'test_te.csv']
    files = ['uid,iid\n0,0\n0,1\n1,2\n1,1\n',
             '2\n5\n3\n',
             '2\n4\n1\n3\n',
             'uid,iid\n2,0\n',
             'uid,iid\n2,1\n',
             'uid,iid\n3,0\n',
             'uid,iid\n3,1\n']

    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "data_path": "NOT USED",
            "proc_path": tmp_folder,
            "seed": 42,
            "threshold": 2.5,
            "separator": " ",
            "u_min": 1,
            "i_min": 1,
            "heldout": 1,
            "test_prop": 0.5,
            "topn": 1
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        for i, f in enumerate(files):
            with open(tmp_folder + "/" + names[i], "w") as tf:
                tf.write(f)

        man = DatasetManager(tmp_d.name)

        assert hasattr(man, "n_items"), "man should have attribute n_items"
        assert hasattr(man, "training_set"), "man should have attribute training_set"
        assert hasattr(man, "validation_set"), "man should have attribute validation_set"
        assert hasattr(man, "test_set"), "man should have attribute test_set"

        assert man.n_items == 3, "number of items should be 3"
        sp_train, none = man.training_set
        sp_vtr, sp_vte = man.validation_set
        sp_ttr, sp_tte = man.test_set

        assert none is None

        assert np.all(sp_train.data == np.array([1., 1., 1., 1.])),\
            "sp_train should be 4 one"
        r, c = sp_train.nonzero()
        assert np.all(r == np.array([0, 0, 1, 1]))
        assert np.all(c == np.array([0, 1, 1, 2]))

        assert np.all(sp_vtr.data == np.array([1.])), "sp_vtr should be a one"
        assert np.all(sp_vte.data == np.array([1.])), "sp_vtt should be a one"
        assert np.all(sp_ttr.data == np.array([1.])), "sp_ttr should be a one"
        assert np.all(sp_tte.data == np.array([1.])), "sp_tte should be a one"

        r, c = sp_vtr.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([0]))

        r, c = sp_vte.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([1]))

        r, c = sp_ttr.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([0]))

        r, c = sp_tte.nonzero()
        assert np.all(r == np.array([0]))
        assert np.all(c == np.array([1]))

def test_nontopn():
    """Test for the module when topn=0
    """
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write("1 1 4\n1 2 5\n1 3 2\n1 5 4\n")
        f.write("2 2 3\n2 3 1\n2 5 4\n")
        f.write("3 1 5\n3 2 5\n3 4 3\n3 5 4\n")
        f.write("4 1 1\n4 3 4\n4 4 2\n4 5 4\n")

    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "data_path": tmp.name,
            "proc_path": tmp_folder,
            "seed": 42,
            "threshold": 2.5,
            "separator": " ",
            "u_min": 1,
            "i_min": 1,
            "heldout": 1,
            "test_prop": 0.5
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        dp = DataProcessing(tmp_d.name)
        dp.process()

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

        train_gt = 'uid,iid,2\n0,0,3\n0,1,4\n1,2,4\n1,1,4\n'
        iid_gt = '2\n5\n3\n'
        uid_gt = '2\n4\n1\n3\n'
        vtr_gt = 'uid,iid,2\n2,0,5\n'
        vte_gt = 'uid,iid,2\n2,1,4\n'
        ttr_gt = 'uid,iid,2\n3,0,5\n'
        tte_gt = 'uid,iid,2\n3,1,4\n'

        assert train_gt == train_file, "the content of train.csv should be %s" %train_gt
        assert iid_gt == iid_file, "the content of unique_iid.txt should be %s" %iid_gt
        assert uid_gt == uid_file, "the content of unique_uid.txt should be %s" %uid_gt
        assert vtr_gt == vtr_file, "the content of validation_tr.csv should be %s" %vtr_gt
        assert vte_gt == vte_file, "the content of validation_te.csv should be %s" %vte_gt
        assert ttr_gt == ttr_file, "the content of test_tr.csv should be %s" %ttr_gt
        assert tte_gt == tte_file, "the content of test_te.csv should be %s" %tte_gt

        reader = DataReader(tmp_d.name)

        sp_data = reader.load_data("full")

        assert np.all(sp_data.data == np.array([3., 4., 4., 4., 5., 4., 5., 4.]))
        r, c = sp_data.nonzero()
        assert np.all(r == np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        assert np.all(c == np.array([0, 1, 1, 2, 0, 1, 0, 1]))
