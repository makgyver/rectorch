"""Unit tests for the rectorch.samplers module
"""
import os
import sys
import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath('..'))

from samplers import Sampler, DataSampler, EmptyConditionedDataSampler, ConditionedDataSampler, \
    CFGAN_TrainingSampler

def test_Sampler():
    """Test the Sampler class
    """
    sampler = Sampler()
    with pytest.raises(NotImplementedError):
        len(sampler)

    with pytest.raises(NotImplementedError):
        for _ in sampler:
            pass

def test_DataSampler():
    """Test the DataSampler class
    """
    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))

    values = np.array([1.])
    rows = np.array([0])
    cols = np.array([0])
    val_tr = csr_matrix((values, (rows, cols)), shape=(1, 3))

    cols = np.array([1])
    val_te = csr_matrix((values, (rows, cols)), shape=(1, 3))

    sampler = DataSampler(train, batch_size=1, shuffle=False)
    assert len(sampler) == 2, "the number of batches should be 2"
    for i, (t, none) in enumerate(sampler):
        assert none is None, "the test part of the training should be None"
        assert isinstance(t, torch.FloatTensor), "t should be of type torch.Tensor"
        if i == 0:
            assert np.all(t.numpy() == np.array([1, 1, 0])), "the tensor t should be [1, 1, 0]"
        else:
            assert np.all(t.numpy() == np.array([0, 1, 1])), "the tensor t should be [0, 1, 1]"

    sampler = DataSampler(val_tr, val_te, batch_size=1, shuffle=True)
    assert len(sampler) == 1, "the number of batches should be 1"

    for tr, te in sampler:
        assert np.all(tr.numpy() == np.array([1, 0, 0])), "the tensor tr should be [1, 0, 0]"
        assert np.all(te.numpy() == np.array([0, 1, 0])), "the tensor te should be [1, 1, 0]"

def test_ConditionedDataSampler():
    """Test the ConditionedDataSampler class
    """
    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))

    values = np.array([1.])
    rows = np.array([0])
    cols = np.array([0])
    val_tr = csr_matrix((values, (rows, cols)), shape=(1, 3))

    cols = np.array([1])
    val_te = csr_matrix((values, (rows, cols)), shape=(1, 3))

    iid2cids = {0:[1], 1:[0, 1], 2:[0]}
    sampler = ConditionedDataSampler(iid2cids, 2, train, batch_size=2, shuffle=False)
    assert len(sampler) == 3, "the number of batches should be 2"
    for i, (tr, te) in enumerate(sampler):
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        if i == 0:
            assert np.all(tr.numpy() == np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]])),\
                "the tensor tr should be [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]"
            assert np.all(te.numpy() == np.array([[1, 1, 0], [0, 1, 1]])),\
                "the tensor te should be [[1, 1, 0], [0, 1, 1]]"
        elif i == 1:
            assert np.all(tr.numpy() == np.array([[1, 1, 0, 1, 0], [1, 1, 0, 0, 1]])),\
                "the tensor tr should be [[1, 1, 0, 1, 0], [1, 1, 0, 0, 1]]"
            assert np.all(te.numpy() == np.array([[0, 1, 0], [1, 1, 0]])),\
                "the tensor te should be [[1, 0, 1], [1, 1, 0]]"
        else:
            assert np.all(tr.numpy() == np.array([[0, 1, 1, 1, 0], [0, 1, 1, 0, 1]])),\
                "the tensor tr should be [[0, 1, 1, 1, 0], [0, 1, 1, 0, 1]]"
            assert np.all(te.numpy() == np.array([[0, 1, 1], [0, 1, 0]])),\
                "the tensor te should be [[0, 1, 1], [0, 1, 0]]"

    np.random.seed(1)
    sampler = ConditionedDataSampler(iid2cids, 2, val_tr, val_te, batch_size=1, shuffle=True)
    assert len(sampler) == 2, "the number of batches should be 2"
    for i, (tr, te) in enumerate(sampler):
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        if i == 0:
            assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 0])),\
                "the tensor tr should be [1, 0, 0, 0, 0]"
            assert np.all(te.numpy() == np.array([0, 1, 0])),\
                "the tensor te should be [0, 1, 0]"
        else:
            assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 1])),\
                "the tensor tr should be [1, 0, 0, 0, 1]"
            assert np.all(te.numpy() == np.array([0, 1, 0])),\
                "the tensor te should be [0, 1, 0]"

def test_EmptyConditionedDataSampler():
    """Test the EmptyConditionedDataSampler class
    """
    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))

    values = np.array([1.])
    rows = np.array([0])
    cols = np.array([0])
    val_tr = csr_matrix((values, (rows, cols)), shape=(1, 3))

    cols = np.array([1])
    val_te = csr_matrix((values, (rows, cols)), shape=(1, 3))

    sampler = EmptyConditionedDataSampler(2, train, batch_size=2, shuffle=False)
    assert len(sampler) == 1, "the number of batches should be 1"
    for tr, te in sampler:
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        assert np.all(tr.numpy() == np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]])),\
            "the tensor tr should be [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]"
        assert np.all(te.numpy() == np.array([[1, 1, 0], [0, 1, 1]])),\
            "the tensor te should be [[1, 1, 0], [0, 1, 1]]"

    np.random.seed(1)
    sampler = EmptyConditionedDataSampler(2, val_tr, val_te, batch_size=1, shuffle=True)
    assert len(sampler) == 1, "the number of batches should be 1"
    for tr, te in sampler:
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 0])),\
            "the tensor tr should be [1, 0, 0, 0, 0]"
        assert np.all(te.numpy() == np.array([0, 1, 0])),\
            "the tensor te should be [0, 1, 0]"

def test_CFGAN_TrainingSampler():
    """Test the CFGAN_TrainingSampler class
    """
    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))

    sampler = CFGAN_TrainingSampler(train, batch_size=1)
    assert len(sampler) == 2, "the number of batches should be 2"
    assert hasattr(sampler, "idxlist"), "the sampler should have the attribute idxlist"
    assert sampler.idxlist == [0, 1], "the idxlist should be only [0, 1]"

    t = None
    for x in sampler:
        t = x
        break

    assert isinstance(t, torch.FloatTensor), "t should be of type torch.Tensor"
    assert np.all(t.numpy() == np.array([1, 1, 0])) or np.all(t.numpy() == np.array([0, 1, 1])),\
        "the next batch should be [1, 1, 0] or [0, 1, 1]"

    t = next(sampler)
    assert np.all(t.numpy() == np.array([1, 1, 0])) or np.all(t.numpy() == np.array([0, 1, 1])),\
        "the next batch should be [1, 1, 0] or [0, 1, 1]"

