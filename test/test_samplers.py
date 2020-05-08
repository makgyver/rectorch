"""Unit tests for the rectorch.samplers module
"""
import os
import sys
import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath('..'))

from samplers import *

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
    assert len(sampler) == 2
    for i, (t, none) in enumerate(sampler):
        assert none is None
        assert isinstance(t, torch.FloatTensor)
        if i == 0:
            assert np.all(t.numpy() == np.array([1, 1, 0]))
        else:
            assert np.all(t.numpy() == np.array([0, 1, 1]))

    sampler = DataSampler(val_tr, val_te, batch_size=1, shuffle=True)
    assert len(sampler) == 1

    for tr, te in sampler:
        assert np.all(tr.numpy() == np.array([1, 0, 0]))
        assert np.all(te.numpy() == np.array([0, 1, 0]))

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
    assert len(sampler) == (1+2+1+2)/2
    for i, (tr, te) in enumerate(sampler):
        assert isinstance(tr, torch.FloatTensor)
        assert isinstance(te, torch.FloatTensor)
        if i == 0:
            assert np.all(tr.numpy() == np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]))
            assert np.all(te.numpy() == np.array([[1, 1, 0], [0, 1, 1]]))
        elif i == 1:
            assert np.all(tr.numpy() == np.array([[1, 1, 0, 1, 0], [1, 1, 0, 0, 1]]))
            assert np.all(te.numpy() == np.array([[0, 1, 0], [1, 1, 0]]))
        else:
            assert np.all(tr.numpy() == np.array([[0, 1, 1, 1, 0], [0, 1, 1, 0, 1]]))
            assert np.all(te.numpy() == np.array([[0, 1, 1], [0, 1, 0]]))

    np.random.seed(1)
    sampler = ConditionedDataSampler(iid2cids, 2, val_tr, val_te, batch_size=1, shuffle=True)
    assert len(sampler) == 2
    for i, (tr, te) in enumerate(sampler):
        assert isinstance(tr, torch.FloatTensor)
        assert isinstance(te, torch.FloatTensor)
        if i == 0:
            assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 0]))
            assert np.all(te.numpy() == np.array([0, 1, 0]))
        else:
            assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 1]))
            assert np.all(te.numpy() == np.array([0, 1, 0]))

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
    assert len(sampler) == 1
    for tr, te in sampler:
        assert isinstance(tr, torch.FloatTensor)
        assert isinstance(te, torch.FloatTensor)
        assert np.all(tr.numpy() == np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]))
        assert np.all(te.numpy() == np.array([[1, 1, 0], [0, 1, 1]]))

    np.random.seed(1)
    sampler = EmptyConditionedDataSampler(2, val_tr, val_te, batch_size=1, shuffle=True)
    assert len(sampler) == 1
    for tr, te in sampler:
        assert isinstance(tr, torch.FloatTensor)
        assert isinstance(te, torch.FloatTensor)
        assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 0]))
        assert np.all(te.numpy() == np.array([0, 1, 0]))

def test_CFGAN_TrainingSampler():
    """Test the CFGAN_TrainingSampler class
    """
    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))

    sampler = CFGAN_TrainingSampler(train, batch_size=1)
    assert len(sampler) == 2
    assert hasattr(sampler, "idxlist")
    assert sampler.idxlist == [0, 1]

    t = None
    for x in sampler:
        t = x
        break

    assert isinstance(t, torch.FloatTensor)
    assert np.all(t.numpy() == np.array([1, 1, 0])) or np.all(t.numpy() == np.array([0, 1, 1]))

    t = next(sampler)
    assert np.all(t.numpy() == np.array([1, 1, 0])) or np.all(t.numpy() == np.array([0, 1, 1]))
