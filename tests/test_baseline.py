"""Unit tests for the rectorch.models.baseline module
"""
import os
import sys
import tempfile
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath('..'))

from rectorch.utils import prepare_for_prediction
from rectorch.models.baseline import Random, Popularity, CF_KOMD, SLIM
from rectorch.data import Dataset
from rectorch.samplers import DictDummySampler, ArrayDummySampler, SparseDummySampler,\
    TensorDummySampler

def test_random():
    """Test the Random class
    """
    rnd = Random(4, 1, False)
    assert rnd.n_items == 4
    assert rnd.seed == 1
    assert not rnd.fixed
    rnd.train()
    p = rnd.predict([1, 2, 3], [[0], [1], [2]])[0]
    assert isinstance(p, torch.torch.FloatTensor)
    assert p.shape == torch.Size([3, 4])
    assert p[0, 0] == -np.inf
    assert p[1, 1] == -np.inf
    assert p[1, 0] != -np.inf
    assert p[0, 3] != p[1, 3]

    rnd = Random(4, 1, True)
    rnd.train()
    p = rnd.predict([1, 2, 3], [[0], [1], [2]])[0]
    assert isinstance(p, torch.torch.FloatTensor)
    assert p.shape == torch.Size([3, 4])
    assert p[0, 0] == -np.inf
    assert p[1, 1] == -np.inf
    assert p[1, 0] != -np.inf
    assert p[0, 3] == p[1, 3]

    tmp = tempfile.NamedTemporaryFile()
    rnd.save_model(tmp.name)

    rnd2 = Random(5, 999, False)
    assert rnd2.n_items == 5
    assert rnd2.seed == 999
    assert not rnd2.fixed
    rnd2 = Random.load_model(tmp.name)
    print(rnd2.n_items)
    assert rnd2.n_items == 4
    assert rnd2.seed == 1
    assert rnd2.fixed

def test_popularity():
    """Test the Popularity class
    """
    values = [1.] * 7
    rows = [0, 0, 1, 1, 1, 2, 2]
    cols = [0, 1, 0, 1, 2, 1, 3]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1, 2:2}
    iids = {0:0, 1:1, 2:2, 3:3}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)

    pop = Popularity(4)
    assert pop.n_items == 4
    assert pop.model is None

    #R = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1]]
    #t = torch.FloatTensor(R)
    dds = DictDummySampler(data)
    pop = Popularity(4)
    pop.train(dds)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    dds = SparseDummySampler(data)
    pop = Popularity(4)
    pop.train(dds)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    pop.train(dds)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    dds = TensorDummySampler(data)
    pop = Popularity(4)
    pop.train(dds)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    dds = TensorDummySampler(data)
    pop = Popularity(4)
    pop.train(dds)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    dds = ArrayDummySampler(data)
    pop = Popularity(4)
    pop.train(dds)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    p = pop.predict([0, 1, 2], [[1], [2], [3]])[0]
    assert isinstance(p, torch.FloatTensor)
    assert p.shape == torch.Size([3, 4])
    assert p[0, 1] == -np.inf
    assert p[1, 2] == -np.inf
    assert p[2, 1] != -np.inf
    assert p[0, 3] == p[1, 3]

    tmp = tempfile.NamedTemporaryFile()
    pop.save_model(tmp.name)

    pop2 = Popularity(4)
    pop2 = Popularity.load_model(tmp.name)
    assert torch.all(pop2.model == pop.model)

def test_cfkomd():
    """Test CF-KOMD.
    """
    values = [1.] * 7
    rows = [0, 0, 1, 1, 1, 2, 2]
    cols = [0, 1, 0, 1, 2, 1, 3]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1, 2:2}
    iids = {0:0, 1:1, 2:2, 3:3}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)

    cfkomd = CF_KOMD()
    assert cfkomd.ker_fun == "linear"
    assert cfkomd.lambda_p == .1
    assert cfkomd.model == {}
    assert cfkomd.disj_degree == 1

    ads = ArrayDummySampler(data)
    cfkomd = CF_KOMD(ker_fun="disjunctive", disj_degree=2)
    cfkomd.train(ads)
    assert isinstance(cfkomd.model, dict)
    assert isinstance(cfkomd.model[0], torch.FloatTensor)

    p = cfkomd.predict([0, 1], np.array([[0, 1, 0, 0], [0, 0, 1, 0]]))[0]
    assert isinstance(p, torch.FloatTensor)
    assert p.shape == torch.Size([2, 4])
    assert p[0, 1] == -np.inf
    assert p[1, 2] == -np.inf

    cfkomd.train(ads, only_test=True)
    assert isinstance(cfkomd.model, dict)
    assert isinstance(cfkomd.model[0], torch.FloatTensor)

    p = cfkomd.predict([0, 1], np.array([[0, 1, 0, 0], [0, 0, 1, 0]]))[0]
    assert isinstance(p, torch.FloatTensor)
    assert p.shape == torch.Size([2, 4])
    assert p[0, 1] == -np.inf
    assert p[1, 2] == -np.inf

    tmp = tempfile.NamedTemporaryFile()
    cfkomd.save_model(tmp.name)

    cfkomd2 = CF_KOMD()
    cfkomd2 = CF_KOMD.load_model(tmp.name)
    assert torch.all(cfkomd2.model[0] == cfkomd.model[0])
    assert torch.all(cfkomd2.model[1] == cfkomd.model[1])


def test_slim():
    """Test SLIM.
    """
    values = [1.] * 7
    rows = [0, 0, 1, 1, 1, 2, 2]
    cols = [0, 1, 0, 1, 2, 1, 3]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1, 2:2}
    iids = {0:0, 1:1, 2:2, 3:3}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)

    slim = SLIM(l1_reg=.1, l2_reg=.1)
    assert slim.l1_reg == .1
    assert slim.l2_reg == .1
    assert slim.model is None

    sds = SparseDummySampler(data)
    slim = SLIM(l1_reg=.1, l2_reg=.1)
    slim.train(sds)

    sds.test()
    t = prepare_for_prediction(*next(iter(sds)))
    p = slim.predict(*t[0])[0]
    assert isinstance(p, torch.FloatTensor)
    assert p.shape == torch.Size([1, 4])
    assert p[0, 0] == -np.inf
    assert p[0, 1] != -np.inf

    tmp = tempfile.NamedTemporaryFile()
    slim.save_model(tmp.name)

    slim2 = SLIM(l1_reg=.3, l2_reg=.2)
    slim2 = SLIM.load_model(tmp.name)
    assert (slim2.model != slim.model).nnz == 0
    assert slim2.l1_reg == .1
    assert slim2.l2_reg == .1
