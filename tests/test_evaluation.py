"""Unit tests for the rectorch.evaluation module
"""
import os
import sys
import numpy as np
import pytest
import torch
import pandas as pd
sys.path.insert(0, os.path.abspath('..'))

from rectorch.data import Dataset
from rectorch.evaluation import evaluate, one_plus_random
from rectorch.validation import ValidFunc, GridSearch, BayesianSearch, RandomSearch, HyperoptSearch
from rectorch.models import RecSysModel
from rectorch.models.nn import MultiVAE
from rectorch.samplers import Sampler, DataSampler

# pylint: disable=W0223

class FakeModel(RecSysModel):
    """Fake model
    """
    def predict(self, x):
        return (x + torch.FloatTensor([[1]*4]), )


class FakeModelArray(RecSysModel):
    """Fake model
    """
    def predict(self, ids, x):
        return (torch.from_numpy(x).float() + torch.FloatTensor([[1]*4]), ids)


class FakeModelDict(RecSysModel):
    """Fake model
    """
    def predict(self, ids, x):
        return (torch.from_numpy(np.array(x)).float() + torch.FloatTensor([[1]*4]), ids)



class FakeSampler(Sampler):
    """Fake sampler
    """
    def __iter__(self):
        scores = [torch.FloatTensor([[4., 3., 2., 1.]]), torch.FloatTensor([[4., 3., 2., 1.]])]
        gt = [torch.FloatTensor([[1., 1., 0., 0.]]), torch.FloatTensor([[0, 0, 1., 1.]])]

        for i in range(2):
            yield scores[i], gt[i]

    def __len__(self):
        return 2


class FakeSamplerArray(Sampler):
    """Fake sampler
    """
    def __iter__(self):
        scores = [np.array([[4., 3., 2., 1.]]), np.array([[4., 3., 2., 1.]])]
        gt = [np.array([[1., 1., 0., 0.]]), np.array([[0, 0, 1., 1.]])]

        for i in range(2):
            yield ([i], scores[i]), gt[i]

    def __len__(self):
        return 2


class FakeSamplerErr(Sampler):
    """Fake sampler
    """
    def __iter__(self):
        for i in range(2):
            yield ([i], np.array([[4., 3., 2., 1.]])), {2:2}

    def __len__(self):
        return 2

class FakeSamplerDict(Sampler):
    """Fake sampler
    """
    def __iter__(self):
        scores = [[[4., 3., 2., 1.]], [[4., 3., 2., 1.]]]
        gt = [[[0, 1]], [[2, 3]]]

        for i in range(2):
            yield ([i], scores[i]), gt[i]

    def __len__(self):
        return 2

def test_evaluate():
    """Test the evaluate function
    """
    model = FakeModel()
    sampl = FakeSampler(None)
    res = evaluate(model, sampl, ["ndcg@3", "recall@2"])

    assert isinstance(res, dict), "'res' should be e dictionary"
    assert "ndcg@3" in res, "'ndcg@3' should be in 'res'"
    assert "recall@2" in res, "'recall@2' should be in 'res'"
    assert res['ndcg@3'][0] == np.array([1.]), "'ndcg@3' for user 0 should be 1"
    eps = np.array([0.0000001])
    assert np.abs(res['ndcg@3'][1] - np.array([0.3065735964])) < eps,\
        "'ndcg@3' for user 1 should be 0.3065735964"
    assert res['recall@2'][0] == np.array([1.]), "'recall@2' for user 0 should be 1"
    assert res['recall@2'][1] == np.array([0.]), "'recall@2' for user 1 should be 0"

def test_one_plus_random():
    """Test the one_plus_random function
    """
    model = FakeModel()
    sampl = FakeSampler(None)
    res = one_plus_random(model, sampl, ["mrr@1", "hit@1"], r=2)
    assert isinstance(res, dict), "'res' should be e dictionary"
    assert "mrr@1" in res, "'mrr@1' should be in 'res'"
    assert "hit@1" in res, "'hit@1' should be in 'res'"
    assert len(res['hit@1']) == 4
    assert len(res['mrr@1']) == 4
    assert res['hit@1'][0] == np.array([1.]), "'hit@1' for user 0 with first item should be 1"
    assert res['hit@1'][1] == np.array([1.]), "'hit@1' for user 0 with second item should be 1"
    assert res['hit@1'][2] == np.array([0.]), "'hit@1' for user 1 with first item should be 0"
    assert res['hit@1'][3] == np.array([0.]), "'hit@1' for user 1 with second item should be 0"
    assert res['mrr@1'][0] == np.array([1.]), "'mrr@1' for user 0 with first item should be 1"
    assert res['mrr@1'][1] == np.array([1.]), "'mrr@1' for user 0 with second item should be 1"
    assert res['mrr@1'][2] == np.array([0.]), "'mrr@1' for user 1 with first item should be 0"
    assert res['mrr@1'][3] == np.array([0.]), "'mrr@1' for user 1 with second item should be 0"

    with pytest.raises(ValueError):
        one_plus_random(model, sampl, ["mrr@1", "hit@1"], r=3)

    model = FakeModelArray()
    sampl = FakeSamplerArray(None)
    res = one_plus_random(model, sampl, ["mrr@1", "hit@1"], r=2)
    assert isinstance(res, dict), "'res' should be e dictionary"
    assert "mrr@1" in res, "'mrr@1' should be in 'res'"
    assert "hit@1" in res, "'hit@1' should be in 'res'"
    assert len(res['hit@1']) == 4
    assert len(res['mrr@1']) == 4
    assert res['hit@1'][0] == np.array([1.]), "'hit@1' for user 0 with first item should be 1"
    assert res['hit@1'][1] == np.array([1.]), "'hit@1' for user 0 with second item should be 1"
    assert res['hit@1'][2] == np.array([0.]), "'hit@1' for user 1 with first item should be 0"
    assert res['hit@1'][3] == np.array([0.]), "'hit@1' for user 1 with second item should be 0"
    assert res['mrr@1'][0] == np.array([1.]), "'mrr@1' for user 0 with first item should be 1"
    assert res['mrr@1'][1] == np.array([1.]), "'mrr@1' for user 0 with second item should be 1"
    assert res['mrr@1'][2] == np.array([0.]), "'mrr@1' for user 1 with first item should be 0"
    assert res['mrr@1'][3] == np.array([0.]), "'mrr@1' for user 1 with second item should be 0"

    model = FakeModelDict()
    sampl = FakeSamplerDict(None)
    res = one_plus_random(model, sampl, ["mrr@1", "hit@1"], r=2)
    assert isinstance(res, dict), "'res' should be e dictionary"
    assert "mrr@1" in res, "'mrr@1' should be in 'res'"
    assert "hit@1" in res, "'hit@1' should be in 'res'"
    assert len(res['hit@1']) == 4
    assert len(res['mrr@1']) == 4
    assert res['hit@1'][0] == np.array([1.]), "'hit@1' for user 0 with first item should be 1"
    assert res['hit@1'][1] == np.array([1.]), "'hit@1' for user 0 with second item should be 1"
    assert res['hit@1'][2] == np.array([0.]), "'hit@1' for user 1 with first item should be 0"
    assert res['hit@1'][3] == np.array([0.]), "'hit@1' for user 1 with second item should be 0"
    assert res['mrr@1'][0] == np.array([1.]), "'mrr@1' for user 0 with first item should be 1"
    assert res['mrr@1'][1] == np.array([1.]), "'mrr@1' for user 0 with second item should be 1"
    assert res['mrr@1'][2] == np.array([0.]), "'mrr@1' for user 1 with first item should be 0"
    assert res['mrr@1'][3] == np.array([0.]), "'mrr@1' for user 1 with second item should be 0"

    model = FakeModelDict()
    sampl = FakeSamplerErr(None)
    with pytest.raises(TypeError):
        res = one_plus_random(model, sampl, ["mrr@1", "hit@1"], r=2)

def test_ValidFunc():
    """Test the ValidFunc class
    """
    vfun = ValidFunc(one_plus_random, r=2)

    model = FakeModel()
    sampl = FakeSampler(None)
    res = vfun(model, sampl, "mrr@1")

    assert isinstance(res, np.ndarray), "'res' should be e dictionary"
    assert res[0] == np.array([1.]), "'mrr@1' for user 0 with first item should be 1"
    assert res[1] == np.array([1.]), "'mrr@1' for user 0 with second item should be 1"
    assert res[2] == np.array([0.]), "'mrr@1' for user 1 with first item should be 0"
    assert res[3] == np.array([0.]), "'mrr@1' for user 1 with second item should be 0"

    with pytest.raises(AssertionError):
        def addfun(a=1, b=2, c=3, d=4):
            return a + b + c + d
        ValidFunc(addfun, b=3)

    ValidFunc(evaluate)
    assert repr(vfun) == str(vfun)

def test_GridSearch():
    """Test the GridSearch class
    """
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    dataset = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)

    gs = GridSearch(MultiVAE, {
        "mvae_net" : ("MultiVAE_net", [{"dec_dims":[10, dataset.n_items]}]),
        "beta" : [.5, 1.],
        "anneal_steps" : [0]
    }, ValidFunc(evaluate), "ndcg@1")
    gs.report()

    assert hasattr(gs, "model_class")
    assert hasattr(gs, "params_grid")
    assert hasattr(gs, "valid_func")
    assert hasattr(gs, "valid_metric")
    assert hasattr(gs, "params_dicts")
    assert hasattr(gs, "size")
    assert hasattr(gs, "valid_scores")
    assert hasattr(gs, "best_model")

    assert gs.model_class == MultiVAE
    assert gs.valid_metric == "ndcg@1"
    assert gs.size == 2
    assert gs.best_model is None

    sampler = DataSampler(dataset, mode="train")
    mod, per = gs.train(sampler, num_epochs=1)

    assert gs.best_model is not None
    assert isinstance(gs.best_model, MultiVAE)
    assert gs.best_model == mod
    assert per in gs.valid_scores

    gs.report()
    print(str(gs))


def test_HyperoptSearch():
    """Test the HyperoptSearch class
    """
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    dataset = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)

    gs = HyperoptSearch(MultiVAE, {
        "mvae_net" : ("MultiVAE_net", [{"dec_dims":[10, dataset.n_items]}]),
        "beta" : (.5, 1.),
        "anneal_steps" : [0, 1]
    }, ValidFunc(evaluate), "ndcg@1", num_eval=4)
    gs.report()

    assert hasattr(gs, "model_class")
    assert hasattr(gs, "params_domains")
    assert hasattr(gs, "valid_func")
    assert hasattr(gs, "valid_metric")
    assert hasattr(gs, "params_dicts")
    assert hasattr(gs, "valid_scores")
    assert hasattr(gs, "best_model")

    assert gs.model_class == MultiVAE
    assert gs.valid_metric == "ndcg@1"
    assert gs.best_model is None

    sampler = DataSampler(dataset, mode="train")
    mod, per = gs.train(sampler, num_epochs=1)
    gs.report()

    assert gs.best_model is not None
    assert isinstance(gs.best_model, MultiVAE)
    assert gs.best_model == mod
    assert per in gs.valid_scores
    print(str(gs))

def test_RandomSearch():
    """Test the RandomSearch class
    """
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    dataset = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)

    gs = RandomSearch(MultiVAE, {
        "mvae_net" : ("MultiVAE_net", [{"dec_dims":[10, dataset.n_items]}]),
        "beta" : (.5, 1.),
        "anneal_steps" : [0, 1]
    }, ValidFunc(evaluate), "ndcg@1", num_eval=4)
    gs.report()

    assert hasattr(gs, "model_class")
    assert hasattr(gs, "params_domains")
    assert hasattr(gs, "valid_func")
    assert hasattr(gs, "valid_metric")
    assert hasattr(gs, "params_dicts")
    assert hasattr(gs, "valid_scores")
    assert hasattr(gs, "best_model")

    assert gs.model_class == MultiVAE
    assert gs.valid_metric == "ndcg@1"
    assert gs.best_model is None

    sampler = DataSampler(dataset, mode="train")
    mod, per = gs.train(sampler, num_epochs=1)
    gs.report()

    assert gs.best_model is not None
    assert isinstance(gs.best_model, MultiVAE)
    assert gs.best_model == mod
    assert per in gs.valid_scores
    print(str(gs))

def test_BayesianSearch():
    """Test the BayesianSearch class
    """
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    dataset = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)

    gs = BayesianSearch(MultiVAE, {
        "mvae_net" : ("MultiVAE_net", [{"dec_dims":[10, dataset.n_items]}]),
        "beta" : (.5, 1.),
        "anneal_steps" : [0, 1]
    }, ValidFunc(evaluate), "ndcg@1", num_eval=4)
    gs.report()

    assert hasattr(gs, "model_class")
    assert hasattr(gs, "params_domains")
    assert hasattr(gs, "valid_func")
    assert hasattr(gs, "valid_metric")
    assert hasattr(gs, "params_dicts")
    assert hasattr(gs, "valid_scores")
    assert hasattr(gs, "best_model")

    assert gs.model_class == MultiVAE
    assert gs.valid_metric == "ndcg@1"
    assert gs.best_model is None

    sampler = DataSampler(dataset, mode="train")
    mod, per = gs.train(sampler, num_epochs=1)
    gs.report()

    assert gs.best_model is not None
    assert isinstance(gs.best_model, MultiVAE)
    assert gs.best_model == mod
    assert per in gs.valid_scores
    print(str(gs))
