"""Unit tests for the rectorch.evaluation module
"""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath('..'))

from evaluation import evaluate
from models import RecSysModel
from samplers import Sampler

class FakeModel(RecSysModel):
    """Fake model
    """
    def predict(self, x, *args, **kwargs):
        return (x + torch.FloatTensor([[1]*4]), )


class FakeSampler(Sampler):
    """Fake sampler
    """
    def __iter__(self):
        scores = [torch.FloatTensor([[4., 3., 2., 1.]]), torch.FloatTensor([[4., 3., 2., 1.]])]
        gt = [torch.FloatTensor([[1., 1., 0., 0.]]), torch.FloatTensor([[0, 0, 1., 1.]])]

        for i in range(2):
            yield scores[i], gt[i]


def test_evaluate():
    """Test the evaluate function
    """
    model = FakeModel()
    sampl = FakeSampler()
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
