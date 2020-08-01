"""Unit tests for the rectorch.metrics module
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('..'))

from rectorch.metrics import Metrics


def test_ndcg_at_k():
    """Test Metric.ndcg_at_k
    """
    scores = np.array([[4., 3., 2., 1.]])
    gt = np.array([[1., 1., 0., 0.]])
    gt_2 = np.array([[0, 0, 1., 1.]])

    assert Metrics.ndcg_at_k(scores, gt, 2) == np.array([1.]), "ndcg@2 should be 1."
    assert Metrics.ndcg_at_k(scores, gt_2, 2) == np.array([0.]), "ndcg@2 should be 0."
    assert Metrics.ndcg_at_k(scores, gt, 3) == np.array([1.]), "ndcg@3 should be 1."
    eps = np.array([0.00001])
    assert np.abs(Metrics.ndcg_at_k(scores, gt_2, 3) - np.array([0.3065735964])) < eps,\
        "ndcg@3 should be 0.3065735964"


def test_recall_at_k():
    """Test Metric.recall_at_k
    """
    scores = np.array([[4., 3., 2., 1., 0.]])
    gt = np.array([[1., 1., 0., 0., 1.]])
    gt_2 = np.array([[0, 0, 1., 1., 1.]])

    assert Metrics.recall_at_k(scores, gt, 2) == np.array([1.]), "recall@2 should be 1."
    assert Metrics.recall_at_k(scores, gt_2, 2) == np.array([0.]), "recall@2 should be 0."
    eps = np.array([0.00001])
    assert np.abs(Metrics.recall_at_k(scores, gt, 3) - np.array([0.6666666])) < eps,\
        "recall@3 should be .66666666"
    assert np.abs(Metrics.recall_at_k(scores, gt_2, 3) - np.array([0.3333333])) < eps,\
        "recall@3 should be 0.3333333"


def test_hit_at_k():
    """Test Metric.hit_at_k
    """
    scores = np.array([[4., 3., 2., 1.], [1., 2., 3., 4.]])
    ground_truth = np.array([[0, 0, 1., 1.], [0, 0, 1., 1.]])
    t1 = Metrics.hit_at_k(scores, ground_truth, 3)
    t2 = Metrics.hit_at_k(scores, ground_truth, 2)
    assert np.all(t1 == np.array([1., 1.]))
    assert np.all(t2 == np.array([0., 1.]))


def test_mrr_at_k():
    """Test Metric.mrr_at_k
    """
    scores = np.array([[4., 2., 3., 1.], [1., 2., 3., 4.]])
    ground_truth = np.array([[0, 0, 1., 1.], [0, 0, 1., 1.]])
    t1 = Metrics.mrr_at_k(scores, ground_truth, 3)
    t2 = Metrics.mrr_at_k(scores, ground_truth, 1)
    assert np.all(t1 == np.array([.5, 1.]))
    assert np.all(t2 == np.array([0., 1.]))


def test_auc():
    """Test Metric.auc
    """
    scores = np.array([[4., 3., 2., 1.], [1., 2., 3., 4.], [1., 2., 3., 4.]])
    ground_truth = np.array([[0, 0, 1., 1.], [0, 0, 1., 1.], [1., 0, 0, 1.]])
    t1 = Metrics.auc(scores, ground_truth)
    assert np.all(t1 == np.array([0., 1., .5]))

def test_ap():
    """Test Metric.ap_at_k
    """
    scores = np.array([[4., 3., 2., 1.], [1., 2., 3., 4.], [1., 2., 3., 4.]])
    ground_truth = np.array([[0, 0, 1., 1.], [0, 0, 1., 1.], [1., 0, 0, 1.]])
    t1 = Metrics.ap_at_k(scores, ground_truth, 2)
    assert np.all(t1 == np.array([0., 1., .5]))


def test_compute():
    """Test Metric.compute
    """
    scores = np.array([[4., 3., 2., 1., 0.]])
    gt = np.array([[1., 1., 0., 0., 1.]])

    res = Metrics.compute(scores, gt, ["recall@2", "recall@3", "ndcg@2"])
    assert isinstance(res, dict), "res should be a dict"
    assert "recall@2" in res, "recall@2 should be in res"
    assert "recall@3" in res, "recall@3 should be in res"
    assert "ndcg@2" in res, "ndcg@2 should be in res"
    assert "ndcg@3" not in res, "ndcg@3 should not be in res"

    res = Metrics.compute(scores, gt, ["recall_at_k", "ndcg_at_k"])
    assert "recall_at_k" in res, "recall_at_k should be in res"
    assert "ndcg_at_k" in res, "ndcg_at_k should be in res"

    res = Metrics.compute(scores, gt, ["precision@10", "precision_at_k"])
    assert not res, "res should be empty"
