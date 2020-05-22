r"""Module containing utility functions to evaluate recommendation engines.
"""
from functools import partial
import inspect
import random
import numpy as np
from .metrics import Metrics

__all__ = ['ValidFunc', 'evaluate', 'one_plus_random']

class ValidFunc():
    """Wrapper class for validation functions.

    When a validation function is passed to the method ``train`` of a
    :class:`rectorch.models.RecSysModel` must have e specific signature, that is three parameters:
    ``model``, ``test_loader`` and ``metric_list``. This class has to be used to adapt any
    evaluation function to this signature by partially initializing potential additional
    arguments.

    Parameters
    ----------
    func : :obj:`function`
        Evaluation function that has to be wrapped. The evaluation function must match the signature
        ``func(model, test_loader, metric_list, **kwargs)``.

    Attributes
    ----------
    func_name : :obj:`str`
        The name of the evalutaion function.
    function : :obj:`function`
        The wrapped evaluation function.

    Examples
    --------
    The :func:`one_plus_random` function has an additional argument ``r`` that must be initialized
    before using as a validation function inside a ``train`` method of a
    :class:`rectorch.models.RecSysModel`.

    >>> from rectorch.evaluation import ValidFunc, one_plus_random
    >>> opr = ValidFunc(one_plus_random, r=5)
    >>> opr
    ValidFunc(fun='one_plus_random', params={'r': 5})

    To call the validation function, simply call the :class:`ValidFunc` object with the required
    arguments.
    """
    def __init__(self, func, **kwargs):
        self.func_name = func.__name__
        self.function = partial(func, **kwargs)

        args = inspect.getfullargspec(self.function).args
        assert args == ["model", "test_loader", "metric_list"],\
            "A (partial) validation function must have the following kwargs: model, test_loader and\
            metric_list"

    def __call__(self, model, test_loader, metric):
        return self.function(model, test_loader, [metric])[metric]

    def __str__(self):
        kwdefargs = inspect.getfullargspec(self.function).kwonlydefaults
        return "ValidFunc(fun='%s', params=%s)" %(self.func_name, kwdefargs)

    def __repr__(self):
        return str(self)


def evaluate(model, test_loader, metric_list):
    r"""Evaluate the given method.

    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided :class:`rectorch.samplers.Sampler`
    (i.e.,  ``test_loader``).

    Parameters
    ----------
    model : :class:`rectorch.models.RecSysModel`
        The model to evaluate.
    test_loader : :class:`rectorch.samplers.Sampler`
        The test set loader.
    metric_list : :obj:`list` of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:

        ``matric_name`` @ ``k``

        where ``matric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method
        :func:`ndcg_at_k <rectorch.metrics.Metrics.ndcg_at_k>` with ``k=10``.

    Returns
    -------
    :obj:`dict` of :obj:`numpy.array`
        Dictionary with the results for each metric in ``metric_list``. Keys are string
        representing the metric, while the value is an array with the value of the metric
        computed on the users.
    """
    results = {m:[] for m in metric_list}
    for _, (data_tr, heldout) in enumerate(test_loader):
        data_tensor = data_tr.view(data_tr.shape[0], -1)
        recon_batch = model.predict(data_tensor)[0].cpu().numpy()
        heldout = heldout.view(heldout.shape[0], -1).cpu().numpy()
        res = Metrics.compute(recon_batch, heldout, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results


def one_plus_random(model, test_loader, metric_list, r=1000):
    r"""One plus random evaluation.

    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided :class:`rectorch.samplers.Sampler`
    (i.e.,  ``test_loader``). The evaluation methodology is one-plus-random [OPR]_ that can be
    summarized as follows. For each user, and for each test items, ``r`` random negative items are
    chosen and the metrics are computed w.r.t. to this subset of items (``r`` + 1 items in total).

    Parameters
    ----------
    model : :class:`rectorch.models.RecSysModel`
        The model to evaluate.
    test_loader : :class:`rectorch.samplers.Sampler`
        The test set loader.
    metric_list : :obj:`list` of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:

        ``matric_name`` @ ``k``

        where ``matric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method
        :func:`ndcg_at_k <rectorch.metrics.Metrics.ndcg_at_k>` with ``k=10``.
    r : :obj:`int`
        Number of negative items to consider in the "random" part.

    Returns
    -------
    :obj:`dict` of :obj:`numpy.array`
        Dictionary with the results for each metric in ``metric_list``. Keys are string
        representing the metric, while the value is an array with the value of the metric
        computed on the users.

    References
    ----------
    .. [OPR] Alejandro Bellogin, Pablo Castells, and Ivan Cantador. Precision-oriented Evaluation of
       Recommender Systems: An Algorithmic Comparison. In Proceedings of the Fifth ACM Conference on
       Recommender Systems (RecSys '11). ACM, New York, NY, USA, 333–336, 2011.
       DOI: https://doi.org/10.1145/2043932.2043996
    """
    results = {m:[] for m in metric_list}
    for _, (data_tr, heldout) in enumerate(test_loader):
        tot = set(range(heldout.shape[1]))
        data_tensor = data_tr.view(data_tr.shape[0], -1)
        recon_batch = model.predict(data_tensor)[0].cpu().numpy()
        heldout = heldout.view(heldout.shape[0], -1).cpu().numpy()

        users, items = heldout.nonzero()
        rows = []
        for u, i in zip(users, items):
            rnd = random.sample(tot - set(list(heldout[u].nonzero()[0])), r)
            rows.append(list(recon_batch[u][[i] + list(rnd)]))

        pred = np.array(rows)
        ground_truth = np.zeros_like(pred)
        ground_truth[:, 0] = 1
        res = Metrics.compute(pred, ground_truth, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results
