r"""Module containing utility functions to evaluate recommendation engines.
"""
import numpy as np
from .metrics import Metrics

__all__ = ['evaluate']

def evaluate(model, test_loader, metric_list):
    r"""Evaluate the given method.

    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided :class:`sampler.Sampler`
    (i.e.,  ``test_loader``).

    Parameters
    ----------
    model : :class:`models.RecSysModel`
        The model to evaluate.
    test_loader : :class:`sampler.Sampler`
        The test set loader.
    metric_list : :obj:`list` of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:

        ``matric_name`` @ ``k``

        where ``matric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method :py:func:`ndcg_at_k <metrics.Metrics.ndcg_at_k>`
        with ``k=10``.

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
