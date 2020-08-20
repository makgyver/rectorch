r"""This module contains classes and methods for perfoming model selection/validation.
"""
import importlib
import itertools
from functools import partial
import inspect
import numpy as np
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin
from rectorch import env
from rectorch.models import RecSysModel

__all__ = ['ValidFunc', 'HPSearch', 'GridSearch', 'BayesianSearch']


class ValidFunc():
    r"""Wrapper class for validation functions.

    When a validation function is passed to the method ``train`` of a
    :class:`rectorch.models.RecSysModel` must have e specific signature, that is three parameters:
    ``model``, ``test_sampler`` and ``metric_list``. This class has to be used to adapt any
    evaluation function to this signature by partially initializing potential additional
    arguments.

    Parameters
    ----------
    func : :obj:`function`
        Evaluation function that has to be wrapped. The evaluation function must match the signature
        ``func(model, test_sampler, metric_list, **kwargs)``.

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
        assert args == ["model", "test_sampler", "metric_list"],\
            "A (partial) validation function must have the following kwargs: model, test_sampler\
            and metric_list"

    def __call__(self, model, test_sampler, metric):
        return self.function(model, test_sampler, [metric])[metric]

    def __str__(self):
        kwdefargs = inspect.getfullargspec(self.function).kwonlydefaults
        return "ValidFunc(fun='%s', params=%s)" %(self.func_name, kwdefargs)

    def __repr__(self):
        return str(self)


class HPSearch(RecSysModel):
    """Abstract hyper-parameter search algorithm.

    Parameters
    ----------
    model_class : class model from :mod:`rectorch.models` module
        The class of the model.
    params_domains : :obj:`dict`
        Dictionary containing the hyper-parametrs' sets for initilizing the searching strategy.
    valid_func : :class:`rectorch.validation.ValidFunc`
        The validation function.
    valid_metric : :obj:`str`
        The metric used during the validation to select the best model.

    Attributes
    ----------
    model_class : model class from :mod:`rectorch.models.nn` module
        The class of the model.
    params_domains : :obj:`dict`
        Dictionary containing the hyper-parametrs' sets for initilizing the searching strategy.
    valid_func : :class:`rectorch.validation.ValidFunc`
        The validation function.
    valid_metric : :obj:`str`
        The metric used during the validation to select the best model.
    params_dicts : :obj:`list` of :obj:`dict`
        List of dictionaries representing the different entries of the grid.
    size : :obj:`int`
        The size of the grid in terms of how many configurations have to be validated.
    valid_scores : :obj:`list` of :obj:`float`
        The scores obtained by the different models. If empty it means that the grid search has
        not been performed yet.
    best_model : trained model from :mod:`rectorch.models.nn` module
        The best performing model on the validation set.
    """
    def __init__(self,
                 model_class,
                 params_domains,
                 valid_func,
                 valid_metric):
        self.model_class = model_class
        self.params_domains = params_domains
        self.valid_func = valid_func
        self.valid_metric = valid_metric
        self.best_model = None
        self.valid_scores = []
        self.params_dicts = []

    def train(self, data_sampler, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        return self.best_model.predict(*args, **kwargs)

    def save_model(self, filepath):
        r"""Save the best model to file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file to save the model.

        Notes
        -----
        Using the ``save_model`` method of the ``HPSearch`` class is actually the same as invoking
        the ``save_model`` of the best model.
        """
        self.best_model.save_model(filepath)

    @classmethod
    def load_model(cls, filepath, model_class):
        r"""Load a model from file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.
        model_class : class model from :mod:`rectorch.models` module
            The class of the model.

        Returns
        -------
        :class:`rectorch.models.RecSysModel`
            A recommendation model.

        Notes
        -----
        Using the ``load_model`` method of the ``HPSearch`` class is actually the same as invoking
        the ``load_model`` of the model class.
        """
        return model_class.load_model(filepath)

    def report(self):
        r"""Output a report of the grid search.

        The output consists of pairs of model's hyper-parameters setting with its achieved
        score on the validation set.
        """
        if not self.valid_scores:
            env.logger.info("Grid search has not been performed, yet! Call the 'train' method.")
        else:
            for i, p in enumerate(self.params_dicts):
                env.logger.info(p, ":", self.valid_scores[i])


class GridSearch(HPSearch):
    r"""Perform a hyper-parameters grid search to select the best setting.

    The GridSearch class is a sub-class of RecSysModel and hence it can be used as a trained model.
    After training, the GridSearch object is like a wrapper for the model class for which
    it has performed model selection.

    Parameters
    ----------
    model_class : class model from :mod:`rectorch.models` module
        The class of the model.
    params_grid : :obj:`dict`
        Dictionary containing the hyper-parametrs' sets for constructing the grid. A key represents
        the name of the parameter in the signature of the class model, while the value is the list
        of values for that hyper-parameters to try. Hyper-parameters
        of a neural network (i.e., object of the module :mod:`rectorch.nets`) must be specified
        using a tuple where the first element is the name of the nets class, while the second the
        list of hyper-parameters for that neural network inside e dictionary.
    valid_func : :class:`rectorch.validation.ValidFunc`
        The validation function.
    valid_metric : :obj:`str`
        The metric used during the validation to select the best model.

    Attributes
    ----------
    model_class : model class from :mod:`rectorch.models.nn` module
        The class of the model.
    params_grid : :obj:`dict`
        Dictionary containing the hyper-parametrs' sets for constructing the grid. A key represents
        the name of the parameter in the signature of the class model, while the value is the list
        of values for that hyper-parameters to try. Hyper-parameters
        of a neural network (i.e., object of the module :mod:`rectorch.nets`) must be specified
        using a tuple where the first element is the name of the nets class, while the second the
        list of hyper-parameters for that neural network inside e dictionary.
    valid_func : :class:`rectorch.validation.ValidFunc`
        The validation function.
    valid_metric : :obj:`str`
        The metric used during the validation to select the best model.
    params_dicts : :obj:`list` of :obj:`dict`
        List of dictionaries representing the different entries of the grid.
    size : :obj:`int`
        The size of the grid in terms of how many configurations have to be validated.
    valid_scores : :obj:`list` of :obj:`float`
        The scores obtained by the different models. If empty it means that the grid search has
        not been performed yet.
    best_model : trained model from :mod:`rectorch.models.nn` module
        The best performing model on the validation set.

    Examples
    --------
    Given a ``dataset`` (of the class :class:`rectorch.data.Dataset`) object:

    >>> from rectorch.evaluation import GridSearch, ValidFunc, evaluate
    >>> from rectorch.models.nn import MultiVAE
    >>> from rectorch.nets import MultiVAE_net
    >>> from rectorch.samplers import DataSampler
    >>> n_items = dataset.n_items
    >>> grid = GridSearch(MultiVAE,
    >>>                   {"mvae_net" : ("MultiVAE_net",
    >>>                                  [{"dec_dims":[50, n_items]}, {"dec_dims":[100, n_items]}]),
    >>>                    "beta" : [.2, .5],
    >>>                    "anneal_steps" : [0, 100]},
    >>>                   ValidFunc(evaluate),
    >>>                   "ndcg@10")
    >>> sampler = DataSampler(dataset, mode="train")
    >>> best_model, best_ndcg10 = grid.train(sampler, num_epochs=10)
    """
    def __init__(self, model_class, params_grid, valid_func, valid_metric):
        super(GridSearch, self).__init__(model_class, params_grid, valid_func, valid_metric)
        self.params_grid = {}
        for k, v in self.params_domains.items():
            if isinstance(v, tuple):
                self.params_grid[k] = [(v[0], vv) for vv in v[1]]
            else:
                self.params_grid[k] = v

        params_list = [[(k, x) for x in v] for k, v in self.params_grid.items()]
        self.params_dicts = [dict(x) for x in list(itertools.product(*params_list))]
        self.size = len(self.params_dicts)

    def _model_generator(self):
        #model_cls = getattr(importlib.import_module("rectorch.models.nn"), self.model_class)
        model_cls = self.model_class
        for params in self.params_dicts:
            for p, v in params.items():
                if isinstance(v, tuple):
                    net_class = getattr(importlib.import_module("rectorch.nets"), v[0])
                    params[p] = net_class(**v[1])
            yield model_cls(**params)

    def train(self, data_sampler, **kwargs):
        r"""Perform the grid search.

        Parameters
        ----------
        data_sampler : :class:`rectorch.samplers.Sampler`
            The data sampler.

        Returns
        -------
        best_model : trained model from :mod:`rectorch.models.nn` module
            The best performing model on the validation set.
        best_perf : :obj:`float`
            The performance of the best model on the validation set.
        """
        best_perf = -np.inf
        best_model = None

        for model in self._model_generator():
            data_sampler.train()
            model.train(data_sampler, **kwargs)

            data_sampler.valid()
            valid_res = self.valid_func(model, data_sampler, self.valid_metric)
            mu_val = np.mean(valid_res)

            self.valid_scores.append(mu_val)
            if mu_val > best_perf:
                best_perf = mu_val
                best_model = model

        self.best_model = best_model
        return self.best_model, best_perf


class BayesianSearch(HPSearch):
    """Bayesian hyper-parameter optimization.

    **UNDOCUMENTED**

    Parameters
    ----------
    model_class : class model from :mod:`rectorch.models` module
        The class of the model.
    params_domains : :obj:`dict`
        Dictionary containing the hyper-parametrs' sets for initilizing the searching strategy.
    valid_func : :class:`rectorch.validation.ValidFunc`
        The validation function.
    valid_metric : :obj:`str`
        The metric used during the validation to select the best model.
    num_eval : :obj:`int` [optional]
        Number of evaluation points, by default 5.

    Attributes
    ----------
    model_class : model class from :mod:`rectorch.models.nn` module
        The class of the model.
    params_domains : :obj:`dict`
        Dictionary containing the hyper-parametrs' sets for initilizing the searching strategy.
    valid_func : :class:`rectorch.validation.ValidFunc`
        The validation function.
    valid_metric : :obj:`str`
        The metric used during the validation to select the best model.
    params_dicts : :obj:`list` of :obj:`dict`
        List of dictionaries representing the different entries of the grid.
    size : :obj:`int`
        The size of the grid in terms of how many configurations have to be validated.
    valid_scores : :obj:`list` of :obj:`float`
        The scores obtained by the different models. If empty it means that the grid search has
        not been performed yet.
    best_model : trained model from :mod:`rectorch.models.nn` module
        The best performing model on the validation set.

    Examples
    --------
    Given a ``dataset`` (of the class :class:`rectorch.data.Dataset`) object:

    >>> from rectorch.evaluation import BayesianSearch, ValidFunc, evaluate
    >>> from rectorch.models.nn import MultiVAE
    >>> from rectorch.samplers import DataSampler
    >>> sampler = DataSampler(dataset, mode="train")
    >>> n_items = dataset.n_items
    >>> params = {"mvae_net" : ("MultiVAE_net", [{"dec_dims":[50, n_items]},
    >>>                                          {"dec_dims":[100, n_items]}]),
    >>>           "beta" : (0., 1.),
    >>>           "anneal_steps" : [0, 100]}
    >>> bs = BayesianSearch(MultiVAE, params, ValidFunc(evaluate), "ndcg@10", 4)
    >>> best_model, best_ndcg10 = bs.train(sampler, num_epochs=2)
    """
    def __init__(self,
                 model_class,
                 params_range,
                 valid_func,
                 valid_metric,
                 num_eval=5):
        super(BayesianSearch, self).__init__(model_class, params_range, valid_func, valid_metric)
        self.num_eval = num_eval
        self.space = {}
        self._params = {}

        for k, v in self.params_domains.items():
            if isinstance(v, tuple):
                if isinstance(v[1], list):
                    net_class = getattr(importlib.import_module("rectorch.nets"), v[0])
                    nets = [net_class(**p) for p in v[1]]
                    self.space[k] = hp.choice(k, nets)
                    self._params[k] = nets
                else:
                    self.space[k] = hp.uniform(k, v[0], v[1])
                    self._params[k] = None
            elif isinstance(v, list):
                self.space[k] = hp.choice(k, v)
                self._params[k] = v
            else:
                raise ValueError()

    def train(self, data_sampler, *args, **kwargs):
        def objective(params):
            model = self.model_class(**params)
            data_sampler.train()
            model.train(data_sampler, *args, **kwargs)
            data_sampler.valid()
            scores = self.valid_func(model, data_sampler, self.valid_metric)
            return {'loss': 1 - np.mean(scores), 'params': params, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.num_eval,
                    trials=trials)

        best_params = {}
        for k, v in best.items():
            best_params[k] = v if self._params[k] is None else self._params[k][v]

        self.best_model = self.model_class(**best_params)
        data_sampler.train()
        self.best_model.train(data_sampler, *args, **kwargs)
        best_result = 1. - max([trial["result"]["loss"] for trial in trials.trials])
        for trial in trials.trials:
            self.valid_scores.append(1. - trial["result"]["loss"])
            self.params_dicts.append(trial["result"]["params"])
        return self.best_model, best_result
