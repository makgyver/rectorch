r"""This module contains some baseline recommender systems.
"""
import os
import time
from collections import Counter
from scipy.sparse import csr_matrix
import torch
import numpy as np
import cvxopt as co
import cvxopt.solvers as solver
from rectorch.models import RecSysModel
from rectorch.utils import md_kernel, kernel_normalization, cvxopt_diag
from rectorch import env

__all__ = ['Random', 'Popularity', 'CF_KOMD']

class Random(RecSysModel):
    r"""Random recommender.

    This recommendation method simply give random recommendations.

    Parameters
    ----------
    n_items : :obj:`int`
        Number of items.
    seed : :obj:`int` [optional]
        The random seed, by default 0.
    fixed : :obj:`bool` [optional]
        Whether the random recommendation is the same for each user or not. By default :obj:`False`.

    Attributes
    ----------
    n_items : :obj:`int`
        Number of items.
    seed : :obj:`int`
        The random seed.
    fixed : :obj:`bool`
        Whether the random recommendation is the same for each user or not.
    """
    def __init__(self, n_items, seed=0, fixed=False):
        super(Random, self).__init__()
        self.n_items = n_items
        self.seed = seed
        self.fixed = fixed

    def train(self, data_sampler=None):
        pass

    def predict(self, users, train_items, remove_train=True):
        torch.random.manual_seed(self.seed)
        if not self.fixed:
            pred = torch.randn(len(users), self.n_items, dtype=torch.float)
        else:
            pred = torch.randn(self.n_items, dtype=torch.float)
            pred = pred.repeat([len(users), 1])

        if remove_train:
            for u in range(len(train_items)):
                pred[u, train_items[u]] = -np.inf
        return (pred, )

    def save_model(self, filepath):
        env.logger.info("Saving model checkpoint to %s...", filepath)
        with open(filepath, "w") as f:
            f.write(str(self.n_items) + "\n")
            f.write(str(self.seed) + "\n")
            f.write(str(1 if self.fixed else 0))
        env.logger.info("Model checkpoint saved!")

    @classmethod
    def load_model(cls, filepath):
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        env.logger.info("Loading model checkpoint from %s...", filepath)
        n_items, seed, fixed = 0, 0, False
        with open(filepath, "r") as f:
            n_items = int(f.readline().strip())
            seed = int(f.readline().strip())
            fixed = bool(f.readline().strip())
        rnd = Random(n_items, seed, fixed)
        env.logger.info("Model checkpoint loaded!")
        return rnd


class Popularity(RecSysModel):
    r"""Popularity-based recommender.

    It recommends the most popular (i.e., rated) items.

    Parameters
    ----------
    n_items : :obj:`int`
        Number of items.

    Attributes
    ----------
    n_items : :obj:`int`
        Number of items.
    model : :class:`torch.FloatTensor`
        The array of items' scores.
    """
    def __init__(self, n_items):
        super(Popularity, self).__init__()
        self.n_items = n_items
        self.model = None

    def train(self, data_sampler, retrain=False):
        r"""Compute the items' popularity.

        Parameters
        ----------
        data_sampler : :class:`rectorch.samplers.DummySampler`
            The training sampler.
        retrain : :obj:`bool` [optional]
            Whether the popularity must be recomputed or not, by default :obj:`False`.
            If :obj:`False` the computation is avoided iff the model is not empty
            (i.e., :obj:`None`).
        """
        if not retrain and self.model is not None:
            return

        train_data = data_sampler.data_tr
        if isinstance(train_data, csr_matrix):
            nparray = np.array(train_data.sum(axis=0)).flatten()
            self.model = torch.from_numpy(nparray).float()
        elif isinstance(train_data, torch.FloatTensor):
            self.model = torch.sum(train_data, 0)
        elif isinstance(train_data, torch.sparse.FloatTensor):
            self.model = torch.sparse.sum(train_data, 0).to_dense()
        elif isinstance(train_data, dict):
            occs = Counter([i for items in train_data.values() for i in items])
            self.model = torch.zeros(self.n_items, dtype=torch.float)
            for i in occs.keys():
                self.model[i] = float(occs[i])
        elif isinstance(train_data, np.ndarray):
            self.model = torch.from_numpy(train_data.sum(axis=0)).float()

    def predict(self, users, train_items, remove_train=True):
        pred = self.model.repeat([len(users), 1])
        if remove_train:
            for u in range(len(users)):
                pred[u, train_items[u]] = -np.inf
        return (pred, )

    def save_model(self, filepath):
        env.logger.info("Saving model checkpoint to %s...", filepath)
        torch.save({"model" : self.model, "n_items": self.n_items}, filepath)
        env.logger.info("Model checkpoint saved!")

    @classmethod
    def load_model(cls, filepath):
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        env.logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath)
        pop = Popularity(checkpoint['n_items'])
        pop.model = checkpoint['model']
        env.logger.info("Model checkpoint loaded!")
        return pop


class CF_KOMD(RecSysModel):
    """CF-KOMD: kernel-based collaborative filtering.

    CF-KOMD is a kernel-based method proposed by Polato et al. [CFKOMD1]_, [CFKOMD2]_, [CFKOMD3]_.
    The method is based on the concept of maximal margin (like in SVM) to compute the ranking
    between items. The optimization problem producing the ranking can be computed in parallel since
    for each user a different (and rather small) optimization problem is created.

    Parameters
    ----------
    lambda_p : :obj:`float` [optional]
        The non-negative regularization hyper-parameter  to penalize non-smooth weights,
        by default 0.1.
    ker_fun : :obj:`string` [optional]
        The kernel function to use. Two kernel functions are currently supported "linear" and
        "disjunctive", by default "linear".
    disj_degree : :obj:`int` [optional]
        The degree of the disjunctive kernel, by default 1. Note: the degree 1 is the same as using
        the linear kernel.

    Attributes
    ----------
    lambda_p : :obj:`float`
        The regularization hyper-parameter to penalize non-smooth weights.
    ker_fun : :obj:`string`
        The kernel function to use. Two kernel functions are currently supported "linear" and
        "disjunctive".
    disj_degree : :obj:`int`
        The degree of the disjunctive kernel.
    model : :obj:`dict` of :class:`torch.Tensor`
        The CF-KOMD model(s).

    References
    ----------
    .. [CFKOMD1] Mirko Polato and Fabio Aiolli. 2018. Boolean kernels for collaborative filtering in
       top-N item recommendation.  Neurocomputing, Elsevier Science Ltd., Vol. 286, pp. 214-225,
       Oxford, UK. DOI: 10.1016/j.neucom.2018.01.057, ISSN: 0925-2312.
    .. [CFKOMD2] Mirko Polato and Fabio Aiolli. 2017. Exploiting sparsity to build efficient kernel
       based collaborative filtering for top-N item recommendation. Neurocomputing, Elsevier Science
       Ltd., Vol. 268, pp. 17-26, Oxford, UK, doi: 10.1016/j.neucom.2016.12.090, ISSN: 0925-2312.
    .. [CFKOMD3] Mirko Polato and Fabio Aiolli. 2016. Kernel based collaborative filtering for very
       large scale top-N item recommendation". In Proceedings of the European Symposium on
       Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN'16),
       Bruges, BE, ISBN: 978-287587027-8, 2016.
    """
    def __init__(self, ker_fun="linear", disj_degree=1, lam=0.1):
        super(CF_KOMD, self).__init__()
        self.lambda_p = lam
        self.ker_fun = ker_fun
        self.disj_degree = disj_degree
        self.model = {}
        solver.options['show_progress'] = False

    def _compute_kernel(self, X):
        if self.ker_fun == "linear":
            return np.dot(X.T, X)
        elif self.ker_fun == "disjunctive":
            assert self.disj_degree >= 1, "The 'disj_degree' must be an integer >= 1."
            return md_kernel(X, self.disj_degree)
        else:
            raise ValueError("'ker_fun' can only be 'linear' or 'disjunctive'.")

    def train(self, data_sampler, only_test=False, verbose=1):
        """Training procedure of CF-KOMD.

        Parameters
        ----------
        data_sampler : :class:`samplers.ArrayDummySampler`
            The sampler object that load the training/validation set in mini-batches.
        only_test : :obj:`bool` [optional]
            Whether to build the model only for the test users, by default :obj:`False`.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum (that depends on the size of
            the training set) verbosity higher values will not have any effect.
        """
        start_time = time.time()

        if only_test:
            data_sampler.test()
            (test_users, _), _ = next(iter(data_sampler))
            data_sampler.train()
        else:
            test_users = range(data_sampler.data.n_users)

        X = data_sampler.data_tr
        env.logger.info("Computing {} kernel".format(self.ker_fun))
        K = self._compute_kernel(X)
        K = kernel_normalization(K)
        K = co.matrix(K)

        q_ = co.matrix(0.0, (K.size[0], 1))
        for i in range(K.size[0]):
            q_[i, 0] = sum(K[i, :]) / float(K.size[0]) #-1

        env.logger.info("Kernel computed!")

        log_delay = max(100, len(test_users) // 10**verbose)
        b = co.matrix(1.0)
        batch_start_time = time.time()
        for i, u in enumerate(test_users):
            if (i+1) % log_delay == 0:
                elapsed = time.time() - batch_start_time
                env.logger.info('| user {}/{} | ms/user {:.2f} |'
                                .format(i + 1, len(test_users), elapsed * 1000 / log_delay))
                batch_start_time = time.time()

            Xp = X[u, :].nonzero()[0].tolist()
            num_pos = len(Xp)

            Kp = K[Xp, Xp]
            kn = q_[Xp, :]

            I = cvxopt_diag(co.matrix(1.0, (num_pos, 1)))
            P = (Kp + self.lambda_p * I)
            q = -kn
            G = -I
            h = co.matrix(0.0, (num_pos, 1))
            A = co.matrix(1.0, (1, num_pos))

            sol = solver.qp(P, q, G, h, A, b)
            self.model[u] = K[Xp, :].T * sol['x'] - q_
            self.model[u] = np.array(self.model[u]).flatten()
            self.model[u] = torch.from_numpy(self.model[u]).float()

        env.logger.info('| training complete | total training time {:.2f} s |'
                        .format(time.time() - start_time))

    def predict(self, users, train_items, remove_train=True):
        pred = torch.stack([self.model[u] for u in users])
        if remove_train:
            for u in range(len(users)):
                pred[u, train_items[u].nonzero()[0]] = -np.inf
        pred = torch.FloatTensor(pred)
        return (pred, )

    def save_model(self, filepath):
        state = {'lam': self.lambda_p,
                 'model': self.model,
                 'ker_fun': self.ker_fun,
                 'disj_degree': self.disj_degree
                }
        env.logger.info("Saving CF_KOMD model to %s...", filepath)
        torch.save(state, filepath)
        env.logger.info("Model saved!")

    @classmethod
    def load_model(cls, filepath):
        assert os.path.isfile(filepath), "The model file %s does not exist." %filepath
        env.logger.info("Loading CF_KOMD model from %s...", filepath)
        state = torch.load(filepath)
        cfkomd = CF_KOMD(lam=state["lam"],
                         ker_fun=state["ker_fun"],
                         disj_degree=state["disj_degree"])
        cfkomd.model = state["model"]
        env.logger.info("Model loaded!")
        return cfkomd
