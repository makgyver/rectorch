r"""This module contains some baseline recommender systems.
"""
import os
from collections import Counter
from scipy.sparse import csr_matrix
import torch
import numpy as np
from rectorch.models import RecSysModel
from rectorch import env

__all__ = ['Random', 'Popularity']

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
