r"""This module conntains some baseline recommender systems.
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

    def train(self):
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

    def load_model(self, filepath):
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        env.logger.info("Loading model checkpoint from %s...", filepath)
        with open(filepath, "r") as f:
            self.n_items = int(f.readline().strip())
            self.seed = int(f.readline().strip())
            self.fixed = bool(f.readline().strip())
        env.logger.info("Model checkpoint loaded!")
        return self.seed


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

    def train(self, train_data, retrain=False):
        r"""Compute the items' popularity.

        Parameters
        ----------
        train_data : :obj:`dict` or :class:`torch.Tensor` or :class:`scipy.sparse.csr_matrix`
            The rating matrix.
        retrain : :obj:`bool` [optional]
            Whether the popularity must be recomputed or not, by default :obj:`False`.
            If :obj:`False` the computation is avoided iff the model is not empty
            (i.e., :obj:`None`).
        """
        if not retrain and self.model is not None:
            return

        if isinstance(train_data, csr_matrix):
            nparray = np.array(train_data.sum(axis=0)).flatten()
            self.model = torch.from_numpy(nparray).float()
        if isinstance(train_data, (torch.FloatTensor, torch.sparse.FloatTensor)):
            self.model = torch.sum(train_data, 0)
        elif isinstance(train_data, dict):
            occs = Counter([i for items in train_data.values() for i in items])
            self.model = torch.zeros(self.n_items, dtype=torch.float)
            for i in occs.keys():
                self.model[i] = float(occs[i])

    def predict(self, users, train_items, remove_train=True):
        pred = self.model.repeat([len(users), 1])
        if remove_train:
            for u in range(len(train_items)):
                pred[u, train_items[u]] = -np.inf
        return (pred, )

    def save_model(self, filepath):
        env.logger.info("Saving model checkpoint to %s...", filepath)
        torch.save({"model" : self.model}, filepath)
        env.logger.info("Model checkpoint saved!")

    def load_model(self, filepath):
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        env.logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath)
        self.model = checkpoint['model']
        env.logger.info("Model checkpoint loaded!")
        return checkpoint
