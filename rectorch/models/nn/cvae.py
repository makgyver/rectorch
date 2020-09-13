r"""Conditioned Variatonal Autoencoder for collaborative filtering.

Conditioned Variational Autoencoder (C-VAE) for constrained top-N item recommendation can
recommend items that have to satisfy a given condition. The architecture is similar to a
standard VAE in which the condition vector is fed into the encoder.
The loss function can be seen in two ways:

* same as in :class:`MultiVAE` but with a different target reconstruction. Infact, the
    network has to reconstruct only those items satisfying a specific condition;
* a modified loss which performs the filtering by itself.

More details about the loss function are given in the paper [CVAE]_.

The training process is almost identical to the one of :class:`MultiVAE` but the sampler
must be a :class:`rectorch.samplers.ConditionedDataSampler`.

References
----------
.. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
   Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
   https://arxiv.org/abs/2004.11141
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
import torch
from torch import nn
import torch.nn.functional as F
from rectorch import env
from rectorch.models.nn.multvae import MultVAE_net, MultVAE, MultVAE_trainer
from rectorch.evaluation import evaluate
from rectorch.validation import ValidFunc
from rectorch.samplers import DataSampler, Sampler

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["ConditionedDataSampler", "EmptyConditionedDataSampler", "CMultVAE_net", "MultVAE"]


class ConditionedDataSampler(DataSampler):
    r"""Data sampler with conditioned filtering used by the
    :class:`rectorch.models.nn.CMultiVAE` model.

    This data sampler is useful when training the :class:`rectorch.models.nn.CMultiVAE` model
    described in [CVAE]_. During the training, each user must be conditioned over all the possible
    conditions (actually the ones that the user knows) so the training set must be modified
    accordingly.

    Parameters
    ----------
    iid2cids : :obj:`dict` { :obj:`int` \: :obj:`list` of :obj:`int` }
        Dictionary that maps each item to the list of all valid conditions for that item. Items
        are referred to with the inner id, and conditions with an integer in the range 0,
        ``n_cond`` -1.
    n_cond : :obj:`int`
        Number of possible conditions.
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must bu randomly shuffled before creating the batches, by default
        :obj:`True`.

    Attributes
    ----------
    all attributes : see **Parameters** section.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    """
    def __init__(self,
                 iid2cids,
                 n_cond,
                 data,
                 mode="train",
                 batch_size=1,
                 shuffle=True):
        super(ConditionedDataSampler, self).__init__(data, mode, batch_size, shuffle)
        self.iid2cids = iid2cids
        self.n_cond = n_cond
        self._compute_conditions()

    def _compute_conditions(self):
        r2cond = {}
        for i, row in enumerate(self.sparse_data_tr):
            _, cols = row.nonzero()
            r2cond[i] = set.union(*[set(self.iid2cids[c]) for c in cols])

        self.examples = [(r, -1) for r in r2cond]
        self.examples += [(r, c) for r in r2cond for c in r2cond[r]]
        self.examples = np.array(self.examples)
        del r2cond

        rows = [m for m in self.iid2cids for _ in range(len(self.iid2cids[m]))]
        cols = [g for m in self.iid2cids for g in self.iid2cids[m]]
        values = np.ones(len(rows))
        self.M = csr_matrix((values, (rows, cols)), shape=(len(self.iid2cids), self.n_cond))

    def _set_mode(self, mode="train", batch_size=None):
        if self.sparse_data_tr is not None:
            super()._set_mode(mode, batch_size)
            self._compute_conditions()
        else:
            super()._set_mode(mode, batch_size)

    def __len__(self):
        return int(np.ceil(len(self.examples) / self.batch_size))

    def __iter__(self):
        n = len(self.examples)
        idxlist = list(range(n))
        if self.shuffle and self.mode == "train":
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            ex = self.examples[idxlist[start_idx:end_idx]]
            rows, cols = [], []
            for i, (_, c) in enumerate(ex):
                if c >= 0:
                    rows.append(i)
                    cols.append(c)

            values = np.ones(len(rows))
            cond_matrix = csr_matrix((values, (rows, cols)), shape=(len(ex), self.n_cond))

            rows_ = [r for r, _ in ex]
            data_tr = hstack([self.sparse_data_tr[rows_], cond_matrix], format="csr")

            if self.sparse_data_te is None:
                self.sparse_data_te = self.sparse_data_tr

            for i, (_, c) in enumerate(ex):
                if c < 0:
                    rows += [i] * self.n_cond
                    cols += range(self.n_cond)

            values = np.ones(len(rows))
            cond_matrix = csr_matrix((values, (rows, cols)), shape=(len(ex), self.n_cond))
            filtered = cond_matrix.dot(self.M.transpose().tocsr()) > 0
            data_te = self.sparse_data_te[rows_].multiply(filtered)

            filter_idx = np.diff(data_te.indptr) != 0
            data_te = data_te[filter_idx]
            data_tr = data_tr[filter_idx]

            data_te = torch.FloatTensor(data_te.toarray())
            data_tr = torch.FloatTensor(data_tr.toarray())

            yield data_tr, data_te


class EmptyConditionedDataSampler(DataSampler):
    r"""Data sampler that returns unconditioned batches used by the
    :class:`rectorch.models.nn.CMultiVAE` model.

    This data sampler is useful when training the :class:`rectorch.models.nn.CMultiVAE` model
    described in [CVAE]_. This sampler is very similar to :class:`DataSampler` with the expection
    that the yielded batches have appended a zero matrix of the size ``batch_size`` :math:`\times`
    ``n_cond``.

    Parameters
    ----------
    n_cond : :obj:`int`
        Number of possible conditions.
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must bu randomly shuffled before creating the batches, by default
        :obj:`True`.

    Attributes
    ----------
    all attributes : see **Parameters** section.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    """
    def __init__(self,
                 cond_size,
                 data,
                 mode="train",
                 batch_size=1,
                 shuffle=True):
        super(EmptyConditionedDataSampler, self).__init__(data, mode, batch_size, shuffle)
        self.cond_size = cond_size

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.sparse_data_tr.shape[0]
        idxlist = list(range(n))
        if self.shuffle and self.mode == "train":
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data_tr = self.sparse_data_tr[idxlist[start_idx:end_idx]]
            cond_matrix = csr_matrix((data_tr.shape[0], self.cond_size))
            data_tr = hstack([data_tr, cond_matrix], format="csr")
            data_tr = torch.FloatTensor(data_tr.toarray())

            if self.sparse_data_te is None:
                self.sparse_data_te = self.sparse_data_tr

            data_te = self.sparse_data_te[idxlist[start_idx:end_idx]]
            data_te = torch.FloatTensor(data_te.toarray())

            yield data_tr, data_te


class CMultVAE_net(MultVAE_net):
    r'''Conditioned Variational Autoencoder network for collaborative filtering.

    The network structure follows the definition as in [CVAE]_. Hidden layers are fully
    connected and *tanh* activated. The output layer of both the encoder and the decoder
    are linearly activated.

    Parameters
    ----------
    cond_dim : :obj:`int`
        The size of the condition vector.
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :class:`AE_net`.
    enc_dims : :obj:`list`, array_like of :obj:`int` or None [optional]
        See :class:`AE_net`.
    dropout : :obj:`float` [optional]
        See :class:`VAE_net`.

    Attributes
    ----------
    cond_dim : :obj:`int`
        See :attr:`cond_dim` parameter.
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`dec_dims` parameter.
    enc_dims : :obj:`list` or array_like
        See :attr:`end_dims` parameter.
    dropout : :obj:`float`
        The dropout layer that is applied to the input during the :meth:`VAE_net.forward`.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    '''
    def __init__(self, cond_dim, dec_dims, enc_dims=None, dropout=0.5):
        super(CMultVAE_net, self).__init__(dec_dims, enc_dims, dropout)
        self.cond_dim = cond_dim

        temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        temp_dims[0] += self.cond_dim
        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
        self.init_weights()

    def encode(self, x):
        h1 = F.normalize(x[:, :-self.cond_dim])
        if self.training:
            h1 = self._dropout_layer(h1)
        h = torch.cat((h1, x[:, -self.cond_dim:]), 1)
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def get_state(self):
        state = super().get_state()
        state["name"] = self.__class__.__name__
        state["params"]["cond_dim"] = self.cond_dim
        return state

class CMultVAE_trainer(MultVAE_trainer):
    pass

class CMultVAE(MultVAE):
    r"""Conditioned Variatonal Autoencoder for collaborative filtering.

    Parameters
    ----------
    iid2cids : :obj:`dict` { :obj:`int` \: :obj:`list` of :obj:`int` }
        Dictionary that maps each item to the list of all valid conditions for that item. Items
        are referred to with the inner id, and conditions with an integer in the range 0,
        ``n_cond`` -1.
    n_cond : :obj:`int`
        Number of possible conditions.
    dec_dims : :obj:`list` or array_like of :obj:`int` or :obj:`None`
        Dimension of the hidden layers of the decoder network, by default :obj:`None`. When
        :obj:`None` the parameter ``trainer`` must be not :obj:`None`.
    enc_dims : :obj:`list` or array_like of :obj:`int` :obj:`None`
        Dimension of the hidden layers of the encoder network, by default :obj:`None`. When
        :obj:`None` the encoder is assumed of having the reversed structure of the decoder.
    dropout : :obj:`float` [optional]
        The dropout rate for the dropout layer that is applied to the input during the
        forward operation. By default 0.5.
    beta : :obj:`float` [optional]
        The :math:`\beta` hyper-parameter of Multi-VAE. When ``anneal_steps > 0`` then this
        value is the value to anneal starting from 0, otherwise the ``beta`` will be fixed to
        the given value for the duration of the training. By default 1.
    anneal_steps : :obj:`int` [optional]
        Number of annealing step for reaching the target value ``beta``, by default 0.
        0 means that no annealing will be performed and the regularization parameter will be
        fixed to ``beta``.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.
    device : :obj:`str` [optional]
        The device where the model must be loaded, by default :obj:`None`. If :obj:`None`, the
        default device (see `rectorch.env.device`) is used.
    trainer : :class:`rectorch.models.nn.multvae.CMultVAE_trainer` [optional]
        The trainer object for performing the learning, by default :obj:`None`. If not :obj:`None`
        it is the only parameters that is taken into account for creating the model.

    Attributes
    ----------
    network : :class:`rectorch.models.nn.multvae.CMultVAE_net`
        The neural network architecture.
    trainer : :class:`rectorch.models.nn.multvae.CMultVAE_trainer`
        The trainer class for performing the learning.
    device : :obj:`str`
        The device where the model must be loaded.
    iid2cids : :obj:`dict` { :obj:`int` \: :obj:`list` of :obj:`int` }
        Dictionary that maps each item to the list of all valid conditions for that item. Items
        are referred to with the inner id, and conditions with an integer in the range 0,
        ``n_cond`` -1.
    n_cond : :obj:`int`
        Number of possible conditions.
    """
    def __init__(self,
                 iid2cid=None,
                 n_cond=0,
                 dec_dims=None,
                 enc_dims=None,
                 dropout=0.5,
                 beta=1.,
                 anneal_steps=0,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(CMultVAE, self).__init__(trainer=trainer)
        else:
            device = torch.device(device) if device is not None else env.device
            network = CMultVAE_net(cond_dim=n_cond,
                                   dec_dims=dec_dims,
                                   enc_dims=enc_dims,
                                   dropout=dropout)
            trainer = CMultVAE_trainer(network,
                                       beta=beta,
                                       anneal_steps=anneal_steps,
                                       device=device,
                                       opt_conf=opt_conf)
            super(CMultVAE, self).__init__(trainer=trainer)
            self.n_cond = n_cond
            self.iid2cid = iid2cid

    def train(self,
              dataset,
              batch_size=1,
              shuffle=True,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=200,
              best_path=None,
              verbose=1,
              seed=None):
        r"""CMultVAE training procedure.

        Parameters
        ----------
        dataset : class:`rectorch.data.Dataset` or :class:`rectorch.samplers.Sampler`
            The dataset or the sampler to use for training/validation.
        batch_size : :obj:`int` [optional]
            The size of the batches, by default 1.
        shuffle : :obj:`bool` [optional]
            Whether the data set must by randomly shuffled before creating the batches, by default
            :obj:`True`.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default :obj:`None`.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`rectorch.validation.ValidFunc` [optional]
            The validation function, by default a standard validation procedure, i.e.,
            :func:`rectorch.evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 200.
        best_path : :obj:`str` or :obj:`None` [optional]
            Where the best model on the validation set will be saved, by default :obj:`None`. When
            set to :obj:`None` the model wont be saved.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        seed : :obj:`int` [optional]
            The random seed to use, by default :obj:`None`. If :obj:`None` no seed will be set.
        """
        if isinstance(dataset, Sampler):
            data_sampler = dataset
        else:
            data_sampler = ConditionedDataSampler(self.iid2cid,
                                                  self.n_cond,
                                                  dataset,
                                                  mode="train",
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)
        super().train(data_sampler,
                      batch_size,
                      shuffle,
                      valid_metric,
                      valid_func,
                      num_epochs,
                      best_path,
                      verbose,
                      seed)

    def predict(self, x, remove_train=True):
        self.network.eval()
        cond_dim = self.network.cond_dim
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[torch.nonzero(x_tensor[:, :-cond_dim], as_tuple=True)] = -np.inf
            return recon_x, mu, logvar

    @classmethod
    def from_state(cls, state):
        trainer = CMultVAE_trainer.from_state(state)
        return CMultVAE(trainer=trainer)
