r"""The ``samplers`` module that contains definitions of sampler classes useful when training neural
network-based models.

The ``samplers`` module is inspired by the :class:`torch.utils.data.DataLoader` class which,
however, is not really efficient because it outputs a single example at a time. The idea behind the
samplers defined in this module is to treat the data set at batches highly improving the efficiency.
Each new sampler must extend the base class :class:`Sampler` implementing all the abstract
methods, in particular :meth:`samplers.Sampler._set_mode`, :meth:`samplers.Sampler.__len__` and
:meth:`samplers.Sampler.__iter__`.
"""
import importlib
import numpy as np
from scipy.sparse import csr_matrix, hstack
import torch
from torch.autograd import Variable

__all__ = ['Sampler', 'DataSampler', 'DummySampler', 'DictDummySampler', 'ArrayDummySampler',\
    'SparseDummySampler', 'TensorDummySampler', 'ConditionedDataSampler',\
    'EmptyConditionedDataSampler', 'CFGAN_Sampler', 'SVAE_Sampler']

#TODO document the way sampler works


class Sampler():
    r"""Sampler base class.

    A sampler is meant to be used as a generator of batches useful in training neural networks.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.

    Attributes
    ----------
    All attributes : see the **Parameters** section.

    Notes
    -----
    Each new sampler must extend this base class implementing all the abstract
    special methods, in particular :meth:`rectorch.samplers.Sampler.__len__` and
    :meth:`rectorch.samplers.Sampler.__iter__`.
    """
    def __init__(self, data, mode="train", batch_size=1):
        self.data = data
        self.mode = mode
        self.batch_size = batch_size

    def _set_mode(self, mode="train", batch_size=None):
        """Change the sampler's mode according to the given parameter.

        Parameters
        ----------
        mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
            Indicates the mode in which the sampler operates.
        """
        raise NotImplementedError

    def train(self, batch_size=None):
        """Set the sampler to training mode.

        Parameters
        ----------
        batch_size : :obj:`int` or :obj:`None` [optional]
            The size of the batches, by default :obj:`None`. If ``None`` no modification will be
            applied to the batch size.
        """
        self._set_mode("train", batch_size)

    def valid(self, batch_size=None):
        r"""Set the sampler to validation mode.

        Parameters
        ----------
        batch_size : :obj:`int` or :obj:`None` [optional]
            The size of the batches, by default :obj:`None`. If ``None`` no modification will be
            applied to the batch size.
        """
        self._set_mode("valid", batch_size)

    def test(self, batch_size=None):
        r"""Set the sampler to test mode.

        Parameters
        ----------
        batch_size : :obj:`int` or :obj:`None` [optional]
            The size of the batches, by default :obj:`None`. If ``None`` no modification will be
            applied to the batch size.
        """
        self._set_mode("test", batch_size)

    def __len__(self):
        r"""Return the number of batches.
        """
        raise NotImplementedError

    def __iter__(self):
        r"""Iterate through the batches yielding a batch at a time.
        """
        raise NotImplementedError

    @classmethod
    def build(cls, dataset, **kwargs):
        r"""Build a sampler according to the given parameters.

        Parameters
        ----------
        dataset : :class:`rectorch.data.Dataset`
            The dataset.

        Returns
        -------
        :class:`rectorch.samplers.Sampler`
            A new data sampler.
        """
        sampler_class = getattr(importlib.import_module("rectorch.samplers"), kwargs["name"])
        del kwargs["name"]
        return sampler_class(dataset, **kwargs)


class DummySampler(Sampler):
    r"""Abstract sampler that simply returns the dataset.

    Notes
    -----
    The value of the attribute ``batch_size`` is always set to 1.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` or :obj:`None` [optional]
        The size of the batches, by default :obj:`None`. If :obj:`None` the batch size will be set
        to the number of users of the active data set.
    shuffle : :obj:`bool` [optional]
        Whether the data set must be shuffled, by default :obj:`False`.
    """
    def __init__(self, data, mode="train", batch_size=None, shuffle=False):
        super(DummySampler, self).__init__(data, mode, batch_size)
        self.data_tr = None
        self.data_val = None
        self.data_te = None
        self._data = None
        self.shuffle = shuffle

    def _set_mode(self, mode="train", batch_size=None):
        assert mode in ["train", "valid", "test"], "Invalid sampler's mode."
        self.mode = mode
        if self.mode == "train":
            self._data = self.data_tr
        elif self.mode == "valid":
            self._data = self.data_val
        else:
            self._data = self.data_te

    def __len__(self):
        if isinstance(self._data, tuple):
            return int(np.ceil(len(self._data[0]) / self.batch_size))
        else:
            return int(np.ceil(len(self._data) / self.batch_size))

    def __iter__(self):
        raise NotImplementedError()


class DictDummySampler(DummySampler):
    r"""Dummy sampler that returns the dataset as a dictionary.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` or :obj:`None` [optional]
        The size of the batches, by default :obj:`None`. If :obj:`None` the batch size will be set
        to the number of users of the active data set.
    shuffle : :obj:`bool` [optional]
        Whether the data set must be shuffled, by default :obj:`False`.
    cold_users : :obj:`bool` [optional]
        Whether the validation/test users have to be included in the training set
        i.e., ``cold_users == False``, by default :obj:`True`.
        Note: it is used only when the dataset has been vertically splitted.

    Attributes
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    batch_size : :obj:`int`
        The size of the batches.
    data_tr : :obj:`dict`
        The training set.
    data_val : :obj:`dict`
        The validation set.
    data_te : :obj:`dict`
        The test set.
    shuffle : :obj:`bool`
        Whether the data set must be shuffled.
    """
    def __init__(self, data, mode="train", batch_size=None, shuffle=False, cold_users=True):
        super(DictDummySampler, self).__init__(data, mode, batch_size, shuffle)
        self.data_tr, self.data_val, self.data_te = data.to_dict(cold_users=cold_users)
        self._set_mode(mode)
        if batch_size is None:
            self.batch_size = len(self._data[0] if isinstance(self._data, tuple) else self._data)

    def __iter__(self):
        n = len(self._data[0] if isinstance(self._data, tuple) else self._data)
        idxlist = list(range(n))
        if self.shuffle and self.mode == "train":
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            users = idxlist[start_idx:end_idx]
            if isinstance(self._data, tuple):
                tr_sets = [self._data[0][u] for u in users]
                te_sets = [self._data[1][u] for u in users]
            else:
                tr_sets = [self.data_tr[u] for u in users]
                te_sets = [self._data[u] for u in users]

            yield (users, tr_sets), None if self.mode == "train" else te_sets


class ArrayDummySampler(DummySampler):
    r"""Dummy sampler that returns the dataset as a numpy array.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` or :obj:`None` [optional]
        The size of the batches, by default :obj:`None`. If :obj:`None` the batch size will be set
        to the number of users of the active data set.
    shuffle : :obj:`bool` [optional]
        Whether the data set must be shuffled, by default :obj:`False`.
    cold_users : :obj:`bool` [optional]
        Whether the validation/test users have to be included in the training set
        i.e., ``cold_users == False``, by default :obj:`True`.
        Note: it is used only when the dataset has been vertically splitted.

    Attributes
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    data_tr : :class:`numpy.ndarray`
        The training set.
    data_val : :class:`numpy.ndarray`
        The validation set.
    data_te : :class:`numpy.ndarray`
        The test set.
    shuffle : :obj:`bool`
        Whether the data set must be shuffled, by default :obj:`False`.
    """
    def __init__(self, data, mode="train", batch_size=None, shuffle=False, cold_users=True):
        super(ArrayDummySampler, self).__init__(data, mode, batch_size, shuffle)
        self.data_tr, self.data_val, self.data_te = data.to_array(cold_users=cold_users)
        self._set_mode(mode)
        if batch_size is None:
            self.batch_size = len(self._data[0] if isinstance(self._data, tuple) else self._data)

    def __iter__(self):
        n = self._data[0].shape[0] if isinstance(self._data, tuple) else self._data.shape[0]
        idxlist = list(range(n))
        if self.shuffle and self.mode == "train":
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            users = idxlist[start_idx:end_idx]
            if isinstance(self._data, tuple):
                tr_sets = self._data[0][users]
                te_sets = self._data[1][users]
            else:
                tr_sets = self.data_tr[users]
                te_sets = self._data[users]

            yield (users, tr_sets), None if self.mode == "train" else te_sets


class SparseDummySampler(ArrayDummySampler):
    r"""Dummy sampler that returns the dataset as a sparse scipy array.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    batch_size : :obj:`int` or :obj:`None` [optional]
        The size of the batches, by default :obj:`None`. If :obj:`None` the batch size will be set
        to the number of users of the active data set.
    shuffle : :obj:`bool` [optional]
        Whether the data set must be shuffled, by default :obj:`False`.
    cold_users : :obj:`bool` [optional]
        Whether the validation/test users have to be included in the training set
        i.e., ``cold_users == False``, by default :obj:`True`.
        Note: it is used only when the dataset has been vertically splitted.

    Attributes
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    data_tr : :class:`scipy.sparse.csr_matrix`
        The training set.
    data_val : :class:`scipy.sparse.csr_matrix`
        The validation set.
    data_te : :class:`scipy.sparse.csr_matrix`
        The test set.
    shuffle : :obj:`bool`
        Whether the data set must be shuffled.
    """
    def __init__(self, data, mode="train", batch_size=None, shuffle=False, cold_users=True):
        super(SparseDummySampler, self).__init__(data, mode, batch_size, shuffle, cold_users)
        self.data_tr, self.data_val, self.data_te = data.to_sparse(cold_users=cold_users)
        self._set_mode(mode)
        if batch_size is None:
            if isinstance(self._data, tuple):
                self.batch_size = self._data[0].shape[0]
            else:
                self.batch_size = self._data.shape[0]

    def __len__(self):
        if isinstance(self._data, tuple):
            return int(np.ceil(self._data[0].shape[0] / self.batch_size))
        else:
            return int(np.ceil(self._data.shape[0] / self.batch_size))


class TensorDummySampler(ArrayDummySampler):
    r"""Dummy sampler that returns the dataset as a pytorch tensor.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` or :obj:`None` [optional]
        The size of the batches, by default :obj:`None`. If :obj:`None` the batch size will be set
        to the number of users of the active data set.
    shuffle : :obj:`bool` [optional]
        Whether the data set must be shuffled, by default :obj:`False`.
    cold_users : :obj:`bool` [optional]
        Whether the validation/test users have to be included in the training set
        i.e., ``cold_users == False``, by default :obj:`True`.
        Note: it is used only when the dataset has been vertically splitted.

    Attributes
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    data_tr : :class:`torch.Tensor`
        The training set.
    data_val : :class:`torch.Tensor`
        The validation set.
    data_te : :class:`torch.Tensor`
        The test set.
    shuffle : :obj:`bool`
        Whether the data set must be shuffled.
    """
    def __init__(self, data, mode="train", batch_size=None, shuffle=False, cold_users=True):
        super(TensorDummySampler, self).__init__(data, mode, batch_size, shuffle)
        self.data_tr, self.data_val, self.data_te = data.to_tensor(cold_users=cold_users)
        self._set_mode(mode)
        if batch_size is None:
            if isinstance(self._data, tuple):
                self.batch_size = self._data[0].shape[0]
            else:
                self.batch_size = self._data.shape[0]

    def __len__(self):
        if isinstance(self._data, tuple):
            return int(np.ceil(self._data[0].shape[0] / self.batch_size))
        else:
            return int(np.ceil(self._data.shape[0] / self.batch_size))


class DataSampler(Sampler):
    r"""This is a standard sampler that returns batches without any particular constraint.

    Bathes are randomly returned with the defined dimension (i.e., ``batch_size``). If ``shuffle``
    is set to :obj:`False` then the sampler returns batches with the same order as in the original
    dataset. When ``sparse_data_te`` is defined then each returned batch is a :obj:`tuple` with
    the training part of the batch and its test/validation counterpart. Otherwise, if
    ``sparse_data_te`` is :obj:`None` then the second element of the yielded tuple will be
    :obj:`None`.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must by randomly shuffled before creating the batches, by default
        :obj:`True`.

    Attributes
    ----------
    all attributes : see **Parameters** section.
    """
    def __init__(self,
                 data,
                 mode="train",
                 batch_size=1,
                 shuffle=True):
        super(DataSampler, self).__init__(data, mode, batch_size)
        self._sptr, self._spval, self._spte = self.data.to_sparse()
        self.sparse_data_tr, self.sparse_data_te = None, None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._set_mode(mode)

    def _set_mode(self, mode="train", batch_size=None):
        assert mode in ["train", "valid", "test"], "Invalid sampler's mode."
        self.mode = mode

        if self.mode == "train":
            self.sparse_data_tr = self._sptr
            self.sparse_data_te = None
        elif self.mode == "valid":
            self.sparse_data_tr = self._spval[0]
            self.sparse_data_te = self._spval[1]
        else:
            self.sparse_data_tr = self._spte[0]
            self.sparse_data_te = self._spte[1]

        if batch_size is not None:
            self.batch_size = batch_size

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
            data_tr = torch.FloatTensor(data_tr.toarray())

            data_te = None
            if self.sparse_data_te is not None:
                data_te = self.sparse_data_te[idxlist[start_idx:end_idx]]
                data_te = torch.FloatTensor(data_te.toarray())

            yield data_tr, data_te


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

    def _set_mode(self, mode="vero", batch_size=None):
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


class CFGAN_Sampler(DataSampler):
    r"""Sampler used for training the generator and discriminator of the CFGAN model.

    The peculiarity of this sampler (see for [CFGAN]_ more details) is that batches are
    continuously picked at random from all the training set.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 64

    Attributes
    ----------
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_tr`` parameter.
    batch_size : :obj:`int`
        See ``batch_size`` parameter.
    idxlist : :obj:`list` of :obj:`int`
        Shuffled list of indexes. After an iteration over the sampler, or after a call to the
        :func:`next` function, the ``idxlist`` contains, in the first ``batch_size`` positions,
        the indexes of the examples that are contained in the current batch.

    References
    ----------
    .. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
       CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
       In Proceedings of the 27th ACM International Conference on Information and Knowledge
       Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
       DOI: https://doi.org/10.1145/3269206.3271743
    """
    def __init__(self,
                 data,
                 mode="train",
                 batch_size=64):
        super(CFGAN_Sampler, self).__init__(data, mode, batch_size, False)
        self.idxlist = list(range(self.sparse_data_tr.shape[0]))

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def _set_mode(self, mode="train", batch_size=None):
        super()._set_mode(mode, batch_size)
        self.idxlist = list(range(self.sparse_data_tr.shape[0]))

    def __iter__(self):
        if self.mode == "train":
            while True:
                np.random.shuffle(self.idxlist)
                data_tr = self.sparse_data_tr[self.idxlist[:self.batch_size]]
                yield torch.FloatTensor(data_tr.toarray())
        else:
            yield from super().__iter__()


class SVAE_Sampler(Sampler):
    r"""Sampler used for training SVAE.

    This sampler yields pairs (``x``, ``y``) where ``x`` is the tensor of indexes of the
    positive items, and ``y`` the target tensor with the (multi-hot) ground truth items.
    This sampler is characterized by batches of size one (a single user at a time).
    Given a user (batch) *u* the returned ground truth tensor is a 3D tensor of dimension
    :math:`1 \times |\mathcal{I}_u|-1 \times m`, where :math:`|\mathcal{I}_u|` is the set
    of rated items by *u*, and *m* the number of items. This tensor represents the ground truth
    for *u* over time, and each slice of the tensor is a different timestamp across all the possible
    time unit for this specific user.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    pred_type : :obj:`str` in the set {``'next_k'``, ``'next'``, ``'postfix'``} [optional]
        The variant of loss used by the model, by default ``'next_k'``. If ``'next'`` then
        only the next item must be predicted, if ``'next_k'`` the next *k* items are considered in
        the ground truth, otherwise (= ``'postfix'``) all the remaining items are taken as ground
        truth.
    k : :obj:`int` [optional]
        The number of item to predict in the ``'next_k'`` variant, by default 1. This parameter
        is not considered when ``pred_type`` is not ``'next_k'``.
    shuffle : :obj:`bool` [optional]
        Whether the data set must by randomly shuffled before creating the batches, by default
        :obj:`True`.
    is_training : :obj:`bool` [optional]
        Whether the sampler is used during training, by default :obj:`True`.

    Attributes
    ----------
    all attributes : see **Parameters** section.
    """
    def __init__(self,
                 data,
                 mode="train",
                 pred_type="next_k",
                 k=1,
                 shuffle=True):
        super(SVAE_Sampler, self).__init__(data, mode)
        if pred_type == "next_k":
            assert k >= 1, "If pred_type == 'next_k' then 'k' must be a positive integer."
        self.pred_type = pred_type
        self.shuffle = shuffle
        self.num_items = data.n_items
        self.k = k

        self._dictr, self._dicval, self._dicte = self.data.to_dict()
        self.dict_data_tr, self.dict_data_te = None, None
        self._set_mode(mode)

    def _set_mode(self, mode="train", batch_size=1):
        assert mode in ["train", "valid", "test"], "Invalid sampler's mode."
        self.mode = mode

        if self.mode == "train":
            self.dict_data_tr = self._dictr
            self.dict_data_te = None
        elif self.mode == "valid":
            self.dict_data_tr = self._dicval[0]
            self.dict_data_te = self._dicval[1]
        else:
            self.dict_data_tr = self._dicte[0]
            self.dict_data_te = self._dicte[1]

    def __len__(self):
        return len(self.dict_data_tr)

    def __iter__(self):
        idxlist = list((self.dict_data_tr.keys()))
        if self.shuffle and self.mode == "train":
            np.random.shuffle(idxlist)

        for _, user in enumerate(idxlist):
            ulen = len(self.dict_data_tr[user])
            y_batch_s = torch.zeros(1, ulen - 1, self.num_items)

            if self.mode == "train":
                if self.pred_type == 'next':
                    for timestep in range(ulen - 1):
                        idx = self.dict_data_tr[user][timestep + 1]
                        y_batch_s[0, timestep, idx] = 1.
                elif self.pred_type == 'next_k':
                    for timestep in range(ulen - 1):
                        idx = self.dict_data_tr[user][timestep + 1:][:self.k]
                        y_batch_s[0, timestep, idx] = 1.
                elif self.pred_type == 'postfix':
                    for timestep in range(ulen - 1):
                        idx = self.dict_data_tr[user][timestep + 1:]
                        y_batch_s[0, timestep, idx] = 1.
            else:
                y_batch_s = torch.zeros(1, 1, self.num_items)
                y_batch_s[0, 0, self.dict_data_te[user]] = 1.

            x_batch = [self.dict_data_tr[user][:-1]]

            x = Variable(torch.LongTensor(x_batch))
            y = Variable(y_batch_s, requires_grad=False)

            yield x, y
