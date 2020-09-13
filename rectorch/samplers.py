r"""The ``samplers`` module that contains definitions of sampler classes useful when training neural
network-based models.

The ``samplers`` module is inspired by the :class:`torch.utils.data.DataLoader` class which,
however, is not really efficient because it outputs a single example at a time. The idea behind the
samplers defined in this module is to treat the data set at batches highly improving the efficiency.
Each new sampler must extend the base class :class:`Sampler` implementing all the abstract
methods, in particular :meth:`Sampler._set_mode`,
:meth:`Sampler.__len__` and :meth:`Sampler.__iter__`.

A sampler object provides the dataset to a model during training, validation and test.
In its most conventional form, a sampler delivers batches of samples that are used to train/test the
model. However, it is also necessary when the dataset is used as a single batch (see the
:class:`rectorch.samplers.DummySampler` family). A sampler can be in three different states:
``train``, ``valid``, and ``test``. These states represent which kind of data the sampler will
deliver when you cycle through it. To change the sampler's state call the ``train()``, ``valid()``,
and ``test()`` methods.
"""
import importlib
import numpy as np
import torch

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['Sampler', 'DataSampler', 'DummySampler', 'DictDummySampler', 'ArrayDummySampler',\
    'SparseDummySampler', 'TensorDummySampler']


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
    special methods, in particular :meth:`Sampler.__len__` and :meth:`Sampler.__iter__`.
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
        assert mode in ["train", "valid", "test"], "Invalid sampler's mode."
        self.mode = mode

    def train(self, batch_size=None):
        """Set the sampler to training mode.

        Parameters
        ----------
        batch_size : :obj:`int` or :obj:`None` [optional]
            The size of the batches, by default :obj:`None`. If :obj:`None` no modification will be
            applied to the batch size.
        """
        self._set_mode("train", batch_size)

    def valid(self, batch_size=None):
        r"""Set the sampler to validation mode.

        Parameters
        ----------
        batch_size : :obj:`int` or :obj:`None` [optional]
            The size of the batches, by default :obj:`None`. If :obj:`None` no modification will be
            applied to the batch size.
        """
        self._set_mode("valid", batch_size)

    def test(self, batch_size=None):
        r"""Set the sampler to test mode.

        Parameters
        ----------
        batch_size : :obj:`int` or :obj:`None` [optional]
            The size of the batches, by default :obj:`None`. If :obj:`None` no modification will be
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
        super()._set_mode(mode)
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
                empty_users = set(np.where(te_sets.sum(axis=1) == 0)[0])
                users = list(set(users) - empty_users)
                tr_sets = tr_sets[users]
                te_sets = te_sets[users]

            yield (users, tr_sets), None if self.mode == "train" else te_sets


class SparseDummySampler(DummySampler):
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
        super(SparseDummySampler, self).__init__(data, mode, batch_size, shuffle)
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
                filter_idx = np.diff(te_sets.indptr) != 0
                tr_sets = tr_sets[filter_idx]
                te_sets = te_sets[filter_idx]
                users = list(np.array(users)[filter_idx])

            yield (users, tr_sets), None if self.mode == "train" else te_sets


class TensorDummySampler(DummySampler):
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
                filter_idx = torch.nonzero(te_sets.sum(axis=1) > 0, as_tuple=False)[:, 0]
                tr_sets = tr_sets[filter_idx]
                te_sets = te_sets[filter_idx]
                users = filter_idx.tolist()

            yield (users, tr_sets), None if self.mode == "train" else te_sets


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
            if isinstance(self._spval, tuple):
                self.sparse_data_tr = self._spval[0]
                self.sparse_data_te = self._spval[1]
            else:
                self.sparse_data_tr = self._sptr
                self.sparse_data_te = self._spval
        else:
            if isinstance(self._spte, tuple):
                self.sparse_data_tr = self._spte[0]
                self.sparse_data_te = self._spte[1]
            else:
                self.sparse_data_tr = self._sptr
                self.sparse_data_te = self._spte

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
