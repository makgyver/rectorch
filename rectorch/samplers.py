r"""The ``samplers`` module that contains definitions of sampler classes useful when training neural
network-based models.

The ``samplers`` module is inspired by the :class:`torch.utils.data.DataLoader` class which,
however, is not really efficient because it outputs a single example at a time. The idea behind the
samplers defined in this module is to treat the data set at batches highly improving the efficiency.
Each new sampler must extend the base class :class:`Sampler` implementing all the abstract special
methods, in particular :meth:`Sampler.__len__` and :meth:`Sampler.__iter__`.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
import torch

__all__ = ['Sampler', 'DataSampler', 'ConditionedDataSampler', 'EmptyConditionedDataSampler',\
    'BalancedConditionedDataSampler', 'CFGAN_TrainingSampler']

class Sampler():
    r"""Sampler base class.

    A sampler is meant to be used as a generator of batches useful in training neural networks.

    Notes
    -----
    Each new sampler must extend this base class implementing all the abstract
    special methods, in particular :meth:`Sampler.__len__` and :meth:`Sampler.__iter__`.
    """
    def __init__(self, *args, **kargs):
        pass

    def __len__(self):
        """Return the number of batches.
        """
        raise NotImplementedError

    def __iter__(self):
        """Iterate through the batches yielding a batch at a time.
        """
        raise NotImplementedError


class DataSampler(Sampler):
    r"""This is a standard sampler that returns batches without any particular constraint.

    Bathes are randomly returned with the defined dimension (i.e., ``batch_size``). If ``shuffle``
    is set to ``False`` then the sampler returns batches with the same order as in the original
    dataset. When ``sparse_data_te`` is defined then each returned batch is a :obj:`tuple` with
    the training part of the batch and its test/validation counterpart. Otherwise, if
    ``sparse_data_te`` is ``None`` then the second element of the yielded tuple will be ``None``.

    Parameters
    ----------
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        The training sparse user-item rating matrix.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix` [optional]
        The test sparse user-item rating matrix. The shape of this matrix must be the same as
        ``sparse_data_tr``. By default ``None``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must bu randomly shuffled before creating the batches, by default
        ``True``.

    Attributes
    ----------
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_tr`` parameter.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_te`` parameter.
    batch_size : :obj:`int`
        See ``batch_size`` parameter.
    shuffle : :obj:`bool`
        See ``shuffle`` parameter.
    """
    def __init__(self,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
        super(DataSampler, self).__init__()
        self.sparse_data_tr = sparse_data_tr
        self.sparse_data_te = sparse_data_te
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.sparse_data_tr.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
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


class ConditionedDataSampler(Sampler):
    r"""Data sampler with conditioned filtering used by the :class:`models.CMultiVAE` model.

    This data sampler is useful when training the :class:`models.CMultiVAE` model described in
    [CVAE]_. During the training, each user must be conditioned over all the possible conditions
    (actually the ones that the user knows) so the training set must be modified accordingly.

    Parameters
    ----------
    iid2cids : :obj:`dict` (key :obj:`int` - value :obj:`list` of :obj:`int`)
        Dictionary that maps each item to the list of all valid conditions for that item. Items
        are referred to with the inner id, and conditions with an integer in the range 0,
        ``n_cond`` -1.
    n_cond : :obj:`int`
        Number of possible conditions.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        The training sparse user-item rating matrix.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix` [optional]
        The test sparse user-item rating matrix. The shape of this matrix must be the same as
        ``sparse_data_tr``. By default ``None``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must bu randomly shuffled before creating the batches, by default
        ``True``.

    Attributes
    ----------
    iid2cids : :obj:`dict` (key :obj:`int` - value :obj:`list` of :obj:`int`)
        See ``iid2cids`` parameter.
    n_cond : :obj:`int`
        See ``n_cond`` parameter.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_tr`` parameter.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_te`` parameter.
    batch_size : :obj:`int`
        See ``batch_size`` parameter.
    shuffle : :obj:`bool`
        See ``shuffle`` parameter.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    """
    def __init__(self,
                 iid2cids,
                 n_cond,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
        super(ConditionedDataSampler, self).__init__()
        self.sparse_data_tr = sparse_data_tr
        self.sparse_data_te = sparse_data_te
        self.iid2cids = iid2cids
        self.batch_size = batch_size
        self.n_cond = n_cond
        self.shuffle = shuffle
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

    def __len__(self):
        return int(np.ceil(len(self.examples) / self.batch_size))

    def __iter__(self):
        n = len(self.examples)
        idxlist = list(range(n))
        if self.shuffle:
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

            rows_ = [r for r,_ in ex]
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


class BalancedConditionedDataSampler(ConditionedDataSampler):
    r"""Sub-sampled version of the :class:`ConditionedDataSampler`.

    This data sampler is useful when training the :class:`models.CMultiVAE` model described in
    [CVAE]_. During the training, each user must be conditioned over all the possible conditions
    (actually the ones that the user knows) so the training set must be modified accordingly.
    This sampler avoids to create all possible user-condition pairs via sub-samplig. The extent
    of this sub-sampling is defined by the parameter ``subsample``. The prefix 'Balanced' is
    due to the way the subsampling is performed. Given a user *u*, for each condition *c* only a
    ``subsample`` fraction of training sample is created for *u* conditioned by *c*.

    Parameters
    ----------
    iid2cids : :obj:`dict` (key :obj:`int` - value :obj:`list` of :obj:`int`)
        Dictionary that maps each item to the list of all valid conditions for that item. Items
        are referred to with the inner id, and conditions with an integer in the range 0,
        ``n_cond`` -1.
    n_cond : :obj:`int`
        Number of possible conditions.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        The training sparse user-item rating matrix.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix` [optional]
        The test sparse user-item rating matrix. The shape of this matrix must be the same as
        ``sparse_data_tr``. By default ``None``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must bu randomly shuffled before creating the batches, by default
        ``True``.
    subsample : :obj:`float` [optional]
        The size of the dimension. It must be a float between (0, 1], by default 0.2.

    Attributes
    ----------
    iid2cids : :obj:`dict` (key :obj:`int` - value :obj:`list` of :obj:`int`)
        See ``iid2cids`` parameter.
    n_cond : :obj:`int`
        See ``n_cond`` parameter.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_tr`` parameter.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_te`` parameter.
    batch_size : :obj:`int`
        See ``batch_size`` parameter.
    shuffle : :obj:`bool`
        See ``shuffle`` parameter.
    subsample : :obj:`float`
        See ``subsample`` parameter.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    """
    def __init__(self,
                 iid2cids,
                 n_cond,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 subsample=.2):
        super(BalancedConditionedDataSampler, self).__init__(iid2cids,
                                                             n_cond,
                                                             sparse_data_tr,
                                                             sparse_data_te,
                                                             batch_size)
        self.subsample = subsample
        self._compute_sampled_conditions()

    def _compute_conditions(self):
        r2cond = {}
        for i, row in enumerate(self.sparse_data_tr):
            _, cols = row.nonzero()
            r2cond[i] = set.union(*[set(self.iid2cids[c]) for c in cols])

        self.examples = {-1 : list(r2cond.keys())}
        for c in range(self.n_cond):
            self.examples[c] = []
            for r in r2cond:
                if c in r2cond[r]:
                    self.examples[c].append(r)
        del r2cond
        self.num_cond_examples = sum([len(self.examples[c]) for c in range(self.n_cond)])

        rows = [m for m in self.iid2cids for _ in range(len(self.iid2cids[m]))]
        cols = [g for m in self.iid2cids for g in self.iid2cids[m]]
        values = np.ones(len(rows))
        self.M = csr_matrix((values, (rows, cols)), shape=(len(self.iid2cids), self.n_cond))

    def _compute_sampled_conditions(self):
        data = [(r, -1) for r in self.examples[-1]]
        m = int(self.num_cond_examples * self.subsample / self.n_cond)

        for c in range(self.n_cond):
            data += [(r, c) for r in np.random.choice(self.examples[c], m)]

        self.examples = np.array(data)

    def __len__(self):
        m = int(self.num_cond_examples * self.subsample) + self.sparse_data_tr.shape[0]
        return int(np.ceil(m / self.batch_size))


class EmptyConditionedDataSampler(Sampler):
    r"""Data sampler that returns unconditioned batches used by the :class:`models.CMultiVAE` model.

    This data sampler is useful when training the :class:`models.CMultiVAE` model described in
    [CVAE]_. This sampler is very similar to :class:`DataSampler` with the expection that the
    yielded batches have appended a zero matrix of the size ``batch_size`` :math:`\\times`
    ``n_cond``.

    Parameters
    ----------
    n_cond : :obj:`int`
        Number of possible conditions.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        The training sparse user-item rating matrix.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix` [optional]
        The test sparse user-item rating matrix. The shape of this matrix must be the same as
        ``sparse_data_tr``. By default ``None``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 1.
    shuffle : :obj:`bool` [optional]
        Whether the data set must bu randomly shuffled before creating the batches, by default
        ``True``.

    Attributes
    ----------
    n_cond : :obj:`int`
        See ``n_cond`` parameter.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_tr`` parameter.
    sparse_data_te : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_te`` parameter.
    batch_size : :obj:`int`
        See ``batch_size`` parameter.
    shuffle : :obj:`bool`
        See ``shuffle`` parameter.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    """
    def __init__(self,
                 cond_size,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
        super(EmptyConditionedDataSampler, self).__init__()
        self.sparse_data_tr = sparse_data_tr
        self.sparse_data_te = sparse_data_te
        self.batch_size = batch_size
        self.cond_size = cond_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.sparse_data_tr.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
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


class CFGAN_TrainingSampler(Sampler):
    r"""Sampler used for training the generator and discriminator of the CFGAN model.

    The peculiarity of this sampler (see for [CFGAN]_ more details) is that batches are
    continuously picked at random from all the training set.

    Parameters
    ----------
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        The training sparse user-item rating matrix.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 64

    Attributes
    ----------
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
                 sparse_data_tr,
                 batch_size=64):
        super(CFGAN_TrainingSampler, self).__init__()
        self.sparse_data_tr = sparse_data_tr
        self.batch_size = batch_size
        n = self.sparse_data_tr.shape[0]
        self.idxlist = list(range(n))

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        np.random.shuffle(self.idxlist)
        data_tr = self.sparse_data_tr[self.idxlist[:self.batch_size]]
        return torch.FloatTensor(data_tr.toarray())
