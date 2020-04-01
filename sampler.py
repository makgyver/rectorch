import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import torch


class Sampler():
    def __init__(self, *args, **kargs):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass


class DataSampler(Sampler):
    def __init__(self,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
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

        for batch_idx, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data_tr = self.sparse_data_tr[idxlist[start_idx:end_idx]]
            data_tr = torch.FloatTensor(data_tr.toarray())

            data_te = None
            if self.sparse_data_te is not None:
                data_te = self.sparse_data_te[idxlist[start_idx:end_idx]]
                data_te = torch.FloatTensor(data_te.toarray())

            yield data_tr, data_te


class ConditionedDataSampler(Sampler):
    def __init__(self,
                 iid2cids,
                 n_cond,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
        self.sparse_data_tr = sparse_data_tr
        self.sparse_data_te = sparse_data_te
        self.iid2cids = iid2cids
        self.batch_size = batch_size
        self.n_cond = n_cond
        self.shuffle = shuffle
        self.compute_conditions()

    def compute_conditions(self):
        r2cond = {}
        for i,row in enumerate(self.sparse_data_tr):
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

        for batch_idx, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            ex = self.examples[idxlist[start_idx:end_idx]]
            rows, cols = [], []
            for i,(r,c) in enumerate(ex):
                if c >= 0:
                    rows.append(i)
                    cols.append(c)

            values = np.ones(len(rows))
            cond_matrix = csr_matrix((values, (rows, cols)), shape=(len(ex), self.n_cond))

            rows_ = [r for r,_ in ex]
            data_tr = hstack([self.sparse_data_tr[rows_], cond_matrix], format="csr")

            if self.sparse_data_te is None:
                self.sparse_data_te = self.sparse_data_tr

            for i,(r,c) in enumerate(ex):
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


class FastConditionedDataSampler(Sampler):
    def __init__(self,
                 iid2cids,
                 n_cond,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
        self.sparse_data_tr = sparse_data_tr
        self.sparse_data_te = sparse_data_te
        self.iid2cids = iid2cids
        self.batch_size = batch_size
        self.n_cond = n_cond
        self.shuffle = shuffle
        self.compute_conditions()

    def compute_conditions(self):
        r2cond = {}
        for i,row in enumerate(self.sparse_data_tr):
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
        self.M_t = self.M.transpose().tocsr()

    def __len__(self):
        return int(np.ceil(len(self.examples) / self.batch_size))

    def __iter__(self):
        n = len(self.examples)
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        empty_cond = csr_matrix((1, self.n_cond))
        full_cond = csr_matrix(np.ones(empty_cond.shape))
        cond_matrix = identity(self.n_cond, format='csr')

        for batch_idx, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            indices = idxlist[start_idx:end_idx]

            ex = self.examples[indices]
            rows = [r for r,_ in ex]
            cols = [c for _,c in ex]
            cmatrix = vstack([cond_matrix, empty_cond])[cols]
            data_tr = hstack([self.sparse_data_tr[rows], cmatrix], format="csr")

            if self.sparse_data_te is None:
                self.sparse_data_te = self.sparse_data_tr

            cmatrix_te = vstack([cond_matrix, full_cond])[cols]
            filtered = cmatrix_te.dot(self.M_t) > 0
            data_te = self.sparse_data_te[rows].multiply(filtered)

            filter_idx = np.diff(data_te.indptr) != 0
            data_te = data_te[filter_idx]
            data_tr = data_tr[filter_idx]

            data_te = torch.FloatTensor(data_te.toarray())
            data_tr = torch.FloatTensor(data_tr.toarray())

            yield data_tr, data_te


class BalancedConditionedDataSampler(ConditionedDataSampler):
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
        self.num_cond_examples = len(self.examples) - self.sparse_data_tr.shape[0]
        self.subsample = subsample

    def compute_conditions(self):
        data = [(r, -1) for r in self.examples[-1]]
        m = int(self.num_cond_examples * self.subsample / self.n_cond)

        for c in self.examples[self.sparse_data_tr.shape[0]:]:
            data += [(r,c) for r in np.random.choice(list(self.examples[c]), m)]

        self.examples = np.array(data)

    def __len__(self):
        m = int(self.num_cond_examples * self.subsample) + self.sparse_data_tr.shape[0]
        return int(np.ceil(m / self.batch_size))

    def __iter__(self):
        self.compute_conditions()
        super().__iter__()


class EmptyConditionedDataSampler(Sampler):
    def __init__(self,
                 cond_size,
                 sparse_data_tr,
                 sparse_data_te=None,
                 batch_size=1,
                 shuffle=True):
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

        for batch_idx, start_idx in enumerate(range(0, n, self.batch_size)):
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
