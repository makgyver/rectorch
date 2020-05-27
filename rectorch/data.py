r"""The ``data`` module manages the reading, writing and loading of the data sets.

The supported data set format is standard `csv
<https://it.wikipedia.org/wiki/Comma-separated_values>`_.
For more information about the expected data set fromat please visit :ref:`csv-format`.
The data processing and loading configurations are managed through the configuration files
as described in :ref:`config-format`.
The data pre-processing phase is highly inspired by `VAE-CF source code
<https://github.com/dawenl/vae_cf>`_, which has been lately used on several other research works.

Examples
--------
This module is mainly meant to be used in the following way:

>>> from rectorch.data import DataProcessing, DatasetManager
>>> dproc = DataProcessing("/path/to/the/config/file")
>>> dproc.process()
>>> man = DatasetManager(dproc.cfg)

See Also
--------
Research paper: `Variational Autoencoders for Collaborative Filtering
<https://arxiv.org/pdf/1802.05814.pdf>`_

Module:
:mod:`configuration`
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from .configuration import DataConfig

__all__ = ['DataProcessing', 'DataReader', 'DatasetManager']

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]  %(message)s",
                    datefmt='%H:%M:%S-%d%m%y',
                    stream=sys.stdout)

logger = logging.getLogger(__name__)


class DataProcessing:
    r"""Class that manages the pre-processing of raw data sets.

    Data sets are expected of being `csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_
    files where each row represents a rating. More details about the allowed format are described
    in :ref:`csv-format`. The pre-processing is performed following the parameters settings defined
    in the data configuration file (see :ref:`config-format` for more information).

    Parameters
    ----------
    data_config : :class:`rectorch.configuration.DataConfig` or :obj:`str`:
        Represents the data pre-processing configurations.
        When ``type(data_config) == str`` is expected to be the path to the data configuration file.
        In that case a :class:`configuration.DataConfig` object is contextually created.

    Raises
    ------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.

    Attributes
    ----------
    cfg : :class:`rectorch.configuration.DataConfig`
        The :class:`rectorch.configuration.DataConfig` object containing the pre-processing
        configurations.
    i2id : :obj:`dict` (key - :obj:`str`, value - :obj:`int`)
        Dictionary which maps the raw item id, i.e., as in the raw `csv` file, to an internal id
        which is an integer between 0 and the total number of items -1.
    u2id : :obj:`dict` (key - :obj:`str`, value - :obj:`int`)
        Dictionary which maps the raw user id, i.e., as in the raw `csv` file, to an internal id
        which is an integer between 0 and the total number of users -1.
    """
    def __init__(self, data_config):
        if isinstance(data_config, DataConfig):
            self.cfg = data_config
        elif isinstance(data_config, str):
            self.cfg = DataConfig(data_config)
        else:
            raise TypeError("'data_config' must be of type 'DataConfig' or 'str'.")

        self.i2id = {}
        self.u2id = {}

    def process(self):
        r"""Perform the entire pre-processing.

        The pre-processing relies on the configurations provided in the data configuration file.
        The full pre-processing follows a specific pipeline (the meaning of each configuration
        parameter is defined in :ref:`config-format`):

        1. Reading the CSV file named ``data_path``;
        2. Filtering the ratings on the basis of the ``threshold``;
        3. Filtering the users and items according to ``u_min`` and ``i_min``, respectively;
        4. Splitting the users in training, validation and test sets;
        5. Splitting the validation and test set user ratings in training and test items according\
            to ``test_prop``;
        6. Creating the id mappings (see :attr:`i2id` and :attr:`u2id`);
        7. Saving the pre-processed data set files in ``proc_path`` folder.

        .. warning:: In step (4) there is the possibility that users in the validation or test set\
           have less than 2 ratings making step (5) inconsistent for those users. For this reason,\
           this set of users is simply discarded.

        .. warning:: In step (5) there is the possibility that users in the validation or test set\
           have a number of items which could cause problems in applying the diviion between\
           training items and test items (e.g., users with 2 ratings and ``test_prop`` = 0.1).\
           In these cases, it is always guaranteed that there is at least one item in the test part\
           of the users.

        The output consists of a series of files saved in ``proc_path``:

        * ``train.csv`` : (`csv` file) the training ratings corresponding to all ratings of the\
            training users;
        * ``validation_tr.csv`` : (`csv` file) the training ratings corresponding to the validation\
            users;
        * ``validation_te.csv`` : (`csv` file) the test ratings corresponding to the validation\
            users;
        * ``test_tr.csv`` : (`csv` file) the training ratings corresponding to the test users;
        * ``test_te.csv`` : (`csv` file) the test ratings corresponding to the test users;
        * ``unique_uid.txt`` : (`txt` file) with the user id mapping. Line numbers represent the\
            internal id, while the string on the corresponding line is the raw id;
        * ``unique_iid.txt`` : (`txt` file) with the item id mapping. Line numbers represent the\
            internal id, while the string on the corresponding line is the raw id;

        """
        np.random.seed(int(self.cfg.seed))

        logger.info("Reading data file %s.", self.cfg.data_path)

        sep = self.cfg.separator if self.cfg.separator else ','
        raw_data = pd.read_csv(self.cfg.data_path, sep=sep, header=self.cfg.header)

        if self.cfg.threshold:
            raw_data = raw_data[raw_data[raw_data.columns.values[2]] > float(self.cfg.threshold)]

        logger.info("Applying filtering.")
        imin, umin = int(self.cfg.i_min), int(self.cfg.u_min)
        raw_data, user_activity, _ = self._filter(raw_data, umin, imin)
        print(raw_data.head())

        unique_uid = user_activity.index
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]
        n_users = unique_uid.size
        n_heldout = self.cfg.heldout

        logger.info("Calculating splits.")
        tr_users = unique_uid[:(n_users - n_heldout * 2)]
        vd_users = unique_uid[(n_users - n_heldout * 2): (n_users - n_heldout)]
        te_users = unique_uid[(n_users - n_heldout):]

        [uhead, ihead] = raw_data.columns.values[:2]
        train_data = raw_data.loc[raw_data[uhead].isin(tr_users)]
        unique_iid = pd.unique(train_data[ihead])

        logger.info("Creating validation and test set.")
        val_data = raw_data.loc[raw_data[uhead].isin(vd_users)]
        val_data = val_data.loc[val_data[ihead].isin(unique_iid)]
        test_data = raw_data.loc[raw_data[uhead].isin(te_users)]
        test_data = test_data.loc[test_data[ihead].isin(unique_iid)]

        vcnt = val_data[[uhead]].groupby(uhead, as_index=False).size()
        tcnt = test_data[[uhead]].groupby(uhead, as_index=False).size()
        val_data = val_data.loc[val_data[uhead].isin(vcnt[vcnt >= 2].index)]
        test_data = test_data.loc[test_data[uhead].isin(tcnt[tcnt >= 2].index)]

        vcnt_diff = len(vcnt) - len(pd.unique(val_data[uhead]))
        tcnt_diff = len(tcnt) - len(pd.unique(test_data[uhead]))
        if vcnt_diff > 0:
            logger.warning("Skipped %d users in validation set.", vcnt_diff)
        if tcnt_diff > 0:
            logger.warning("Skipped %d users in test set.", tcnt_diff)

        val_data_tr, val_data_te = self._split_train_test(val_data)
        test_data_tr, test_data_te = self._split_train_test(test_data)

        val_us = list(val_data.groupby(uhead).count().index)
        te_us = list(test_data.groupby(uhead).count().index)
        us = val_us + te_us

        unique_uid = list(unique_uid)
        todel = [u for u in unique_uid[len(tr_users):] if u not in us]
        for u in todel:
            unique_uid.remove(u)

        self.i2id = dict((iid, i) for (i, iid) in enumerate(unique_iid))
        self.u2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

        logger.info("Saving unique_iid.txt.")
        pro_dir = self.cfg.proc_path
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, 'unique_iid.txt'), 'w') as f:
            for iid in unique_iid:
                f.write('%s\n' % iid)

        logger.info("Saving unique_uid.txt.")
        with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
            for uid in unique_uid:
                f.write('%s\n' % uid)

        train_data = self._numerize(train_data, self.u2id, self.i2id)
        val_data_tr = self._numerize(val_data_tr, self.u2id, self.i2id)
        val_data_te = self._numerize(val_data_te, self.u2id, self.i2id)
        test_data_tr = self._numerize(test_data_tr, self.u2id, self.i2id)
        test_data_te = self._numerize(test_data_te, self.u2id, self.i2id)

        logger.info("Saving all the files.")
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
        val_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
        val_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
        logger.info("Preprocessing complete!")

    def _filter(self, data, min_u=5, min_i=0):
        def get_count(data, idx):
            return data[[idx]].groupby(idx, as_index=False).size()

        [uhead, ihead] = data.columns.values[:2]
        if min_i > 0:
            icnt = get_count(data, ihead)
            data = data[data[ihead].isin(icnt.index[icnt >= min_i])]

        if min_u > 0:
            ucnt = get_count(data, uhead)
            data = data[data[uhead].isin(ucnt.index[ucnt >= min_u])]

        ucnt, icnt = get_count(data, uhead), get_count(data, ihead)
        return data, ucnt, icnt

    def _numerize(self, data, u2id, i2id):
        [uhead, ihead] = data.columns.values[:2]
        uid = data[uhead].apply(lambda x: u2id[x])
        iid = data[ihead].apply(lambda x: i2id[x])
        if self.cfg.topn:
            return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])
        else:
            dic_data = {'uid': uid, 'iid': iid}
            for c in data.columns.values[2:]:
                dic_data[c] = data[c]
            cols = ['uid', 'iid'] + list(data.columns[2:])
            return pd.DataFrame(data=dic_data, columns=cols)

    def _split_train_test(self, data):
        np.random.seed(self.cfg.seed)
        test_prop = float(self.cfg.test_prop) if self.cfg.test_prop else 0.2
        uhead = data.columns.values[0]
        data_grouped_by_user = data.groupby(uhead)
        tr_list, te_list = [], []

        for _, group in data_grouped_by_user:
            n_items_u = len(group)
            if n_items_u > 1:
                idx = np.zeros(n_items_u, dtype='bool')
                sz = max(int(test_prop * n_items_u), 1)
                idx[np.random.choice(n_items_u, size=sz, replace=False).astype('int64')] = True
                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                # This should never be True
                logger.warning("Skipped user in test set: number of ratings <= 1.")

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        return data_tr, data_te


class DataReader():
    r"""Utility class for reading pre-processed dataset.

    The reader assumes that the data set has been previously pre-processed using
    :meth:`DataProcessing.process`. To avoid malfunctioning, the same configuration file used for
    the pre-processing should be used to load the data set. Once a reader is created it is possible
    to load to the training, validation and test set using :meth:`load_data`.

    Parameters
    ----------
    data_config : :class:`rectorch.configuration.DataConfig` or :obj:`str`:
        Represents the data pre-processing configurations.
        When ``type(data_config) == str`` is expected to be the path to the data configuration file.
        In that case a :class:`DataConfig` object is contextually created.

    Attributes
    ----------
    cfg : :class:`rectorch.configuration.DataConfig`
        Object containing the loading configurations.
    n_items : :obj:`int`
        The number of items in the data set.

    Raises
    ------
    :class:`TypeError`
        Raised when ``data_config`` is neither a :obj:`str` nor a
        :class:`rectorch.configuration.DataConfig`.
    """
    def __init__(self, data_config):
        if isinstance(data_config, DataConfig):
            self.cfg = data_config
        elif isinstance(data_config, str):
            self.cfg = DataConfig(data_config)
        else:
            raise TypeError("'data_config' must be of type 'DataConfig' or 'str'.")
        self.n_items = self._load_n_items()

    def load_data(self, datatype='train'):
        r"""Load (part of) the pre-processed data set.

        Load from the pre-processed file the data set, or part of it, accordingly to the
        ``datatype``.

        Parameters
        ----------
        datatype : :obj:`str` in {``'train'``, ``'validation'``, ``'test'``, ``'full'``} [optional]
            String representing the type of data that has to be loaded, by default ``'train'``.
            When ``datatype`` is equal to ``'full'`` the entire data set is loaded into a sparse
            matrix.

        Returns
        -------
        :obj:`scipy.sparse.csr_matrix` or :obj:`tuple` of :obj:`scipy.sparse.csr_matrix`
            The data set or part of it. When ``datatype`` is ``'full'`` or ``'train'`` a single
            sparse matrix is returned representing the full data set or the training set,
            respectively. While, if ``datatype`` is ``'validation'`` or ``'test'`` a pair of
            sparse matrices are returned. The first matrix is the training part (i.e., for each
            user its training set of items), and the second matrix is the test part (i.e., for each
            user its test set of items).

        Raises
        ------
        :class:`ValueError`
            Raised when ``datatype`` does not match any of the valid strings.
        """
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_train_test_data(datatype)
        elif datatype == 'test':
            return self._load_train_test_data(datatype)
        elif datatype == 'full':
            tr = self._load_train_data()
            val_tr, val_te = self._load_train_test_data("validation")
            te_tr, te_te = self._load_train_test_data("test")
            val = val_tr + val_te
            te = te_tr + te_te
            return sparse.vstack([tr, val, te])
        else:
            raise ValueError("Possible datatype values are 'train', 'validation', 'test', 'full'.")

    def _load_n_items(self):
        unique_iid = []
        with open(os.path.join(self.cfg.proc_path, 'unique_iid.txt'), 'r') as f:
            for line in f:
                unique_iid.append(line.strip())
        return len(unique_iid)

    def _load_train_data(self):
        path = os.path.join(self.cfg.proc_path, 'train.csv')
        data = pd.read_csv(path)
        n_users = data['uid'].max() + 1

        rows, cols = data['uid'], data['iid']
        if self.cfg.topn:
            values = np.ones_like(rows)
        else:
            values = data[data.columns.values[2]]

        data = sparse.csr_matrix((values, (rows, cols)),
                                 dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def _load_train_test_data(self, datatype='test'):
        tr_path = os.path.join(self.cfg.proc_path, f'{datatype}_tr.csv')
        te_path = os.path.join(self.cfg.proc_path, f'{datatype}_te.csv')

        data_tr = pd.read_csv(tr_path)
        data_te = pd.read_csv(te_path)

        start_idx = min(data_tr['uid'].min(), data_te['uid'].min())
        end_idx = max(data_tr['uid'].max(), data_te['uid'].max())

        rows_tr, cols_tr = data_tr['uid'] - start_idx, data_tr['iid']
        rows_te, cols_te = data_te['uid'] - start_idx, data_te['iid']

        if self.cfg.topn:
            values_tr = np.ones_like(rows_tr)
            values_te = np.ones_like(rows_te)
        else:
            values_tr = data_tr[data_tr.columns.values[2]]
            values_te = data_te[data_tr.columns.values[2]]

        data_tr = sparse.csr_matrix((values_tr, (rows_tr, cols_tr)),
                                    dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((values_te, (rows_te, cols_te)),
                                    dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))

        tr_idx = np.diff(data_tr.indptr) != 0
        #te_idx = np.diff(data_te.indptr) != 0
        #keep_idx = tr_idx * te_idx
        return data_tr[tr_idx], data_te[tr_idx]

    def _to_dict(self, data, col="timestamp"):
        data = data.sort_values(col)
        imin = data["uid"].min()
        #ugly but it works
        grouped = data.groupby(by="uid")
        grouped = grouped.apply(lambda x: x.sort_values(col)).reset_index(drop=True)
        grouped = grouped.groupby(by="uid")
        return {idx - imin : list(group["iid"]) for idx, group in grouped}

    def _split_train_test(self, data, col):
        np.random.seed(self.cfg.seed)
        test_prop = float(self.cfg.test_prop) if self.cfg.test_prop else 0.2
        uhead = data.columns.values[0]
        #ugly but it works
        data_grouped_by_user = data.groupby(uhead)
        data_grouped_by_user = data_grouped_by_user.apply(lambda x: x.sort_values(col))
        data_grouped_by_user = data_grouped_by_user.reset_index(drop=True)
        data_grouped_by_user = data_grouped_by_user.groupby(uhead)
        tr_list, te_list = [], []

        for _, group in data_grouped_by_user:
            n_items_u = len(group)
            idx = np.zeros(n_items_u, dtype='bool')
            sz = max(int(test_prop * n_items_u), 1)
            idx[-sz:] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        return data_tr, data_te

    def load_data_as_dict(self, datatype='train', col="timestamp"):
        r"""Load the data as a dictionary

        The loaded dictionary has users as keys and lists of items as values. An entry
        of the dictionary represents the list of rated items (sorted by ``col``) by the user,
        i.e., the key.

        Parameters
        ----------
        datatype : :obj:`str` in {``'train'``, ``'validation'``, ``'test'``} [optional]
            String representing the type of data that has to be loaded, by default ``'train'``.
        col : :obj:`str` of :obj:`None` [optional]
            The name of the column on which items are ordered, by default "timestamp". If
            :obj:`None` no ordered is applied.

        Returns
        -------
        :obj:`dict` (key - :obj:`int`, value - :obj:`list` of :obj:`int`) or :obj:`tuple` of :obj:`dict`
            When ``datatype`` is ``'train'`` a single dictionary is returned representing the
            training set. While, if ``datatype`` is ``'validation'`` or ``'test'`` a pair of
            dictionaries returned. The first dictionary is the training part (i.e., for each
            user its training set of items), and the second dictionart is the test part (i.e., for
            each user its test set of items).
        """
        if datatype == 'train':
            path = os.path.join(self.cfg.proc_path, 'train.csv')
            data = pd.read_csv(path)
            return self._to_dict(data, col)
        elif datatype == 'validation':
            path_tr = os.path.join(self.cfg.proc_path, 'validation_tr.csv')
            path_te = os.path.join(self.cfg.proc_path, 'validation_te.csv')
        elif datatype == 'test':
            path_tr = os.path.join(self.cfg.proc_path, 'test_tr.csv')
            path_te = os.path.join(self.cfg.proc_path, 'test_te.csv')
        elif datatype == 'full':
            data_list = [pd.read_csv(os.path.join(self.cfg.proc_path, 'train.csv')),
                         pd.read_csv(os.path.join(self.cfg.proc_path, 'validation_tr.csv')),
                         pd.read_csv(os.path.join(self.cfg.proc_path, 'validation_te.csv')),
                         pd.read_csv(os.path.join(self.cfg.proc_path, 'test_tr.csv')),
                         pd.read_csv(os.path.join(self.cfg.proc_path, 'test_te.csv'))]
            combined = pd.concat(data_list)
            return self._to_dict(combined, col)
        else:
            raise ValueError("Possible datatype values are 'train', 'validation', 'test', 'full'.")

        data_tr = pd.read_csv(path_tr)
        data_te = pd.read_csv(path_te)

        combined = pd.concat([data_tr, data_te], ignore_index=True)
        combined = combined.sort_values(col)
        data_tr, data_te = self._split_train_test(combined, col)

        return self._to_dict(data_tr, col), self._to_dict(data_te, col)


class DatasetManager():
    """Helper class for handling data sets.

    Given the configuration file, :class:`DatasetManager` automatically load training, validation,
    and test sets that will be accessible from its attributes. It also gives the possibility of
    loading the data set into only a training and a test set. In this latter case, training,
    validation and the training part of the test set are merged together to form a bigger training
    set. The test set will be only the test part of the test set.

    Parameters
    ----------
    config_file : :class:`rectorch.configuration.DataConfig` or :obj:`str`:
        Represents the data pre-processing configurations.
        When ``type(config_file) == str`` is expected to be the path to the data configuration file.
        In that case a :class:`DataConfig` object is contextually created.

    Attributes
    ----------
    n_items : :obj:`int`
        Number of items in the data set.
    training_set : :obj:`tuple` of :obj:`scipy.sparse.csr_matrix`
        The first matrix is the sparse training set matrix, while the  second element of the tuple
        is :obj:`None`.
    validation_set : :obj:`tuple` of :obj:`scipy.sparse.csr_matrix`
        The first matrix is the training part of the validation set (i.e., for each user its
        training set of items), and the second matrix is the test part of the validation set (i.e.,
        for each user its test set of items).
    test_set : :obj:`tuple` of :obj:`scipy.sparse.csr_matrix`
        The first matrix is the training part of the test set (i.e., for each user its
        training set of items), and the second matrix is the test part of the test set (i.e.,
        for each user its test set of items).
    """
    def __init__(self, config_file):
        reader = DataReader(config_file)
        train_data = reader.load_data('train')
        vad_data_tr, vad_data_te = reader.load_data('validation')
        test_data_tr, test_data_te = reader.load_data('test')

        self.n_items = reader.n_items
        self.training_set = (train_data, None)
        self.validation_set = (vad_data_tr, vad_data_te)
        self.test_set = (test_data_tr, test_data_te)

    def get_train_and_test(self):
        r"""Return a training and a test set.

        Load the data set into only a training and a test set. Training, validation and the training
        part of the test set are merged together to form a bigger training set.
        The test set will be only the test part of the test set. The training part of the test users
        are the last ``t`` rows of the training matrix, where ``t`` is the number of test users.

        Returns
        -------
        :obj:`tuple` of :obj:`scipy.sparse.csr_matrix`
            The first matrix is the training set, the second one is the test set.
        """
        tr = sparse.vstack([self.training_set[0], sum(self.validation_set), self.test_set[0]])
        shape = tr.shape[0] - self.test_set[1].shape[0], tr.shape[1]
        te = sparse.vstack([sparse.csr_matrix(shape), self.test_set[1]])
        return tr, te
