r"""This module contains recommender systems based on matrix factorization or related techniques.
"""
import os
import torch
import numpy as np
from rectorch.models import RecSysModel
from rectorch.samplers import ArrayDummySampler, TensorDummySampler, SparseDummySampler
from rectorch import env

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["EASE", "ADMM_Slim"]


class EASE(RecSysModel):
    r"""Embarrassingly Shallow AutoEncoders for Sparse Data (EASE) model.

    This model has been proposed in [EASE]_ and it can be summarized as follows.
    Given the rating matrix :math:`\mathbf{X} \in \mathbb{R}^{n \times m}` with *n* users and *m*
    items, EASE aims at solving the following optimization problem:

    :math:`\min_{\mathbf{B}} \|\mathbf{X}-\mathbf{X} \mathbf{B}\|_{F}^{2}+\
    \lambda \cdot\|\mathbf{B}\|_{F}^{2}`

    subject to :math:`\operatorname{diag}(\mathbf{B})=0`.

    where :math:`\mathbf{B} \in \mathbb{R}^{m \times m}` is like a kernel matrix between items.
    Then, a prediction for a user-item pair *(u,j)* will be computed by
    :math:`S_{u j}=\mathbf{X}_{u,:} \cdot \mathbf{B}_{:, j}`

    It can be shown that estimating :math:`\mathbf{B}` can be done in closed form by computing

    :math:`\hat{\mathbf{B}}=(\mathbf{X}^{\top} \mathbf{X}+\lambda \mathbf{I})^{-1} \cdot\
    (\mathbf{X}^{\top} \mathbf{X}-\mathbf{I}^\top \gamma)`

    where :math:`\gamma \in \mathbb{R}^m` is the vector of Lagragian multipliers, and
    :math:`\mathbf{I}` is the identity matrix.

    Parameters
    ----------
    lam : :obj:`float` [optional]
        The regularization hyper-parameter, by default 100.

    Attributes
    ----------
    lam : :obj:`float`
        See ``lam`` parameter.
    model : :class:`torch.FloatTensor`
        Represent the model, i.e.m the matrix score **S**. If the model has not been trained yet
        ``model`` is set to :obj:`None`.

    References
    ----------
    .. [EASE] Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data.
       In The World Wide Web Conference (WWW '19). Association for Computing Machinery,
       New York, NY, USA, 3251–3257. DOI: https://doi.org/10.1145/3308558.3313710
    """
    def __init__(self, lam=100.):
        self.lam = lam
        self.model = None

    def train(self, data_sampler):
        """Training of the EASE model.

        Parameters
        ----------
        data_sampler : :class:`rectorch.samplers.DummySampler`
            The training sampler.
        """
        env.logger.info("EASE - start tarining (lam=%.4f)", self.lam)
        if isinstance(data_sampler, ArrayDummySampler):
            X = data_sampler.data_tr
        elif isinstance(data_sampler, TensorDummySampler):
            X = data_sampler.data_tr.numpy()
        elif isinstance(data_sampler, SparseDummySampler):
            X = data_sampler.data_tr.toarray()
        else:
            raise ValueError("Wrong sampler type.")

        G = np.dot(X.T, X)
        env.logger.info("EASE - linear kernel computed")
        diag_idx = np.diag_indices(G.shape[0])
        G[diag_idx] += self.lam
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        B[diag_idx] = 0
        del P
        self.model = np.dot(X, B)
        self.model = torch.from_numpy(self.model).float()
        env.logger.info("EASE - training complete")

    def predict(self, ids_te_users, test_tr, remove_train=True):
        r"""Prediction using the EASE model.

        For the EASE model the prediction list for a user *u* is done by computing

        :math:`S_{u}=\mathbf{X}_{u,:} \cdot \mathbf{B}`.

        However, in the **rectorch** implementation the prediction is simply a look up in the score
        matrix *S*.

        Parameters
        ----------
        ids_te_users : array_like
            List of the test user indexes.
        test_tr : :class:`scipy.sparse.csr_matrix` or :class:`numpy.ndarray`
            Training portion of the test users.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default :obj:`True`. Removing
            the training items means set their scores to :math:`-\infty`.

        Returns
        -------
        pred, : :obj:`tuple` with a single element
            pred : :class:`numpy.ndarray`
                The items' score (on the columns) for each user (on the rows).
        """
        pred = self.model[ids_te_users, :]
        if remove_train:
            pred[test_tr.nonzero()] = -np.inf
        return (pred, )

    def get_state(self):
        state = {'lambda': self.lam,
                 'model': self.model
                }
        return state

    def save_model(self, filepath):
        env.logger.info("Saving EASE model to %s...", filepath)
        torch.save(self.get_state(), filepath)
        env.logger.info("Model saved!")

    @classmethod
    def from_state(cls, state):
        ease = EASE(lam=state["lambda"])
        ease.model = state["model"]
        return ease

    @classmethod
    def load_model(cls, filepath):
        assert os.path.isfile(filepath), "The model file %s does not exist." %filepath
        env.logger.info("Loading EASE model from %s...", filepath)
        #state = np.load(filepath, allow_pickle=True)[()]
        state = torch.load(filepath)
        ease = EASE.from_state(state)
        env.logger.info("Model loaded!")
        return ease

    def __str__(self):
        s = "EASE(lambda=%.4f" % self.lam
        if self.model is not None:
            s += ", model size=(%d, %d))" %self.model.shape
        else:
            s += ") - not trained yet!"
        return s


class ADMM_Slim(RecSysModel):
    r"""ADMM SLIM: Sparse Recommendations for Many Users.

    ADMM SLIM [ADMMS]_ is a model similar to SLIM [SLIM]_ in which the objective function is solved
    using Alternating Directions Method of Multipliers (ADMM). In particular,
    given the rating matrix :math:`\mathbf{X} \in \mathbb{R}^{n \times m}` with *n* users and *m*
    items, ADMM SLIM aims at solving the following optimization problem:

    :math:`\min_{B,C,\Gamma} \frac{1}{2}\|X-X B\|_{F}^{2}+\frac{\lambda_{2}}{2} \cdot\|B\|_{F}^{2}+\
    \lambda_{1} \cdot\|C\|_{1} +\
    \langle\Gamma, B-C\rangle_{F}+\frac{\rho}{2} \cdot\|B-C\|_{F}^{2}`

    with :math:`\textrm{diag}(B)=0`, :math:`\Gamma \in \mathbb{R}^{m \times m}`, and the entry of
    *C* are all greater or equal than 0.

    The prediction for a user-item pair *(u,j)* is then computed by
    :math:`S_{u j}=\mathbf{X}_{u,:} \cdot \mathbf{B}_{:, j}`.


    Parameters
    ----------
    lambda1 : :obj:`float` [optional]
        Elastic net regularization hyper-parameters :math:`\lambda_1`, by default 5.
    lambda2 : :obj:`float` [optional]
        Elastic net regularization hyper-parameters :math:`\lambda_2`, by default 1e3.
    rho : :obj:`float` [optional]
        The penalty hyper-parameter :math:`\rho>0` that applies to :math:`\|B-C\|^2_F`,
        by default 1e5.
    nn_constr : :obj:`bool` [optional]
        Whether to keep the non-negativity constraint, by default :obj:`True`.
    l1_penalty : :obj:`bool` [optional]
        Whether to keep the L1 penalty, by default :obj:`True`. When ``l1_penalty = False`` then
        is like to set :math:`\lambda_1 = 0`.
    item_bias : :obj:`bool` [optional]
        Whether to model the item biases, by default :obj:`False`. When ``item_bias = True`` then
        the scoring function for the user-item pair *(u,i)* becomes:
        :math:`S_{ui}=(\mathbf{X}_{u,:} - \mathbf{b})\cdot \mathbf{B}_{:, i} + \mathbf{b}_i`.

    Attributes
    ----------
    model : :class:`torch.FloatTensor`
        The ADSMM model. If the model has not been trained yet ``model`` is set to :obj:`None`.
    other attributes : see the **Parameters** section.


    References
    ----------
    .. [ADMMS] Harald Steck, Maria Dimakopoulou, Nickolai Riabov, and Tony Jebara. 2020.
       ADMM SLIM: Sparse Recommendations for Many Users. In Proceedings of the 13th International
       Conference on Web Search and Data Mining (WSDM '20). Association for Computing Machinery,
       New York, NY, USA, 555–563. DOI: https://doi.org/10.1145/3336191.3371774
    .. [SLIM] X. Ning and G. Karypis. 2011. SLIM: Sparse Linear Methods for Top-N Recommender
       Systems. In Proceedings of the IEEE 11th International Conference on Data Mining,
       Vancouver,BC, 2011, pp. 497-506. DOI: https://doi.org/10.1109/ICDM.2011.134.
    """
    def __init__(self,
                 lambda1=5.,
                 lambda2=1e3,
                 rho=1e5,
                 nn_constr=True,
                 l1_penalty=True,
                 item_bias=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.rho = rho
        self.nn_constr = nn_constr
        self.l1_penalty = l1_penalty
        self.item_bias = item_bias
        self.model = None


    def train(self, data_sampler, num_iter=50, verbose=1):
        r"""Training of ADMM SLIM.

        The training procedure of ADMM SLIM highly depends on the setting of the
        hyper-parameters. By setting them in specific ways it is possible to define different
        variants of the algorithm. That are:

        1. (Vanilla) ADMM SLIM - :math:`\lambda_1, \lambda_2, \rho>0`, :attr:`item_bias` =
        :obj:`False`, and both :attr:`nn_constr` and :attr:`l1_penalty` set to :obj:`True`;

        2. ADMM SLIM w/o non-negativity constraint over C - :attr:`nn_constr` = :obj:`False` and
        :attr:`l1_penalty` set to :obj:`True`;

        3. ADMM SLIM w/o the L1 penalty - :attr:`l1_penalty` = :obj:`False` and
        :attr:`nn_constr` set to :obj:`True`;

        4. ADMM SLIM w/o L1 penalty and non-negativity constraint: :attr:`nn_constr` =
        :attr:`l1_penalty` = :obj:`False`.

        All these variants can also be combined with the inclusion of the item biases by setting
        :attr:`item_bias` to :obj:`True`.

        Parameters
        ----------
        data_sampler : :class:`rectorch.samplers.DummySampler`
            The training sampler.
        num_iter : :obj:`int` [optional]
            Maximum number of training iterations, by default 50. This argument has no effect
            if both :attr:`nn_constr` and :attr:`l1_penalty` are set to :obj:`False`.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        """
        def _soft_threshold(a, k):
            return np.maximum(0., a - k) - np.maximum(0., -a - k)

        if isinstance(data_sampler, ArrayDummySampler):
            X = data_sampler.data_tr
        #elif isinstance(data_sampler, TensorDummySampler):
        #    X = data_sampler.data_tr.numpy()
        elif isinstance(data_sampler, SparseDummySampler):
            X = data_sampler.data_tr.toarray()
        else:
            raise ValueError("Wrong sampler type.")

        if self.item_bias:
            b = X.sum(axis=0)
            X = X - np.outer(np.ones(X.shape[0]), b)

        XtX = X.T.dot(X)
        env.logger.info("ADMM_Slim - linear kernel computed")
        diag_indices = np.diag_indices(XtX.shape[0])
        XtX[diag_indices] += self.lambda2 + self.rho
        P = np.linalg.inv(XtX)
        env.logger.info("ADMM_Slim - inverse of XtX computed")

        if not self.nn_constr and not self.l1_penalty:
            C = np.eye(P.shape[0]) - P * np.diag(1. / np.diag(P))
        else:
            XtX[diag_indices] -= self.lambda2 + self.rho
            B_aux = P.dot(XtX)
            Gamma = np.zeros(XtX.shape, dtype=float)
            C = np.zeros(XtX.shape, dtype=float)

            log_delay = max(5, num_iter // (10*verbose))
            for j in range(num_iter):
                B_tilde = B_aux + P.dot(self.rho * C - Gamma)
                gamma = np.diag(B_tilde) / np.diag(P)
                B = B_tilde - P * np.diag(gamma)
                C = _soft_threshold(B + Gamma / self.rho, self.lambda1 / self.rho)
                if self.nn_constr and self.l1_penalty:
                    C = np.maximum(C, 0.)
                elif self.nn_constr and not self.l1_penalty:
                    C = np.maximum(B, 0.)
                Gamma += self.rho * (B - C)
                if not (j+1) % log_delay:
                    env.logger.info("| iteration %d/%d |", j+1, num_iter)

        self.model = np.dot(X, C)
        if self.item_bias:
            self.model += b

        self.model = torch.from_numpy(self.model).float()

    def predict(self, ids_te_users, test_tr, remove_train=True):
        r"""Prediction using the ADMM_Slim model.

        Parameters
        ----------
        ids_te_users : array_like
            List of the test user indexes.
        test_tr : :class:`scipy.sparse.csr_matrix` or :class:`numpy.ndarray`
            Training portion of the test users.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default :obj:`True`. Removing
            the training items means set their scores to :math:`-\infty`.

        Returns
        -------
        pred, : :obj:`tuple` with a single element
            pred : :class:`numpy.ndarray`
                The items' score (on the columns) for each user (on the rows).
        """
        pred = self.model[ids_te_users, :]
        if remove_train:
            pred[test_tr.nonzero()] = -np.inf
        return (pred, )

    def get_state(self):
        state = {'lambda1': self.lambda1,
                 'lambda2': self.lambda2,
                 'rho' : self.rho,
                 'model': self.model,
                 'nn_constr' : self.nn_constr,
                 'l1_penalty' : self.l1_penalty,
                 'item_bias' : self.item_bias
                }
        return state

    def save_model(self, filepath):
        env.logger.info("Saving ADMM_Slim model to %s...", filepath)
        torch.save(self.get_state(), filepath)
        env.logger.info("Model saved!")

    @classmethod
    def from_state(cls, state):
        admm = ADMM_Slim(lambda1=state["lambda1"],
                         lambda2=state["lambda2"],
                         rho=state["rho"],
                         nn_constr=state["nn_constr"],
                         l1_penalty=state["l1_penalty"],
                         item_bias=state["item_bias"])
        admm.model = state["model"]
        return admm

    @classmethod
    def load_model(cls, filepath):
        assert os.path.isfile(filepath), "The model file %s does not exist." %filepath
        env.logger.info("Loading ADMM_Slim model from %s...", filepath)
        #state = np.load(filepath, allow_pickle=True)[()]
        state = torch.load(filepath)
        admm = ADMM_Slim.from_state(state)
        env.logger.info("Model loaded!")
        return admm

    def __str__(self):
        s = "ADMM_Slim(lambda1=%.4f, lamdba2=%.4f" %(self.lambda1, self.lambda2)
        s += ", rho=%.4f" %self.rho
        s += ", non_negativity=%s" %self.nn_constr
        s += ", L1_penalty=%s" %self.l1_penalty
        s += ", item_bias=%s" %self.item_bias
        if self.model is not None:
            s += ", model size=(%d, %d))" %self.model.shape
        else:
            s += ") - not trained yet!"
        return s
