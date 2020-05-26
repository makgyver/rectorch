r"""This module includes the training algorithm for a bunch of state-of-the-art recommender systems.

Each new model must be a sub-class of the abstract class :class:`RecSysModel`. Moreover,
if the model is a standard neural network (NN) then it is advisable to inherit from
:class:`TorchNNTrainer` that offers a good base structure to develop a new NN training algorithm.
In these first releases of **rectorch** all models will be located in this module, but in the future
we plan to improve the structure of the library creating sub-modules.

Currently the implemented training algorithms are:

* :class:`MultiDAE`: Denoising Autoencoder for Collaborative filtering with Multinomial prior (in the paper *Mult-DAE*) [VAE]_;
* :class:`MultiVAE`: Variational Autoencoder for Collaborative filtering with Multinomial prior (in the paper *Mult-VAE*) [VAE]_;
* :class:`CMultiVAE`: Conditioned Variational Autoencoder (in the paper *C-VAE*) [CVAE]_;
* :class:`CFGAN`: Collaborative Filtering with Generative Adversarial Networks [CFGAN]_;
* :class:`EASE`: Embarrassingly shallow autoencoder for sparse data [EASE]_.
* :class:`ADMM_Slim`: ADMM SLIM: Sparse Recommendations for Many Users [ADMMS]_.
* :class:`SVAE`: Sequential Variational Autoencoders for Collaborative Filtering [SVAE]_.

It is also implemented a generic Variational autoencoder trainer (:class:`VAE`) based on the classic
loss function *cross-entropy* based reconstruction loss, plus the KL loss.

See Also
--------
Modules:
:mod:`nets <rectorch.nets>`
:mod:`samplers <rectorch.samplers>`

References
----------
.. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
   Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
   World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
   Committee, Republic and Canton of Geneva, CHE, 689–698.
   DOI: https://doi.org/10.1145/3178876.3186150
.. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
   Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
   https://arxiv.org/abs/2004.11141
.. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
   CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
   In Proceedings of the 27th ACM International Conference on Information and Knowledge
   Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
   DOI: https://doi.org/10.1145/3269206.3271743
.. [EASE] Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data.
   In The World Wide Web Conference (WWW ’19). Association for Computing Machinery,
   New York, NY, USA, 3251–3257. DOI: https://doi.org/10.1145/3308558.3313710
.. [ADMMS] Harald Steck, Maria Dimakopoulou, Nickolai Riabov, and Tony Jebara. 2020.
   ADMM SLIM: Sparse Recommendations for Many Users. In Proceedings of the 13th International
   Conference on Web Search and Data Mining (WSDM ’20). Association for Computing Machinery,
   New York, NY, USA, 555–563. DOI: https://doi.org/10.1145/3336191.3371774
.. [SVAE] Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi. 2019.
   Sequential Variational Autoencoders for Collaborative Filtering. In Proceedings of the Twelfth
   ACM International Conference on Web Search and Data Mining (WSDM ’19). Association for Computing
   Machinery, New York, NY, USA, 600–608. DOI: https://doi.org/10.1145/3289600.3291007
"""
import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .evaluation import ValidFunc, evaluate

__all__ = ['RecSysModel', 'TorchNNTrainer', 'AETrainer', 'VAE', 'MultiVAE', 'MultiDAE',\
    'CMultiVAE', 'EASE', 'CFGAN', 'SVAE']

logger = logging.getLogger(__name__)


class RecSysModel():
    r"""Abstract base class that any Recommendation model must inherit from.
    """
    def train(self, train_data, **kwargs):
        r"""Training procedure.

        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be queried for predictions.

        Parameters
        ----------
        train_data : :class:`rectorch.samplers.Sampler` or :class:`scipy.sparse.csr_matrix` or\
            :class:`torch.Tensor`
            This object represents the training data. If the training procedure is based on
            mini-batches, then ``train_data`` should be a :class:`rectorch.samplers.Sampler`.
        **kargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            training.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        r"""Perform the prediction using a trained model.

        The prediction is preformed over a generic input ``x`` and the method should be callable
        after the training procedure (:meth:`RecSysModel.train`).

        Parameters
        ----------
        x : :class:`rectorch.samplers.Sampler` or :class:`scipy.sparse.csr_matrix` or\
            :class:`torch.Tensor`
            The input for which the prediction has to be computed.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def save_model(self, filepath, *args, **kwargs):
        r"""Save the model to file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file to save the model.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def load_model(self, filepath, *args, **kwargs):
        r"""Load the model from file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()


class TorchNNTrainer(RecSysModel):
    r"""Abstract class representing a neural network-based model.

    This base class assumes that the model can be trained using a standard backpropagation
    procedure. It is not meant to manage complex training patterns, such as alternate training
    between more than one network as done with Generative Adversarial Networks. Thus, it assumes
    that there is a neural network (i.e., :class:`torch.nn.Module`)for which the parameters must be
    learned.

    Parameters
    ----------
    net : :class:`torch.nn.Module`
        The neural network architecture.
    learning_rate : :obj:`float` [optional]
        The learning rate for the optimizer, by default 1e-3.

    Attributes
    ----------
    network : :class:`torch.nn.Module`
        The neural network architecture.
    learning_rate : :obj:`float`
        The learning rate for the optimizer.
    optimizer : :class:`torch.optim.Optimizer`
        Optimizer used for performing the training.
    device : :class:`torch.device`
        Device where the pytorch tensors are saved.
    """
    def __init__(self, net, learning_rate=1e-3):
        self.network = net
        self.learning_rate = learning_rate
        self.optimizer = None #to be initialized in the sub-classes

        if next(self.network.parameters()).is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def loss_function(self, prediction, ground_truth, *args, **kwargs):
        r"""The loss function that the model wants to minimize.

        Parameters
        ----------
        prediction : :class:`torch.Tensor`
            The prediction tensor.
        ground_truth : :class:`torch.Tensor`
            The ground truth tensor that the model should have reconstructed correctly.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for computing the
            loss.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for computing the
            loss.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=100,
              verbose=1,
              **kwargs):
        r"""Training of a neural network-based model.

        Parameters
        ----------
        train_data : :class:`rectorch.samplers.Sampler`
            The sampler object that load the training set in mini-batches.
        valid_data : :class:`rectorch.samplers.Sampler` [optional]
            The sampler object that load the validation set in mini-batches, by default ``None``.
            If the model does not have any validation procedure set this parameter to ``None``.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default ``None``.
            If ``valid_data`` is not ``None`` then ``valid_metric`` must be not ``None``.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`evaluation.ValidFunc` [optional]
            The validation function, by default a standard validation procedure, i.e.,
            :func:`evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 100.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum (that depends on the size of
            the training set) verbosity higher values will not have any effect.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def train_epoch(self, epoch, train_data, *args, **kwargs):
        r"""Training of a single epoch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        train_data : :class:`rectorch.samplers.Sampler`
            The sampler object that load the training set in mini-batches.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            training.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            training.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def train_batch(self, epoch, tr_batch, te_batch, *args, **kwargs):
        r"""Training of a single batch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        tr_batch : :class:`torch.Tensor`
            Traning part of the current batch.
        te_batch : :class:`torch.Tensor` or ``None``
            Test part of the current batch, if any, otherwise ``None``.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            training on the batch.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            training on the batch.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        s = self.__class__.__name__ + "(\n"
        for k, v in self.__dict__.items():
            sv = "\n".join(["  "+line for line in str(str(v)).split("\n")])[2:]
            s += "  %s = %s,\n" % (k, sv)
        s = s[:-2] + "\n)"
        return s

    def __repr__(self):
        return str(self)


class AETrainer(TorchNNTrainer):
    r"""Base class for Autoencoder-based models.

    Parameters
    ----------
    ae_net : :class:`torch.nn.Module`
        The autoencoder neural network.
    learning_rate : :obj:`float` [optional]
        The learning rate for the optimizer, by default 1e-3.

    Attributes
    ----------
    optimizer : :class:`torch.optim.Adam`
        The optimizer is an Adam optimizer with default parameters and learning rate equals to
        ``learning_rate``.

    other attributes : see the base class :class:`TorchNNTrainer`.
    """
    def __init__(self, ae_net, learning_rate=1e-3):
        super(AETrainer, self).__init__(ae_net, learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def loss_function(self, prediction, ground_truth):
        r"""Vanilla Autoencoder loss function.

        This is a standard Mean Squared Error (squared L2 norm) loss, that is

        :math:`\mathcal{L} = \frac{1}{L} \sum_{i=1}^L (x_i - y_i)^2`

        where L is the batch size x and y are the ground truth and the prediction, respectively.

        Parameters
        ----------
        prediction : :class:`torch.Tensor`
            The reconstructed input, i.e., the output of the variational autoencoder. It is meant
            to be the reconstruction over a batch.
        ground_truth : :class:`torch.Tensor`
            The input, and hence the target tensor. It is meant to be a batch size input.
        mu : :class:`torch.Tensor`
            The mean part of latent space for the given ``x``. Together with ``logvar`` represents
            the representation of the input ``x`` before the reparameteriation trick.
        logvar : :class:`torch.Tensor`
            The (logarithm of the) variance part of latent space for the given ``x``. Together with
            ``mu`` represents the representation of the input ``x`` before the reparameteriation
            trick.

        Returns
        -------
        :class:`torch.Tensor`
            Tensor (:math:`1 \times 1`) representing the average loss incurred over the input
            batch.
        """
        return torch.nn.MSELoss()(ground_truth, prediction)

    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=100,
              verbose=1):
        try:
            for epoch in range(1, num_epochs + 1):
                self.train_epoch(epoch, train_data, verbose)
                if valid_data is not None:
                    assert valid_metric is not None, \
                                "In case of validation 'valid_metric' must be provided"
                    valid_res = valid_func(self, valid_data, valid_metric)
                    mu_val = np.mean(valid_res)
                    std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                    logger.info('| epoch %d | %s %.3f (%.4f) |',
                                epoch, valid_metric, mu_val, std_err_val)
        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')


    def train_epoch(self, epoch, train_loader, verbose=1):
        self.network.train()
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(train_loader) // 10**verbose)

        for batch_idx, (data, gt) in enumerate(train_loader):
            partial_loss += self.train_batch(data, gt)
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                logger.info('| epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                            epoch, (batch_idx+1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            partial_loss / log_delay)
                train_loss += partial_loss
                partial_loss = 0.0
                start_time = time.time()
        total_loss = (train_loss + partial_loss) / len(train_loader)
        time_diff = time.time() - epoch_start_time
        logger.info("| epoch %d | loss %.4f | total time: %.2fs |", epoch, total_loss, time_diff)

    def train_batch(self, tr_batch, te_batch=None):
        r"""Training of a single batch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        tr_batch : :class:`torch.Tensor`
            Traning part of the current batch.
        te_batch : :class:`torch.Tensor` or ``None`` [optional]
            Test part of the current batch, if any, otherwise ``None``, by default ``None``.

        Returns
        -------
        :obj:`float`
            The loss incurred in the batch.
        """
        data_tensor = tr_batch.view(tr_batch.shape[0], -1).to(self.device)
        self.optimizer.zero_grad()
        recon_batch = self.network(data_tensor)
        loss = self.loss_function(recon_batch, data_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x, remove_train=True):
        r"""Perform the prediction using a trained Autoencoder.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input for which the prediction has to be computed.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default True. Removing
            the training items means set their scores to :math:`-\infty`.

        Returns
        -------
        recon_x, : :obj:`tuple` with a single element
            recon_x : :class:`torch.Tensor`
                The reconstructed input, i.e., the output of the autoencoder.
                It is meant to be the reconstruction over the input batch ``x``.
        """
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x = self.network(x_tensor)
            if remove_train:
                recon_x[tuple(x_tensor.nonzero().t())] = -np.inf
            return (recon_x, )

    def save_model(self, filepath, cur_epoch):
        r"""Save the model to file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file to save the model.
        cur_epoch : :obj:`int`
            The last training epoch.
        """
        state = {'epoch': cur_epoch,
                 'state_dict': self.network.state_dict(),
                 'optimizer': self.optimizer.state_dict()
                }
        self._save_checkpoint(filepath, state)

    def _save_checkpoint(self, filepath, state):
        logger.info("Saving model checkpoint to %s...", filepath)
        torch.save(state, filepath)
        logger.info("Model checkpoint saved!")

    def load_model(self, filepath):
        r"""Load the model from file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.

        Returns
        -------
        :obj:`dict`
            A dictionary that summarizes the state of the model when it has been saved.
            Note: not all the information about the model are stored in the saved 'checkpoint'.
        """
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Model checkpoint loaded!")
        return checkpoint


class VAE(AETrainer):
    r"""Class representing a standard Variational Autoencoder.

    The learning follows standard backpropagation minimizing the loss described in [KINGMA]_.
    See :meth:`VAE.loss_function` for more details.

    Notes
    -----
    Parameters and Attributes are the same as in the base class :class:`AETrainer`.

    References
    ----------
    .. [KINGMA] Kingma, Diederik P and Welling, Max Auto-Encoding Variational Bayes, 2013.
       arXiv pre-print: https://arxiv.org/abs/1312.6114.
    """
    def loss_function(self, recon_x, x, mu, logvar):
        r"""Standard VAE loss function.

        This method implements the loss function described in [KINGMA]_ assuming a Gaussian latent,
        that is:

        :math:`\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{KL}`

        where

        :math:`\mathcal{L}_{rec} = -\frac{1}{L}\
        \sum_{l} E_{\sim q_{\theta}(z | x_{i})}[\log p(x_{i} | z^{(i, l)})]`

        and

        :math:`\mathcal{L}_{KL} = -\frac{1}{2} \sum_{j=1}^{J}\
        [1+\log (\sigma_{i}^{2})-\sigma_{i}^{2}-\mu_{i}^{2}]`

        with J is the dimension of the latent vector z, and L is the number of samples
        stochastically drawn according to reparameterization trick.

        Parameters
        ----------
        recon_x : :class:`torch.Tensor`
            The reconstructed input, i.e., the output of the variational autoencoder. It is meant
            to be the reconstruction over a batch.
        x : :class:`torch.Tensor`
            The input, and hence the target tensor. It is meant to be a batch size input.
        mu : :class:`torch.Tensor`
            The mean part of latent space for the given ``x``. Together with ``logvar`` represents
            the representation of the input ``x`` before the reparameteriation trick.
        logvar : :class:`torch.Tensor`
            The (logarithm of the) variance part of latent space for the given ``x``. Together with
            ``mu`` represents the representation of the input ``x`` before the reparameteriation
            trick.

        Returns
        -------
        :class:`torch.Tensor`
            Tensor (:math:`1 \times 1`) representing the average loss incurred over the input
            batch.

        References
        ----------
        .. [KINGMA] Kingma, Diederik P and Welling, Max Auto-Encoding Variational Bayes, 2013.
           arXiv pre-print: https://arxiv.org/abs/1312.6114.
        """
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + KLD

    def train_batch(self, tr_batch, te_batch=None):
        data_tensor = tr_batch.view(tr_batch.shape[0], -1).to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, var = self.network(data_tensor)
        loss = self.loss_function(recon_batch, data_tensor, mu, var)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x, remove_train=True):
        r"""Perform the prediction using a trained Variational Autoencoder.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input batch tensor for which the prediction must be computed.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default True. Removing
            the training items means set their scores to :math:`-\infty`.

        Returns
        -------
        recon_x, mu, logvar : :obj:`tuple`
            recon_x : :class:`torch.Tensor`
                The reconstructed input, i.e., the output of the variational autoencoder.
                It is meant to be the reconstruction over the input batch ``x``.
            mu : :class:`torch.Tensor`
                The mean part of latent space for the given ``x``. Together with ``logvar``
                represents the representation of the input ``x`` before the reparameteriation trick.
            logvar : :class:`torch.Tensor`
                The (logarithm of the) variance part of latent space for the given ``x``. Together
                with ``mu`` represents the representation of the input ``x`` before the
                reparameteriation trick.
        """
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[tuple(x_tensor.nonzero().t())] = -np.inf
            return recon_x, mu, logvar


class MultiDAE(AETrainer):
    r"""Denoising Autoencoder with multinomial likelihood for collaborative filtering.

    This model has been proposed in [VAE]_ as a baseline method to compare with Mult-VAE.
    The model represent a standard denoising autoencoder in which the data is assumed to be
    multinomial distributed.

    Parameters
    ----------
    mdae_net : :class:`torch.nn.Module`
        The autoencoder neural network.
    lam : :obj:`float` [optional]
        The regularization hyper-parameter :math:`\lambda` as defined in [VAE]_, by default 0.2.
    learning_rate : :obj:`float` [optional]
        The learning rate for the optimizer, by default 1e-3.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    """
    def __init__(self,
                 mdae_net,
                 lam=0.2,
                 learning_rate=1e-3):
        super(MultiDAE, self).__init__(mdae_net, learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=0.001)
        self.lam = lam

    def loss_function(self, recon_x, x):
        r"""Multinomial likelihood denoising autoencoder loss.

        Since the model assume a multinomial distribution over the input, then te reconstruction
        loss must be modified with respect to a vanilla VAE. In particular,
        the MultiDAE loss function is a combination of a reconstruction loss and a regularization
        loss, i.e.,

        :math:`\mathcal{L}(\mathbf{x}_{u} ; \theta, \phi) =\
        \mathcal{L}_{rec}(\mathbf{x}_{u} ; \theta, \phi) + \lambda\
        \mathcal{L}_{reg}(\mathbf{x}_{u} ; \theta, \phi)`

        where

        :math:`\mathcal{L}_{rec}(\mathbf{x}_{u} ; \theta, \phi) =\
        \mathbb{E}_{q_{\phi}(\mathbf{z}_{u} | \mathbf{x}_{u})}[\log p_{\theta}\
        (\mathbf{x}_{u} | \mathbf{z}_{u})]`

        and

        :math:`\mathcal{L}_{reg}(\mathbf{x}_{u} ; \theta, \phi) = \| \theta \|_2 + \| \phi \|_2`,

        with :math:`\mathbf{x}_u` the input vector and :math:`\mathbf{z}_u` the latent vector
        representing the user *u*.

        Parameters
        ----------
        recon_x : :class:`torch.Tensor`
            The reconstructed input, i.e., the output of the variational autoencoder. It is meant
            to be the reconstruction over a batch.
        x : :class:`torch.Tensor`
            The input, and hence the target tensor. It is meant to be a batch size input.

        Returns
        -------
        :class:`torch.Tensor`
            Tensor (:math:`1 \times 1`) representing the average loss incurred over the input
            batch.
        """
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        l2_reg = 0
        for W in self.network.parameters():
            l2_reg += W.norm(2)

        return BCE + self.lam * l2_reg


class MultiVAE(VAE):
    r"""Variational Autoencoder for collaborative Filtering.

    MultiVAE (dubbed Mult-VAE in [VAE]_) is a vanilla VAE in which the data distribution is
    assumed to be multinomial and the objective function is an under-regularized version
    of the standard VAE loss function. Specifically, the Kullbach-Liebler divergence term is
    weighted by an hyper-parameter (:math:`\beta`) that shows to improve de recommendations'
    quality when < 1. So, the regularization term is weighted less giving to the model more freedom
    in representing the input in the latent space. More details about this loss are given in
    :meth:`MultiVAE.loss_function`.

    Parameters
    ----------
    mvae_net : :class:`torch.nn.Module`
        The variational autoencoder neural network.
    beta : :obj:`float` [optional]
        The :math:`\beta` hyper-parameter of Multi-VAE. When ``anneal_steps > 0`` then this
        value is the value to anneal starting from 0, otherwise the ``beta`` will be fixed to
        the given value for the duration of the training. By default 1.
    anneal_steps : :obj:`int` [optional]
        Number of annealing step for reaching the target value ``beta``, by default 0.
        0 means that no annealing will be performed and the regularization parameter will be
        fixed to ``beta``.
    learning_rate : :obj:`float` [optional]
        The learning rate for the optimizer, by default 1e-3.

    Attributes
    ----------
    anneal_steps : :obj:`int`
        See ``anneal_steps`` parameter.
    self.annealing : :obj:`bool`
        Whether the annealing is active or not. It is implicitely set to ``True`` if
        ``anneal_steps > 0``, otherwise is set to ``False``.
    gradient_updates : :obj:`int`
        Number of gradient updates since the beginning of the training. Once
        ``gradient_updates >= anneal_steps``, then the annealing is complete and the used
        :math:`\beta` in the loss function is ``beta``.
    beta : :obj:`float`
        See ``beta`` parameter.
    optimizer : :class:`torch.optim.Adam`
        The optimizer is an Adam optimizer with default parameters and learning rate equals to
        ``learning_rate``.

    other attributes : see the base class :class:`VAE`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
        Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
        World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
        Committee, Republic and Canton of Geneva, CHE, 689–698.
        DOI: https://doi.org/10.1145/3178876.3186150
    """
    def __init__(self,
                 mvae_net,
                 beta=1.,
                 anneal_steps=0,
                 learning_rate=1e-3):
        super(MultiVAE, self).__init__(mvae_net, learning_rate=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=learning_rate,
                                    weight_decay=0.0)
        self.anneal_steps = anneal_steps
        self.annealing = anneal_steps > 0
        self.gradient_updates = 0.
        self.beta = beta

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        r"""VAE for collaborative filtering loss function.

        MultiVAE assumes a multinomial distribution over the input and this is reflected in the loss
        function. The loss is a :math:`\beta` ELBO (Evidence Lower BOund) in which the
        regularization part is weighted by a hyper-parameter :math:`\beta`. Moreover, as in
        MultiDAE, the reconstruction loss is based on the multinomial likelihood.
        Specifically, the loss function of MultiVAE is defined as:

        :math:`\mathcal{L}_{\beta}(\mathbf{x}_{u} ; \theta, \phi)=\
        \mathbb{E}_{q_{\phi}(\mathbf{z}_{u} | \mathbf{x}_{u})}[\log p_{\theta}\
        (\mathbf{x}_{u} | \mathbf{z}_{u})]-\beta \cdot \operatorname{KL}(q_{\phi}\
        (\mathbf{z}_{u} | \mathbf{x}_{u}) \| p(\mathbf{z}_{u}))`

        Parameters
        ----------
        recon_x : :class:`torch.Tensor`
            The reconstructed input, i.e., the output of the variational autoencoder. It is meant
            to be the reconstruction over a batch.
        x : :class:`torch.Tensor`
            The input, and hence the target tensor. It is meant to be a batch size input.
        mu : :class:`torch.Tensor`
            The mean part of latent space for the given ``x``. Together with ``logvar`` represents
            the representation of the input ``x`` before the reparameteriation trick.
        logvar : :class:`torch.Tensor`
            The (logarithm of the) variance part of latent space for the given ``x``. Together with
            ``mu`` represents the representation of the input ``x`` before the reparameteriation
            trick.
        beta : :obj:`float` [optional]
            The current :math:`\beta` regularization hyper-parameter, by default 1.0.

        Returns
        -------
        :class:`torch.Tensor`
            Tensor (:math:`1 \times 1`) representing the average loss incurred over the input
            batch.
        """
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + beta * KLD

    def train_batch(self, tr_batch, te_batch=None):
        data_tensor = tr_batch.view(tr_batch.shape[0], -1).to(self.device)
        if te_batch is None:
            gt_tensor = data_tensor
        else:
            gt_tensor = te_batch.view(te_batch.shape[0], -1).to(self.device)

        if self.annealing:
            anneal_beta = min(self.beta, 1. * self.gradient_updates / self.anneal_steps)
        else:
            anneal_beta = self.beta

        self.optimizer.zero_grad()
        recon_batch, mu, var = self.network(data_tensor)
        loss = self.loss_function(recon_batch, gt_tensor, mu, var, anneal_beta)
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1.
        return loss.item()

    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=200,
              best_path="chkpt_best.pth",
              verbose=1):
        r"""Training procedure for Multi-VAE.

        The training of MultiVAE follows pretty much the same as a standard VAE with the only
        difference in the (possible) annealing of the hyper-parameter :math:`\beta`. This model
        also offer the possibility of keeping track of the best performing model in validation
        by setting the validation data (``valid_data``) and metric (``valid_metric``). The model
        will be saved in the file indicated in the parameter ``best_path``.

        Parameters
        ----------
        train_data : :class:`rectorch.samplers.Sampler`
            The sampler object that load the training set in mini-batches.
        valid_data : :class:`rectorch.samplers.Sampler` [optional]
            The sampler object that load the validation set in mini-batches, by default ``None``.
            If the model does not have any validation procedure set this parameter to ``None``.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default ``None``.
            If ``valid_data`` is not ``None`` then ``valid_metric`` must be not ``None``.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`evaluation.ValidFunc` [optional]
            The validation function, by default a standard validation procedure, i.e.,
            :func:`evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 100.
        best_path : :obj:`str` [optional]
            String representing the path where to save the best performing model on the validation
            set. By default ``"chkpt_best.pth"``.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum (that depends on the size of
            the training set) verbosity higher values will not have any effect.
        """
        try:
            best_perf = -1. #Assume the higher the better >= 0
            for epoch in range(1, num_epochs + 1):
                self.train_epoch(epoch, train_data, verbose)
                if valid_data:
                    assert valid_metric is not None, \
                                "In case of validation 'valid_metric' must be provided"
                    valid_res = valid_func(self, valid_data, valid_metric)
                    mu_val = np.mean(valid_res)
                    std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                    logger.info('| epoch %d | %s %.3f (%.4f) |',
                                epoch, valid_metric, mu_val, std_err_val)

                    if best_perf < mu_val:
                        self.save_model(best_path, epoch)
                        best_perf = mu_val

        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def save_model(self, filepath, cur_epoch):
        state = {'epoch': cur_epoch,
                 'state_dict': self.network.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'gradient_updates': self.gradient_updates
                }
        self._save_checkpoint(filepath, state)

    def load_model(self, filepath):
        checkpoint = super().load_model(filepath)
        self.gradient_updates = checkpoint['gradient_updates']
        return checkpoint


class CMultiVAE(MultiVAE):
    r"""Conditioned Variatonal Autoencoder for collaborative filtering.

    Conditioned Variational Autoencoder (C-VAE) for constrained top-N item recommendation can
    recommend items that have to satisfy a given condition. The architecture is similar to a
    standard VAE in which the condition vector is fed into the encoder.
    The loss function can be seen in two ways:

    - same as in :class:`MultiVAE` but with a different target reconstruction. Infact, the\
        network has to reconstruct only those items satisfying a specific condition;
    - a modified loss which performs the filtering by itself.

    More details about the loss function are given in the paper [CVAE]_.

    The training process is almost identical to the one of :class:`MultiVAE` but the sampler
    must be a :class:`samplers.ConditionedDataSampler`.

    Notes
    -----
    For parameters and attributes please refer to :class:`MultiVAE`.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    """
    def __init__(self,
                 cmvae_net,
                 beta=1.,
                 anneal_steps=0,
                 learning_rate=1e-3):
        super(CMultiVAE, self).__init__(cmvae_net,
                                        beta=beta,
                                        anneal_steps=anneal_steps,
                                        learning_rate=learning_rate)

    def predict(self, x, remove_train=True):
        self.network.eval()
        cond_dim = self.network.cond_dim
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[tuple(x_tensor[:, :-cond_dim].nonzero().t())] = -np.inf
            return recon_x, mu, logvar


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
    model : :class:`numpy.ndarray`
        Represent the model, i.e.m the matrix score **S**. If the model has not been trained yet
        ``model`` is set to ``None``.

    References
    ----------
    .. [EASE] Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data.
       In The World Wide Web Conference (WWW ’19). Association for Computing Machinery,
       New York, NY, USA, 3251–3257. DOI: https://doi.org/10.1145/3308558.3313710
    """
    def __init__(self, lam=100.):
        self.lam = lam
        self.model = None

    def train(self, train_data):
        """Training of the EASE model.

        Parameters
        ----------
        train_data : :class:`scipy.sparse.csr_matrix`
            The training data.
        """
        logger.info("EASE - start tarining (lam=%.4f)", self.lam)
        X = train_data.toarray()
        G = np.dot(X.T, X)
        logger.info("EASE - linear kernel computed")
        diag_idx = np.diag_indices(G.shape[0])
        G[diag_idx] += self.lam
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        B[diag_idx] = 0
        del P
        self.model = np.dot(X, B)
        logger.info("EASE - training complete")

    def predict(self, ids_te_users, test_tr, remove_train=True):
        r"""Prediction using the EASE model.

        For the EASE model the prediction list for a user *u* is done by computing

        :math:`S_{u}=\mathbf{X}_{u,:} \cdot \mathbf{B}`.

        However, in the **rectorch** implementation the prediction is simply a look up is the score
        matrix *S*.

        Parameters
        ----------
        ids_te_users : array_like
            List of the test user indexes.
        test_tr : :class:`scipy.sparse.csr_matrix`
            Training portion of the test users.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default True. Removing
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

    def save_model(self, filepath):
        state = {'lambda': self.lam,
                 'model': self.model
                }
        logger.info("Saving EASE model to %s...", filepath)
        np.save(filepath, state)
        logger.info("Model saved!")

    def load_model(self, filepath):
        assert os.path.isfile(filepath), "The model file %s does not exist." %filepath
        logger.info("Loading EASE model from %s...", filepath)
        state = np.load(filepath, allow_pickle=True)[()]
        self.lam = state["lambda"]
        self.model = state["model"]
        logger.info("Model loaded!")
        return state

    def __str__(self):
        s = "EASE(lambda=%.4f" % self.lam
        if self.model is not None:
            s += ", model size=(%d, %d))" %self.model.shape
        else:
            s += ") - not trained yet!"
        return s

    def __repr__(self):
        return str(self)


class CFGAN(RecSysModel):
    r"""A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.

    This recommender systems is based on GAN where the (conditioned) generator aims at generating
    the ratings of users while the discriminiator tries to discriminate between genuine users
    profile and generated ones.
    The two networks try to optimize the followng loss functions:

    - Discriminator (D):
        :math:`J^{D}=-\sum_{u}\left(\log D(\mathbf{r}_{u} | \
        \mathbf{c}_{u})+\log (1-D(\hat{\mathbf{r}}_{u}\
        \odot(\mathbf{e}_{u}+\mathbf{k}_{u}) | \mathbf{c}_{u}))\right)`
    - Generator (G):
        :math:`J^{G}=\sum_{u}\left(\log (1-D(\hat{\mathbf{r}}_{u}\
        \odot(\mathbf{e}_{u}+\mathbf{k}_{u}) | \mathbf{c}_{u}))+\
        \alpha \cdot \sum_{j}(r_{uj}-\hat{r}_{uj})^{2}\right)`

    where :math:`\mathbf{c}_u` is the condition vector, :math:`\mathbf{k}_u` is an n-dimensional
    indicator vector such that :math:`k_{uj} = 1` iff
    *j* is a negative sampled item, :math:`\hat{\mathbf{r}}_u` is a fake user profile, and
    :math:`\mathbf{e}_u` is the masking vector to remove the non-rated items.
    For more details please refer to the original paper [CFGAN]_.

    Parameters
    ----------
    generator : :class:`torch.nn.Module`
        The generator neural network. The expected architecture is
        :math:`[m, l_1, \dots, l_h, m]` where *m* is the number of items :math:`l_i, i \in [1,h]`
        is the number of nodes of the hidden layer *i*.
    discriminator : :class:`torch.nn.Module`
        The discriminator neural network. The expected architecture is
        :math:`[2m, l_1, \dots, l_h, 1]` where *m* is the number of items :math:`l_i, i \in [1,h]`
        is the number of nodes of the hidden layer *i*.
    alpha : :obj:`float` [optional]
        The ZR-coefficient, by default .1.
    s_pm : :obj:`float` [optional]
        The sampling parameter for the partial-masking approach, by default .7.
    s_zr : :obj:`float` [optional]
        The sampling parameter for the zero-reconstruction regularization, by default .5.
    learning_rate : :obj:`float` [optional]
        The optimization learning rate, by default 0.001.

    Attributes
    ----------
    generator : :class:`torch.nn.Module`
        See ``generator`` parameter.
    discriminator : :class:`torch.nn.Module`
        See ``discriminator`` parameter.
    alpha : :obj:`float`
        See ``alpha`` paramater.
    s_pm : :obj:`float`
        See ``s_pm`` paramater.
    s_zr : :obj:`float`
        See ``s_zr`` paramater.
    learning_rate : :obj:`float`
        See ``learning_rate`` paramater.
    opt_g : :class:`torch.optim.Adam`
        Optimizer used for performing the training of the generator.
    opt_d : :class:`torch.optim.Adam`
        Optimizer used for performing the training of the discriminator.

    References
    ----------
    .. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
       CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
       In Proceedings of the 27th ACM International Conference on Information and Knowledge
       Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
       DOI: https://doi.org/10.1145/3269206.3271743
    """
    def __init__(self,
                 generator,
                 discriminator,
                 alpha=.1,
                 s_pm=.7,
                 s_zr=.5,
                 learning_rate=0.001):
        self.generator = generator
        self.discriminator = discriminator

        #TODO: check this # pylint: disable=fixme
        if next(self.generator.parameters()).is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.s_pm = s_pm
        self.s_zr = s_zr
        self.loss = torch.nn.BCELoss()
        self.regularization_loss = torch.nn.MSELoss(reduction="sum")
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_items = self.generator.input_dim

        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)


    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=1000,
              g_steps=5,
              d_steps=5,
              verbose=1):
        r"""Training procedure of CFGAN.

        The training works in an alternate way between generator and discriminator.
        The number of training batches in each epochs are ``g_steps`` and ``d_steps``, respectively.

        Parameters
        ----------
        train_data : :class:`samplers.CFGAN_TrainingSampler`
            The sampler object that load the training set in mini-batches.
        valid_data : :class:`samplers.DataSampler` [optional]
            The sampler object that load the validation set in mini-batches, by default ``None``.
            If the model does not have any validation procedure set this parameter to ``None``.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default ``None``.
            If ``valid_data`` is not ``None`` then ``valid_metric`` must be not ``None``.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`evaluation.ValidFunc` [optional]
            The validation function, by default a standard validation procedure, i.e.,
            :func:`evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 1000.
        g_steps : :obj:`int` [optional]
            Number of steps for a generator epoch, by default 5.
        d_steps : :obj:`int` [optional]
            Number of steps for a discriminator epoch, by default 5.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum (that depends on the size of
            the training set) verbosity higher values will not have any effect.
        """
        self.discriminator.train()
        self.generator.train()

        start_time = time.time()
        log_delay = max(10, num_epochs // 10**verbose)
        loss_d, loss_g = 0, 0
        try:
            for epoch in range(1, num_epochs+1):
                for _ in range(1, g_steps+1):
                    loss_g += self.train_gen_batch(next(train_data).to(self.device))

                for _ in range(1, d_steps+1):
                    loss_d += self.train_disc_batch(next(train_data).to(self.device))

                if epoch % log_delay == 0:
                    loss_g /= (g_steps * log_delay)
                    loss_d /= (d_steps * log_delay)
                    elapsed = time.time() - start_time
                    logger.info('| epoch {} | ms/batch {:.2f} | loss G {:.6f} | loss D {:.6f} |'
                                .format(epoch, elapsed * 1000 / log_delay, loss_g, loss_d))
                    start_time = time.time()
                    loss_g, loss_d = 0, 0

                    if valid_data is not None:
                        assert valid_metric is not None, \
                                    "In case of validation 'valid_metric' must be provided"
                        valid_res = valid_func(self, valid_data, valid_metric)
                        mu_val = np.mean(valid_res)
                        std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                        logger.info('| epoch %d | %s %.3f (%.4f) |',
                                    epoch, valid_metric, mu_val, std_err_val)

        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')


    def train_gen_batch(self, batch):
        r"""Training on a single batch for the generator.

        Parameters
        ----------
        batch : :class:`torch.Tensor`
            The current batch.

        Returns
        -------
        :obj:`float`
            The loss incurred in the current batch by the generator.
        """
        batch = batch.to(self.device)
        real_label = torch.ones(batch.shape[0], 1).to(self.device)

        mask = batch.clone()
        size = int(self.s_pm * self.n_items)
        for u in range(batch.shape[0]):
            rand_it = np.random.choice(self.n_items, size, replace=False)
            mask[u, rand_it] = 1
        mask = mask.to(self.device)

        if self.alpha > 0:
            mask_zr = batch.clone()
            size = int(self.s_zr * self.n_items)
            for u in range(batch.shape[0]):
                rand_it = np.random.choice(self.n_items, size, replace=False)
                mask_zr[u, rand_it] = 1
            mask_zr = mask_zr.to(self.device)

        fake_data = self.generator(batch)
        g_reg_loss = 0
        if self.alpha > 0:
            g_reg_loss = self.regularization_loss(fake_data, mask_zr)

        fake_data = torch.mul(fake_data, mask)
        disc_on_fake = self.discriminator(fake_data, batch)
        g_loss = self.loss(disc_on_fake, real_label)
        g_loss = g_loss + self.alpha * g_reg_loss
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()

        return g_loss.item()


    def train_disc_batch(self, batch):
        r"""Training on a single batch for the discriminator.

        Parameters
        ----------
        batch : :class:`torch.Tensor`
            The current batch.

        Returns
        -------
        :obj:`float`
            The loss incurred in the batch by the discriminator.
        """
        batch = batch.to(self.device)
        real_label = torch.ones(batch.shape[0], 1).to(self.device)
        fake_label = torch.zeros(batch.shape[0], 1).to(self.device)

        mask = batch.clone()
        size = int(self.s_pm * self.n_items)
        for u in range(batch.shape[0]):
            rand_it = np.random.choice(self.n_items, size, replace=False)
            mask[u, rand_it] = 1
        mask = mask.to(self.device)

        disc_on_real = self.discriminator(batch, batch)
        d_loss_real = self.loss(disc_on_real, real_label)

        fake_data = self.generator(batch)
        fake_data = torch.mul(fake_data, mask)
        disc_on_fake = self.discriminator(fake_data, batch)
        d_loss_fake = self.loss(disc_on_fake, fake_label)

        d_loss = d_loss_real + d_loss_fake
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()

        return d_loss.item()

    def predict(self, x, remove_train=True):
        self.generator.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            pred = self.generator(x_tensor)
            if remove_train:
                pred[tuple(x_tensor.nonzero().t())] = -np.inf
        return (pred, )

    def __str__(self):
        s = self.__class__.__name__ + "(\n"
        for k, v in self.__dict__.items():
            sv = "\n".join(["  "+line for line in str(str(v)).split("\n")])[2:]
            s += "  %s = %s,\n" % (k, sv)
        s = s[:-2] + "\n)"
        return s

    def __repr__(self):
        return str(self)

    def save_model(self, filepath, cur_epoch):
        state = {'epoch': cur_epoch,
                 'state_dict_g': self.generator.state_dict(),
                 'state_dict_d': self.discriminator.state_dict(),
                 'optimizer_g': self.opt_g.state_dict(),
                 'optimizer_d': self.opt_g.state_dict()
                }
        logger.info("Saving CFGAN model to %s...", filepath)
        torch.save(state, filepath)
        logger.info("Model saved!")

    def load_model(self, filepath):
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath)
        self.generator.load_state_dict(checkpoint['state_dict_g'])
        self.discriminator.load_state_dict(checkpoint['state_dict_d'])
        self.opt_g.load_state_dict(checkpoint['optimizer_g'])
        self.opt_d.load_state_dict(checkpoint['optimizer_d'])
        logger.info("Model checkpoint loaded!")
        return checkpoint


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
        Whether to keep the non-negativity constraint, by default ``True``.
    l1_penalty : :obj:`bool` [optional]
        Whether to keep the L1 penalty, by default ``True``. When ``l1_penalty = False`` then
        is like to set :math:`\lambda_1 = 0`.
    item_bias : :obj:`bool` [optional]
        Whether to model the item biases, by default ``False``. When ``item_bias = True`` then
        the scoring function for the user-item pair *(u,i)* becomes:
        :math:`S_{ui}=(\mathbf{X}_{u,:} - \mathbf{b})\cdot \mathbf{B}_{:, i} + \mathbf{b}_i`.

    Attributes
    ----------
    See the parameters' section.

    References
    ----------
    .. [ADMMS] Harald Steck, Maria Dimakopoulou, Nickolai Riabov, and Tony Jebara. 2020.
       ADMM SLIM: Sparse Recommendations for Many Users. In Proceedings of the 13th International
       Conference on Web Search and Data Mining (WSDM ’20). Association for Computing Machinery,
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


    def train(self, train_data, num_iter=50, verbose=1):
        r"""Training of ADMM SLIM.

        The training procedure of ADMM SLIM highly depends on the setting of the
        hyper-parameters. By setting them in specific ways it is possible to define different
        variants of the algorithm. That are:

        1. (Vanilla) ADMM SLIM - :math:`\lambda_1, \lambda_2, \rho>0`, :attr:`item_bias` =
        ``False``, and both :attr:`nn_constr` and :attr:`l1_penalty` set to ``True``;

        2. ADMM SLIM w/o non-negativity constraint over C - :attr:`nn_constr` = ``False`` and
        :attr:`l1_penalty` set to ``True``;

        3. ADMM SLIM w/o the L1 penalty - :attr:`l1_penalty` = ``False`` and
        :attr:`nn_constr` set to ``True``;

        4. ADMM SLIM w/o L1 penalty and non-negativity constraint: :attr:`nn_constr` =
        :attr:`l1_penalty` = ``False``.

        All these variants can also be combined with the inclusion of the item biases by setting
        :attr:`item_bias` to ``True``.

        Parameters
        ----------
        train_data : :class:`scipy.sparse.csr_matrix`
            The training data.
        num_iter : :obj:`int` [optional]
            Maximum number of training iterations, by default 50. This argument has no effect
            if both :attr:`nn_constr` and :attr:`l1_penalty` are set to ``False``.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum (that depends on the size of
            the training set) verbosity higher values will not have any effect.
        """
        def _soft_threshold(a, k):
            return np.maximum(0., a - k) - np.maximum(0., -a - k)

        X = train_data.toarray()
        if self.item_bias:
            b = X.sum(axis=0)
            X = X - np.outer(np.ones(X.shape[0]), b)

        XtX = X.T.dot(X)
        logger.info("ADMM_Slim - linear kernel computed")
        diag_indices = np.diag_indices(XtX.shape[0])
        XtX[diag_indices] += self.lambda2 + self.rho
        P = np.linalg.inv(XtX)
        logger.info("ADMM_Slim - inverse of XtX computed")

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
                    logger.info("| iteration %d/%d |", j+1, num_iter)

        self.model = np.dot(X, C)
        if self.item_bias:
            self.model += b

    def predict(self, ids_te_users, test_tr, remove_train=True):
        pred = self.model[ids_te_users, :]
        if remove_train:
            pred[test_tr.nonzero()] = -np.inf
        return (pred, )

    def save_model(self, filepath):
        state = {'lambda1': self.lambda1,
                 'lambda2': self.lambda2,
                 'rho' : self.rho,
                 'model': self.model,
                 'nn_constr' : self.nn_constr,
                 'l1_penalty' : self.l1_penalty,
                 'item_bias' : self.item_bias
                }
        logger.info("Saving ADMM_Slim model to %s...", filepath)
        np.save(filepath, state)
        logger.info("Model saved!")

    def load_model(self, filepath):
        assert os.path.isfile(filepath), "The model file %s does not exist." %filepath
        logger.info("Loading ADMM_Slim model from %s...", filepath)
        state = np.load(filepath, allow_pickle=True)[()]
        self.lambda1 = state["lambda1"]
        self.lambda2 = state["lambda2"]
        self.rho = state["rho"]
        self.nn_constr = state["nn_constr"]
        self.l1_penalty = state["l1_penalty"]
        self.item_bias = state["item_bias"]
        self.model = state["model"]
        logger.info("Model loaded!")
        return state

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

    def __repr__(self):
        return str(self)


#TODO documentation
class SVAE(MultiVAE):
    r"""Sequential Variational Autoencoders for Collaborative Filtering.

    **UNDOCUMENTED** [SVAE]_

    Parameters
    ----------
    mvae_net : :class:`torch.nn.Module`
        The variational autoencoder neural network.
    beta : :obj:`float` [optional]
        The :math:`\beta` hyper-parameter of Multi-VAE. When ``anneal_steps > 0`` then this
        value is the value to anneal starting from 0, otherwise the ``beta`` will be fixed to
        the given value for the duration of the training. By default 1.
    anneal_steps : :obj:`int` [optional]
        Number of annealing step for reaching the target value ``beta``, by default 0.
        0 means that no annealing will be performed and the regularization parameter will be
        fixed to ``beta``.
    learning_rate : :obj:`float` [optional]
        The learning rate for the optimizer, by default 1e-3.

    References
    ----------
    .. [SVAE] Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi. 2019.
        Sequential Variational Autoencoders for Collaborative Filtering. In Proceedings of the
        Twelfth ACM International Conference on Web Search and Data Mining (WSDM '19).
        Association for Computing Machinery, New York, NY, USA, 600–608.
        DOI: https://doi.org/10.1145/3289600.3291007
    """
    def __init__(self,
                 svae_net,
                 beta=1.,
                 anneal_steps=0,
                 learning_rate=1e-3):
        super(SVAE, self).__init__(svae_net,
                                   beta=beta,
                                   anneal_steps=anneal_steps,
                                   learning_rate=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=learning_rate,
                                    weight_decay=5e-3)

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        likelihood_n = -torch.sum(torch.sum(F.log_softmax(recon_x, -1) * x.view(recon_x.shape), -1))
        likelihood_d = float(torch.sum(x[0, :recon_x.shape[2]]))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return likelihood_n / likelihood_d + beta * KLD

    def predict(self, x, remove_train=True):
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[0, -1, x_tensor] = -np.inf
            return recon_x[:, -1, :], mu, logvar
