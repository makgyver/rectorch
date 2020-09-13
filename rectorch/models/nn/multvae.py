r"""Variational Autoencoder for collaborative Filtering.

MultVAE (dubbed Mult-VAE in [VAE]_) is a vanilla VAE in which the input data distribution is
assumed to be multinomial and the objective function is an under-regularized version
of the standard VAE loss function. Specifically, the Kullbach-Liebler divergence term is
weighted by an hyper-parameter (:math:`\beta`) that shows to improve the recommendations'
quality when < 1. So, the regularization term is weighted less giving to the model more freedom
in representing the input in the latent space. More details about this loss are given in
:meth:`rectorch.models.nn.multvae.MultVAE_trainer.loss_function`.

References
----------
.. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
   Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
   World Wide Web Conference (WWW '18). International World Wide Web Conferences Steering
   Committee, Republic and Canton of Geneva, CHE, 689–698.
   DOI: https://doi.org/10.1145/3178876.3186150
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rectorch import env, set_seed
from rectorch.models.nn import NeuralModel, VAE_net, VAE_trainer
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

__all__ = ["MultVAE_net", "MultVAE_trainer", "MultVAE"]


class MultVAE_net(VAE_net):
    r'''Variational Autoencoder network for collaborative filtering.

    The network structure follows the definition as in [VAE]_. Hidden layers are fully
    connected and *tanh* activated. The output layer of both the encoder and the decoder
    are linearly activated.

    Parameters
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        Dimension of the hidden layers of the decoder network.
    enc_dims : :obj:`list` or array_like of :obj:`int` :obj:`None`
        Dimension of the hidden layers of the encoder network, by default :obj:`None`. When
        :obj:`None` the encoder is assumed of having the reversed structure of the decoder.
    dropout : :obj:`float` [optional]
        The dropout rate for the dropout layer that is applied to the input during the
        forward operation. By default 0.5.

    Attributes
    ----------
    dropout_rate : :obj:`float`
        The dropout rate for the dropout layer that is applied to the input during the
        forward operation.
    other attributes : see the base class :class:`rectorch.models.nn.VAE_net`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    '''

    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultVAE_net, self).__init__(dec_dims, enc_dims)
        self.dropout_rate = dropout
        self._dropout_layer = nn.Dropout(dropout)

    def encode(self, x):
        h = F.normalize(x)
        if self.training:
            h = self._dropout_layer(h)
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def _reparameterize(self, mu, logvar):
        if self.training:
            return super()._reparameterize(mu, logvar)
        else:
            return mu

    def decode(self, z):
        h = z
        for _, layer in enumerate(self.dec_layers[:-1]):
            h = torch.tanh(layer(h))
        return self.dec_layers[-1](h)

    def get_state(self):
        state = super().get_state()
        state["name"] = self.__class__.__name__
        state["params"]["dropout"] = self.dropout_rate
        return state


class MultVAE_trainer(VAE_trainer):
    r"""Trainer class for Mult-VAE model.

    Parameters
    ----------
    mvae_net : :class:`rectorch.models.nn.NeuralNet`
        The variational autoencoder neural network.
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

    Attributes
    ----------
    anneal_steps : :obj:`int`
        Number of annealing step for reaching the target value ``beta``.
        0 means that no annealing will be performed and the regularization parameter will be
        fixed to ``beta``.
    annealing : :obj:`bool`
        Whether the annealing is active or not. It is implicitely set to :obj:`True` if
        ``anneal_steps > 0``, otherwise is set to :obj:`False`.
    gradient_updates : :obj:`int`
        Number of gradient updates since the beginning of the training. Once
        ``gradient_updates >= anneal_steps``, then the annealing is complete and the used
        :math:`\beta` in the loss function is ``beta``.
    beta : :obj:`float`
        See ``beta`` parameter.
    optimizer : :class:`torch.optim.Optimizer`
        The optimizer is initialized according to the given configurations in ``opt_conf``.
    other attributes : see the base class :class:`rectorch.models.nn.VAE_trainer`.
    """
    def __init__(self,
                 mvae_net,
                 beta=1.,
                 anneal_steps=0,
                 device=None,
                 opt_conf=None):
        super(MultVAE_trainer, self).__init__(mvae_net, device, opt_conf)
        self.anneal_steps = anneal_steps
        self.annealing = anneal_steps > 0
        self.gradient_updates = 0.
        self.beta = beta

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        r"""Beta-VAE loss function.

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

    def get_state(self):
        state = super().get_state()
        state["params"]["beta"] = self.beta
        state["params"]["anneal_steps"] = self.anneal_steps
        state["gradient_updates"] = self.gradient_updates
        return state

    @classmethod
    def from_state(cls, state):
        trainer = super().from_state(state)
        trainer.gradient_updates = state["gradient_updates"]
        return trainer


class MultVAE(NeuralModel):
    r"""Variational Autoencoder for collaborative filtering.

    Parameters
    ----------
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
    trainer : :class:`rectorch.models.nn.multvae.MultVAE_trainer` [optional]
        The trainer object for performing the learning, by default :obj:`None`. If not :obj:`None`
        it is the only parameters that is taken into account for creating the model.

    Attributes
    ----------
    network : :class:`rectorch.models.nn.multvae.MultVAE_net`
        The neural network architecture.
    trainer : :class:`rectorch.models.nn.multvae.MultVAE_trainer`
        The trainer class for performing the learning.
    device : :obj:`str`
        The device where the model must be loaded.
    """
    def __init__(self,
                 dec_dims=None,
                 enc_dims=None,
                 dropout=0.5,
                 beta=1.,
                 anneal_steps=0,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(MultVAE, self).__init__(trainer.network, trainer, trainer.device)
        else:
            device = torch.device(device) if device is not None else env.device
            network = MultVAE_net(dec_dims=dec_dims,
                                  enc_dims=enc_dims,
                                  dropout=dropout)
            trainer = MultVAE_trainer(network,
                                      beta=beta,
                                      anneal_steps=anneal_steps,
                                      device=device,
                                      opt_conf=opt_conf)
            super(MultVAE, self).__init__(network, trainer, device)

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
        r"""Mult-VAE training procedure.

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
        set_seed(seed)
        if isinstance(dataset, Sampler):
            data_sampler = dataset
        else:
            data_sampler = DataSampler(dataset,
                                       mode="train",
                                       batch_size=batch_size,
                                       shuffle=shuffle)
        try:
            best_perf = -1. #Assume the higher the better >= 0
            for epoch in range(1, num_epochs + 1):
                data_sampler.train()
                self.trainer.train_epoch(epoch, data_sampler, verbose)
                if valid_metric is not None:
                    data_sampler.valid()
                    valid_res = valid_func(self, data_sampler, valid_metric)
                    mu_val = np.mean(valid_res)
                    env.logger.info('| epoch %d | %s %.3f (%.4f) |',
                                    epoch,
                                    valid_metric,
                                    mu_val,
                                    np.std(valid_res) / np.sqrt(len(valid_res)))
                    if best_perf < mu_val:
                        if best_path:
                            self.save_model(best_path)
                        best_perf = mu_val
        except KeyboardInterrupt:
            env.logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def predict(self, x, remove_train=True):
        r"""Perform the prediction using a trained Variational Autoencoder.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input batch tensor for which the prediction must be computed.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default :obj:`True`. Removing
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
                recon_x[torch.nonzero(x_tensor, as_tuple=True)] = -np.inf
            return recon_x, mu, logvar

    @classmethod
    def from_state(cls, state):
        trainer = MultVAE_trainer.from_state(state)
        return MultVAE(trainer=trainer)

    @classmethod
    def load_model(cls, filepath, device=None):
        state = torch.load(filepath)
        if device:
            state["device"] = device
        return cls.from_state(state)
