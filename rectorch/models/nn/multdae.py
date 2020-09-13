r"""Denoising Autoencoder with multinomial likelihood for collaborative filtering.

This model has been proposed in [VAE]_ as a baseline method to compare with Mult-DAE.
The model represent a standard denoising autoencoder in which the data is assumed of being
multinomial distributed.

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
from torch.nn.init import normal_ as normal_init
from torch.nn.init import xavier_uniform_ as xavier_init
from rectorch import env, set_seed
from rectorch.models.nn import NeuralModel, AE_net, AE_trainer
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

__all__ = ["MultDAE_net", "MultDAE_trainer", "MultDAE"]


class MultDAE_net(AE_net):
    r"""Denoising Autoencoder network for collaborative filtering.

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
    other attributes : see the base class :class:`rectorch.models.nn.AE_net`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    """
    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultDAE_net, self).__init__(dec_dims, enc_dims)
        self.dropout_rate = dropout
        self._dropout_layer = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.enc_dims[:-1], self.enc_dims[1:])])

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
        self.init_weights()

    def encode(self, x):
        h = F.normalize(x)
        if self.training:
            h = self._dropout_layer(h)
        for _, layer in enumerate(self.enc_layers):
            h = torch.tanh(layer(h))
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.dec_layers):
            h = layer(h)
            if i != len(self.dec_layers) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        r"""Initialize the weights of the network.

        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        for layer in self.enc_layers:
            xavier_init(layer.weight)
            normal_init(layer.bias)

        for layer in self.dec_layers:
            xavier_init(layer.weight)
            normal_init(layer.bias)

    def get_state(self):
        state = super().get_state()
        state["name"] = self.__class__.__name__
        state["params"]["dropout"] = self.dropout_rate
        return state


class MultDAE_trainer(AE_trainer):
    r"""Denoising Autoencoder with multinomial likelihood for collaborative filtering.

    This model has been proposed in [VAE]_ as a baseline method to compare with Mult-DAE.
    The model represent a standard denoising autoencoder in which the data is assumed of being
    multinomial distributed.

    Parameters
    ----------
    mdae_net : :class:`torch.nn.Module`
        The autoencoder neural network.
    lam : :obj:`float` [optional]
        The regularization hyper-parameter :math:`\lambda` as defined in [VAE]_, by default 0.2.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW '18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    """
    def __init__(self,
                 mdae_net,
                 lam=0.2,
                 device=None,
                 opt_conf=None):
        super(MultDAE_trainer, self).__init__(mdae_net, device, opt_conf)
        self.lam = lam

    def loss_function(self, recon_x, x):
        r"""Multinomial likelihood denoising autoencoder loss.

        Since the model assume a multinomial distribution over the input, then the reconstruction
        loss must be modified with respect to a vanilla AE. In particular,
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

    def get_state(self):
        state = super().get_state()
        state["params"]["lam"] = self.lam
        return state


class MultDAE(NeuralModel):
    r"""Mult-DAE model as in Variational Autoencoder for collaborative filtering.

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
    lam : :obj:`float` [optional]
        The regularization hyper-parameter :math:`\lambda` as defined in [VAE]_, by default 0.2.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.
    device : :obj:`str` [optional]
        The device where the model must be loaded, by default :obj:`None`. If :obj:`None`, the
        default device (see `rectorch.env.device`) is used.
    trainer : :class:`rectorch.models.nn.multdae.MultDAE_trainer` [optional]
        The trainer object for performing the learning, by default :obj:`None`. If not :obj:`None`
        it is the only parameters that is taken into account for creating the model.

    Attributes
    ----------
    network : :class:`rectorch.models.nn.multdae.MultDAE_net`
        The neural network architecture.
    trainer : :class:`rectorch.models.nn.multdae.MultDAE_trainer`
        The trainer class for performing the learning.
    device : :obj:`str`
        The device where the model must be loaded.
    
    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW '18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    """
    def __init__(self,
                 dec_dims=None,
                 enc_dims=None,
                 dropout=0.5,
                 lam=0.2,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(MultDAE, self).__init__(trainer.network, trainer, trainer.device)
        else:
            device = torch.device(device) if device is not None else env.device
            network = MultDAE_net(dec_dims=dec_dims,
                                  enc_dims=enc_dims,
                                  dropout=dropout)
            trainer = MultDAE_trainer(network,
                                      lam=lam,
                                      device=device,
                                      opt_conf=opt_conf)
            super(MultDAE, self).__init__(network, trainer, device)

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
        r"""Mult-DAE training procedure.

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
        r"""Perform the prediction using a trained Mult-DAE.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input for which the prediction has to be computed.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default :obj:`True`. Removing
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
                recon_x[torch.nonzero(x_tensor, as_tuple=True)] = -np.inf
            return (recon_x, )

    @classmethod
    def from_state(cls, state):
        trainer = MultDAE_trainer.from_state(state)
        return MultDAE(trainer=trainer)

    @classmethod
    def load_model(cls, filepath, device=None):
        state = torch.load(filepath)
        if device:
            state["device"] = device
        return cls.from_state(state)
