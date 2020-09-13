r"""RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback.

RecVAE [RecVAE]_ introduces several novel ideas to improve Mult-VAE [MultVAE]_, including a
novel composite prior distribution for the latent codes, a new approach to setting the
:math:`\beta` hyperparameter for the :math:`\beta`-VAE framework, and a new approach to
training based on alternating updates.

References
----------
.. [RecVAE] Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, and Sergey
   I. Nikolenko. 2020. RecVAE: A New Variational Autoencoder for Top-N Recommendations
   with Implicit Feedback. In Proceedings of the 13th International Conference on Web
   Search and Data Mining (WSDM '20). Association for Computing Machinery, New York, NY, USA,
   528–536. DOI: https://doi.org/10.1145/3336191.3371831
.. [MultVAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
   Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
   World Wide Web Conference (WWW '18). International World Wide Web Conferences Steering
   Committee, Republic and Canton of Geneva, CHE, 689–698.
   DOI: https://doi.org/10.1145/3178876.3186150
"""
import time
from importlib import import_module
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rectorch import env, set_seed
from rectorch.models.nn import NeuralModel, NeuralNet, TorchNNTrainer
from rectorch.evaluation import evaluate
from rectorch.validation import ValidFunc
from rectorch.samplers import DataSampler, Sampler
from rectorch.utils import log_norm_pdf, swish, init_optimizer

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["RecVAE_CompositePrior_net", "RecVAE_Encoder_net", "RecVAE_net", "RecVAE_trainer",\
    "RecVAE"]


class RecVAE_CompositePrior_net(NeuralNet):
    r"""Composite prior network of the RecVAE model.

    Parameters
    ----------
    input_dim : :obj:`int`
        The dimension of the input, i.e., the number of items.
    hidden_dim : :obj:`int`
        The dimension of the hidden layers.
    latent_dim : :obj:`int`
        The dimension of the latent space.
    mixture_weights : :obj:`list` of :obj:`float`
        Weights in the mixture of the priors (:math:`\alpha` in [RecVAE]_).

    References
    ----------
    .. [RecVAE] Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, and Sergey
       I. Nikolenko. 2020. RecVAE: A New Variational Autoencoder for Top-N Recommendations
       with Implicit Feedback. In Proceedings of the 13th International Conference on Web
       Search and Data Mining (WSDM '20). Association for Computing Machinery, New York, NY, USA,
       528–536. DOI: https://doi.org/10.1145/3336191.3371831
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, mixture_weights):
        super(RecVAE_CompositePrior_net, self).__init__()
        self.mixture_weights = mixture_weights

        param_no_grad = lambda dim: nn.Parameter(torch.Tensor(1, dim), requires_grad=False)
        self.mu_prior = param_no_grad(latent_dim)
        self.logvar_prior = param_no_grad(latent_dim)
        self.logvar_uniform_prior = param_no_grad(latent_dim)
        self.encoder_old = RecVAE_Encoder_net(input_dim, hidden_dim, latent_dim)
        self.encoder_old.requires_grad_(False)
        self.init_weights()

    def init_weights(self):
        self.logvar_uniform_prior.data.fill_(10)
        self.logvar_prior.data.fill_(0)
        self.mu_prior.data.fill_(0)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        density_per_gaussian = torch.stack(gaussians, dim=-1)
        return torch.logsumexp(density_per_gaussian, dim=-1)

    def get_state(self):
        pass


class RecVAE_Encoder_net(NeuralNet):
    r"""Encoder network of the RecVAE model.

    Parameters
    ----------
    input_dim : :obj:`int`
        The dimension of the input, i.e., the number of items.
    hidden_dim : :obj:`int`
        The dimension of the hidden layers.
    latent_dim : :obj:`int`
        The dimension of the latent space.
    num_hidden : :obj:`int` [optional]
        Number of hidden layers, by default 4.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hidden=4):
        super(RecVAE_Encoder_net, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden)]
        self.layers = nn.ModuleList(layers)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.init_weights()

    def forward(self, x, dropout_rate):
        xnorm = x.pow(2).sum(dim=-1).sqrt()
        x = x / xnorm[:, None]
        x = F.dropout(x, p=dropout_rate, training=self.training)

        hprev = [self.ln(swish(self.layers[0](x)))]
        for layer in self.layers[1:]:
            hnext = self.ln(swish(layer(hprev[-1])) + sum(hprev))
            hprev.append(hnext)

        return self.fc_mu(hprev[-1]), self.fc_logvar(hprev[-1])

    def init_weights(self):
        pass

    def get_state(self):
        pass


class RecVAE_net(NeuralNet):
    r"""RecVAE neural network model.

    Parameters
    ----------
    input_dim : :obj:`int`
        The dimension of the input, i.e., the number of items.
    hidden_dim : :obj:`int`
        The dimension of the hidden layers.
    latent_dim : :obj:`int`
        The dimension of the latent space.
    enc_num_hidden : :obj:`int` [optional]
        Number of hidden layers in the encoder network, by default 4.
    prior_mixture_weights : :obj:`list` of :obj:`float` [optional]
        Weights in the mixture of the priors (:math:`\alpha` in [RecVAE]_),
        by default [3/20, 3/4, 1/10].

    References
    ----------
    .. [RecVAE] Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, and Sergey
       I. Nikolenko. 2020. RecVAE: A New Variational Autoencoder for Top-N Recommendations
       with Implicit Feedback. In Proceedings of the 13th International Conference on Web
       Search and Data Mining (WSDM '20). Association for Computing Machinery, New York, NY, USA,
       528–536. DOI:https://doi.org/10.1145/3336191.3371831
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 enc_num_hidden=4,
                 prior_mixture_weights=None):#[3/20, 3/4, 1/10]):
        super(RecVAE_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.enc_num_hidden = enc_num_hidden
        if prior_mixture_weights is None:
            self.prior_mixture_weights = [1/3, 1/3, 1/3]
        else:
            self.prior_mixture_weights = prior_mixture_weights
        self.encoder = RecVAE_Encoder_net(input_dim, hidden_dim, latent_dim, enc_num_hidden)
        self.prior = RecVAE_CompositePrior_net(input_dim,
                                               hidden_dim,
                                               latent_dim,
                                               self.prior_mixture_weights)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.init_weights()

    def init_weights(self):
        pass

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, user_ratings, dropout_rate=0.5):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self._reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        return x_pred, z, mu, logvar

    def update_prior(self):
        r"""Update the weights in the prior network.
        """
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "input_dim" : self.input_dim,
                "hidden_dim" : self.hidden_dim,
                "latent_dim" : self.latent_dim,
                "enc_num_hidden" : self.enc_num_hidden,
                "prior_mixture_weights" : self.prior_mixture_weights
            }
        }
        return state


class RecVAE_trainer(TorchNNTrainer):
    """Trainer class for the RecVAE model.

    Parameters
    ----------
    recvae_net : :class:`rectorch.nets.RecVAE_net`
        The RecVAE neural network architecture.
    beta : :obj:`float` [optional]
        KL-divergence scaling factor, by default 0.
    gamma : :obj:`float` [optional]
        KL-divergence scaling factor that substitute ``beta`` when set to a positive value,
        by default 1. In this case, gamma is multiplied by the L1 norm of the input batch.
    device : :obj:`str` [optional]
        The device where the model must be loaded, by default :obj:`None`. If :obj:`None`, the
        default device (see `rectorch.env.device`) is used.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.
    """
    def __init__(self,
                 recvae_net,
                 beta=0.,
                 gamma=1.,
                 device=None,
                 opt_conf=None):
        super(RecVAE_trainer, self).__init__(recvae_net, device, opt_conf)
        self.beta = beta
        self.gamma = gamma
        self.optimizer = [init_optimizer(self.network.encoder.parameters(), opt_conf),
                          init_optimizer(self.network.decoder.parameters(), opt_conf)]
        self.opt_enc = self.optimizer[0]
        self.opt_dec = self.optimizer[1]

    def loss_function(self, recon_x, x, z, mu, logvar):
        r"""RecVAE loss function.

        Parameters
        ----------
        recon_x : :class:`torch.Tensor`
            The reconstructed input, i.e., the output of the variational autoencoder. It is meant
            to be the reconstruction over a batch.
        x : :class:`torch.Tensor`
            The input, and hence the target tensor. It is meant to be a batch size input.
        z : :class:`torch.Tensor`
            The output of the encoder network.
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
        kl_weight = self.gamma * x.sum(dim=-1) if self.gamma > 0 else self.beta
        mll = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1).mean()
        kld = (log_norm_pdf(z, mu, logvar) - self.network.prior(x, z))
        kld = kld.sum(dim=-1).mul(kl_weight).mean()
        return -(mll - kld)

    def train_batch(self, tr_batch, optimizer, te_batch=None, dropout=0.):
        r"""Training of a single batch.

        Parameters
        ----------
        tr_batch : :class:`torch.Tensor`
            Traning part of the current batch.
        optimizer : :class:`torch.optim.Optimizer`
            The optimizer to use for this batch.
        te_batch : :class:`torch.Tensor` or :obj:`None` [optional]
            Test part of the current batch, if any, otherwise :obj:`None`, by default :obj:`None`.
        dropout : :obj:`float` [optional]
            The dropout rate for the encoder network (if necessary), default 0.

        Returns
        -------
        :obj:`float`
            The loss incurred in the batch.
        """
        data_tensor = tr_batch.view(tr_batch.shape[0], -1).to(self.device)
        if te_batch is None:
            gt_tensor = data_tensor
        else:
            gt_tensor = te_batch.view(te_batch.shape[0], -1).to(self.device)

        optimizer.zero_grad()
        recon_batch, z, mu, var = self.network(data_tensor, dropout)
        loss = self.loss_function(recon_batch, gt_tensor, z, mu, var)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_inner_epoch(self, epoch, data_sampler, net_part, verbose=1):
        r"""Training of a single epoch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        data_sampler : :class:`rectorch.samplers.Sampler`
            The sampler object that load the training set in mini-batches.
        net_part : :obj:`str` in the set {``"enc"``, ``"dec"``}
            The part of the network to train. ``"enc"`` means encoder and ``"dec"`` decoder.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        """
        optimizer = self.opt_enc if net_part == "enc" else self.opt_dec
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(data_sampler) // 10**verbose)
        dropout = .5 if net_part == "enc" else 0
        for batch_idx, (data, gt) in enumerate(data_sampler):

            partial_loss += self.train_batch(data, optimizer, gt, dropout)
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                env.logger.info('| %s - epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                                net_part, epoch, (batch_idx+1), len(data_sampler),
                                elapsed * 1000 / log_delay,
                                partial_loss / log_delay)
                train_loss += partial_loss
                partial_loss = 0.0
                start_time = time.time()
        total_loss = (train_loss + partial_loss) / len(data_sampler)
        time_diff = time.time() - epoch_start_time
        env.logger.info("| %s - epoch %d | loss %.4f | total time: %.2fs |",
                        net_part, epoch, total_loss, time_diff)
        return total_loss

    def train_epoch(self, epoch, data_sampler, enc_epochs, dec_epochs, verbose=1):
        r"""Train an epoch for the whole network (both encoder and decoder).

        Parameters
        ----------
        epoch : :obj:`int`
            Number of (meta) epochs.
        data_sampler : :class:`rectorch.samplers.DataSampler`
            The training sampler.
        enc_epochs : :obj:`int`
            Number of training epochs for each meta-epoch for the encoder.
        dec_epochs : :obj:`int`
            Number of training epochs for each meta-epoch for the decoder.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        """
        self.network.train()
        total_loss = 0
        start_time = time.time()

        env.logger.info("| meta-epoch %d start |", epoch)
        for ee in range(enc_epochs):
            total_loss += self.train_inner_epoch(ee + 1, data_sampler, "enc", verbose)

        self.network.update_prior()

        for de in range(dec_epochs):
            total_loss += self.train_inner_epoch(de + 1, data_sampler, "dec", verbose)

        total_loss /= enc_epochs + dec_epochs
        time_diff = time.time() - start_time
        env.logger.info("| meta-epoch %d | loss %.4f | total time: %.2fs |",
                        epoch, total_loss, time_diff)
        self.current_epoch += 1

    def get_state(self):
        state = {
            'epoch': self.current_epoch,
            'network': self.network.get_state(),
            'optimizer_e': self.opt_enc.state_dict(),
            'optimizer_d': self.opt_dec.state_dict(),
            'params' : {
                'opt_conf' : self.opt_conf,
                'gamma' : self.gamma,
                'beta' : self.beta
            }
        }
        return state

    @classmethod
    def from_state(cls, state):
        net_class = getattr(import_module(cls.__module__), state["network"]["name"])
        trainer_class = getattr(import_module(cls.__module__), cls.__name__)
        net = net_class(**state['network']['params'])
        net.load_state_dict(state['network']['state'])
        trainer = trainer_class(net, **state['params'])
        trainer.opt_dec.load_state_dict(state['optimizer_d'])
        trainer.opt_enc.load_state_dict(state['optimizer_e'])
        trainer.current_epoch = state['epoch']
        return trainer


class RecVAE(NeuralModel):
    r"""RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback.

    Parameters
    ----------
    input_dim : :obj:`int`
        The dimension of the input, i.e., the number of items.
    hidden_dim : :obj:`int`
        The dimension of the hidden layers.
    latent_dim : :obj:`int`
        The dimension of the latent space.
    enc_num_hidden : :obj:`int` [optional]
        Number of hidden layers in the encoder network, by default 4.
    prior_mixture_weights : :obj:`list` of :obj:`float` [optional]
        Weights in the mixture of the priors (:math:`\alpha` in [RecVAE]_),
        by default [3/20, 3/4, 1/10].
    beta : :obj:`float` [optional]
        KL-divergence scaling factor, by default 0.
    gamma : :obj:`float` [optional]
        KL-divergence scaling factor that substitute ``beta`` when set to a positive value,
        by default 1. In this case, gamma is multiplied by the L1 norm of the input batch.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.
    device : :obj:`str` [optional]
        The device where the model must be loaded, by default :obj:`None`. If :obj:`None`, the
        default device (see `rectorch.env.device`) is used.
    trainer : :class:`rectorch.models.nn.multvae.RecVAE_trainer` [optional]
        The trainer object for performing the learning, by default :obj:`None`. If not :obj:`None`
        it is the only parameters that is taken into account for creating the model.

    Attributes
    ----------
    network : :class:`rectorch.models.nn.multvae.RecVAE_net`
        The neural network architecture.
    trainer : :class:`rectorch.models.nn.multvae.RecVAE_trainer`
        The trainer class for performing the learning.
    device : :obj:`str`
        The device where the model must be loaded.

    References
    ----------
    .. [RecVAE] Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, and Sergey
       I. Nikolenko. 2020. RecVAE: A New Variational Autoencoder for Top-N Recommendations
       with Implicit Feedback. In Proceedings of the 13th International Conference on Web
       Search and Data Mining (WSDM '20). Association for Computing Machinery, New York, NY, USA,
       528–536. DOI:https://doi.org/10.1145/3336191.3371831
    """
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 latent_dim=None,
                 enc_num_hidden=4,
                 prior_mixture_weights=None,
                 beta=0.,
                 gamma=1.,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(RecVAE, self).__init__(trainer.network, trainer, trainer.device)
        else:
            device = torch.device(device) if device is not None else env.device
            network = RecVAE_net(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 latent_dim=latent_dim,
                                 enc_num_hidden=enc_num_hidden,
                                 prior_mixture_weights=prior_mixture_weights)
            trainer = RecVAE_trainer(network,
                                     beta=beta,
                                     gamma=gamma,
                                     device=device,
                                     opt_conf=opt_conf)
            super(RecVAE, self).__init__(network, trainer, device)

    def train(self,
              dataset,
              batch_size=1,
              shuffle=True,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=200,
              enc_epochs=3,
              dec_epochs=1,
              best_path=None,
              verbose=1,
              seed=None):
        r"""Training procedure for RecVAE.

        The training of RecVAE works in an alternated fashion. At each (meta) epoch the encoder is
        trained for a fixed number of epochs (``enc_epochs``), then the decoder is trained for a
        number of epochs (``dec_epochs``).

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
        enc_epochs : :obj:`int` [optional]
            Number of training epochs for each meta-epoch for the encoder, by default 3.
        dec_epochs : :obj:`int` [optional]
            Number of training epochs for each meta-epoch for the decoder, by default 1.
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
            best_perf = -1
            for epoch in range(1, num_epochs + 1):
                data_sampler.train()
                self.trainer.train_epoch(epoch, data_sampler, enc_epochs, dec_epochs, verbose)
                if valid_metric is not None:
                    data_sampler.valid()
                    valid_res = valid_func(self, data_sampler, valid_metric)
                    mu_val = np.mean(valid_res)
                    std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                    env.logger.info('| epoch %d | %s %.3f (%.4f) |',
                                    epoch, valid_metric, mu_val, std_err_val)

                    if best_perf < mu_val:
                        if best_path:
                            self.save_model(best_path)
                        best_perf = mu_val
        except KeyboardInterrupt:
            env.logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def predict(self, x, remove_train=True):
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, _, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[torch.nonzero(x_tensor, as_tuple=True)] = -np.inf
            return recon_x, mu, logvar

    @classmethod
    def from_state(cls, state):
        trainer = RecVAE_trainer.from_state(state)
        return RecVAE(trainer=trainer)

    @classmethod
    def load_model(cls, filepath, device=None):
        state = torch.load(filepath)
        if device:
            state["device"] = device
        return cls.from_state(state)
