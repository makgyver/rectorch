"""This module includes the implementation of many recommender systems based on neural networks.

Each new model must be a sub-class of the abstract class :class:`RecSysModel`. Moreover,
if the model is a standard neural network (NN) then it is advisable to inherit from
:class:`NeuralModel`. Each neural network based model (represented as a separate module) is thought
of being composed of four parts, i.e., classes: a neural network (subclass of :class:`NeuralNet`),
a training algorithm (subclass of :class:`TorchNNTrainer`), a data sampler (subclass of
:class:`Sampler`) and a model class (subclass of :class:`NeuralModel`). If a standard sampler fits
the need of the learning algorithm then the sampler can be omitted from the module.

Currently the implemented neural network-based models are:

* :mod:`rectorch.models.nn.multdae`: Denoising Autoencoder for Collaborative filtering with
  Multinomial prior (in the paper *Mult-DAE*) [VAE]_;
* :mod:`rectorch.models.nn.multvae`: Variational Autoencoder for Collaborative filtering with
  Multinomial prior(in the paper *Mult-VAE*) [VAE]_;
* :mod:`rectorch.models.nn.cvae`: Conditioned Variational Autoencoder (in the paper *C-VAE*)
  [CVAE]_;
* :mod:`rectorch.models.nn.cfgan`: Collaborative Filtering with Generative Adversarial Networks
  [CFGAN]_;
* :mod:`rectorch.models.nn.svae`: Sequential Variational Autoencoders for Collaborative Filtering
  [SVAE]_.
* :mod:`rectorch.models.nn.recvae`: RecVAE: A New Variational Autoencoder for Top-N Recommendations
  with Implicit Feedback [RecVAE]_.

It is also implemented a generic Variational autoencoder trainer (:class:`VAE`) based on the classic
loss function *cross-entropy* based reconstruction loss, plus the KL loss.

See Also
--------
Modules:
:mod:`models <rectorch.models>`
:mod:`samplers <rectorch.samplers>`

References
----------
.. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
   Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
   World Wide Web Conference (WWW '18). International World Wide Web Conferences Steering
   Committee, Republic and Canton of Geneva, CHE, 689–698.
   DOI: https://doi.org/10.1145/3178876.3186150
.. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
   Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
   https://arxiv.org/abs/2004.11141
.. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
   CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
   In Proceedings of the 27th ACM International Conference on Information and Knowledge
   Management (CIKM '18). Association for Computing Machinery, New York, NY, USA, 137–146.
   DOI: https://doi.org/10.1145/3269206.3271743
.. [SVAE] Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi. 2019.
   Sequential Variational Autoencoders for Collaborative Filtering. In Proceedings of the Twelfth
   ACM International Conference on Web Search and Data Mining (WSDM '19). Association for Computing
   Machinery, New York, NY, USA, 600–608. DOI: https://doi.org/10.1145/3289600.3291007
.. [RecVAE] Ilya Shenbin, Anton Alekseev, Elena Tutubalina, Valentin Malykh, and Sergey
   I. Nikolenko. 2020. RecVAE: A New Variational Autoencoder for Top-N Recommendations
   with Implicit Feedback. In Proceedings of the 13th International Conference on Web
   Search and Data Mining (WSDM '20). Association for Computing Machinery, New York, NY, USA,
   528–536. DOI: https://doi.org/10.1145/3336191.3371831
"""
import os
import time
from importlib import import_module
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal_init
from torch.nn.init import xavier_uniform_ as xavier_init
from rectorch import env, StatefulObject
from rectorch.models import RecSysModel
from rectorch.utils import init_optimizer
from rectorch.evaluation import evaluate
from rectorch.validation import ValidFunc
from rectorch.samplers import ArrayDummySampler, TensorDummySampler, SparseDummySampler

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["NeuralNet", "AE_net", "VAE_net", "TorchNNTrainer", "AE_trainer", "VAE_trainer",\
    "NeuralModel", "multvae", "multdae", "cvae", "recvae", "svae", "cfgan"]


class NeuralNet(nn.Module, StatefulObject):
    """Abstract class representing a generic neural network.

    This abstract class must be inherited anytime a new neural network is defined.
    The following methods must be implemented in the sub-classes:

    - :meth:`forward`
    - :meth:`init_weights`
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def init_weights(self):
        r"""Initialize the weights of the network.
        """
        raise NotImplementedError()

    @classmethod
    def from_state(cls, state):
        r"""Create a new network from the given state.

        The state is a dictionary containing all the necesssary information to build a
        ``NeuralNet`` equivalent to the one that generated the state (thorugh the method
        :meth:`get_state`).

        Parameters
        ----------
        state : :obj:`dict`
            The network's state dictionary  useful to replicate the saved network.
        """
        net_class = getattr(import_module(cls.__module__), cls.__name__)
        net = net_class(**state["params"])
        net.load_state_dict(state["state"])
        return net


class AE_net(NeuralNet):
    r"""Abstract Autoencoder network.

    This abstract class must be inherited anytime a new autoencoder network is defined.
    The following methods must be implemented in the sub-classes:

    - :meth:`encode`
    - :meth:`decode`

    Parameters
    ----------
    dec_dims : list or array_like
        Dimensions of the decoder network. ``dec_dims[0]`` indicates the dimension of the latent
        space, and ``dec_dims[-1]`` indicates the dimension of the input space.
    enc_dims : list, array_like or :obj:`None` [optional]
        Dimensions of the encoder network. ``end_dims[0]`` indicates the dimension of the input
        space, and ``end_dims[-1]`` indicates the dimension of the latent space.
        If evaluates to False, ``enc_dims = dec_dims[::-1]``. By default :obj:`None`.

    Attributes
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`dec_dims` in the **Parameter** section.
    enc_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`end_dims` in the **Parameter** section.
    """
    def __init__(self, dec_dims, enc_dims=None):
        super(AE_net, self).__init__()
        self.enc_dims = enc_dims if enc_dims else dec_dims[::-1]
        self.dec_dims = dec_dims

    def encode(self, x):
        r"""Forward propagate the input in the encoder network.

        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor.
        """
        raise NotImplementedError()

    def decode(self, z):
        r"""Forward propagate the latent represenation in the decoder network.

        Parameters
        ----------
        z : :py:class:`torch.Tensor`
            The latent tensor.
        """
        raise NotImplementedError()

    def forward(self, x):
        r"""Forward propagate the input in the entire network.

        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor to feed to the network.
        """
        z = self.encode(x)
        return self.decode(z)

    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "enc_dims" : self.enc_dims,
                "dec_dims" : self.dec_dims,
            }
        }
        return state


class VAE_net(AE_net):
    r"""Variational Autoencoder network.

    Layers are fully connected and *tanh* activated with the exception of the ouput layers of
    both the encoder and decoder that are linearly activated.

    Notes
    -----
    See :class:`AE_net` for parameters and attributes.
    """
    def __init__(self, dec_dims, enc_dims=None):
        super(VAE_net, self).__init__(dec_dims, enc_dims)

        # Last dimension of enc- network is for mean and variance
        temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
        self.init_weights()

    def encode(self, x):
        r"""Apply the encoder network of the Variational Autoencoder.

        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor

        Returns
        -------
        mu, logvar : :obj:`tuple` of :py:class:`torch.Tensor`
            The tensors in the latent space representing the mean and standard deviation (actually
            the logarithm of the variance) of the probability distributions over the
            latent variables.
        """
        h = x
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def decode(self, z):
        r"""Apply the decoder network to the sampled latent representation.

        Parameters
        ----------
        z : :py:class:`torch.Tensor`
            The sampled (trhough the reparameterization trick) latent tensor.

        Returns
        -------
        :class:`torch.Tensor`
            The output tensor of the decoder network.
        """
        h = z
        for i, layer in enumerate(self.dec_layers):
            h = layer(h)
            if i != len(self.dec_layers) - 1:
                h = torch.tanh(h)
        return torch.sigmoid(h)

    def _reparameterize(self, mu, var):
        if self.training:
            std = torch.exp(0.5*var)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, x):
        r"""Apply the full Variational Autoencoder network to the input.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        x', mu, logvar : :obj:`tuple` of :py:class:`torch.Tensor`
            The reconstructed input (x') along with the intermediate tensors in the latent space
            representing the mean and standard deviation (actually the logarithm of the variance)
            of the probability distributions over the latent variables.
        """
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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


class TorchNNTrainer(StatefulObject):
    r"""Abstract class representing a neural network-based training.

    This base class assumes that the model can be trained using a standard backpropagation
    procedure. It is not meant to manage complex training patterns, such as alternate training
    between more than one network as done with Generative Adversarial Networks. Thus, it assumes
    that there is a neural network (i.e., :class:`torch.nn.Module`) for which the parameters must be
    learned.

    Parameters
    ----------
    net : :class:`rectorch.model.nn.NeuralNet`
        The neural network architecture.
    device : :class:`torch.device`
        Device where the pytorch tensors are saved.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.

    Attributes
    ----------
    network : :class:`rectorch.model.nn.NeuralNet`
        The neural network architecture.
    learning_rate : :obj:`float`
        The learning rate for the optimizer.
    optimizer : :class:`torch.optim.Optimizer`
        Optimizer(s) used for performing the training.
    device : :class:`torch.device`
        Device where the pytorch tensors are saved.
    """
    def __init__(self, net, device, opt_conf=None):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.network = net.to(self.device)
        self.optimizer = init_optimizer(self.network.parameters(), opt_conf)
        self.current_epoch = 0
        self.opt_conf = opt_conf

    def loss_function(self, *args, **kwargs):
        r"""The loss function that the model wants to minimize.

        Parameters
        ----------
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

    def train_epoch(self, epoch, data_sampler, *args, **kwargs):
        r"""Training of a single epoch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        data_sampler : :class:`rectorch.samplers.Sampler`
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

    def train_batch(self, epoch, tr_batch, *args, **kwargs):
        r"""Training of a single batch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        tr_batch : :class:`torch.Tensor`
            Traning part of the current batch.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            training on the batch.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            training on the batch.
        """
        raise NotImplementedError()

    def get_state(self):
        state = {
            'params': {
                'device' : self.device,
                'opt_conf' : self.opt_conf
            },
            'epoch': self.current_epoch,
            'network': self.network.get_state(),
            'optimizer' : self.optimizer.state_dict(),
        }
        return state

    @classmethod
    def from_state(cls, state):
        r"""Load the model from file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.

        Returns
        -------
        :class:`TorchNNTrainer`
            An object of type that is a sub-class of :class:`TorchNNTrainer`.
        """
        net_class = getattr(import_module(cls.__module__), state["network"]["name"])
        trainer_class = getattr(import_module(cls.__module__), cls.__name__)
        net = net_class(**state['network']['params'])
        net.load_state_dict(state['network']['state'])
        trainer = trainer_class(net, **state['params'])
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.current_epoch = state['epoch']
        return trainer


class AE_trainer(TorchNNTrainer):
    r"""Base class for Autoencoder-based models.

    Parameters
    ----------
    ae_net : :class:`torch.nn.Module`
        The autoencoder neural network.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.

    Attributes
    ----------
    all attributes : see the base class :class:`TorchNNTrainer`.
    """

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

    def train_epoch(self, epoch, data_sampler, verbose=1):
        self.network.train()
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(data_sampler) // 10**verbose)

        for batch_idx, (data, gt) in enumerate(data_sampler):
            partial_loss += self.train_batch(data, gt)
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                env.logger.info('| epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                                epoch, (batch_idx+1), len(data_sampler),
                                elapsed * 1000 / log_delay,
                                partial_loss / log_delay)
                train_loss += partial_loss
                partial_loss = 0.0
                start_time = time.time()
        total_loss = (train_loss + partial_loss) / len(data_sampler)
        time_diff = time.time() - epoch_start_time
        env.logger.info("| epoch %d | loss %.4f | total time: %.2fs |",
                        epoch, total_loss, time_diff)
        self.current_epoch += 1

    def train_batch(self, tr_batch, te_batch=None):
        r"""Training of a single batch.

        Parameters
        ----------
        epoch : :obj:`int`
            Epoch's number.
        tr_batch : :class:`torch.Tensor`
            Traning portion of the current batch.
        te_batch : :class:`torch.Tensor` [optional]
            Test portion of the current batch, by default :obj:`None`.

        Returns
        -------
        :obj:`float`
            The loss incurred in the batch.
        """
        data_tensor = tr_batch.view(tr_batch.shape[0], -1).to(self.device)
        if te_batch is not None:
            ground_truth = te_batch.view(te_batch.shape[0], -1).to(self.device)
        else:
            ground_truth = data_tensor
        self.optimizer.zero_grad()
        recon_batch = self.network(data_tensor)
        loss = self.loss_function(recon_batch, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class VAE_trainer(AE_trainer):
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
        .. [KINGMA] Kingma, Diederik P and Welling, Max. Auto-Encoding Variational Bayes, 2013.
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


class NeuralModel(RecSysModel):
    r"""Recommender system based on neural network.

    Parameters
    ----------
    network : :class:`rectorch.models.nn.NeuralNet`
        The neural network architecture.
    trainer : :class:`rectorch.models.nn.TorchNNTrainer`
        The trainer class for performing the learning.
    device : :obj:`str`
        The device where the model must be loaded.

    Attributes
    ----------
    all attributes : see **Parameter** Section
    """
    def __init__(self, network, trainer, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.network = network
        self.trainer = trainer

    def train(self,
              dataset,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=100,
              verbose=1):
        r"""Training procedure for a neural network based model.

        Parameters
        ----------
        dataset : :class:`rectorch.data.Dataset` or :class:`rectorch.samplers.Sampler`
            The dataset or the sampler to use for training/validation.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default :obj:`None`.
            If ``valid_metric`` is set to :obj:`None` the validation step is skipped.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`rectorch.validation.ValidFunc` [optional]
            The validation function, by default it set to the standard validation procedure, i.e.,
            :func:`rectorch.evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 100.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        """
        raise NotImplementedError()

    def predict(self, x, remove_train=True):
        r"""Perform the prediction using a trained Variational Autoencoder.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input batch tensor for which the prediction must be computed.
        remove_train : :obj:`bool` [optional]
            Whether to remove the training set from the prediction, by default :obj:`True`. Removing
            the training items means set their scores to :math:`-\infty`.
        """
        raise NotImplementedError()

    def get_state(self):
        return self.trainer.get_state()

    def save_model(self, filepath):
        env.logger.info("Saving %s model checkpoint to %s...", self.__class__.__name__, filepath)
        torch.save(self.get_state(), filepath)
        env.logger.info("Model checkpoint saved!")

    #@classmethod
    #def from_state(cls, state):
    #   pass

    @classmethod
    def load_model(cls, filepath, device=None):
        r"""Load the model from file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.

        Returns
        -------
        :class:`rectorch.models.nn.TorchNNTrainer`
            An object of type that is a sub-class of :class:`rectorch.models.nn.TorchNNTrainer`.
        """
        raise NotImplementedError()
