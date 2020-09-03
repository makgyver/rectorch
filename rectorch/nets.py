r"""This module contains definitions of the neural newtork architectures used by
the **rectorch** models.

See Also
--------
Modules:
:mod:`models <rectorch.models>`
:mod:`models.nn <rectorch.models.nn>`
"""
import importlib
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal_init
from torch.nn.init import xavier_uniform_ as xavier_init
from rectorch.utils import swish, log_norm_pdf

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['AE_net', 'MultiDAE_net', 'VAE_net', 'MultiVAE_net', 'CMultiVAE_net', 'CFGAN_G_net',\
    'CFGAN_D_net', 'SVAE_net']

class NeuralNet(nn.Module):
    """Abstract class representing a generic neural network.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def init_weights(self):
        r"""Initialize the weights of the network.
        """
        raise NotImplementedError()

    def get_state(self):
        r"""Get the state of the network as a dictionary.

        The state contains all useful information to construct a new network from scratch
        that is identical to the current network.
        """
        raise NotImplementedError()

    @classmethod
    def from_state(cls, state):
        r"""Create a new network from the given state.

        Parameters
        ----------
        state : :obj:`dict`
            The network's state dictionary  useful to replicate the saved network.
        """
        net_class = getattr(importlib.import_module("rectorch.nets"), cls.__name__)
        net = net_class(**state["params"])
        net.load_state_dict(state["state"])
        return net


class AE_net(NeuralNet):
    r"""Abstract Autoencoder network.

    This abstract class must be inherited anytime a new autoencoder network is defined.
    The following methods must be implemented in the sub-classes:

    - :meth:`encode`
    - :meth:`decode`
    - :meth:`init_weights`

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
        See :attr:`dec_dims` parameter.
    enc_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`end_dims` parameter.
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
            The input tensor
        """
        raise NotImplementedError()

    def decode(self, z):
        r"""Forward propagate the latent represenation in the decoder network.

        Parameters
        ----------
        z : :py:class:`torch.Tensor`
            The latent tensor
        """
        raise NotImplementedError()

    def forward(self, x):
        r"""Forward propagate the input in the network.

        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor to feed to the network.
        """
        z = self.encode(x)
        return self.decode(z)


class CDAE_net(AE_net):
    r"""Collaborative Deonising AutoEncoder (CDAE).

    The CDAE network architecture follows the definition in [CDAE]_.
    Both encoder and decoder do not have any hidden layer. The dimension of
    the input (and output) is :attr:`n_users` + :attr:`n_items`.

    Parameters
    ----------
    n_items : :obj:`int`
        Number of items.
    n_users : :obj:`int`
        Number of users.
    latent_size : :obj:`int` [optional]
        Dimension of the latent space, by default 50.
    dropout : :obj:`float` [optional]
        Dropout (noise) percentage defined in the interval [0,1], by default 0.5.
    sigmoid_hidden : :obj:`bool` [optional]
        Whether the activation function of the output layer of the encoder is a sigmoid, default
        :obj:`False`.
    sigmoid_out : :obj:`bool` [optional]
        Whether the activation function of the output layer of the decoder is a sigmoid, default
        :obj:`False`.

    References
    ----------
    .. [CDAE] Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester. 2016.
       Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
       In Proceedings of the Ninth ACM International Conference on Web Search
       and Data Mining (WSDM ’16). Association for Computing Machinery,
       New York, NY, USA, 153–162. DOI: https://doi.org/10.1145/2835776.2835837
    """
    def __init__(self,
                 n_items,
                 n_users,
                 latent_size=50,
                 dropout=0.5,
                 sigmoid_hidden=False,
                 sigmoid_out=False):
        super(CDAE_net, self).__init__([latent_size, n_items], [n_items + n_users, latent_size])
        self.dropout = nn.Dropout(dropout)

        self.n_items = n_items
        self.n_users = n_users
        self.latent_size = latent_size
        self.enc_layer = nn.Linear(self.enc_dims[0], self.enc_dims[1])
        self.dec_layer = nn.Linear(self.dec_dims[0], self.dec_dims[1])
        self.sigmoid_hidden = sigmoid_hidden
        self.sigmoid_out = sigmoid_out

        self.init_weights()

    def encode(self, x):
        r"""Apply the encoder network to the input.

        The forward operation of the CDAE encoder network computes:

        :math:`h(W^\top \tilde{\mathbf{x}_i} + \mathbf{x}_u + \mathbf{b})`

        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor

        Returns
        -------
        :py:class:`torch.Tensor`
            The tensor in the latent space after the application of the encoder.
        """
        if self.training:
            x[:self.n_items] *= 1. / (1. - self.dropout.p)
            x[:self.n_items] = self.dropout(x[:self.n_items])

        x = self.enc_layer(x)
        return torch.sigmoid(x) if self.sigmoid_hidden else x

    def decode(self, z):
        z = self.dec_layer(z)
        return torch.sigmoid(z) if self.sigmoid_out else z

    def init_weights(self):
        r"""Initialize the weights of the network.

        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        xavier_init(self.enc_layer.weight)
        normal_init(self.enc_layer.bias)
        xavier_init(self.dec_layer.weight)
        normal_init(self.dec_layer.bias)

    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "enc_dims" : self.enc_dims,
                "dec_dims" : self.dec_dims,
                "n_items" : self.n_items,
                "n_users" : self.n_users,
                "dropout" : self.dropout.p,
                "sigmoid_hidden" : self.sigmoid_hidden,
                "sigmoid_out" : self.sigmoid_out
            }
        }
        return state


class MultiDAE_net(AE_net):
    r"""Denoising Autoencoder network for collaborative filtering.

    The network structure follows the definition as in [VAE]_. Hidden layers are fully
    connected and *tanh* activated. The output layer of both the encoder and the decoder
    are linearly activated.

    Parameters
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :class:`AE_net`.
    enc_dims : :obj:`list`, array_like of :obj:`int` or None [optional]
        See :class:`AE_net`.
    dropout : :obj:`float` [optional]
        The dropout probability (in the range [0,1]), by default 0.5.

    Attributes
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`dec_dims` parameter.
    enc_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`end_dims` parameter.
    dropout : :obj:`float`
        The dropout layer that is applied to the input during the :meth:`AE_net.forward`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    """
    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultiDAE_net, self).__init__(dec_dims, enc_dims)
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.enc_dims[:-1], self.enc_dims[1:])])

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
        self.init_weights()

    def encode(self, x):
        h = F.normalize(x)
        if self.training:
            h = self.dropout(h)
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
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "enc_dims" : self.enc_dims,
                "dec_dims" : self.dec_dims,
                "dropout" : self.dropout.p
            }
        }
        return state


class VAE_net(AE_net):
    r"""Variational Autoencoder network.

    Layers are fully connected and ReLU activated with the exception of the ouput layers of
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
            The input tensor

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

class MultiVAE_net(VAE_net):
    r'''Variational Autoencoder network for collaborative filtering.

    The network structure follows the definition as in [VAE]_. Hidden layers are fully
    connected and *tanh* activated. The output layer of both the encoder and the decoder
    are linearly activated.

    Parameters
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :class:`AE_net`.
    enc_dims : :obj:`list`, array_like of :obj:`int` or None [optional]
        See :class:`AE_net`.
    dropout : :obj:`float` [optional]
        See :class:`VAE_net`

    Attributes
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`dec_dims` parameter.
    enc_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`end_dims` parameter.
    dropout : :obj:`float`
        The dropout layer that is applied to the input during the :meth:`VAE_net.forward`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    '''

    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultiVAE_net, self).__init__(dec_dims, enc_dims)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        h = F.normalize(x)
        if self.training:
            h = self.dropout(h)
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
        state["params"]["dropout"] = self.dropout.p
        return state


class CMultiVAE_net(MultiVAE_net):
    r'''Conditioned Variational Autoencoder network for collaborative filtering.

    The network structure follows the definition as in [CVAE]_. Hidden layers are fully
    connected and *tanh* activated. The output layer of both the encoder and the decoder
    are linearly activated.

    Parameters
    ----------
    cond_dim : :obj:`int`
        The size of the condition vector.
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :class:`AE_net`.
    enc_dims : :obj:`list`, array_like of :obj:`int` or None [optional]
        See :class:`AE_net`.
    dropout : :obj:`float` [optional]
        See :class:`VAE_net`.

    Attributes
    ----------
    cond_dim : :obj:`int`
        See :attr:`cond_dim` parameter.
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`dec_dims` parameter.
    enc_dims : :obj:`list` or array_like
        See :attr:`end_dims` parameter.
    dropout : :obj:`float`
        The dropout layer that is applied to the input during the :meth:`VAE_net.forward`.

    References
    ----------
    .. [CVAE] Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
       Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
       https://arxiv.org/abs/2004.11141
    '''
    def __init__(self, cond_dim, dec_dims, enc_dims=None, dropout=0.5):
        super(CMultiVAE_net, self).__init__(dec_dims, enc_dims, dropout)
        self.cond_dim = cond_dim

        temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        temp_dims[0] += self.cond_dim
        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
        self.init_weights()

    def encode(self, x):
        h1 = F.normalize(x[:, :-self.cond_dim])
        if self.training:
            h1 = self.dropout(h1)
        h = torch.cat((h1, x[:, -self.cond_dim:]), 1)
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def get_state(self):
        state = super().get_state()
        state["name"] = self.__class__.__name__
        state["params"]["cond_dim"] = self.cond_dim
        return state


class CFGAN_G_net(NeuralNet):
    r"""Generator network of the CFGAN model.

    The generator newtork of CFGAN is a simple Multi Layer perceptron. Each internal layer is
    fully connected and ReLU activated. The output layer insted has a sigmoid as activation
    funciton. See [CFGAN]_ for a full description.

    Parameters
    ----------
    layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the network ordered from the input to the output.

    Attributes
    ----------
    layers_dim : :obj:`list` of :obj:`int`
        See the :attr:`layers_dim` parameter.
    input_dim : :obj:`int`
        The dimension of the output of the generator, i.e., the input of the discriminator.
    latent_dim : :obj:`int`
        The dimension of the latent space, i.e., the dimension of the input of the generator.

    References
    ----------
    .. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
       CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
       In Proceedings of the 27th ACM International Conference on Information and Knowledge
       Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
       DOI: https://doi.org/10.1145/3269206.3271743
    """
    def __init__(self, layers_dim):
        super(CFGAN_G_net, self).__init__()
        self.latent_dim = layers_dim[0]
        self.input_dim = layers_dim[-1]
        self.layers_dim = layers_dim

        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.ReLU(True)]

        dims = zip(self.layers_dim[:-2], self.layers_dim[1:])
        layers = [layer for d_in, d_out in dims for layer in block(d_in, d_out)]
        layers += [nn.Linear(*self.layers_dim[-2:]), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

    def forward(self, z):
        r"""Apply the generator network to the input.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor to be forwarded.

        Returns
        -------
        :class:`torch.Tensor`
            The output tensor results of the application of the generator network.
        """
        return self.model(z)

    def init_weights(self, layer):
        r"""Initialize the weights of the network.

        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        if isinstance(layer, nn.Linear):
            xavier_init(layer.weight)
            normal_init(layer.bias)

    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "layers_dim" : self.layers_dim
            }
        }
        return state


class CFGAN_D_net(NeuralNet):
    r"""Discriminator network of the CFGAN model.

    The discriminator newtork of CFGAN is a simple Multi Layer perceptron. Each internal layer is
    fully connected and ReLU activated. The output layer insted has a sigmoid as activation
    funciton. See [CFGAN]_ for a full description.

    Parameters
    ----------
    layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the network ordered from the input to the output.

    Attributes
    ----------
    layers_dim : :obj:`list` of :obj:`int`
        See the :attr:`layers_dim` parameter.
    input_dim : :obj:`int`
        The dimension of the input of the discriminator.

    References
    ----------
    .. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
       CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
       In Proceedings of the 27th ACM International Conference on Information and Knowledge
       Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
       DOI: https://doi.org/10.1145/3269206.3271743
    """
    def __init__(self, layers_dim):
        super(CFGAN_D_net, self).__init__()
        assert layers_dim[-1] == 1, "Discriminator must output a single node"
        self.input_dim = layers_dim[0]
        self.layers_dim = layers_dim

        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.ReLU(True)]

        dims = zip(self.layers_dim[:-2], self.layers_dim[1:])
        layers = [layer for d_in, d_out in dims for layer in block(d_in, d_out)]
        layers += [nn.Linear(*self.layers_dim[-2:]), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

    def forward(self, x, cond):
        r"""Apply the discriminator network to the input.

        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor to be forwarded.
        cond : :py:class:`torch.Tensor`
            The condition tensor. Note that must hold that ``x.shape[0] == cond.shape[0]``.

        Returns
        -------
        :py:class:`torch.Tensor`
            The output tensor results of the application of the discriminator to the input
            concatenated with the condition.
        """
        return self.model(torch.cat((x, cond), dim=1))

    def init_weights(self, layer):
        r"""Initialize the weights of the network.

        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        if isinstance(layer, nn.Linear):
            xavier_init(layer.weight)
            normal_init(layer.bias)

    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "layers_dim" : self.layers_dim
            }
        }
        return state


class SVAE_net(VAE_net):
    """Sequential Variational Autoencoders for Collaborative Filtering.

    **UNDOCUMENTED** [SVAE]_

    Parameters
    ----------
    n_items : :obj:`int`
        Number of items.
    embed_size : :obj:`int`
        Size of the embedding for the items.
    rnn_size : :obj:`int`
        Size of the recurrent layer if the GRU part of the network.
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :class:`AE_net`.
    enc_dims : :obj:`list`, array_like of :obj:`int` or None [optional]
        See :class:`AE_net`.

    Attributes
    ----------
    all attributes : see **Parameters** section.

    References
    ----------
    .. [SVAE] Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi. 2019.
       Sequential Variational Autoencoders for Collaborative Filtering. In Proceedings of the
       Twelfth ACM International Conference on Web Search and Data Mining (WSDM '19).
       Association for Computing Machinery, New York, NY, USA, 600–608.
       DOI: https://doi.org/10.1145/3289600.3291007
    """
    def __init__(self, n_items, embed_size, rnn_size, dec_dims, enc_dims):
        super(SVAE_net, self).__init__(dec_dims, enc_dims)
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.n_items = n_items
        self.embed_size = embed_size
        self.rnn_size = rnn_size
        self.item_embed = nn.Embedding(n_items, embed_size)

        self.gru = nn.GRU(embed_size, rnn_size, batch_first=True, num_layers=1)
        self.init_weights()

    def forward(self, x):
        in_shape = x.shape
        x = self.item_embed(x.view(-1)) # [seq_len x embed_size]
        rnn_out, _ = self.gru(x.view(in_shape[0], in_shape[1], -1)) # [1 x seq_len x rnn_size]
        rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1) # [seq_len x rnn_size]
        mu, logvar = self.encode(rnn_out) # [seq_len x hidden_size]
        z = self._reparameterize(mu, logvar) # [seq_len x latent_size]
        dec_out = self.decode(z)  # [seq_len x total_items]
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1) # [1 x seq_len x total_items]
        return dec_out, mu, logvar

    def decode(self, z):
        h = z
        for _, layer in enumerate(self.dec_layers[:-1]):
            h = torch.tanh(layer(h))
        return self.dec_layers[-1](h)

    def init_weights(self):
        for layer in self.enc_layers:
            nn.init.xavier_normal_(layer.weight)
        for layer in self.dec_layers:
            nn.init.xavier_normal_(layer.weight)

    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "n_items" : self.n_items,
                "enc_dims" : self.enc_dims,
                "dec_dims" : self.dec_dims,
                "embed_size" : self.embed_size,
                "rnn_size" : self.rnn_size
            }
        }
        return state


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
