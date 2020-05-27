r"""This module contains definitions of the neural newtork architectures used by
the **rectorch** models.

See Also
--------
Modules:
:mod:`models <models>`
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal_init
from torch.nn.init import xavier_uniform_ as xavier_init

__all__ = ['AE_net', 'MultiDAE_net', 'VAE_net', 'MultiVAE_net', 'CMultiVAE_net', 'CFGAN_G_net',\
    'CFGAN_D_net', 'SVAE_net']

logger = logging.getLogger(__name__)


class AE_net(nn.Module):
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
        if enc_dims:
            #assert enc_dims[0] == dec_dims[-1], \
            #            "In and Out dimensions must equal to each other"
            #assert enc_dims[-1] == dec_dims[0], \
            #            "Latent dimension for encoder and decoder network mismatches."
            self.enc_dims = enc_dims
        else:
            self.enc_dims = dec_dims[::-1]

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

    def init_weights(self):
        r"""Initialize the weights of the network.
        """
        raise NotImplementedError()


#TODO check this network
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
    latent_size : :obj:`int`, optional
        Dimension of the latent space, by default 50.
    dropout : :obj:`float`, optional
        Dropout (noise) percentage defined in the interval [0,1], by default 0.5.

    References
    ----------
    .. [CDAE] Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester. 2016.
       Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
       In Proceedings of the Ninth ACM International Conference on Web Search
       and Data Mining (WSDM ’16). Association for Computing Machinery,
       New York, NY, USA, 153–162. DOI: https://doi.org/10.1145/2835776.2835837
    """
    def __init__(self, n_items, n_users, latent_size=50, dropout=0.5):
        super(CDAE_net, self).__init__([latent_size, n_items], [n_items+n_users, latent_size])
        self.dropout = nn.Dropout(dropout)

        self.n_items = n_items
        self.enc_layer = nn.Linear(self.enc_dims[0], self.enc_dims[1])
        self.dec_layer = nn.Linear(self.dec_dims[0], self.dec_dims[1])

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
            x[:self.n_items] *= 1. / (1.-self.dropout.p)
            x[:self.n_items] = self.dropout(x[:self.n_items])

        x = torch.sigmoid(self.enc_layer(x))
        return x

    def decode(self, z):
        return torch.sigmoid(self.dec_layer(z))

    def init_weights(self):
        r"""Initialize the weights of the network.

        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        xavier_init(self.enc_layer.weight)
        normal_init(self.enc_layer.bias)
        xavier_init(self.dec_layer.weight)
        normal_init(self.dec_layer.bias)


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
    dropout : :obj:`float`, optional
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
        :py:class:`torch.Tensor`
            The output tensor of the decoder network.
        """
        h = z
        for i, layer in enumerate(self.dec_layers):
            h = layer(h)
            if i != len(self.dec_layers) - 1:
                h = torch.tanh(h)
        return torch.sigmoid(h)

    def _reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std

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
    dropout : :obj:`float`, optional
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
    dropout : :obj:`float`, optional
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


class CFGAN_G_net(nn.Module):
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


class CFGAN_D_net(nn.Module):
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
    See *Parameres* section.

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
        #temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        #self.enc_layers = nn.ModuleList(
        #    [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

        #self.dec_layers = nn.ModuleList(
        #    [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
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
