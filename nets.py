import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal_init
from torch.nn.init import xavier_uniform_ as xavier_init
import torch.optim as optim

logger = logging.getLogger(__name__)


class AE_net(nn.Module):
    def __init__(self, dec_dims, enc_dims=None):
        super(AE_net, self).__init__()
        if enc_dims:
            assert enc_dims[0] == dec_dims[-1], "In and Out dimensions must equal to each other"
            assert enc_dims[-1] == dec_dims[0], "Latent dimension for encoder and decoder network mismatches."
            self.enc_dims = enc_dims
        else:
            self.enc_dims = dec_dims[::-1]

        self.dec_dims = dec_dims

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, z):
        raise NotImplementedError()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def init_weights(self):
        raise NotImplementedError()


class CDAE_net(AE_net):
    def __init__(self, n_items, n_users, latent_size=50, dropout=0.5):
        super(AE_net, self).__init__([latent_size, n_items], [n_items+n_users, latent_size])
        self.dropout = nn.Dropout(dropout)

        self.n_items = n_items
        self.enc_layer = nn.Linear(self.enc_dims[0], self.enc_dims[1])
        self.dec_layer = nn.Linear(self.dec_dims[0], self.dec_dims[1])

        self.init_weights()

    def encode(self, x):
        if self.training:
            x[:self.n_items] *= 1. / (1.-self.dropout.p)
            x[:self.n_items] = self.dropout(x[:self.n_items])

        x = torch.sigmoid(self.enc_layer(x))
        return x

    def decode(self, z):
        return torch.sigmoid(self.dec_layer(z))

    def init_weights(self):
        xavier_init(self.enc_layer.weight)
        normal_init(self.enc_layer.bias)
        xavier_init(self.dec_layer.weight)
        normal_init(self.dec_layer.bias)


class MultiDAE_net(AE_net):
    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultiDAE_net, self).__init__(dec_dims, enc_dims)
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.enc_dims[:-1], self.enc_dims[1:])]
        )

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])]
        )
        self.init_weights()

    def encode(self, x):
        h = F.normalize(x)
        if self.training:
            h = self.dropout(h)
        for i, layer in enumerate(self.enc_layers):
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
        for layer in self.enc_layers:
            xavier_init(layer.weight)
            normal_init(layer.bias)

        for layer in self.dec_layers:
            xavier_init(layer.weight)
            normal_init(layer.bias)


class VAE_net(AE_net):
    def __init__(self, dec_dims, enc_dims=None):
        super(VAE_net, self).__init__(dec_dims, enc_dims)

        # Last dimension of enc- network is for mean and variance
        temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])]
        )

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])]
        )
        self.init_weights()

    def encode(self, x):
        h = x
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = F.relu(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.dec_layers):
            h = layer(h)
            if i != len(self.dec_layers) - 1:
                h = torch.relu(h)
        return h

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def init_weights(self):
        for layer in self.enc_layers:
            xavier_init(layer.weight)
            normal_init(layer.bias)

        for layer in self.dec_layers:
            xavier_init(layer.weight)
            normal_init(layer.bias)


class MultiVAE_net(VAE_net):
    '''
    Autoencoder architecture as described in
    "Variational Autoencoders for Collaborative Filtering"
    https://arxiv.org/abs/1802.05814
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

    def reparameterize(self, mu, logvar):
        if self.training:
            return super().reparameterize(mu, logvar)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.dec_layers[:-1]):
            h = torch.tanh(layer(h))
        return self.dec_layers[-1](h)


class CMultiVAE_net(MultiVAE_net):
    def __init__(self, cond_dim, dec_dims, enc_dims=None, dropout=0.5):
        super(CMultiVAE_net, self).__init__(dec_dims, enc_dims, dropout)
        self.cond_dim = cond_dim

        temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        temp_dims[0] += self.cond_dim
        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])]
        )

        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])]
        )
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
        return self.model(z)

    def init_weights(self, layer):
        if type(layer) in [nn.Linear]:
            xavier_init(layer.weight)
            normal_init(layer.bias)


class CFGAN_D_net(nn.Module):
    def __init__(self, layers_dim):
        super(CFGAN_D_net, self).__init__()
        assert layers_dim[-1] == 1, "Discriminator must output a single node"
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

    def forward(self, x, cond):
        return self.model(torch.cat((x, cond), dim=1))

    def init_weights(self, layer):
        if type(layer) in [nn.Linear]:
            xavier_init(layer.weight)
            normal_init(layer.bias)
