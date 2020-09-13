r"""Sequential Variational Autoencoders (SVAE) for Collaborative Filtering.

SVAE [SVAE]_ introduces a recurrent version of the VAE, where temporal dependencies are taken
into account and passed through a recurrent neural network (RNN). At each time-step of the RNN,
the sequence is fed through a series of fully-connected layers, the output of which models the
probability distribution of the most likely future preferences.

References
----------
.. [SVAE] Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi. 2019.
   Sequential Variational Autoencoders for Collaborative Filtering. In Proceedings of the
   Twelfth ACM International Conference on Web Search and Data Mining (WSDM '19).
   Association for Computing Machinery, New York, NY, USA, 600–608.
   DOI: https://doi.org/10.1145/3289600.3291007
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from rectorch import env
from rectorch.models.nn import VAE_net
from rectorch.models.nn.multvae import MultVAE, MultVAE_trainer
from rectorch.samplers import Sampler
from rectorch.evaluation import evaluate
from rectorch.validation import ValidFunc

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["SVAE_Sampler", "SVAE_net", "SVAE_trainer", "SVAE"]


class SVAE_Sampler(Sampler):
    r"""Sampler used for training SVAE.

    This sampler yields pairs (``x``, ``y``) where ``x`` is the tensor of indexes of the
    positive items, and ``y`` the target tensor with the (multi-hot) ground truth items.
    This sampler is characterized by batches of size one (a single user at a time).
    Given a user (batch) *u* the returned ground truth tensor is a 3D tensor of dimension
    :math:`1 \times |\mathcal{I}_u|-1 \times m`, where :math:`|\mathcal{I}_u|` is the set
    of rated items by *u*, and *m* the number of items. This tensor represents the ground truth
    for *u* over time, and each slice of the tensor is a different timestamp across all the possible
    time unit for this specific user.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    pred_type : :obj:`str` in the set {``'next_k'``, ``'next'``, ``'postfix'``} [optional]
        The variant of loss used by the model, by default ``'next_k'``. If ``'next'`` then
        only the next item must be predicted, if ``'next_k'`` the next *k* items are considered in
        the ground truth, otherwise (= ``'postfix'``) all the remaining items are taken as ground
        truth.
    k : :obj:`int` [optional]
        The number of item to predict in the ``'next_k'`` variant, by default 1. This parameter
        is not considered when ``pred_type`` is not ``'next_k'``.
    shuffle : :obj:`bool` [optional]
        Whether the data set must by randomly shuffled before creating the batches, by default
        :obj:`True`.
    is_training : :obj:`bool` [optional]
        Whether the sampler is used during training, by default :obj:`True`.

    Attributes
    ----------
    all attributes : see **Parameters** section.
    """
    def __init__(self,
                 data,
                 mode="train",
                 pred_type="next_k",
                 k=1,
                 shuffle=True):
        super(SVAE_Sampler, self).__init__(data, mode)
        if pred_type == "next_k":
            assert k >= 1, "If pred_type == 'next_k' then 'k' must be a positive integer."
        self.pred_type = pred_type
        self.shuffle = shuffle
        self.num_items = data.n_items
        self.k = k

        self._dictr, self._dicval, self._dicte = self.data.to_dict()
        self.dict_data_tr, self.dict_data_te = None, None
        self._set_mode(mode)

    def _set_mode(self, mode="train", batch_size=1):
        assert mode in ["train", "valid", "test"], "Invalid sampler's mode."
        self.mode = mode

        if self.mode == "train":
            self.dict_data_tr = self._dictr
            self.dict_data_te = None
        elif self.mode == "valid":
            if isinstance(self._dicval, tuple):
                self.dict_data_tr = self._dicval[0]
                self.dict_data_te = self._dicval[1]
            else:
                self.dict_data_tr = self._dictr
                self.dict_data_te = self._dicval
        else:
            if isinstance(self._dicte, tuple):
                self.dict_data_tr = self._dicte[0]
                self.dict_data_te = self._dicte[1]
            else:
                self.dict_data_tr = self._dictr
                self.dict_data_te = self._dicte

    def __len__(self):
        return len(self.dict_data_tr)

    def __iter__(self):
        idxlist = list((self.dict_data_tr.keys()))
        if self.shuffle and self.mode == "train":
            np.random.shuffle(idxlist)

        for _, user in enumerate(idxlist):
            ulen = len(self.dict_data_tr[user])
            y_batch_s = torch.zeros(1, ulen - 1, self.num_items)

            if self.mode == "train":
                if self.pred_type == 'next':
                    for timestep in range(ulen - 1):
                        idx = self.dict_data_tr[user][timestep + 1]
                        y_batch_s[0, timestep, idx] = 1.
                elif self.pred_type == 'next_k':
                    for timestep in range(ulen - 1):
                        idx = self.dict_data_tr[user][timestep + 1:][:self.k]
                        y_batch_s[0, timestep, idx] = 1.
                elif self.pred_type == 'postfix':
                    for timestep in range(ulen - 1):
                        idx = self.dict_data_tr[user][timestep + 1:]
                        y_batch_s[0, timestep, idx] = 1.
            else:
                y_batch_s = torch.zeros(1, 1, self.num_items)
                y_batch_s[0, 0, self.dict_data_te[user]] = 1.

            x_batch = [self.dict_data_tr[user][:-1]]

            x = Variable(torch.LongTensor(x_batch))
            y = Variable(y_batch_s, requires_grad=False)

            yield x, y

class SVAE_net(VAE_net):
    """Sequential Variational Autoencoders for Collaborative Filtering.

    The network structure follows the definition as in [SVAE]_. Items are embedded into a latent
    space and then fed into a Gated Recurrent Unit (GRU). Finally, a VAE network takes as input
    the GRU representation and returns the items' scores.

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


class SVAE_trainer(MultVAE_trainer):
    r"""Trainer class for SVAE model.

    Parameters
    ----------
    mvae_net : :class:`rectorch.models.nn.NeuralNet`
        The variational autoencoder neural network.
    beta : :obj:`float` [optional]
        The :math:`\beta` hyper-parameter of SVAE. When ``anneal_steps > 0`` then this
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
    optimizer : :class:`torch.optim.Optimizer`
        The optimizer is initialized according to the given configurations in ``opt_conf``.
    other attributes : see the **Parameters** section.

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
                 device=None,
                 opt_conf=None):
        super(SVAE_trainer, self).__init__(svae_net,
                                           beta=beta,
                                           anneal_steps=anneal_steps,
                                           device=device,
                                           opt_conf=opt_conf)

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        likelihood_n = -torch.sum(torch.sum(F.log_softmax(recon_x, -1) * x.view(recon_x.shape), -1))
        likelihood_d = float(torch.sum(x[0, :recon_x.shape[2]]))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return likelihood_n / likelihood_d + beta * KLD


class SVAE(MultVAE):
    r"""Variational Autoencoder for collaborative filtering.

    Parameters
    ----------
    n_items : :obj:`int`
        Number of items.
    embed_size : :obj:`int`
        Size of the embedding for the items.
    rnn_size : :obj:`int`
        Size of the recurrent layer if the GRU part of the network.
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
    network : :class:`rectorch.models.nn.multvae.SVAE_net`
        The neural network architecture.
    trainer : :class:`rectorch.models.nn.multvae.SVAE_trainer`
        The trainer class for performing the learning.
    device : :obj:`str`
        The device where the model must be loaded.
    """
    def __init__(self,
                 n_items=None,
                 embed_size=None,
                 rnn_size=None,
                 dec_dims=None,
                 enc_dims=None,
                 dropout=0.5,
                 beta=1.,
                 anneal_steps=0,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(SVAE, self).__init__(trainer=trainer)
        else:
            device = torch.device(device) if device is not None else env.device
            network = SVAE_net(n_items=n_items,
                               embed_size=embed_size,
                               rnn_size=rnn_size,
                               dec_dims=dec_dims,
                               enc_dims=enc_dims)
            trainer = SVAE_trainer(network,
                                   beta=beta,
                                   anneal_steps=anneal_steps,
                                   device=device,
                                   opt_conf=opt_conf)
            super(SVAE, self).__init__(trainer=trainer)

    def train(self,
              dataset,
              pred_type="next_k",
              k=1,
              shuffle=True,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=200,
              best_path=None,
              verbose=1,
              seed=None):
        r"""SVAE training procedure.

        Parameters
        ----------
        dataset : class:`rectorch.data.Dataset` or :class:`rectorch.samplers.Sampler`
            The dataset or the sampler to use for training/validation.
        batch_size : :obj:`int` [optional]
            The size of the batches, by default 1.
        pred_type : :obj:`str` in the set {``'next_k'``, ``'next'``, ``'postfix'``} [optional]
            The variant of loss used by the model, by default ``'next_k'``. If ``'next'`` then
            only the next item must be predicted, if ``'next_k'`` the next *k* items are considered
            in the ground truth, otherwise (= ``'postfix'``) all the remaining items are taken as
            ground truth.
        k : :obj:`int` [optional]
            The number of item to predict in the ``'next_k'`` variant, by default 1. This parameter
            is not considered when ``pred_type`` is not ``'next_k'``.
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
        if isinstance(dataset, Sampler):
            data_sampler = dataset
        else:
            data_sampler = SVAE_Sampler(dataset,
                                        mode="train",
                                        pred_type=pred_type,
                                        k=k,
                                        shuffle=shuffle)
        super().train(data_sampler,
                      1,
                      shuffle,
                      valid_metric,
                      valid_func,
                      num_epochs,
                      best_path,
                      verbose,
                      seed)

    def predict(self, x, remove_train=True):
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[0, -1, x_tensor] = -np.inf
            return recon_x[:, -1, :], mu, logvar

    @classmethod
    def from_state(cls, state):
        trainer = SVAE_trainer.from_state(state)
        return SVAE(trainer=trainer)
