r"""A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.

This recommender system is based on GAN where the (conditioned) generator aims at generating
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

References
----------
.. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
   CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
   In Proceedings of the 27th ACM International Conference on Information and Knowledge
   Management (CIKM '18). Association for Computing Machinery, New York, NY, USA, 137–146.
   DOI: https://doi.org/10.1145/3269206.3271743
"""
import time
from importlib import import_module
import numpy as np
import torch
from torch import nn
from torch.nn.init import normal_ as normal_init
from torch.nn.init import xavier_uniform_ as xavier_init
from rectorch import env
from rectorch.samplers import DataSampler, Sampler
from rectorch.models.nn import NeuralNet, TorchNNTrainer, NeuralModel
from rectorch.evaluation import evaluate
from rectorch.validation import ValidFunc
from rectorch.utils import init_optimizer

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["CFGAN_G_net", "CFGAN_D_net", "CFGAN_Sampler", "CFGAN_trainer", "CFGAN"]


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


class CFGAN_Sampler(DataSampler):
    r"""Sampler used for training the generator and discriminator of the CFGAN model.

    The peculiarity of this sampler (see for [CFGAN]_ more details) is that batches are
    continuously picked at random from all the training set.

    Parameters
    ----------
    data : :class:`rectorch.data.Dataset`
        The dataset from which the sampler samples the ratings.
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``} [optional]
        Indicates the mode in which the sampler operates, by default ``'train'``.
    batch_size : :obj:`int` [optional]
        The size of the batches, by default 64

    Attributes
    ----------
    mode : :obj:`str` in the set {``'train'``, ``'valid'``, ``'test'``}
        Indicates the mode in which the sampler operates.
    sparse_data_tr : :obj:`scipy.sparse.csr_matrix`
        See ``sparse_data_tr`` parameter.
    batch_size : :obj:`int`
        See ``batch_size`` parameter.
    idxlist : :obj:`list` of :obj:`int`
        Shuffled list of indexes. After an iteration over the sampler, or after a call to the
        :func:`next` function, the ``idxlist`` contains, in the first ``batch_size`` positions,
        the indexes of the examples that are contained in the current batch.

    References
    ----------
    .. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
       CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
       In Proceedings of the 27th ACM International Conference on Information and Knowledge
       Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
       DOI: https://doi.org/10.1145/3269206.3271743
    """
    def __init__(self,
                 data,
                 mode="train",
                 batch_size=64):
        super(CFGAN_Sampler, self).__init__(data, mode, batch_size, False)
        self.idxlist = list(range(self.sparse_data_tr.shape[0]))

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def _set_mode(self, mode="train", batch_size=None):
        super()._set_mode(mode, batch_size)
        self.idxlist = list(range(self.sparse_data_tr.shape[0]))

    def __iter__(self):
        if self.mode == "train":
            while True:
                np.random.shuffle(self.idxlist)
                data_tr = self.sparse_data_tr[self.idxlist[:self.batch_size]]
                yield torch.FloatTensor(data_tr.toarray())
        else:
            yield from super().__iter__()


class CFGAN_trainer(TorchNNTrainer):
    r"""Trainer class for the CFGAN model.

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
        The ZR-coefficient, by default 0.1.
    s_pm : :obj:`float` [optional]
        The sampling parameter for the partial-masking approach, by default 0.7.
    s_zr : :obj:`float` [optional]
        The sampling parameter for the zero-reconstruction regularization, by default 0.5.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.

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
    opt_g : :class:`torch.optim.Optimizer`
        Optimizer used for performing the training of the generator.
    opt_d : :class:`torch.optim.Optimizer`
        Optimizer used for performing the training of the discriminator.
    """
    def __init__(self,
                 generator,
                 discriminator,
                 alpha=.1,
                 s_pm=.7,
                 s_zr=.5,
                 device=None,
                 opt_conf=None):
        super(CFGAN_trainer, self).__init__(generator, device, opt_conf)
        self.generator = self.network
        self.discriminator = discriminator.to(self.device)

        self.s_pm = s_pm
        self.s_zr = s_zr
        self.alpha = alpha
        self.n_items = self.generator.input_dim
        self.optimizer = [init_optimizer(self.generator.parameters(), opt_conf),
                          init_optimizer(self.discriminator.parameters(), opt_conf)]
        self.opt_g = self.optimizer[0]
        self.opt_d = self.optimizer[1]

    def loss_function(self, x, y, reg=False):
        return torch.nn.MSELoss(reduction="sum")(x, y) if reg else torch.nn.BCELoss()(x, y)

    def train_epoch(self, epoch, data_sampler, g_steps, d_steps):
        data_sampler.train()
        it_sampler = iter(data_sampler)
        loss_d, loss_g = 0, 0
        for _ in range(1, g_steps+1):
            loss_g += self.train_batch(next(it_sampler).to(self.device), True)

        for _ in range(1, d_steps+1):
            loss_d += self.train_batch(next(it_sampler).to(self.device), False)

        self.current_epoch += 1
        return loss_g, loss_d

    def train_batch(self, tr_batch, generator=True):
        return self.train_gen_batch(tr_batch) if generator else self.train_disc_batch(tr_batch)

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
            g_reg_loss = self.loss_function(fake_data, mask_zr, reg=True)

        fake_data = torch.mul(fake_data, mask)
        disc_on_fake = self.discriminator(fake_data, batch)
        g_loss = self.loss_function(disc_on_fake, real_label)
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
        d_loss_real = self.loss_function(disc_on_real, real_label)

        fake_data = self.generator(batch)
        fake_data = torch.mul(fake_data, mask)
        disc_on_fake = self.discriminator(fake_data, batch)
        d_loss_fake = self.loss_function(disc_on_fake, fake_label)

        d_loss = d_loss_real + d_loss_fake
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()
        return d_loss.item()

    def get_state(self):
        state = {
            'epoch': self.current_epoch,
            'network_g': self.generator.get_state(),
            'network_d': self.discriminator.get_state(),
            'optimizer_g': self.opt_g.state_dict(),
            'optimizer_d': self.opt_g.state_dict(),
            'params': {
                'alpha' : self.alpha,
                's_pm' : self.s_pm,
                's_zr' : self.s_zr,
                'opt_conf' : self.opt_conf
            }
        }
        return state

    @classmethod
    def from_state(cls, state):
        gen_class = getattr(import_module(cls.__module__), state["network_g"]["name"])
        net_g = gen_class(**state['network_g']['params'])
        disc_class = getattr(import_module(cls.__module__), state["network_d"]["name"])
        net_d = disc_class(**state['network_d']['params'])
        trainer = CFGAN_trainer(net_g,
                                net_d,
                                **state['params'])
        trainer.generator.load_state_dict(state["network_g"]['state'])
        trainer.discriminator.load_state_dict(state["network_d"]['state'])
        trainer.opt_g.load_state_dict(state['optimizer_g'])
        trainer.opt_d.load_state_dict(state['optimizer_d'])
        trainer.current_epoch = state['epoch']
        return trainer


class CFGAN(NeuralModel):
    r"""CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.

    Parameters
    ----------
    gan_layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the generator network ordered from the input to the output.
    dic_layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the discriminator network ordered from the input to the
        output.
    alpha : :obj:`float` [optional]
        The ZR-coefficient, by default 0.1.
    s_pm : :obj:`float` [optional]
        The sampling parameter for the partial-masking approach, by default 0.7.
    s_zr : :obj:`float` [optional]
        The sampling parameter for the zero-reconstruction regularization, by default 0.5.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.
    device : :obj:`str` [optional]
        The device where the model must be loaded, by default :obj:`None`. If :obj:`None`, the
        default device (see `rectorch.env.device`) is used.
    trainer : :class:`rectorch.models.nn.multvae.CFGAN_trainer` [optional]
        The trainer object for performing the learning, by default :obj:`None`. If not :obj:`None`
        it is the only parameters that is taken into account for creating the model.
    """
    def __init__(self,
                 gen_layers_dim=None,
                 dis_layers_dim=None,
                 alpha=.1,
                 s_pm=.7,
                 s_zr=.5,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(CFGAN, self).__init__(trainer.network, trainer, trainer.device)
        else:
            device = torch.device(device) if device is not None else env.device
            gen = CFGAN_G_net(gen_layers_dim)
            dis = CFGAN_D_net(dis_layers_dim)
            trainer = CFGAN_trainer(gen,
                                    dis,
                                    alpha=alpha,
                                    s_pm=s_pm,
                                    s_zr=s_zr,
                                    device=device,
                                    opt_conf=opt_conf)
            super(CFGAN, self).__init__(trainer.network, trainer, device)

    def train(self,
              dataset,
              batch_size=64,
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
        dataset : :class:`rectorch.data.Dataset` or :class:`rectorch.samplers.CFGAN_TrainingSampler`
            The dataset/sampler object that load the training/validation set in mini-batches.
        batch_size : :obj:`int` [optional]
            The size of the batches, by default 64.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default :obj:`None`.
            If ``valid_data`` is not :obj:`None` then ``valid_metric`` must be not :obj:`None`.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`rectorch.validation.ValidFunc` [optional]
            The validation function, by default a standard validation procedure, i.e.,
            :func:`rectorch.evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 1000.
        g_steps : :obj:`int` [optional]
            Number of steps for a generator epoch, by default 5.
        d_steps : :obj:`int` [optional]
            Number of steps for a discriminator epoch, by default 5.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        """
        if isinstance(dataset, Sampler):
            data_sampler = dataset
        else:
            data_sampler = CFGAN_Sampler(dataset,
                                         mode="train",
                                         batch_size=batch_size)

        start_time = time.time()
        log_delay = max(10, num_epochs // 10**verbose)
        loss_d, loss_g = 0, 0
        try:
            for epoch in range(1, num_epochs+1):
                lg, ld = self.trainer.train_epoch(epoch, data_sampler, g_steps, d_steps)
                loss_g += lg
                loss_d += ld

                if epoch % log_delay == 0:
                    loss_g /= (g_steps * log_delay)
                    loss_d /= (d_steps * log_delay)
                    elapsed = time.time() - start_time
                    env.logger.info('| epoch {} | ms/batch {:.2f} | loss G {:.6f} | loss D {:.6f} |'
                                    .format(epoch, elapsed * 1000 / log_delay, loss_g, loss_d))
                    start_time = time.time()
                    loss_g, loss_d = 0, 0

                    if valid_metric is not None:
                        data_sampler.valid()
                        valid_res = valid_func(self, data_sampler, valid_metric)
                        mu_val = np.mean(valid_res)
                        std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                        env.logger.info('| epoch %d | %s %.3f (%.4f) |',
                                        epoch, valid_metric, mu_val, std_err_val)

        except KeyboardInterrupt:
            env.logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def predict(self, x, remove_train=True):
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            pred = self.network(x_tensor)
            if remove_train:
                pred[torch.nonzero(x_tensor, as_tuple=True)] = -np.inf
        return (pred, )

    @classmethod
    def from_state(cls, state):
        trainer = CFGAN_trainer.from_state(state)
        return CFGAN(trainer=trainer)

    @classmethod
    def load_model(cls, filepath, device=None):
        state = torch.load(filepath)
        if device:
            state["device"] = device
        return cls.from_state(state)
