"""Unit tests for the rectorch.models.nn module
"""
import os
import sys
import tempfile
import pytest
import torch
import pandas as pd
import warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.abspath('..'))

from rectorch.data import Dataset
from rectorch.models import RecSysModel
from rectorch.models.mf import EASE, ADMM_Slim
from rectorch.samplers import DataSampler
from rectorch.models.nn.multvae import MultVAE, MultVAE_net
from rectorch.models.nn.multdae import MultDAE, MultDAE_net
from rectorch.models.nn.cvae import ConditionedDataSampler, CMultVAE, CMultVAE_net
from rectorch.models.nn.recvae import RecVAE_net, RecVAE
from rectorch.models.nn.svae import SVAE_Sampler, SVAE, SVAE_net
from rectorch.models.nn.cfgan import CFGAN, CFGAN_D_net, CFGAN_G_net, CFGAN_Sampler
from rectorch.models.nn import TorchNNTrainer, AE_trainer, VAE_trainer, VAE_net, NeuralModel
from rectorch.samplers import ArrayDummySampler

def test_RecSysModel():
    """Test the RecSysModel class
    """
    model = RecSysModel()

    with pytest.raises(NotImplementedError):
        model.train(None)
    with pytest.raises(NotImplementedError):
        model.predict(None)
    with pytest.raises(NotImplementedError):
        model.save_model(None)
    with pytest.raises(NotImplementedError):
        RecSysModel.load_model(None)

def test_NeuralModel():
    """Test the RecSysModel class
    """
    model = NeuralModel(None, None, "cpu")

    with pytest.raises(NotImplementedError):
        model.train(None)
    with pytest.raises(NotImplementedError):
        RecSysModel.load_model(None)


def test_TorchNNTrainer():
    """Test the TorchNNTrainer class
    """
    net = MultDAE_net([1, 2], [2, 1], .1)
    model = TorchNNTrainer(net, device="cpu")

    with pytest.raises(NotImplementedError):
        model.train_batch(None, None, None)

    with pytest.raises(NotImplementedError):
        model.train_epoch(None, None, None)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.optimizer, torch.optim.Adam),\
        "optimizer should be of type torch.optim.Adam"
    assert str(model) == repr(model)

    x = torch.FloatTensor([[1, 1], [2, 2]])
    with pytest.raises(NotImplementedError):
        model.loss_function(None, None)
        model.train(None, None)
        model.train_epoch(0, None)
        model.train_batch(0, None, None)
        model.predict(x)

def create_sampler(rows, cols):
    """Create a toy DataSampler
    """
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = df_tr.copy()#pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = df_tr.copy()#pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    data = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)
    return DataSampler(data, mode="train", batch_size=1, shuffle=False)

def test_AETrainer():
    """Test the AETrainer class
    """
    net = MultDAE_net([1, 2], [2, 1], .1)
    model = AE_trainer(net, device="cpu")

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    gt = torch.FloatTensor([[1, 1], [2, 1]])
    pred = torch.FloatTensor([[1, 1], [1, 1]])
    assert model.loss_function(pred, gt) == torch.FloatTensor([.25]), "the loss should be .25"

    sampler = create_sampler([0, 0, 1], [0, 1, 1])
    '''
    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    '''
    model.train_epoch(0, sampler, verbose=4)
    '''
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name)

    model2 = AETrainer.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"
    '''

def test_VAE():
    """Test the VAE class
    """
    net = VAE_net([1, 2], [2, 1])
    model = VAE_trainer(net, device="cpu")

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    gt = torch.FloatTensor([[1, 1], [2, 1]])
    pred = torch.FloatTensor([[1, 1], [1, 1]])
    torch.manual_seed(12345)
    mu, logvar = model.network.encode(gt)
    pred = torch.sigmoid(pred)
    assert model.loss_function(pred, gt, mu, logvar) != torch.FloatTensor([.0]),\
        "the loss should not be 0"


    sampler = create_sampler([0, 0, 1], [0, 1, 1])
    '''
    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    '''
    model.train_epoch(0, sampler, verbose=4)
    '''
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name)

    model2 = VAE.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"
    '''

def test_MultDAE():
    """Test the MultDAE class
    """
    #net = MultDAE_net([1, 2], [2, 1], dropout=.1)
    model = MultDAE([1, 2], [2, 1], dropout=.1)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    #assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    #assert hasattr(model, "lam"), "model should have the attribute lam"
    #assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert model.trainer.lam == .2, "lambda should be .2"
    assert isinstance(model.trainer.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    gt = torch.FloatTensor([[1, 1], [2, 1]])
    pred = torch.FloatTensor([[1, 1], [1, 1]])
    torch.manual_seed(12345)
    assert model.trainer.loss_function(pred, gt) != torch.FloatTensor([.0]),\
        "the loss should not be 0"

    sampler = create_sampler([0, 0, 1], [0, 1, 1])

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model = MultDAE([1, 2], [2, 1], dropout=.2)
    model.train(sampler.data,
                valid_metric="ndcg@1",
                num_epochs=10)
    model.save_model(tmp.name)

    model2 = MultDAE.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"


def test_MultiVAE():
    """Test the MultVAE class
    """
    #net = MultVAE_net([1, 2], [2, 1], .1)
    model = MultVAE(dec_dims=[1,2], enc_dims=[2,1])

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "trainer"), "model should have the attribute trainer"
    #assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.trainer.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    gt = torch.FloatTensor([[1, 1], [2, 1]])
    pred = torch.FloatTensor([[1, 1], [1, 1]])
    torch.manual_seed(12345)
    mu, logvar = model.trainer.network.encode(gt)
    pred = torch.sigmoid(pred)
    assert model.trainer.loss_function(pred, gt, mu, logvar) != torch.FloatTensor([.0]),\
        "the loss should not be 0"

    sampler = create_sampler([0, 0, 1], [0, 1, 1])

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name)

    model2 = MultVAE.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    tmp2 = tempfile.NamedTemporaryFile()
    #net = MultVAE_net([1, 2], [2, 1], .1)
    model = MultVAE([1, 2], [2, 1], .5, 1., 5)
    model.train(sampler.data,
                valid_metric="ndcg@1",
                num_epochs=10)
    model.save_model(tmp2.name)

    model2 = MultVAE.load_model(tmp2.name)
    assert model2.trainer.gradient_updates > 0,\
        "the loaded model should have been saved after some gradient updates"


def test_CMultVAE():
    """Test the CMultVAE class
    """
    iid2cids = {0:[1], 1:[0, 1], 2:[0]}
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = df_tr.copy()#pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = df_tr.copy()#pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    data = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)

    sampler = ConditionedDataSampler(iid2cids, 2, data, mode="train", batch_size=1, shuffle=False)

    model = CMultVAE(iid2cids, 2, [1, 3], dropout=.1)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    #assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    #assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.trainer.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    x = torch.FloatTensor([[1, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    gt = torch.FloatTensor([[1, 1, 1], [2, 1, 1]])
    pred = torch.FloatTensor([[1, 1, 1], [1, 1, 1]])
    torch.manual_seed(12345)
    mu, logvar = model.network.encode(x)
    pred = torch.sigmoid(pred)
    assert model.trainer.loss_function(pred, gt, mu, logvar) != torch.FloatTensor([.0]),\
        "the loss should not be 0"

    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name)

    model2 = CMultVAE.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    tmp2 = tempfile.NamedTemporaryFile()
    model = CMultVAE(iid2cids, 2, [1, 3], [3, 1], .1, 5)
    model.train(sampler,
                valid_metric="ndcg@1",
                num_epochs=10,
                best_path=tmp2.name)

    model2 = CMultVAE.load_model(tmp2.name)
    assert model2.trainer.gradient_updates > 0,\
        "the loaded model should have been saved after some gradient updates"


def test_EASE():
    """Test the EASE class
    """
    ease = EASE(200.)
    assert hasattr(ease, "lam"), "ease should have the attribute lam"
    assert hasattr(ease, "model"), "ease should have the attribute model"
    assert ease.lam == 200, "lambda should be 200"
    assert ease.model is None, "before the training the inner model should be None"
    assert repr(ease) == str(ease)

    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    cols = [0, 1, 1, 2, 2, 3, 3, 4, 4, 1]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    data = Dataset(df_tr, None, (df_tr.copy(), df_tr.copy()), uids, iids)
    sampler = ArrayDummySampler(data)
    #X = csr_matrix(np.random.randint(2, size=(10, 5)), dtype="float64")

    ease.train(sampler)
    assert isinstance(ease.model, torch.FloatTensor),\
        "after training the model should be a pytorch tensor"

    X = sampler.data_tr
    pr = ease.predict([1, 3, 4], X[[1, 3, 4]])[0]
    assert pr.shape == (3, 5), "the shape of the prediction whould be 3 x 5"
    tmp = tempfile.NamedTemporaryFile()
    ease.save_model(tmp.name)
    ease2 = EASE.load_model(tmp.name)
    assert torch.all(ease2.model == ease.model), "the two model should be the same"
    os.remove(tmp.name)
    assert repr(ease) == str(ease)

def test_CFGAN():
    """Test of the CFGAN class
    """
    n_items = 3
    cfgan = CFGAN([n_items, 5, n_items], [n_items*2, 5, 1], alpha=.03, s_pm=.5, s_zr=.7)

    assert hasattr(cfgan.trainer, "generator")
    assert hasattr(cfgan.trainer, "discriminator")
    assert hasattr(cfgan.trainer, "s_pm")
    assert hasattr(cfgan.trainer, "s_zr")
    assert hasattr(cfgan.trainer, "alpha")
    assert hasattr(cfgan.trainer, "n_items")
    assert hasattr(cfgan.trainer, "opt_g")
    assert hasattr(cfgan.trainer, "opt_d")
    assert cfgan.trainer.s_pm == .5
    assert cfgan.trainer.s_zr == .7
    assert cfgan.trainer.alpha == .03
    assert cfgan.trainer.n_items == 3
    assert isinstance(cfgan.trainer.opt_d, torch.optim.Adam)
    assert isinstance(cfgan.trainer.opt_g, torch.optim.Adam)

    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    data = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)
    sampler = CFGAN_Sampler(data, batch_size=1)

    cfgan.train(sampler, valid_metric="ndcg@1", num_epochs=10, g_steps=1, d_steps=1, verbose=4)
    pred = cfgan.predict(torch.FloatTensor([[0, 1, 1], [1, 1, 0]]))[0]
    assert pred.shape == (2, 3)

    tmp = tempfile.NamedTemporaryFile()
    cfgan.save_model(tmp.name)

    cfgan2 = CFGAN.load_model(tmp.name)
    assert str(cfgan2) == repr(cfgan2)

def test_ADMM_Slim():
    """Test the ADMM_Slim class
    """
    slim = ADMM_Slim(lambda1=5.,
                     lambda2=1e3,
                     rho=1e5,
                     nn_constr=True,
                     l1_penalty=True,
                     item_bias=False)
    assert hasattr(slim, "lambda1"), "admm_slim should have the attribute lambda1"
    assert hasattr(slim, "lambda2"), "admm_slim should have the attribute lambda2"
    assert hasattr(slim, "rho"), "admm_slim should have the attribute rho"
    assert hasattr(slim, "l1_penalty"), "admm_slim should have the attribute l1_penalty"
    assert hasattr(slim, "nn_constr"), "admm_slim should have the attribute nn_constr"
    assert hasattr(slim, "item_bias"), "admm_slim should have the attribute item_bias"
    assert hasattr(slim, "model"), "sladmm_slimim should have the attribute model"
    assert slim.lambda1 == 5, "lambda1 should be 5"
    assert slim.lambda2 == 1e3, "lambda2 should be 1000"
    assert slim.rho == 1e5, "rho should be 10000"
    assert slim.nn_constr, "nn_constr should be True"
    assert slim.l1_penalty, "l1_penalty should be True"
    assert not slim.item_bias, "item_bias should be False"
    assert slim.model is None, "before the training the inner model should be None"
    assert repr(slim) == str(slim)

    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    cols = [0, 1, 1, 2, 2, 3, 3, 4, 4, 1]
    values = [1.] * len(cols)
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    uids = {i:i for i in range(len(set(rows)))}
    iids = {i:i for i in range(len(set(cols)))}
    data = Dataset(df_tr, None, (df_tr.copy(), df_tr.copy()), uids, iids)
    sampler = ArrayDummySampler(data)
    X = sampler.data_tr

    slim.train(sampler)
    assert isinstance(slim.model, torch.FloatTensor),\
        "after training the model should be a numpy matrix"
    pr = slim.predict([1, 3, 4], X[[1, 3, 4]])[0]
    assert pr.shape == (3, 5), "the shape of the prediction whould be 3 x 5"
    tmp = tempfile.NamedTemporaryFile()
    slim.save_model(tmp.name)
    slim2 = ADMM_Slim.load_model(tmp.name)
    assert torch.all(slim2.model == slim.model), "the two model should be the same"
    os.remove(tmp.name)
    assert repr(slim) == str(slim)

    slim2 = ADMM_Slim(nn_constr=False, l1_penalty=True, item_bias=False)
    slim2.train(sampler)
    slim2 = ADMM_Slim(nn_constr=True, l1_penalty=False, item_bias=False)
    slim2.train(sampler)
    slim2 = ADMM_Slim(nn_constr=False, l1_penalty=False, item_bias=False)
    slim2.train(sampler)
    slim2 = ADMM_Slim(nn_constr=False, l1_penalty=False, item_bias=True)
    slim2.train(sampler)


def test_SVAE():
    """Test the SVAE class
    """
    total_items = 7

    model = SVAE(n_items=total_items,
                   embed_size=2,
                   rnn_size=2,
                   dec_dims=[2, total_items],
                   enc_dims=[2, 2])

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model.trainer, "optimizer"), "model should have the attribute optimizer"
    #assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.trainer.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    values = [1.] * 19
    rows = [0] * 7 + [1] * 7 + [2] * 5
    cols = list(range(7)) + list(range(6, -1, -1)) + [2, 1, 6, 0, 3]
    tt = list(range(7)) + list(range(7)) + list(range(5))
    df_tr = pd.DataFrame(list(zip(rows, cols, values, tt)),
                         columns=['uid', 'iid', 'rating', 'time'])

    values = [1.] * 10
    rows = [0] * 4 + [1] * 4 + [2] * 2
    cols = [0, 1, 2, 3, 6, 5, 4, 3, 1, 6]
    tt = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    df_te_tr = pd.DataFrame(list(zip(rows, cols, values, tt)),
                            columns=['uid', 'iid', 'rating', 'time'])

    values = [1.] * 8
    rows = [0, 0, 0, 1, 1, 1, 2, 2]
    cols = [4, 5, 6, 2, 1, 0, 0, 3]
    tt = [0, 1, 2, 0, 1, 2, 0, 1]
    df_te_te = pd.DataFrame(list(zip(rows, cols, values, tt)),
                            columns=['uid', 'iid', 'rating', 'time'])

    uids = {0:0, 1:1, 2:2}
    iids = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
    data = Dataset(df_tr, (df_te_tr, df_te_te), (df_te_tr, df_te_te), uids, iids)

    sampler = SVAE_Sampler(data,
                           mode="train",
                           pred_type="next_k",
                           k=2,
                           shuffle=False)

    x = torch.LongTensor([[1, 2, 5]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, valid_metric="ndcg@1", verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name)
    model2 = SVAE.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"


def test_RecVAE():
    """Test the RecVAE class
    """
    model = RecVAE(2, 4, 2)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model.trainer, "opt_dec"), "model should have the attribute opt_dec"
    assert hasattr(model.trainer, "opt_enc"), "model should have the attribute opt_enc"
    #assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.trainer.opt_enc, torch.optim.Adam), "opt_enc should be of Adam type"
    assert isinstance(model.trainer.opt_dec, torch.optim.Adam), "opt_dec should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    #gt = torch.FloatTensor([[1, 1], [2, 1]])
    #pred = torch.FloatTensor([[1, 1], [1, 1]])

    sampler = create_sampler([0, 0, 1], [0, 1, 1])

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name)

    model2 = RecVAE.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    tmp2 = tempfile.NamedTemporaryFile()
    model = RecVAE(2, 4, 3, gamma=.03)
    model.train(sampler,
                valid_metric="ndcg@1",
                num_epochs=10)

    model.save_model(tmp2.name)
    model2 = RecVAE.load_model(tmp2.name)
    assert model2.trainer.current_epoch > 0
