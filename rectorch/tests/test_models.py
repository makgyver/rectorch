"""Unit tests for the rectorch.models module
"""
import os
import sys
import tempfile
import pytest
import torch
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath('..'))

from models import RecSysModel, TorchNNTrainer, AETrainer, VAE, MultiDAE, MultiVAE, CMultiVAE,\
    EASE, CFGAN
from nets import MultiDAE_net, VAE_net, MultiVAE_net, CMultiVAE_net, CFGAN_D_net, CFGAN_G_net
from samplers import DataSampler, ConditionedDataSampler, CFGAN_TrainingSampler

def test_RecSysModel():
    """Test the RecSysModel class
    """
    model = RecSysModel()

    with pytest.raises(NotImplementedError):
        model.train(None)
        model.predict(None)
        model.save_model(None)
        model.load_model(None)

def test_TorchNNTrainer():
    """Test the TorchNNTrainer class
    """
    net = MultiDAE_net([1, 2], [2, 1], .1)
    model = TorchNNTrainer(net)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "learning_rate"), "model should have the attribute learning_rate"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.learning_rate == 1e-3, "the learning rate should be 1e-3"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert model.optimizer is None, "optimizer should be None"
    assert str(model) == repr(model)

    x = torch.FloatTensor([[1, 1], [2, 2]])
    with pytest.raises(NotImplementedError):
        model.loss_function(None, None)
        model.train(None, None)
        model.train_epoch(0, None)
        model.train_batch(0, None, None)
        model.predict(x)


def test_AETrainer():
    """Test the AETrainer class
    """
    net = MultiDAE_net([1, 2], [2, 1], .1)
    model = AETrainer(net)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "learning_rate"), "model should have the attribute learning_rate"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.learning_rate == 1e-3, "the learning rate should be 1e-3"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    gt = torch.FloatTensor([[1, 1], [2, 1]])
    pred = torch.FloatTensor([[1, 1], [1, 1]])
    assert model.loss_function(pred, gt) == torch.FloatTensor([.25]), "the loss should be .25"

    values = np.array([1., 1., 1.])
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 1])
    train = csr_matrix((values, (rows, cols)))
    sampler = DataSampler(train, batch_size=1, shuffle=False)

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name, 1)

    net = MultiDAE_net([1, 2], [2, 1], .1)
    model2 = AETrainer(net)
    model2.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    sampler = DataSampler(train, train, batch_size=1, shuffle=False)

    res = model.validate(sampler, "ndcg@1")
    assert isinstance(res, np.ndarray), "results should the be a numpy array"
    assert len(res) == 2, "results should be of length 2"

def test_VAE():
    """Test the VAE class
    """
    net = VAE_net([1, 2], [2, 1])
    model = VAE(net)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "learning_rate"), "model should have the attribute learning_rate"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.learning_rate == 1e-3, "the learning rate should be 1e-3"
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

    values = np.array([1., 1., 1.])
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 1])
    train = csr_matrix((values, (rows, cols)))
    sampler = DataSampler(train, batch_size=1, shuffle=False)

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name, 1)

    net = VAE_net([1, 2], [2, 1])
    model2 = VAE(net)
    model2.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    sampler = DataSampler(train, train, batch_size=1, shuffle=False)

    res = model.validate(sampler, "ndcg@1")
    assert isinstance(res, np.ndarray), "results should the be a numpy array"
    assert len(res) == 2, "results should be of length 2"

def test_MultiDAE():
    """Test the MultiDAE class
    """
    net = MultiDAE_net([1, 2], [2, 1], dropout=.1)
    model = MultiDAE(net)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "learning_rate"), "model should have the attribute learning_rate"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert hasattr(model, "lam"), "model should have the attribute lam"
    assert model.learning_rate == 1e-3, "the learning rate should be 1e-3"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert model.lam == .2, "lambda should be .2"
    assert isinstance(model.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    gt = torch.FloatTensor([[1, 1], [2, 1]])
    pred = torch.FloatTensor([[1, 1], [1, 1]])
    torch.manual_seed(12345)
    assert model.loss_function(pred, gt) != torch.FloatTensor([.0]),\
        "the loss should not be 0"

    values = np.array([1., 1., 1.])
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 1])
    train = csr_matrix((values, (rows, cols)))
    sampler = DataSampler(train, batch_size=1, shuffle=False)

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name, 1)

    net = MultiDAE_net([1, 2], [2, 1], dropout=.1)
    model2 = MultiDAE(net)
    model2.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    sampler = DataSampler(train, train, batch_size=1, shuffle=False)

    res = model.validate(sampler, "ndcg@1")
    assert isinstance(res, np.ndarray), "results should the be a numpy array"
    assert len(res) == 2, "results should be of length 2"

def test_MultiVAE():
    """Test the MultiVAE class
    """
    net = MultiVAE_net([1, 2], [2, 1], .1)
    model = MultiVAE(net)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "learning_rate"), "model should have the attribute learning_rate"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.learning_rate == 1e-3, "the learning rate should be 1e-3"
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

    values = np.array([1., 1., 1.])
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 1])
    train = csr_matrix((values, (rows, cols)))
    sampler = DataSampler(train, batch_size=1, shuffle=False)

    x = torch.FloatTensor([[1, 1], [2, 2]])
    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name, 1)

    net = MultiVAE_net([1, 2], [2, 1], .1)
    model2 = MultiVAE(net)
    model2.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    sampler = DataSampler(train, train, batch_size=1, shuffle=False)

    res = model.validate(sampler, "ndcg@1")
    assert isinstance(res, np.ndarray), "results should the be a numpy array"
    assert len(res) == 2, "results should be of length 2"

    tmp2 = tempfile.NamedTemporaryFile()
    net = MultiVAE_net([1, 2], [2, 1], .1)
    model = MultiVAE(net, 1., 5)
    model.train(sampler, sampler, "ndcg@1", 10, tmp2.name)

    net2 = MultiVAE_net([1, 2], [2, 1], .1)
    model2 = MultiVAE(net2, 1., 5)
    assert model2.gradient_updates == 0,\
        "after initialization there should not be any gradient updates"
    model2.load_model(tmp2.name)
    assert model2.gradient_updates > 0,\
        "the loaded model should have been saved after some gradient updates"


def test_CMultiVAE():
    """Test the CMultiVAE class
    """
    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))

    iid2cids = {0:[1], 1:[0, 1], 2:[0]}
    sampler = ConditionedDataSampler(iid2cids, 2, train, batch_size=2, shuffle=False)

    net = CMultiVAE_net(2, [1, 3], dropout=.1)
    model = CMultiVAE(net)

    assert hasattr(model, "network"), "model should have the attribute newtork"
    assert hasattr(model, "device"), "model should have the attribute device"
    assert hasattr(model, "learning_rate"), "model should have the attribute learning_rate"
    assert hasattr(model, "optimizer"), "model should have the attribute optimizer"
    assert model.learning_rate == 1e-3, "the learning rate should be 1e-3"
    assert model.network == net, "the network should be the same as the parameter"
    assert model.device == torch.device("cpu"), "the device should be cpu"
    assert isinstance(model.optimizer, torch.optim.Adam), "optimizer should be of Adam type"
    assert str(model) == repr(model), "repr and str should have the same effect"

    x = torch.FloatTensor([[1, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    gt = torch.FloatTensor([[1, 1, 1], [2, 1, 1]])
    pred = torch.FloatTensor([[1, 1, 1], [1, 1, 1]])
    torch.manual_seed(12345)
    mu, logvar = model.network.encode(x)
    pred = torch.sigmoid(pred)
    assert model.loss_function(pred, gt, mu, logvar) != torch.FloatTensor([.0]),\
        "the loss should not be 0"

    model.predict(x, True)
    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    model.train(sampler, num_epochs=10, verbose=4)
    torch.manual_seed(12345)
    out_2 = model.predict(x, False)[0]

    assert not torch.all(out_1.eq(out_2)), "the outputs should be different"

    tmp = tempfile.NamedTemporaryFile()
    model.save_model(tmp.name, 1)

    net = CMultiVAE_net(2, [1, 3], dropout=.1)
    model2 = CMultiVAE(net)
    model2.load_model(tmp.name)

    torch.manual_seed(12345)
    out_1 = model.predict(x, False)[0]
    torch.manual_seed(12345)
    out_2 = model2.predict(x, False)[0]
    assert torch.all(out_1.eq(out_2)), "the outputs should be the same"

    res = model.validate(sampler, "ndcg@1")
    assert isinstance(res, np.ndarray), "results should be a numpy array"
    assert len(res) == 6, "results should be of length 6"

    tmp2 = tempfile.NamedTemporaryFile()
    net = CMultiVAE_net(2, [1, 3], [3, 1], .1)
    model = CMultiVAE(net, 1., 5)
    model.train(sampler, sampler, "ndcg@1", 10, tmp2.name)

    net2 = CMultiVAE_net(2, [1, 3], [3, 1], .1)
    model2 = CMultiVAE(net2, 1., 5)
    assert model2.gradient_updates == 0,\
        "after initialization there should not be any gradient updates"
    model2.load_model(tmp2.name)
    assert model2.gradient_updates > 0,\
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

    X = csr_matrix(np.random.randint(2, size=(10, 5)), dtype="float64")
    ease.train(X)
    assert isinstance(ease.model, np.ndarray), "after training the model should be a numpy matrix"
    pr = ease.predict([2, 4, 5], X[[2, 4, 5]])[0]
    assert pr.shape == (3, 5), "the shape of the prediction whould be 3 x 5"
    tmp = tempfile.NamedTemporaryFile()
    ease.save_model(tmp.name)
    ease2 = EASE(200.)
    ease2.load_model(tmp.name + ".npy")
    assert np.all(ease2.model == ease.model), "the two model should be the same"
    os.remove(tmp.name + ".npy")
    assert repr(ease) == str(ease)

def test_CFGAN():
    n_items = 3
    gen = CFGAN_G_net([n_items, 5, n_items])
    disc = CFGAN_D_net([n_items*2, 5, 1])
    cfgan = CFGAN(gen, disc, alpha=.03, s_pm=.5, s_zr=.7, learning_rate=0.001)

    assert hasattr(cfgan, "generator")
    assert hasattr(cfgan, "discriminator")
    assert hasattr(cfgan, "s_pm")
    assert hasattr(cfgan, "s_zr")
    assert hasattr(cfgan, "loss")
    assert hasattr(cfgan, "alpha")
    assert hasattr(cfgan, "learning_rate")
    assert hasattr(cfgan, "n_items")
    assert hasattr(cfgan, "opt_g")
    assert hasattr(cfgan, "opt_d")
    assert cfgan.generator == gen
    assert cfgan.discriminator == disc
    assert cfgan.s_pm == .5
    assert cfgan.s_zr == .7
    assert cfgan.learning_rate == 1e-3
    assert cfgan.alpha == .03
    assert cfgan.n_items == 3
    assert isinstance(cfgan.loss, torch.nn.BCELoss)
    assert isinstance(cfgan.regularization_loss, torch.nn.MSELoss)
    assert isinstance(cfgan.opt_d, torch.optim.Adam)
    assert isinstance(cfgan.opt_g, torch.optim.Adam)

    values = np.array([1., 1., 1., 1.])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 1, 2])
    train = csr_matrix((values, (rows, cols)))
    sampler = CFGAN_TrainingSampler(train, 1)

    values = np.array([1.])
    rows = np.array([0])
    cols = np.array([0])
    val_tr = csr_matrix((values, (rows, cols)), shape=(1, 3))

    cols = np.array([1])
    val_te = csr_matrix((values, (rows, cols)), shape=(1, 3))

    vsampler = DataSampler(val_tr, val_te, batch_size=1, shuffle=False)
    cfgan.train(sampler, vsampler, "ndcg@1", 10, 1, 1, 4)

    tmp = tempfile.NamedTemporaryFile()
    cfgan.save_model(tmp.name, 10)

    gen2 = CFGAN_G_net([n_items, 5, n_items])
    disc2 = CFGAN_D_net([n_items*2, 5, 1])
    cfgan2 = CFGAN(gen2, disc2, alpha=.03, s_pm=.5, s_zr=.7, learning_rate=0.001)
    chkpt = cfgan2.load_model(tmp.name)
    assert chkpt["epoch"] == 10
    assert cfgan2.generator != gen
    assert cfgan2.discriminator != disc
    assert str(cfgan) == repr(cfgan)
