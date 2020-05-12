"""Unit tests for the rectorch.nets module
"""
import os
import sys
import pytest
import torch
sys.path.insert(0, os.path.abspath('..'))

from nets import AE_net, MultiDAE_net, MultiVAE_net, CMultiVAE_net, CFGAN_G_net, CFGAN_D_net

def test_AE_net():
    """Test the AE_net class
    """
    net = AE_net([1, 2], [2, 1])
    x = torch.FloatTensor([[1, 1], [2, 2]])

    with pytest.raises(NotImplementedError):
        net.encode(x)

    with pytest.raises(NotImplementedError):
        net.decode(x)

    with pytest.raises(NotImplementedError):
        net.forward(x)

    with pytest.raises(NotImplementedError):
        net.init_weights()

    assert hasattr(net, "enc_dims"), "Missing enc_dims attribute"
    assert hasattr(net, "dec_dims"), "Missing dec_dims attribute"

def test_MultiDAE_net():
    """Test the MultiDAE_net class
    """
    net = MultiDAE_net([1, 2], [2, 1], .1)
    x = torch.FloatTensor([[1, 1], [2, 2]])
    y = net(x)

    assert hasattr(net, "enc_dims"), "Missing enc_dims attribute"
    assert hasattr(net, "dec_dims"), "Missing dec_dims attribute"
    assert hasattr(net, "dropout"), "Missing dropout attribute"
    assert hasattr(net, "dec_layers"), "Missing dec_layers attribute"
    assert hasattr(net, "enc_layers"), "Missing end_layers attribute"
    assert isinstance(net.dropout, torch.nn.Dropout), "dropout must be a torch.nn.Dropout"
    assert net.dropout.p == .1, "dropout probability must be equal to .1"
    assert isinstance(y, torch.FloatTensor), "y should be a torch.FloatTensor"
    assert y.shape == x.shape, "The shape of x and y should be the same"


def test_MultiVAE_net():
    """Test the MultiVAE_net class
    """
    net = MultiVAE_net([1, 2], [2, 1], .1)
    x = torch.FloatTensor([[1, 1], [2, 2]])
    torch.manual_seed(98765)
    mu, logvar = net.encode(x)
    torch.manual_seed(98765)
    y, mu2, logvar2 = net(x)

    assert hasattr(net, "enc_dims"), "Missing enc_dims attribute"
    assert hasattr(net, "dec_dims"), "Missing dec_dims attribute"
    assert hasattr(net, "dropout"), "Missing dropout attribute"
    assert hasattr(net, "dec_layers"), "Missing dec_layers attribute"
    assert hasattr(net, "enc_layers"), "Missing end_layers attribute"
    assert isinstance(net.dropout, torch.nn.Dropout), "dropout must be a torch.nn.Dropout"
    assert net.dropout.p == .1, "dropout probability must be equal to .1"
    assert isinstance(y, torch.FloatTensor), "y should be a torch.FloatTensor"
    assert isinstance(mu, torch.FloatTensor), "mu should be a torch.FloatTensor"
    assert isinstance(logvar, torch.FloatTensor), "logvar should be a torch.FloatTensor"
    assert isinstance(mu2, torch.FloatTensor), "mu2 should be a torch.FloatTensor"
    assert isinstance(logvar2, torch.FloatTensor), "logvar2 should be a torch.FloatTensor"
    assert mu.equal(mu2), "mu and mu2 should be equal"
    assert logvar.equal(logvar2), "logvar and logvar2 should be equal"
    assert y.shape == x.shape, "The shape of x and y should be the same"


def test_CMultiVAE_net():
    """Test the CMultiVAE_net class
    """
    net = CMultiVAE_net(1, [1, 2], [2, 1], .1)
    x = torch.FloatTensor([[1, 1, 1], [2, 2, 0]])
    torch.manual_seed(98765)
    mu, logvar = net.encode(x)
    torch.manual_seed(98765)
    y, mu2, logvar2 = net(x)

    assert hasattr(net, "enc_dims"), "Missing enc_dims attribute"
    assert hasattr(net, "dec_dims"), "Missing dec_dims attribute"
    assert hasattr(net, "dropout"), "Missing dropout attribute"
    assert hasattr(net, "dec_layers"), "Missing dec_layers attribute"
    assert hasattr(net, "enc_layers"), "Missing end_layers attribute"
    assert isinstance(net.dropout, torch.nn.Dropout), "dropout must be a torch.nn.Dropout"
    assert net.dropout.p == .1, "dropout probability must be equal to .1"
    assert isinstance(y, torch.FloatTensor), "y should be a torch.FloatTensor"
    assert isinstance(mu, torch.FloatTensor), "mu should be a torch.FloatTensor"
    assert isinstance(logvar, torch.FloatTensor), "logvar should be a torch.FloatTensor"
    assert isinstance(mu2, torch.FloatTensor), "mu2 should be a torch.FloatTensor"
    assert isinstance(logvar2, torch.FloatTensor), "logvar2 should be a torch.FloatTensor"
    assert mu.equal(mu2), "mu and mu2 should be equal"
    assert logvar.equal(logvar2), "logvar and logvar2 should be equal"
    assert y.shape == torch.Size([2, 2]), "The shape of y should be torch.Size([2, 2])"


def test_CFGAN_G_net():
    """Test the CFGAN_G_net class
    """
    net = CFGAN_G_net([2, 3, 4])
    x = torch.FloatTensor([[1, 1], [2, 2]])
    y = net(x)

    assert hasattr(net, "latent_dim"), "Missing latent_dim attribute"
    assert hasattr(net, "input_dim"), "Missing input_dim attribute"
    assert hasattr(net, "layers_dim"), "Missing layers_dim attribute"
    assert net.latent_dim == 2, "latent_dim should be 2"
    assert net.input_dim == 4, "input_dim should be 4"
    assert net.layers_dim == [2, 3, 4], "layers_dim should be [2, 3, 4]"
    assert y.shape == torch.Size([2, 4]), "The shape of y should be torch.Size([2, 4])"


def test_CFGAN_D_net():
    """Test the CFGAN_D_net class
    """
    net = CFGAN_D_net([4, 3, 1])
    x = torch.FloatTensor([[1, 1], [2, 2]])
    y = net(x, x)

    assert hasattr(net, "input_dim"), "Missing input_dim attribute"
    assert hasattr(net, "layers_dim"), "Missing layers_dim attribute"
    assert net.input_dim == 4, "input_dim should be 4"
    assert net.layers_dim == [4, 3, 1], "layers_dim should be [4, 3, 1]"
    assert y.shape == torch.Size([2, 1]), "The shape of y should be torch.Size([2, 1])"
