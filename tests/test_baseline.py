"""Unit tests for the rectorch.models.baseline module
"""
import os
import sys
import tempfile
import torch
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath('..'))

from rectorch.models.baseline import Random, Popularity

def test_random():
    """Test the Random class
    """
    rnd = Random(4, 1, False)
    assert rnd.n_items == 4
    assert rnd.seed == 1
    assert not rnd.fixed
    rnd.train()
    p = rnd.predict([1, 2, 3], [[0], [1], [2]])
    assert isinstance(p, torch.torch.FloatTensor)
    assert p.shape == torch.Size([3, 4])
    assert p[0, 0] == -np.inf
    assert p[1, 1] == -np.inf
    assert p[1, 0] != -np.inf
    assert p[0, 3] != p[1, 3]

    rnd = Random(4, 1, True)
    rnd.train()
    p = rnd.predict([1, 2, 3], [[0], [1], [2]])
    assert isinstance(p, torch.torch.FloatTensor)
    assert p.shape == torch.Size([3, 4])
    assert p[0, 0] == -np.inf
    assert p[1, 1] == -np.inf
    assert p[1, 0] != -np.inf
    assert p[0, 3] == p[1, 3]

    tmp = tempfile.NamedTemporaryFile()
    rnd.save_model(tmp.name)

    rnd2 = Random(5, 999, False)
    assert rnd2.n_items == 5
    assert rnd2.seed == 999
    assert not rnd2.fixed
    rnd2.load_model(tmp.name)
    assert rnd2.n_items == 4
    assert rnd2.seed == 1
    assert rnd2.fixed

def test_popularity():
    """Test the Popularity class
    """
    pop = Popularity(4)
    assert pop.n_items == 4
    assert pop.model is None

    R = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1]]
    t = torch.FloatTensor(R)
    pop = Popularity(4)
    pop.train(t)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    t = torch.sparse.torch.FloatTensor(R)
    pop = Popularity(4)
    pop.train(t)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.torch.FloatTensor([2, 3, 1, 1]))
    old_model = pop.model
    t[0, 2] = 1.0
    pop.train(t)
    assert torch.all(pop.model == old_model)

    t = csr_matrix(R)
    pop = Popularity(4)
    pop.train(t)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    t = {0:[0, 1], 1:[0, 1, 2], 2:[1, 3]}
    pop = Popularity(4)
    pop.train(t)
    assert isinstance(pop.model, torch.FloatTensor)
    assert torch.all(pop.model == torch.FloatTensor([2, 3, 1, 1]))

    p = pop.predict([0, 1, 2], [[1], [2], [3]])
    assert isinstance(p, torch.FloatTensor)
    assert p.shape == torch.Size([3, 4])
    assert p[0, 1] == -np.inf
    assert p[1, 2] == -np.inf
    assert p[2, 1] != -np.inf
    assert p[0, 3] == p[1, 3]

    tmp = tempfile.NamedTemporaryFile()
    pop.save_model(tmp.name)

    pop2 = Popularity(4)
    pop2.load_model(tmp.name)
    assert torch.all(pop2.model == pop.model)
