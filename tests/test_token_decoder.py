import torch

from model.token_decoder.gaussian_head import SharedGaussianHead
from model.token_decoder.latent_split import LatentSplitter
from model.token_decoder.token_initializer import TokenInitializer
from model.token_decoder.token_refiner import TokenRefiner


def test_fixed_query_initialization_and_gaussian_readout():
    evidence = torch.randn(2, 40, 32)
    initializer = TokenInitializer(8, 32, 4, num_layers=2)
    head = SharedGaussianHead(32, 1, 2, 4.0, -6.9, -1.2, -2.0)
    z = initializer(evidence)
    gaussians = head(z)
    assert z.shape == (2, 8, 32)
    assert gaussians.xyz.shape == (2, 8, 3)
    assert gaussians.feature.shape == (2, 8, 4, 3)
    assert gaussians.opacity.shape == (2, 8, 9, 1)


def test_dense_split_and_refinement_shapes():
    z = torch.randn(2, 8, 32)
    evidence = torch.randn(2, 40, 32)
    splitter = LatentSplitter(32, 4)
    refiner = TokenRefiner(32, 4, num_layers=4)
    split = splitter(z, evidence, dense=True)
    refined = refiner(split.latent, evidence)
    assert split.latent.shape == (2, 16, 32)
    assert split.parent_index.shape == (2, 16)
    assert refined.shape == (2, 16, 32)
