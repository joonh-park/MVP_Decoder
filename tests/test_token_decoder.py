import torch

from model.token_decoder.attention import CompetitiveSlotAttention
from model.token_decoder.gaussian_head import SharedGaussianHead
from model.token_decoder.latent_split import LatentSplitter
from model.token_decoder.token_initializer import TokenInitializer
from model.token_decoder.token_refiner import TokenRefiner


def test_fixed_query_initialization_and_gaussian_readout():
    evidence = torch.randn(2, 40, 32)
    initializer = TokenInitializer(8, 32, 4, num_layers=2)
    head = SharedGaussianHead(32, 1, 4.0, -6.9, -1.2, -2.0)
    z = initializer(evidence)
    gaussians = head(z)
    assert z.shape == (2, 8, 32)
    assert gaussians.xyz.shape == (2, 8, 3)
    assert gaussians.feature.shape == (2, 8, 4, 3)
    assert gaussians.opacity.shape == (2, 8, 1)
    assert torch.all((gaussians.opacity > 0) & (gaussians.opacity < 1))


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


def test_configurable_cross_slot_initialization():
    evidence = torch.randn(2, 40, 32)
    initializer = TokenInitializer(
        8,
        32,
        4,
        layer_specs=[{"type": "cross"}, {"type": "slot", "repeat": 2}],
        query_chunk_size=3,
        evidence_chunk_size=7,
    )
    output = initializer(evidence)
    layer_types = [layer.attention_type for layer in initializer.stack.layers]
    assert output.shape == (2, 8, 32)
    assert layer_types == ["cross", "slot", "slot"]
    assert torch.isfinite(output).all()


def test_slot_attention_evidence_chunking_is_exact():
    slots = torch.randn(2, 7, 32)
    evidence = torch.randn(2, 19, 32)
    full = CompetitiveSlotAttention(32, 4, evidence_chunk_size=19).eval()
    chunked = CompetitiveSlotAttention(32, 4, evidence_chunk_size=5).eval()
    chunked.load_state_dict(full.state_dict())
    with torch.no_grad():
        expected = full(slots, evidence)
        actual = chunked(slots, evidence)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
