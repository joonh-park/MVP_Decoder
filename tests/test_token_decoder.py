import torch

from model.token_decoder.attention import CompetitiveSlotAttention
from model.token_decoder.gaussian_head import SharedGaussianHead
from model.token_decoder.latent_split import LatentSplitter
from model.token_decoder.token_initializer import TokenInitializer
from model.token_decoder.token_refiner import TokenRefiner


def test_fixed_query_initialization_and_gaussian_readout():
    evidence = torch.randn(2, 40, 32)
    initializer = TokenInitializer(8, 32, 4, num_layers=2)
    head = SharedGaussianHead(32, 1, 4.0, 0.001, 0.03, -2.0)
    z = initializer(evidence)
    gaussians = head(z)
    assert z.shape == (2, 8, 32)
    assert gaussians.xyz.shape == (2, 8, 3)
    assert gaussians.feature.shape == (2, 8, 4, 3)
    assert gaussians.opacity.shape == (2, 8, 1)
    assert torch.all((gaussians.opacity > 0) & (gaussians.opacity < 1))


def test_gt_ray_gaussian_positions_stay_on_pose_conditioned_ray():
    head = SharedGaussianHead(
        32,
        1,
        4.0,
        0.001,
        0.03,
        -2.0,
        position_mode="gt_ray",
        depth_min=0.1,
        depth_max=5.0,
        depth_init=2.0,
        anchor_dim=8,
        anchor_query_chunk_size=3,
    )
    z = torch.randn(2, 7, 32)
    evidence = torch.randn(2, 1, 32)
    center_ray = torch.tensor(
        [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, -1.0, 0.0]
    ).view(1, 1, 1, 1, 9).expand(2, -1, -1, -1, -1)

    gaussians = head(z, evidence=evidence, center_ray=center_ray)

    assert torch.allclose(gaussians.xyz[..., 0], torch.ones(2, 7), atol=1e-6)
    assert torch.allclose(gaussians.xyz[..., 1], torch.full((2, 7), 2.0), atol=1e-6)
    depth = gaussians.xyz[..., 2] - 3.0
    assert torch.all((depth > 0.1) & (depth < 5.0))
    assert gaussians.scale.exp().max() <= 0.03


def test_c3g_scale_parameterization_and_clamping():
    head = SharedGaussianHead(32, 0, 4.0, 0.001, 0.03, -2.0)
    z = torch.randn(2, 8, 32)

    gaussians = head(z)

    actual_scale = gaussians.scale.exp()
    assert torch.all(actual_scale > 0)
    assert actual_scale.max() <= 0.03


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
