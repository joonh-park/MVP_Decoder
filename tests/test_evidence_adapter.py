import torch

from model.token_decoder.evidence_adapter import EvidenceAdapter


def test_evidence_adapter_flattens_view_and_patch_axes():
    adapter = EvidenceAdapter(feature_dim=16, token_dim=32, geometry_dim=8)
    feature = torch.randn(2, 3, 4, 5, 16)
    ray = torch.randn(2, 3, 4, 5, 9)
    output = adapter(feature, ray)
    assert output.shape == (2, 3 * 4 * 5, 32)
    assert adapter.ray_encoder[-1].out_features == 8
    assert adapter.fusion_mlp[0].in_features == 40
