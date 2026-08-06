import torch

from model.token_decoder.evidence_adapter import EvidenceAdapter


def test_evidence_adapter_flattens_view_and_patch_axes():
    adapter = EvidenceAdapter(feature_dim=16, token_dim=32)
    feature = torch.randn(2, 3, 4, 5, 16)
    ray = torch.randn(2, 3, 4, 5, 9)
    output = adapter(feature, ray)
    assert output.shape == (2, 3 * 4 * 5, 32)
