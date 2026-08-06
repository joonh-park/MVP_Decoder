from abc import ABC, abstractmethod

from torch import nn

from model.backbones.evidence_output import EvidenceOutput


class EvidenceBackbone(nn.Module, ABC):
    """Stable interface shared by posed and future unposed backbones."""

    @abstractmethod
    def forward(self, images, intrinsics, c2w) -> EvidenceOutput:
        raise NotImplementedError
