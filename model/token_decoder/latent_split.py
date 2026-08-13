import torch
from torch import nn

from model.token_decoder.attention import CrossAttentionBlock
from model.token_decoder.types import SplitOutput


class LatentSplitter(nn.Module):
    """Dense-train latent splitter; adaptive selection can reuse its score head."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        query_chunk_size: int = 0,
    ):
        super().__init__()
        self.conditioner = CrossAttentionBlock(
            dim,
            num_heads,
            mlp_ratio,
            dropout=dropout,
            query_chunk_size=query_chunk_size,
        )
        self.norm = nn.LayerNorm(dim)
        self.score_head = nn.Linear(dim, 1)
        self.child_delta = nn.Linear(dim, 2 * dim)
        self.child_embedding = nn.Parameter(torch.empty(2, dim))
        nn.init.normal_(self.child_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        z: torch.Tensor,
        error_evidence: torch.Tensor,
        dense: bool = True,
        threshold: float = 0.5,
    ) -> SplitOutput:
        conditioned = self.conditioner(z, error_evidence)
        normalized = self.norm(conditioned)
        scores = self.score_head(normalized).squeeze(-1)
        delta = self.child_delta(normalized).view(*z.shape[:2], 2, z.shape[-1])
        children = z.unsqueeze(2) + delta + self.child_embedding.view(1, 1, 2, -1)

        if dense:
            latent = children.flatten(1, 2)
            parent_index = torch.arange(z.shape[1], device=z.device).repeat_interleave(2)
            parent_index = parent_index.unsqueeze(0).expand(z.shape[0], -1)
            return SplitOutput(latent=latent, scores=scores, parent_index=parent_index)

        split_mask = scores.sigmoid() >= threshold
        if not torch.all(split_mask.sum(dim=1) == split_mask.sum(dim=1)[0]):
            raise ValueError("Adaptive split currently requires equal output counts within a batch")
        outputs = []
        parent_indices = []
        for batch_index in range(z.shape[0]):
            keep = ~split_mask[batch_index]
            split = split_mask[batch_index]
            outputs.append(
                torch.cat([z[batch_index, keep], children[batch_index, split].flatten(0, 1)], dim=0)
            )
            indices = torch.arange(z.shape[1], device=z.device)
            parent_indices.append(
                torch.cat([indices[keep], indices[split].repeat_interleave(2)], dim=0)
            )
        return SplitOutput(
            latent=torch.stack(outputs),
            scores=scores,
            parent_index=torch.stack(parent_indices),
        )
