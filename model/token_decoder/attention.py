from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, length, dim = x.shape
    return x.view(batch, length, num_heads, dim // num_heads).transpose(1, 2)


class CrossAttention(nn.Module):
    """Standard Q-to-KV cross-attention with optional query chunking."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        query_chunk_size: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.dropout = dropout
        self.query_chunk_size = query_chunk_size
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _attend(self, q, k, v):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )

    def forward(self, query: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.q_proj(query), self.num_heads)
        k = _split_heads(self.k_proj(evidence), self.num_heads)
        v = _split_heads(self.v_proj(evidence), self.num_heads)
        chunk_size = self.query_chunk_size
        if chunk_size > 0 and q.shape[-2] > chunk_size:
            output = torch.cat(
                [self._attend(chunk, k, v) for chunk in q.split(chunk_size, dim=-2)],
                dim=-2,
            )
        else:
            output = self._attend(q, k, v)
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.out_proj(output)


class SelfCrossAttention(nn.Module):
    """Update queries from joint query/evidence KV with separate projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        query_chunk_size: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.dropout = dropout
        self.query_chunk_size = query_chunk_size
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.query_k_proj = nn.Linear(dim, dim, bias=bias)
        self.query_v_proj = nn.Linear(dim, dim, bias=bias)
        self.evidence_k_proj = nn.Linear(dim, dim, bias=bias)
        self.evidence_v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _attend(self, q, k, v):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )

    def forward(self, query: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.q_proj(query), self.num_heads)
        query_k = _split_heads(self.query_k_proj(query), self.num_heads)
        query_v = _split_heads(self.query_v_proj(query), self.num_heads)
        evidence_k = _split_heads(self.evidence_k_proj(evidence), self.num_heads)
        evidence_v = _split_heads(self.evidence_v_proj(evidence), self.num_heads)
        k = torch.cat((query_k, evidence_k), dim=-2)
        v = torch.cat((query_v, evidence_v), dim=-2)

        chunk_size = self.query_chunk_size
        if chunk_size > 0 and q.shape[-2] > chunk_size:
            output = torch.cat(
                [self._attend(chunk, k, v) for chunk in q.split(chunk_size, dim=-2)],
                dim=-2,
            )
        else:
            output = self._attend(q, k, v)
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.out_proj(output)


class CompetitiveSlotAttention(nn.Module):
    """Cross-attention where each evidence token competitively selects a slot.

    The normalization matches C3G's EKSA/slot-style routing: normalize over
    slots for each evidence token, then normalize each slot over its assigned
    evidence. Evidence chunks are accumulated exactly, so the full
    [slot, evidence] attention matrix is never materialized.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        evidence_chunk_size: int = 1024,
        epsilon: float = 1e-8,
        null_slot: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if evidence_chunk_size <= 0:
            raise ValueError("evidence_chunk_size must be positive")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.evidence_chunk_size = evidence_chunk_size
        self.epsilon = epsilon
        self.null_slot = null_slot
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _assign(self, logits: torch.Tensor) -> torch.Tensor:
        max_logits = logits.amax(dim=-2, keepdim=True).detach()
        exp_logits = (logits - max_logits).exp()
        denominator = exp_logits.sum(dim=-2, keepdim=True)
        if self.null_slot:
            denominator = denominator + (-max_logits).exp()
        return exp_logits / (denominator + self.epsilon)

    def forward(self, slots: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.q_proj(slots), self.num_heads)
        k = _split_heads(self.k_proj(evidence), self.num_heads)
        v = _split_heads(self.v_proj(evidence), self.num_heads)
        scale = self.head_dim**-0.5

        update_sum = torch.zeros_like(q)
        weight_sum = torch.zeros_like(q[..., :1])
        for k_chunk, v_chunk in zip(
            k.split(self.evidence_chunk_size, dim=-2),
            v.split(self.evidence_chunk_size, dim=-2),
        ):
            logits = torch.matmul(q, k_chunk.transpose(-1, -2)) * scale
            assignment = self._assign(logits)
            if self.training and self.dropout > 0:
                assignment = F.dropout(assignment, p=self.dropout)
            update_sum = update_sum + torch.matmul(assignment, v_chunk)
            weight_sum = weight_sum + assignment.sum(dim=-1, keepdim=True)

        output = update_sum / (weight_sum + self.epsilon)
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm evidence-to-token update; evidence is never updated."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_type: str = "cross",
        dropout: float = 0.0,
        query_chunk_size: int = 0,
        evidence_chunk_size: int = 1024,
        slot_epsilon: float = 1e-8,
        slot_null: bool = True,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.query_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        if attention_type == "cross":
            self.cross_attn = CrossAttention(
                dim,
                num_heads,
                dropout=dropout,
                query_chunk_size=query_chunk_size,
            )
        elif attention_type == "self_cross":
            self.cross_attn = SelfCrossAttention(
                dim,
                num_heads,
                dropout=dropout,
                query_chunk_size=query_chunk_size,
            )
        elif attention_type == "slot":
            self.cross_attn = CompetitiveSlotAttention(
                dim,
                num_heads,
                dropout=dropout,
                evidence_chunk_size=evidence_chunk_size,
                epsilon=slot_epsilon,
                null_slot=slot_null,
            )
        else:
            raise ValueError(
                f"Unknown attention type '{attention_type}'; expected "
                "'cross', 'self_cross', or 'slot'"
            )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, z: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        z = z + self.cross_attn(self.query_norm(z), self.evidence_norm(evidence))
        z = z + self.ffn(self.ffn_norm(z))
        return z


def build_attention_layers(
    layer_specs: Sequence[str | Mapping],
    *,
    dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    query_chunk_size: int,
    evidence_chunk_size: int,
    slot_epsilon: float,
    slot_null: bool,
) -> nn.ModuleList:
    """Build an ordered attention stack from compact config specifications."""

    layers = []
    for raw_spec in layer_specs:
        spec = {"type": raw_spec} if isinstance(raw_spec, str) else dict(raw_spec)
        attention_type = spec.pop("type")
        repeat = int(spec.pop("repeat", 1))
        if repeat <= 0:
            raise ValueError(f"Layer repeat must be positive, got {repeat}")
        settings = {
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "query_chunk_size": query_chunk_size,
            "evidence_chunk_size": evidence_chunk_size,
            "slot_epsilon": slot_epsilon,
            "slot_null": slot_null,
        }
        unknown = set(spec) - set(settings)
        if unknown:
            raise ValueError(f"Unknown layer settings: {sorted(unknown)}")
        settings.update(spec)
        layers.extend(
            CrossAttentionBlock(
                dim=dim,
                attention_type=attention_type,
                **settings,
            )
            for _ in range(repeat)
        )
    if not layers:
        raise ValueError("At least one decoder layer must be configured")
    return nn.ModuleList(layers)


class TokenLayerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_specs: Sequence[str | Mapping],
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        query_chunk_size: int = 0,
        evidence_chunk_size: int = 1024,
        slot_epsilon: float = 1e-8,
        slot_null: bool = True,
    ):
        super().__init__()
        self.layers = build_attention_layers(
            layer_specs,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            query_chunk_size=query_chunk_size,
            evidence_chunk_size=evidence_chunk_size,
            slot_epsilon=slot_epsilon,
            slot_null=slot_null,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, evidence)
        return self.output_norm(z)
