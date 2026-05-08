"""
Universal Hopfield Network (Millidge et al., ICML 2022)
=========================================================

Modern Hopfield retrieval 을 3-stage framework 로 명시 분해.
Brain Bet Ledger v0.2 의 T1-2 lens 구현.

3-Stage 분해:
    Stage 1 (Similarity):  Q · K^T
    Stage 2 (Separation):  β · softmax 또는 α-entmax (T1-1 Sparse Hopfield)
    Stage 3 (Projection):  Top-K selection
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Optional
import torch
import torch.nn.functional as F


# Stage 1: Similarity Functions
def similarity_dot(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q.unsqueeze(-2), k.transpose(-2, -1)).squeeze(-2)

def similarity_cosine(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q_norm = F.normalize(q, dim=-1)
    k_norm = F.normalize(k, dim=-1)
    return similarity_dot(q_norm, k_norm)

def similarity_euclidean(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    diff = q.unsqueeze(-2) - k
    return -torch.norm(diff, p=2, dim=-1)

SIMILARITY_FUNCTIONS: dict[str, Callable] = {
    "dot": similarity_dot,
    "cosine": similarity_cosine,
    "euclidean": similarity_euclidean,
}


# Stage 2: Separation Functions
def separation_softmax(scores: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.softmax(beta * scores, dim=-1)

def separation_entmax(scores: torch.Tensor, beta: float = 1.0,
                       alpha: float = 1.5) -> torch.Tensor:
    """α-entmax separation (Hu et al. NeurIPS 2023, Sparse Modern Hopfield)."""
    try:
        from entmax import entmax15, sparsemax
    except ImportError:
        return separation_softmax(scores, beta)
    if alpha == 1.5:
        return entmax15(beta * scores, dim=-1)
    elif alpha == 2.0:
        return sparsemax(beta * scores, dim=-1)
    else:
        from entmax import entmax_bisect
        return entmax_bisect(beta * scores, alpha=alpha, dim=-1)

SEPARATION_FUNCTIONS: dict[str, Callable] = {
    "softmax": separation_softmax,
    "entmax": separation_entmax,
    "entmax15": separation_entmax,
    "sparsemax": lambda s, beta=1.0: separation_entmax(s, beta, alpha=2.0),
}


# Stage 3: Projection
@dataclass
class ProjectionResult:
    weights: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_scores: torch.Tensor
    top1_score: float
    top1_top2_gap: float
    sparsity: float


def projection_topk(weights: torch.Tensor, top_k: int) -> ProjectionResult:
    top_k_result = weights.topk(k=min(top_k, weights.shape[-1]), dim=-1)
    top1 = top_k_result.values[..., 0].mean().item() if weights.ndim > 0 else \
           top_k_result.values[0].item()
    if top_k_result.values.shape[-1] >= 2:
        top2 = top_k_result.values[..., 1].mean().item() if weights.ndim > 0 else \
               top_k_result.values[1].item()
        gap = top1 - top2
    else:
        gap = top1
    nonzero_mask = (weights > 1e-8).float()
    sparsity = nonzero_mask.mean().item()
    return ProjectionResult(
        weights=weights,
        top_k_indices=top_k_result.indices,
        top_k_scores=top_k_result.values,
        top1_score=top1,
        top1_top2_gap=gap,
        sparsity=sparsity,
    )


# Universal Hopfield Retrieval
@dataclass
class HopfieldConfig:
    similarity: Literal["dot", "cosine", "euclidean"] = "dot"
    separation: Literal["softmax", "entmax", "sparsemax"] = "entmax"
    beta: float = 1.0
    alpha: float = 1.5
    top_k: int = 128


class UniversalHopfield:
    def __init__(self, config: HopfieldConfig):
        self.config = config
        self.similarity_fn = SIMILARITY_FUNCTIONS[config.similarity]
        self.separation_fn = SEPARATION_FUNCTIONS[config.separation]

    def retrieve(self, query: torch.Tensor, keys: torch.Tensor,
                 beta_override: Optional[float] = None) -> ProjectionResult:
        beta = beta_override if beta_override is not None else self.config.beta
        scores = self.similarity_fn(query, keys)
        if self.config.separation == "entmax":
            weights = self.separation_fn(scores, beta=beta, alpha=self.config.alpha)
        else:
            weights = self.separation_fn(scores, beta=beta)
        return projection_topk(weights, top_k=self.config.top_k)


class MultiUpdateHopfield(UniversalHopfield):
    """2B lens: Multi-update retrieval — fallback for hard queries."""
    def __init__(self, config: HopfieldConfig,
                 confidence_threshold: float = 0.85,
                 max_iterations: int = 3):
        super().__init__(config)
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations

    def retrieve(self, query: torch.Tensor, keys: torch.Tensor,
                 values: Optional[torch.Tensor] = None,
                 beta_override: Optional[float] = None) -> ProjectionResult:
        current_query = query
        result = None
        for iteration in range(self.max_iterations):
            result = super().retrieve(current_query, keys, beta_override)
            if result.top1_score >= self.confidence_threshold:
                break
            if values is not None:
                retrieved_value = torch.matmul(
                    result.weights.unsqueeze(-2), values
                ).squeeze(-2)
                current_query = retrieved_value
            else:
                break
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Hopfield smoke test")
    print("=" * 60)
    torch.manual_seed(42)
    D, N = 128, 1024
    query = torch.randn(D)
    keys = torch.randn(N, D)
    config = HopfieldConfig(separation="entmax", beta=2.0, top_k=128)
    h = UniversalHopfield(config)
    result = h.retrieve(query, keys)
    print(f"  Top-1: {result.top1_score:.4f}")
    print(f"  Gap:   {result.top1_top2_gap:.4f}")
    print(f"  Sparsity: {result.sparsity:.4f}")
    print("✓ Done")
