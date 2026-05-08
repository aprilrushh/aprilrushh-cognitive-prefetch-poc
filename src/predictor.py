"""
Cognitive Predictor — PoC v2의 핵심 발명품
============================================
Layer N의 Q vector로 Layer N+1에서 attention 받을 KV indices를 예측.
Universal Hopfield 3-Stage + Sparse Hopfield (α-entmax) + Multi-head Union.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch

from hopfield import (
    HopfieldConfig, UniversalHopfield, MultiUpdateHopfield, ProjectionResult,
)


@dataclass
class PredictionResult:
    predicted_indices: torch.Tensor
    union_size: int
    per_head_indices: list
    per_head_top1_scores: list
    per_head_top1_top2_gaps: list
    mean_top1_score: float
    min_top1_score: float
    confidence_level: str
    multi_update_triggered: bool = False
    multi_update_iterations: int = 1
    mean_sparsity: float = 0.0
    total_time_ms: float = 0.0
    similarity_time_ms: float = 0.0
    separation_time_ms: float = 0.0
    union_time_ms: float = 0.0
    used_full_prefetch_fallback: bool = False


class CognitivePredictor:
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    MEDIUM_CONFIDENCE_THRESHOLD = 0.75
    MULTI_UPDATE_TRIGGER_THRESHOLD = 0.85
    FALLBACK_TRIGGER_THRESHOLD = 0.50

    def __init__(self, config: Optional[HopfieldConfig] = None,
                 enable_multi_update: bool = True,
                 enable_full_prefetch_fallback: bool = True):
        if config is None:
            config = HopfieldConfig(
                similarity="dot", separation="entmax",
                beta=2.0, alpha=1.5, top_k=256,
            )
        self.config = config
        self.enable_multi_update = enable_multi_update
        self.enable_full_prefetch_fallback = enable_full_prefetch_fallback
        self.hopfield = UniversalHopfield(config)
        if enable_multi_update:
            self.multi_hopfield = MultiUpdateHopfield(
                config,
                confidence_threshold=self.MULTI_UPDATE_TRIGGER_THRESHOLD,
                max_iterations=3,
            )

    def predict(self, current_q: torch.Tensor, target_keys: torch.Tensor,
                target_values: Optional[torch.Tensor] = None,
                beta_override: Optional[float] = None) -> PredictionResult:
        import time
        start_ns = time.perf_counter_ns()
        batch_size, num_heads, head_dim = current_q.shape
        num_kv_heads = target_keys.shape[1]
        seq_len = target_keys.shape[2]
        heads_per_kv_group = num_heads // num_kv_heads

        per_head_indices = []
        per_head_top1_scores = []
        per_head_top1_gaps = []
        per_head_sparsity = []
        multi_update_count = 0

        for kv_head_idx in range(num_kv_heads):
            q_head_start = kv_head_idx * heads_per_kv_group
            q_head_end = q_head_start + heads_per_kv_group
            group_q = current_q[:, q_head_start:q_head_end, :].mean(dim=1)
            head_keys = target_keys[:, kv_head_idx, :, :]

            if self.enable_multi_update and target_values is not None:
                head_values = target_values[:, kv_head_idx, :, :]
                result = self.hopfield.retrieve(group_q, head_keys, beta_override=beta_override)
                if result.top1_score < self.MULTI_UPDATE_TRIGGER_THRESHOLD:
                    result = self.multi_hopfield.retrieve(
                        group_q, head_keys, head_values, beta_override=beta_override
                    )
                    multi_update_count += 1
            else:
                result = self.hopfield.retrieve(group_q, head_keys, beta_override=beta_override)

            per_head_indices.append(result.top_k_indices)
            per_head_top1_scores.append(result.top1_score)
            per_head_top1_gaps.append(result.top1_top2_gap)
            per_head_sparsity.append(result.sparsity)

        union_start_ns = time.perf_counter_ns()
        all_indices = torch.stack(per_head_indices, dim=0)
        all_indices = all_indices.permute(1, 0, 2)
        all_indices_flat = all_indices.reshape(batch_size, -1)

        if batch_size == 1:
            union_indices = torch.unique(all_indices_flat[0])
            union_indices = union_indices.unsqueeze(0)
        else:
            union_lists = [torch.unique(all_indices_flat[b]) for b in range(batch_size)]
            max_size = max(len(u) for u in union_lists)
            union_indices = torch.full(
                (batch_size, max_size), -1,
                dtype=torch.long, device=current_q.device
            )
            for b, u in enumerate(union_lists):
                union_indices[b, :len(u)] = u

        union_time_ms = (time.perf_counter_ns() - union_start_ns) / 1e6
        union_size = union_indices.shape[-1]

        mean_top1 = sum(per_head_top1_scores) / len(per_head_top1_scores)
        min_top1 = min(per_head_top1_scores)
        mean_sparsity = sum(per_head_sparsity) / len(per_head_sparsity)

        if mean_top1 >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
        elif mean_top1 >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        used_fallback = False
        if self.enable_full_prefetch_fallback and \
           min_top1 < self.FALLBACK_TRIGGER_THRESHOLD:
            full_indices = torch.arange(
                seq_len, device=current_q.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
            union_indices = full_indices
            union_size = seq_len
            used_fallback = True

        total_time_ms = (time.perf_counter_ns() - start_ns) / 1e6

        return PredictionResult(
            predicted_indices=union_indices,
            union_size=union_size,
            per_head_indices=per_head_indices,
            per_head_top1_scores=per_head_top1_scores,
            per_head_top1_top2_gaps=per_head_top1_gaps,
            mean_top1_score=mean_top1,
            min_top1_score=min_top1,
            confidence_level=confidence_level,
            multi_update_triggered=(multi_update_count > 0),
            multi_update_iterations=multi_update_count,
            mean_sparsity=mean_sparsity,
            total_time_ms=total_time_ms,
            union_time_ms=union_time_ms,
            used_full_prefetch_fallback=used_fallback,
        )


@dataclass
class HitRateMetrics:
    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    weighted_hit_rate: float
    actual_top_k_size: int
    predicted_size: int
    intersection_size: int


def compute_hit_rate(predicted_indices: torch.Tensor,
                     actual_attention: torch.Tensor,
                     top_k: int = 32) -> HitRateMetrics:
    if actual_attention.ndim == 3:
        actual_attention = actual_attention.mean(dim=1)
    batch_size = actual_attention.shape[0]
    actual_top_k = actual_attention.topk(k=top_k, dim=-1).indices
    actual_top_k_weights = actual_attention.topk(k=top_k, dim=-1).values

    hit_rates, recalls, precisions, weighted_hits = [], [], [], []
    intersection_sizes, predicted_sizes = [], []

    for b in range(batch_size):
        pred = set(predicted_indices[b].cpu().tolist())
        pred = pred - {-1}
        actual = set(actual_top_k[b].cpu().tolist())
        intersection = pred & actual
        hit_rate = len(intersection) / top_k if top_k > 0 else 0.0
        recall = len(intersection) / len(actual) if actual else 0.0
        precision = len(intersection) / len(pred) if pred else 0.0
        actual_indices = actual_top_k[b].cpu().tolist()
        actual_weights = actual_top_k_weights[b].cpu().tolist()
        weighted = sum(w for idx, w in zip(actual_indices, actual_weights) if idx in pred)
        hit_rates.append(hit_rate)
        recalls.append(recall)
        precisions.append(precision)
        weighted_hits.append(weighted)
        intersection_sizes.append(len(intersection))
        predicted_sizes.append(len(pred))

    return HitRateMetrics(
        hit_rate_at_k=sum(hit_rates) / len(hit_rates),
        recall_at_k=sum(recalls) / len(recalls),
        precision_at_k=sum(precisions) / len(precisions),
        weighted_hit_rate=sum(weighted_hits) / len(weighted_hits),
        actual_top_k_size=top_k,
        predicted_size=sum(predicted_sizes) // len(predicted_sizes),
        intersection_size=sum(intersection_sizes) // len(intersection_sizes),
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Cognitive Predictor smoke test (Llama 70B simulation)")
    print("=" * 70)
    torch.manual_seed(42)

    # Llama 70B architecture
    batch_size = 1
    num_heads = 64
    num_kv_heads = 8
    head_dim = 128
    seq_len = 32_768

    print(f"\n  Model: Llama 70B (heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim})")
    print(f"  Context: {seq_len:,}")
    print(f"  GPU: ", "cuda" if torch.cuda.is_available() else "cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_q = torch.randn(batch_size, num_heads, head_dim, device=device) * 0.5
    target_keys = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device) * 0.5
    target_values = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device) * 0.5

    predictor = CognitivePredictor()
    print("\n  β scheduling effect:")
    print(f"  {'β':<8} {'Mean top1':<12} {'Min top1':<12} {'Union':<10} {'Save'}")
    print("  " + "-" * 50)
    for beta in [1.0, 2.0, 5.0, 10.0]:
        result = predictor.predict(current_q, target_keys, target_values, beta_override=beta)
        savings = 100 * (1 - result.union_size / seq_len)
        print(f"  {beta:<8.1f} {result.mean_top1_score:<12.4f} "
              f"{result.min_top1_score:<12.4f} {result.union_size:<10,} {savings:.1f}%")

    print("\n✓ Done")
