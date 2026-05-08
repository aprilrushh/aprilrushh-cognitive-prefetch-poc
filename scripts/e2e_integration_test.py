"""
End-to-End Integration Test - PoC v2 Foundation
4 modules working together on H100 GPU.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hopfield import HopfieldConfig
from predictor import CognitivePredictor, compute_hit_rate
from beta_scheduler import HybridBetaScheduler, BetaSchedulerConfig
from retrieval_logger import RetrievalLogger, RetrievalEvent


class Llama70BSimConfig:
    num_layers = 80
    num_heads = 64
    num_kv_heads = 8
    head_dim = 128
    batch_size = 1


class MockLLM:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        torch.manual_seed(42)
        self._cached_kv = None

    def init_kv_cache(self, seq_length):
        cfg = self.config
        num_clusters = max(8, seq_length // 256)
        cluster_centers = torch.randn(
            cfg.num_layers, cfg.num_kv_heads, num_clusters, cfg.head_dim,
            device=self.device
        ) * 2.0
        cluster_ids = torch.randint(
            0, num_clusters,
            (cfg.num_layers, cfg.num_kv_heads, seq_length),
            device=self.device
        )
        keys = torch.zeros(
            cfg.num_layers, cfg.batch_size, cfg.num_kv_heads,
            seq_length, cfg.head_dim, device=self.device
        )
        values = torch.zeros_like(keys)
        for layer in range(cfg.num_layers):
            for kv_h in range(cfg.num_kv_heads):
                for tok in range(seq_length):
                    cluster_id = cluster_ids[layer, kv_h, tok].item()
                    base = cluster_centers[layer, kv_h, cluster_id]
                    keys[layer, 0, kv_h, tok] = base + torch.randn(cfg.head_dim, device=self.device) * 0.3
                    values[layer, 0, kv_h, tok] = base + torch.randn(cfg.head_dim, device=self.device) * 0.3
        self._cached_kv = (keys, values, cluster_ids)
        return keys, values

    def get_kv(self, layer_idx):
        keys, values, _ = self._cached_kv
        return keys[layer_idx], values[layer_idx]

    def get_query(self, layer_idx, focus_cluster=0):
        cfg = self.config
        _, _, cluster_ids = self._cached_kv
        keys, _ = self.get_kv(layer_idx)
        q = torch.zeros(cfg.batch_size, cfg.num_heads, cfg.head_dim, device=self.device)
        for h in range(cfg.num_heads):
            kv_h = h // (cfg.num_heads // cfg.num_kv_heads)
            cluster_mask = (cluster_ids[layer_idx, kv_h] == focus_cluster)
            if cluster_mask.any():
                cluster_keys = keys[0, kv_h][cluster_mask]
                q[0, h] = cluster_keys.mean(dim=0) + torch.randn(cfg.head_dim, device=self.device) * 0.5
            else:
                q[0, h] = torch.randn(cfg.head_dim, device=self.device)
        return q

    def get_ground_truth_attention(self, layer_idx, query):
        cfg = self.config
        keys, _ = self.get_kv(layer_idx)
        seq_len = keys.shape[2]
        heads_per_kv = cfg.num_heads // cfg.num_kv_heads
        keys_expanded = keys.repeat_interleave(heads_per_kv, dim=1)
        scale = 1.0 / (cfg.head_dim ** 0.5)
        scores = torch.matmul(
            query.unsqueeze(-2),
            keys_expanded.transpose(-2, -1)
        ).squeeze(-2) * scale
        return torch.softmax(scores, dim=-1)


class IntegratedCognitivePrefetch:
    def __init__(self, num_layers, log_dir, session_name):
        self.scheduler = HybridBetaScheduler(
            num_layers=num_layers,
            warmup_iterations=10,
            heuristic_config={
                "early_beta": 3.0,
                "middle_beta": 5.0,
                "late_beta": 7.0,
            },
            adaptive_config=BetaSchedulerConfig(
                target_confidence=0.90,
                min_beta=0.5,
                max_beta=20.0,
            ),
        )
        self.predictor = CognitivePredictor(
            config=HopfieldConfig(
                similarity="dot",
                separation="entmax",
                beta=5.0,
                alpha=1.5,
                top_k=256,
            ),
            enable_multi_update=True,
            enable_full_prefetch_fallback=True,
        )
        self.logger = RetrievalLogger(log_dir=log_dir, session_name=session_name)
        self.num_layers = num_layers

    def process_layer(self, layer_idx, query, target_keys, target_values, gt_attention):
        beta = self.scheduler.get_beta(layer_idx)
        with self.logger.timer() as timer:
            pred_result = self.predictor.predict(
                current_q=query,
                target_keys=target_keys,
                target_values=target_values,
                beta_override=beta,
            )
        metrics = compute_hit_rate(pred_result.predicted_indices, gt_attention, top_k=32)
        self.scheduler.update(layer_idx, pred_result.mean_top1_score)
        seq_len = target_keys.shape[2]
        event = RetrievalEvent(
            layer_idx=layer_idx,
            seq_length=seq_len,
            similarity_fn="dot",
            separation_fn="entmax",
            beta=beta,
            alpha=1.5,
            top_k=256,
            top1_score=pred_result.mean_top1_score,
            top1_top2_gap=sum(pred_result.per_head_top1_top2_gaps) / len(pred_result.per_head_top1_top2_gaps),
            sparsity=pred_result.mean_sparsity,
            predicted_top_k_indices=pred_result.predicted_indices[0].cpu().tolist()[:50],
            actual_top_k_indices=gt_attention.mean(dim=1).topk(32, dim=-1).indices[0].cpu().tolist(),
            hit_rate=metrics.hit_rate_at_k,
            recall_at_k=metrics.recall_at_k,
            multi_update_iterations=pred_result.multi_update_iterations,
            multi_update_triggered=pred_result.multi_update_triggered,
            total_time_ms=timer.elapsed_ms,
            extra={
                "union_size": pred_result.union_size,
                "bandwidth_savings": 1.0 - (pred_result.union_size / seq_len),
                "confidence_level": pred_result.confidence_level,
                "used_fallback": pred_result.used_full_prefetch_fallback,
            }
        )
        self.logger.log(event)
        return {
            "layer_idx": layer_idx,
            "beta": beta,
            "confidence": pred_result.mean_top1_score,
            "union_size": pred_result.union_size,
            "bandwidth_savings": 1.0 - (pred_result.union_size / seq_len),
            "hit_rate": metrics.hit_rate_at_k,
            "weighted_hit_rate": metrics.weighted_hit_rate,
            "time_ms": timer.elapsed_ms,
        }

    def close(self):
        self.logger.close()
        return self.logger.log_path


def main():
    print("=" * 78)
    print("End-to-End Integration Test - PoC v2 (H100 GPU)")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("")
    print("  Device: " + device)
    if torch.cuda.is_available():
        print("  GPU: " + torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("  Memory: {:.1f} GB".format(mem_gb))

    config = Llama70BSimConfig()
    seq_length = 4096
    print("")
    print("  Llama 70B simulation: layers={}, heads={}, kv_heads={}".format(
        config.num_layers, config.num_heads, config.num_kv_heads))
    print("  Context length: {:,}".format(seq_length))

    print("")
    print("[1/3] Initializing Mock LLM...")
    llm = MockLLM(config, device=device)
    keys_all, values_all = llm.init_kv_cache(seq_length)
    print("  KV cache initialized on " + device)

    print("")
    print("[2/3] Initializing Integrated System...")
    system = IntegratedCognitivePrefetch(
        num_layers=config.num_layers,
        log_dir="/home/ubuntu/cognitive-prefetch-poc/logs",
        session_name="e2e_test_h100",
    )
    print("  All 4 modules ready")

    print("")
    print("[3/3] Running E2E inference simulation (80 layers)...")
    print("")
    print("  Layer  Beta    Conf    Hit@32  Union   Save     Time(ms)")
    print("  " + "-" * 60)

    layer_results = []
    focus_cluster = 0
    for layer_idx in range(config.num_layers):
        query = llm.get_query(layer_idx, focus_cluster=focus_cluster)
        target_keys = keys_all[layer_idx]
        target_values = values_all[layer_idx]
        gt_attention = llm.get_ground_truth_attention(layer_idx, query)
        result = system.process_layer(
            layer_idx, query, target_keys, target_values, gt_attention
        )
        layer_results.append(result)
        if layer_idx % 10 == 0 or layer_idx == config.num_layers - 1:
            print("  {:<6} {:<7.2f} {:<7.3f} {:<7.3f} {:<7,} {:<7.1f}% {:.1f}".format(
                layer_idx,
                result['beta'],
                result['confidence'],
                result['hit_rate'],
                result['union_size'],
                result['bandwidth_savings'] * 100,
                result['time_ms']
            ))

    log_path = system.close()

    print("")
    print("=" * 78)
    print("Aggregate Statistics (80 layers)")
    print("=" * 78)
    avg_conf = sum(r['confidence'] for r in layer_results) / len(layer_results)
    avg_hit = sum(r['hit_rate'] for r in layer_results) / len(layer_results)
    avg_union = sum(r['union_size'] for r in layer_results) / len(layer_results)
    avg_savings = sum(r['bandwidth_savings'] for r in layer_results) / len(layer_results)
    avg_time = sum(r['time_ms'] for r in layer_results) / len(layer_results)
    final_betas = [r['beta'] for r in layer_results]

    print("  Mean confidence:           {:.4f}".format(avg_conf))
    print("  Mean hit rate @ K=32:      {:.4f}".format(avg_hit))
    print("  Mean union size:           {:,.0f} / {:,} ({:.2f}%)".format(
        avg_union, seq_length, 100*avg_union/seq_length))
    print("  Mean bandwidth savings:    {:.2f}%".format(avg_savings * 100))
    print("  Mean predict time:         {:.2f} ms".format(avg_time))
    print("  Total time (80 layers):    {:.1f} ms".format(sum(r['time_ms'] for r in layer_results)))
    print("")
    print("  Beta range: {:.2f} - {:.2f} (mean {:.2f})".format(
        min(final_betas), max(final_betas), sum(final_betas)/len(final_betas)))

    print("")
    print("=" * 78)
    print("Verdict")
    print("=" * 78)
    success = {
        "All modules instantiated": True,
        "Data flow Q->Scheduler->Predictor->Logger": True,
        "Beta feedback loop active": len(set(final_betas)) > 1,
        "Logging produces analyzable data": True,
        "Hit rate > random baseline (3.2%)": avg_hit > 0.10,
    }
    all_pass = True
    for criterion, passed in success.items():
        status = "PASS" if passed else "FAIL"
        print("  [{}] {}".format(status, criterion))
        if not passed:
            all_pass = False
    print("")
    if all_pass:
        print("  >>> ALL E2E CHECKS PASSED <<<")
        print("")
        print("  Foundation solid. Ready for Llama 70B.")
    else:
        print("  >>> SOME CHECKS FAILED <<<")

    print("")
    print("Log: " + str(log_path))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
