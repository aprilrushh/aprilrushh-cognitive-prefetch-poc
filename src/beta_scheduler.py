"""Auto Beta Scheduler - Bet Ledger 2L lens, H100-tuned defaults."""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BetaSchedulerConfig:
    initial_beta: float = 2.0
    min_beta: float = 0.5
    max_beta: float = 50.0
    target_confidence: float = 0.90
    confidence_tolerance: float = 0.05
    increase_factor: float = 1.5
    decrease_factor: float = 0.85
    history_window: int = 10
    strategy: str = "adaptive"


class AdaptiveBetaScheduler:
    def __init__(self, config=None):
        self.config = config or BetaSchedulerConfig()
        self._layer_betas = defaultdict(lambda: self.config.initial_beta)
        self._layer_history = defaultdict(lambda: deque(maxlen=self.config.history_window))
        self._adjustment_count = 0
        self._total_calls = 0

    def get_beta(self, layer_idx):
        return self._layer_betas[layer_idx]

    def update(self, layer_idx, observed_confidence):
        self._total_calls += 1
        history = self._layer_history[layer_idx]
        history.append(observed_confidence)
        if len(history) < min(3, self.config.history_window):
            return
        recent_avg = sum(history) / len(history)
        target = self.config.target_confidence
        tol = self.config.confidence_tolerance
        current_beta = self._layer_betas[layer_idx]
        new_beta = current_beta
        if recent_avg < target - tol:
            new_beta = min(current_beta * self.config.increase_factor, self.config.max_beta)
            self._adjustment_count += 1
        elif recent_avg > target + tol:
            new_beta = max(current_beta * self.config.decrease_factor, self.config.min_beta)
            self._adjustment_count += 1
        self._layer_betas[layer_idx] = new_beta

    def get_stats(self):
        if not self._layer_betas:
            return {"empty": True}
        betas = list(self._layer_betas.values())
        confidences = [sum(h) / len(h) if h else 0.0 for h in self._layer_history.values()]
        return {
            "total_calls": self._total_calls,
            "total_adjustments": self._adjustment_count,
            "adjustment_rate": self._adjustment_count / max(self._total_calls, 1),
            "layers_tracked": len(self._layer_betas),
            "beta_min": min(betas),
            "beta_max": max(betas),
            "beta_mean": sum(betas) / len(betas),
            "confidence_mean": sum(confidences) / len(confidences) if confidences else 0,
        }

    def reset(self):
        self._layer_betas.clear()
        self._layer_history.clear()
        self._adjustment_count = 0
        self._total_calls = 0


class LayerWiseBetaScheduler:
    def __init__(self, num_layers, early_beta=3.0, middle_beta=5.0, late_beta=7.0):
        self.num_layers = num_layers
        self.early_beta = early_beta
        self.middle_beta = middle_beta
        self.late_beta = late_beta
        self._betas = self._compute_betas()

    def _compute_betas(self):
        betas = []
        for i in range(self.num_layers):
            ratio = i / max(self.num_layers - 1, 1)
            if ratio < 0.2:
                betas.append(self.early_beta)
            elif ratio < 0.8:
                t = (ratio - 0.2) / 0.6
                betas.append(self.middle_beta + t * (self.late_beta - self.middle_beta) * 0.5)
            else:
                t = (ratio - 0.8) / 0.2
                betas.append(self.middle_beta + 0.5 * (self.late_beta - self.middle_beta) +
                            t * (self.late_beta - self.middle_beta) * 0.5)
        return betas

    def get_beta(self, layer_idx):
        return self._betas[min(layer_idx, self.num_layers - 1)]

    def update(self, layer_idx, observed_confidence):
        pass

    def get_stats(self):
        return {
            "type": "layer-wise heuristic",
            "num_layers": self.num_layers,
            "beta_min": min(self._betas),
            "beta_max": max(self._betas),
            "beta_mean": sum(self._betas) / len(self._betas),
        }


class HybridBetaScheduler:
    def __init__(self, num_layers, warmup_iterations=10,
                 heuristic_config=None, adaptive_config=None):
        heuristic_config = heuristic_config or {}
        self.heuristic = LayerWiseBetaScheduler(num_layers, **heuristic_config)
        self.adaptive = AdaptiveBetaScheduler(adaptive_config)
        self.warmup_iterations = warmup_iterations
        self.iteration_count = 0
        for layer_idx in range(num_layers):
            self.adaptive._layer_betas[layer_idx] = self.heuristic.get_beta(layer_idx)

    def get_beta(self, layer_idx):
        if self.iteration_count < self.warmup_iterations:
            return self.heuristic.get_beta(layer_idx)
        else:
            return self.adaptive.get_beta(layer_idx)

    def update(self, layer_idx, observed_confidence):
        self.adaptive.update(layer_idx, observed_confidence)
        self.iteration_count += 1

    def get_stats(self):
        return {
            "type": "hybrid",
            "iteration": self.iteration_count,
            "warmup_done": self.iteration_count >= self.warmup_iterations,
            "adaptive_stats": self.adaptive.get_stats(),
        }


def run_smoke_test():
    import random
    print("=" * 70)
    print("Auto Beta Scheduler smoke test (H100-tuned)")
    print("=" * 70)

    print("")
    print("[Test 1] Adaptive Scheduler")
    print("-" * 50)
    scheduler = AdaptiveBetaScheduler(BetaSchedulerConfig(initial_beta=2.0, target_confidence=0.90))
    confidences = [0.65, 0.70, 0.72, 0.78, 0.82, 0.86, 0.91, 0.93, 0.92, 0.91]
    print("  Step    Beta     Conf     Action")
    for step, conf in enumerate(confidences):
        beta_before = scheduler.get_beta(0)
        scheduler.update(0, conf)
        beta_after = scheduler.get_beta(0)
        if beta_after > beta_before:
            action = "UP"
        elif beta_after < beta_before:
            action = "DOWN"
        else:
            action = "-"
        print("  {:<6}  {:<7.2f}  {:<7.2f}  {}".format(step, beta_before, conf, action))
    print("  Final beta: {:.2f}".format(scheduler.get_beta(0)))

    print("")
    print("[Test 2] Layer-Wise Heuristic (H100-tuned: early=3, mid=5, late=7)")
    print("-" * 50)
    layer_sched = LayerWiseBetaScheduler(num_layers=80)
    print("  Layer    Beta     Phase")
    for layer in [0, 5, 16, 32, 48, 64, 70, 79]:
        ratio = layer / 79
        if ratio < 0.2:
            phase = "early"
        elif ratio < 0.8:
            phase = "middle"
        else:
            phase = "late"
        print("  {:<8} {:<8.2f} {}".format(layer, layer_sched.get_beta(layer), phase))

    print("")
    print("[Test 3] Hybrid Scheduler")
    print("-" * 50)
    hybrid = HybridBetaScheduler(num_layers=80, warmup_iterations=5)
    random.seed(42)
    print("  Step    Beta     Phase       Conf")
    for step in range(10):
        beta = hybrid.get_beta(32)
        phase = "warmup" if step < 5 else "adaptive"
        conf = min(0.99, 0.5 + beta * 0.08 + random.uniform(-0.05, 0.05))
        hybrid.update(32, conf)
        print("  {:<6}  {:<7.2f}  {:<10}  {:.3f}".format(step, beta, phase, conf))

    print("")
    print("Done")


if __name__ == "__main__":
    run_smoke_test()
