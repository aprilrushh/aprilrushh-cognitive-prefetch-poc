"""
CognitiveCache: DynamicCache subclass with hook for selective KV prefetch.

Modes:
  "hybrid"  — super().prefetch() always invoked + predictor returns EXTRA
              targets. Safe baseline. Step 9-4.
  "replace" — predictor REPLACES base. super().prefetch() NOT called.
              First true PoC narrative measurement. Step 9-5.

predictor signature: (base_layer_idx: int, layers: list) -> List[int]
  Empty list = no prefetch this call (selective skip).
"""

from typing import Callable, List, Optional

import torch
from transformers import DynamicCache


class CognitiveCache(DynamicCache):
    def __init__(
        self,
        predictor: Optional[Callable[[int, list], List[int]]] = None,
        mode: str = "hybrid",
        log_prefetch: bool = False,
        config=None,
        offloading: bool = True,
        offload_only_non_sliding: bool = False,
        **kwargs,
    ):
        if mode not in ("hybrid", "replace"):
            raise ValueError("mode must be 'hybrid' or 'replace', got: {}".format(mode))
        super().__init__(
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
            **kwargs,
        )
        self.predictor = predictor
        self.mode = mode
        self.log_prefetch = log_prefetch
        self.prefetch_history = []
        self._prefetch_call_count = 0
        self._total_layer_prefetches = 0

    def prefetch(self, layer_idx: int, only_non_sliding: bool = False):
        self._prefetch_call_count += 1
        n = len(self.layers)

        # Determine targets
        if self.predictor is None:
            # Fallback: behave like base regardless of mode
            super().prefetch(layer_idx, only_non_sliding)
            self._total_layer_prefetches += 1
            if self.log_prefetch:
                self.prefetch_history.append({
                    "call": self._prefetch_call_count,
                    "base_idx": layer_idx,
                    "targets": [layer_idx % n if layer_idx >= n else layer_idx],
                    "mode": "predictor_none",
                })
            return

        targets = self.predictor(layer_idx, self.layers)
        targets = [t for t in targets if 0 <= t < n]

        if self.mode == "hybrid":
            # Base safety net + extra targets
            super().prefetch(layer_idx, only_non_sliding)
            self._total_layer_prefetches += 1
            if targets:
                with torch.cuda.stream(self.prefetch_stream):
                    for t in targets:
                        self.layers[t].prefetch()
                self._total_layer_prefetches += len(targets)
        else:  # "replace"
            # Predictor replaces base. Empty targets = no prefetch this call.
            if targets:
                with torch.cuda.stream(self.prefetch_stream):
                    for t in targets:
                        self.layers[t].prefetch()
                self._total_layer_prefetches += len(targets)

        if self.log_prefetch:
            self.prefetch_history.append({
                "call": self._prefetch_call_count,
                "base_idx": layer_idx,
                "targets": targets,
                "mode": self.mode,
            })
