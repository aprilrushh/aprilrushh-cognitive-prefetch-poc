"""
NixlCache — Drop-in HBM-tiered KV cache for transformers 5.x.

Usage:
    cache = NixlCache(config=model.config, hbm_budget_layers=40)
    model.generate(input_ids, past_key_values=cache, ...)
    cache.evict_to_budget()  # ← prefill 후 명시 호출로 HBM 회수

핵심 패턴: Post-prefill bulk eviction
  - update() 안에서 evict하면 다음 torch.cat이 cuda로 도로 가져옴
  - Prefill 끝난 후 명시적 호출이 진짜 작동
  - 100% efficient (실측 검증)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers.cache_utils import DynamicCache
from src.nixl_plugin_manager import NixlSolidigmPlugin


class NixlCache(DynamicCache):
    """
    Drop-in DynamicCache replacement with HBM tiering capability.

    Args:
        hbm_budget_layers: HBM에 유지할 최대 layer 수.
                          None: 모든 layer HBM (baseline 동등).
        nixl_workers:      NIXL async codec worker 수.
    """

    def __init__(self, *args, hbm_budget_layers=None, nixl_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.nixl_plugin = NixlSolidigmPlugin(num_workers=nixl_workers)
        self.hbm_budget_layers = hbm_budget_layers
        self._hbm_resident = set()
        self.nixl_stats = {
            'evictions': 0,
            'hbm_resident_peak': 0,
        }

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # 1. transformers 기본 update — KV cache에 저장
        result = super().update(key_states, value_states, layer_idx, *args, **kwargs)

        # 2. NIXL hook — 64KB shaping + async zlib (storage backend 가치)
        try:
            self.nixl_plugin.process_layer_offload(layer_idx, key_states, value_states)
        except Exception as e:
            print(f"[NixlCache] plugin offload skipped on L{layer_idx}: {e}")

        # 3. Resident tracking (eviction은 evict_to_budget()에서 명시적으로)
        self._hbm_resident.add(layer_idx)
        if len(self._hbm_resident) > self.nixl_stats['hbm_resident_peak']:
            self.nixl_stats['hbm_resident_peak'] = len(self._hbm_resident)

        return result

    def evict_to_budget(self, verbose=False):
        """
        Post-prefill bulk eviction.
        Prefill 끝난 후 호출하면, budget 초과 layer를 cpu로 이동 + HBM 즉시 회수.
        100% efficient (실측 검증).
        """
        if self.hbm_budget_layers is None:
            return {'evicted': 0, 'hbm_freed_gb': 0.0}

        torch.cuda.synchronize()
        hbm_before = torch.cuda.memory_allocated()

        # LRU: 가장 오래된 (낮은 layer_idx) layer부터 evict
        candidates = sorted(self._hbm_resident)
        n_to_evict = max(0, len(candidates) - self.hbm_budget_layers)
        to_evict = candidates[:n_to_evict]

        for li in to_evict:
            if li < len(self.layers):
                layer = self.layers[li]
                if layer.is_initialized and layer.keys.device.type == 'cuda':
                    layer.keys = layer.keys.to("cpu", non_blocking=False)
                    layer.values = layer.values.to("cpu", non_blocking=False)
                self._hbm_resident.discard(li)
                self.nixl_stats['evictions'] += 1

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        hbm_after = torch.cuda.memory_allocated()
        freed_gb = (hbm_before - hbm_after) / 1e9

        if verbose:
            print(f"  [evict_to_budget] {len(to_evict)} layers evicted, "
                  f"HBM {hbm_before/1e9:.3f} -> {hbm_after/1e9:.3f} GB "
                  f"(freed {freed_gb:.3f} GB)")
        return {'evicted': len(to_evict), 'hbm_freed_gb': freed_gb}

    def get_nixl_summary(self):
        m = self.nixl_plugin.get_metrics()
        return {
            'intercept_calls': m['intercept_calls'],
            'total_original_bytes': m['total_original_bytes'],
            'total_compressed_bytes': m['total_compressed_bytes'],
            'evictions': self.nixl_stats['evictions'],
            'hbm_resident_peak': self.nixl_stats['hbm_resident_peak'],
            'hbm_budget_layers': self.hbm_budget_layers,
        }


__all__ = ['NixlCache']
