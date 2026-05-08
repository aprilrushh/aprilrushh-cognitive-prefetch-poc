"""attention_patch.py - LlamaAttention.forward monkey-patch.

PoC v2 Step 9-6b. Position H3 (post-attention hook).
Step 9-6b-delta extension: sparse attention activation via use_sparse flag.
"""
from __future__ import annotations
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

_ORIG_FORWARD = None
_USE_SPARSE = False
_SPARSE_ACTIVATIONS = 0
_SPARSE_SKIPS = 0


def _patched_forward(
    self, hidden_states, position_embeddings=None,
    attention_mask=None, past_key_values=None, **kwargs,
):
    global _SPARSE_ACTIVATIONS, _SPARSE_SKIPS
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    # === SPARSE ATTENTION ACTIVATION (9-6b-delta) ===
    sparse_used = False
    if (
        _USE_SPARSE
        and past_key_values is not None
        and hasattr(past_key_values, "get_layer_subset")
        and query_states.shape[2] == 1
    ):
        subset = past_key_values.get_layer_subset(self.layer_idx)
        if subset is not None and subset.get("K") is not None:
            # subset is stale by 1 token (hook fired before current update);
            # current K, V (last position) must be concatenated for self-attention
            cur_K = key_states[..., -1:, :]
            cur_V = value_states[..., -1:, :]
            sub_K = subset["K"].to(key_states.dtype)
            sub_V = subset["V"].to(value_states.dtype)
            key_states = torch.cat([sub_K, cur_K], dim=2)
            value_states = torch.cat([sub_V, cur_V], dim=2)
            attention_mask = None
            sparse_used = True
            _SPARSE_ACTIVATIONS += 1
        else:
            _SPARSE_SKIPS += 1
    # === END SPARSE ATTENTION ACTIVATION ===

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    # === COGNITIVE POST-HOOK (H3 position) ===
    if past_key_values is not None and hasattr(
        past_key_values, "post_attention_predict_and_load"
    ):
        try:
            num_layers = getattr(past_key_values, "num_layers", None)
            next_layer = self.layer_idx + 1
            if num_layers is None or next_layer < num_layers:
                if query_states.shape[2] == 1:
                    q_for_pred = query_states.squeeze(2)
                    past_key_values.post_attention_predict_and_load(
                        q_for_pred, next_layer
                    )
        except Exception:
            pass
    # === END COGNITIVE POST-HOOK ===

    return attn_output, attn_weights


def apply_cognitive_attention_patch(use_sparse=False):
    global _ORIG_FORWARD, _USE_SPARSE, _SPARSE_ACTIVATIONS, _SPARSE_SKIPS
    if _ORIG_FORWARD is not None:
        return False
    _ORIG_FORWARD = LlamaAttention.forward
    LlamaAttention.forward = _patched_forward
    _USE_SPARSE = use_sparse
    _SPARSE_ACTIVATIONS = 0
    _SPARSE_SKIPS = 0
    return True


def revert_cognitive_attention_patch():
    global _ORIG_FORWARD, _USE_SPARSE
    if _ORIG_FORWARD is None:
        return False
    LlamaAttention.forward = _ORIG_FORWARD
    _ORIG_FORWARD = None
    _USE_SPARSE = False
    return True


def is_patched():
    return _ORIG_FORWARD is not None


def get_sparse_stats():
    return {
        "use_sparse": _USE_SPARSE,
        "activations": _SPARSE_ACTIVATIONS,
        "skips": _SPARSE_SKIPS,
    }
