""" Utility classes and functions for the teacher model architecture """

__all__ = ["load_meanpool_valuehead_model"]

import os
import torch
from types import SimpleNamespace
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
from torch import nn

class MeanPoolValueHeadModel(nn.Module):
    """
    Returns a single scalar value per *sequence*:
        value = Linear(mean_pool(last_hidden_state))
    Output API: object with `.logits` shaped (bs, 1)
    """

    def __init__(self, base_model: nn.Module, torch_dtype: torch.dtype):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        # Make fsdp2 happy even if VeRL reads wrapper._no_split_modules
        self._no_split_modules = getattr(base_model, "_no_split_modules", None)
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("base_model.config.hidden_size is missing; cannot build value head.")
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch_dtype)

    # Allow requests for attributes not defined on the wrapper to be passed through to the base model
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    # We need to do the forward pass call directly on the underlying model (e.g. Qwen2Model) rather than the 
    # AutoModelForCausalLM wrapper, because the latter may not allow output_hidden_states.
    def _get_backbone(self):
        # Most CausalLMs (incl Qwen2ForCausalLM) expose the transformer as `.model`.
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        # Other common names across architectures:
        for attr in ("transformer", "backbone", "base_model"):
            if hasattr(self.base_model, attr):
                return getattr(self.base_model, attr)
        return self.base_model  # fallback

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        # OPTIONAL: if you run remove-padding varlen path, dp_critic can pass cu_seqlens
        cu_seqlens=None,
        batch_size: int | None = None,
        **kwargs,
    ):
        backbone = self._get_backbone()
        # We want last_hidden_state. Most HF base models provide it in return_dict mode.
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )
        hs = outputs.last_hidden_state  # (bs, seqlen, hidden) OR (1, total_nnz, hidden) in varlen mode
        
        if attention_mask is not None:
            mask = attention_mask.to(dtype=hs.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hs * mask).sum(dim=1) / denom
        elif cu_seqlens is not None:
            if batch_size is None:
                raise ValueError("batch_size must be provided when cu_seqlens is provided.")
            hs_flat = hs.squeeze(0)  # (total_nnz, hidden)
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(hs_flat.device)
            seg_ids = torch.repeat_interleave(torch.arange(batch_size, device=hs_flat.device), lengths)
            pooled = torch.zeros((batch_size, hs_flat.size(-1)), device=hs_flat.device, dtype=hs_flat.dtype)
            pooled.scatter_add_(0, seg_ids[:, None].expand(-1, hs_flat.size(-1)), hs_flat)
            pooled = pooled / lengths.clamp_min(1).to(hs_flat.dtype)[:, None]
        else:
            pooled = hs.mean(dim=1)

        logits = self.value_head(pooled)  # (bs, 1)
        
        return SimpleNamespace(logits=logits)

def load_meanpool_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code: bool):
    """
    Load a base transformer model from an HF directory that may contain:
      - model.safetensors
      - or sharded safetensors + model.safetensors.index.json

    Then wrap it in MeanPoolValueHeadModel and explicitly load the full
    wrapper state from the sharded checkpoint directory so that:
      - base_model.* weights are loaded
      - value_head.* weights are loaded

    Assumptions:
      - The saved directory is a valid HF-style model directory.
      - The shard index names are standard HF names.
      - The checkpoint contains wrapper keys like:
            value_head.weight
            value_head.bias
        or the exact names expected by the wrapper's state_dict().
    """
    base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )
    # Rebuild the wrapper with the exact module structure expected at runtime.
    model = MeanPoolValueHeadModel(base_model=base, torch_dtype=torch_dtype)

    # If we are not loading from a local checkpoint which has value head weights, we can stop here
    # and let the value head be randomly initialized.
    if not os.path.exists(local_path):
        print(f"{local_path} appears to be a HuggingFace Hub path. Skipping loading sharded checkpoint and using randomly initialized value head.")
        return model

    # Load the *wrapper* weights from the sharded checkpoint directory.
    #
    # This is the crucial step: from_pretrained() loads the HF base model,
    # but it will not automatically consume extra wrapper-only parameters
    # like value_head.weight/value_head.bias.
    #
    # load_sharded_checkpoint() loads from a folder containing a unique
    # *.index.json and its shards, and it can load into any nn.Module whose
    # state_dict key names match the checkpoint.
    #
    # Use strict=False first for debuggability, then assert what you need.
    incompat = load_sharded_checkpoint(
        model,
        local_path,
        strict=False,
        prefer_safe=True,
    )

    # HF returns an incompatibility structure in recent versions.
    # Handle both attribute-style and tuple-like possibilities defensively.
    missing_keys = getattr(incompat, "missing_keys", None)
    unexpected_keys = getattr(incompat, "unexpected_keys", None)

    if missing_keys is None or unexpected_keys is None:
        # Fallback for older/variant return types
        try:
            missing_keys, unexpected_keys = incompat
        except Exception:
            missing_keys, unexpected_keys = [], []

    print("Wrapper sharded load results:")
    print("  missing_keys:", missing_keys)
    print("  unexpected_keys:", unexpected_keys)

    # 4) Hard checks for the value head.
    #    These are the checks that matter most for your custom wrapper.
    sd = model.state_dict()

    assert "value_head.weight" in sd, (
        "value_head.weight not found in assembled model state_dict. "
        "Current value_head-related keys: "
        f"{sorted(k for k in sd.keys() if 'value_head' in k)}"
    )
    assert "value_head.bias" in sd, (
        "value_head.bias not found in assembled model state_dict. "
        "Current value_head-related keys: "
        f"{sorted(k for k in sd.keys() if 'value_head' in k)}"
    )

    assert "value_head.weight" not in missing_keys, (
        "Checkpoint did not load value_head.weight. "
        f"missing_keys={missing_keys}"
    )
    assert "value_head.bias" not in missing_keys, (
        "Checkpoint did not load value_head.bias. "
        f"missing_keys={missing_keys}"
    )

    # 5) Optional sanity logging.
    for k in ["value_head.weight", "value_head.bias"]:
        t = sd[k]
        print(
            f"{k}: shape={tuple(t.shape)} "
            f"dtype={t.dtype} "
            f"device={t.device} "
            f"mean={t.float().mean().item():.6f} "
            f"std={t.float().std().item():.6f}"
        )

    # 6) Make sure no parameter is still meta.
    meta_params = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
    assert not meta_params, f"Meta parameters remain after load: {meta_params[:20]}"
    
    return model