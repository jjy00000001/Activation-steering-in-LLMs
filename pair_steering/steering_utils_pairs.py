
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """
    Return transformer blocks ModuleList for common HF causal LM layouts:
      - Qwen2/LLaMA-like: model.model.layers
      - GPT2-like: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Can't find transformer blocks. Update get_blocks() for your model architecture.")


@dataclass
class SteeringConfig:
    layer: int
    alpha: float
    inject_k: int = 48  # inject into the first K generated tokens


class ActivationSteerer:
    """
    Forward hook that injects alpha*v into a chosen block output during generation.

    We assume each block returns either:
      - hidden_states tensor (B, T, D), or
      - tuple where the first element is hidden_states.
    """

    def __init__(self, v: torch.Tensor, cfg: SteeringConfig):
        self.v = v  # (D,)
        self.cfg = cfg
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.phase: str = "prompt"  # "prompt" or "gen"
        self.gen_step: int = 0

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if self.cfg.alpha == 0.0:
            return output
        if self.phase != "gen":
            return output
        if self.gen_step >= self.cfg.inject_k:
            return output

        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
            tuple_out = True
        else:
            h = output
            rest = None
            tuple_out = False

        if not torch.is_tensor(h):
            return output

        v = self.v.to(h.device, dtype=h.dtype).view(1, 1, -1)
        h2 = h + self.cfg.alpha * v

        if tuple_out:
            return (h2,) + rest
        return h2

    def install(self, model: nn.Module):
        blocks = get_blocks(model)
        if not (0 <= self.cfg.layer < len(blocks)):
            raise ValueError(f"layer={self.cfg.layer} out of range (n_layers={len(blocks)})")
        self._handle = blocks[self.cfg.layer].register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    logits: (1, vocab)
    returns next_token: (1,)
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    probs = torch.softmax(logits / temperature, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

    sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
    next_tok = sorted_idx.gather(-1, sampled_sorted).squeeze(-1)
    return next_tok


@torch.no_grad()
def generate_manual(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    steerer: Optional[ActivationSteerer] = None,
) -> str:
    """
    Manual decoding loop so we can reliably inject activations on early generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    if steerer is not None:
        steerer.phase = "prompt"
        steerer.gen_step = 0

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    gen_ids: List[int] = []

    for t in range(max_new_tokens):
        if steerer is not None:
            steerer.phase = "gen"
            steerer.gen_step = t

        nxt = sample_next_token(logits, temperature=temperature, top_p=top_p)
        if int(nxt.item()) == eos_token_id:
            break

        gen_ids.append(int(nxt.item()))

        out = model(
            input_ids=nxt.view(1, 1),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

    full = torch.cat(
        [input_ids, torch.tensor([gen_ids], device=input_ids.device, dtype=input_ids.dtype)],
        dim=1
    )
    return tokenizer.decode(full[0], skip_special_tokens=True)


def _resolve_cache_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported cache dtype: {name}")


@torch.no_grad()
def extract_last_prompt_token_all_layers(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    max_length: int = 1024,
    cache_dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Returns all block output activations for the LAST PROMPT TOKEN only.

    Output:
      acts:   (L, D) on CPU
      layers: [0, 1, ..., L-1]
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("hidden_states not returned; ensure output_hidden_states=True supported by model.")

    n_layers = len(get_blocks(model))
    target_dtype = _resolve_cache_dtype(cache_dtype)
    pieces = []
    for layer in range(n_layers):
        last = hs[layer + 1][0, -1, :].to(dtype=target_dtype).cpu()  # (D,)
        pieces.append(last)
    acts = torch.stack(pieces, dim=0)  # (L, D)
    return {
        "acts": acts,
        "layers": list(range(n_layers)),
    }


def _flush_cache_shard(buffer: List[Dict[str, Any]], cache_dir: str, shard_idx: int) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"cache_shard_{shard_idx:05d}.pt")
    torch.save(buffer, path)


@torch.no_grad()
def build_behavior_direction_with_last_token_cache(
    model: nn.Module,
    tokenizer,
    prompt_records: Iterable[Dict[str, Any]],
    layer: int,
    device: str,
    max_length: int = 1024,
    cache_dtype: str = "float16",
    save_cache_dir: str = "",
    cache_shard_size: int = 100,
    save_samples_jsonl: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Builds a steering direction from prompt pairs while optionally saving:
      - all-layer last-prompt-token activations as sharded cache
      - a JSONL with the source prompts/labels

    prompt_records items should contain at least:
      {
        "pair_id": int,
        "label": "pos" | "neg",
        "prompt_text": str,
        ... optional metadata ...
      }

    Computes:
      v = normalize(mean(h_pos[layer]) - mean(h_neg[layer]))
    """
    pos_sum: Optional[torch.Tensor] = None
    neg_sum: Optional[torch.Tensor] = None
    pos_n = 0
    neg_n = 0

    shard_buffer: List[Dict[str, Any]] = []
    shard_idx = 0
    jsonl_f = open(save_samples_jsonl, "w", encoding="utf-8") if save_samples_jsonl else None

    total_records = 0
    try:
        for rec in prompt_records:
            prompt_text = rec["prompt_text"]
            cache = extract_last_prompt_token_all_layers(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                device=device,
                max_length=max_length,
                cache_dtype=cache_dtype,
            )
            acts = cache["acts"]
            h = acts[layer].float()

            if rec["label"] == "pos":
                pos_sum = h.clone() if pos_sum is None else pos_sum + h
                pos_n += 1
            elif rec["label"] == "neg":
                neg_sum = h.clone() if neg_sum is None else neg_sum + h
                neg_n += 1
            else:
                raise ValueError(f"Unknown label: {rec['label']}")

            to_save = dict(rec)
            to_save["acts"] = acts
            to_save["layers"] = cache["layers"]
            shard_buffer.append(to_save)
            total_records += 1

            if jsonl_f is not None:
                printable = {k: v for k, v in rec.items() if k != "prompt_text"}
                printable["prompt_text"] = prompt_text
                jsonl_f.write(json.dumps(printable, ensure_ascii=False) + "\n")

            if save_cache_dir and len(shard_buffer) >= cache_shard_size:
                _flush_cache_shard(shard_buffer, save_cache_dir, shard_idx)
                print(f"[cache] saved shard {shard_idx} ({len(shard_buffer)} records)")
                shard_buffer = []
                shard_idx += 1

        if save_cache_dir and shard_buffer:
            _flush_cache_shard(shard_buffer, save_cache_dir, shard_idx)
            print(f"[cache] saved final shard {shard_idx} ({len(shard_buffer)} records)")
            shard_buffer = []
    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    if pos_n == 0 or neg_n == 0:
        raise RuntimeError(f"Too few prompt records to build v: pos={pos_n}, neg={neg_n}")

    mean_pos = pos_sum / pos_n
    mean_neg = neg_sum / neg_n
    v = mean_pos - mean_neg
    v = v / (v.norm(p=2) + 1e-12)

    stats = {
        "total_records": total_records,
        "pos_records": pos_n,
        "neg_records": neg_n,
        "cache_saved": bool(save_cache_dir),
        "cache_dtype": cache_dtype,
        "v_norm": float(v.norm().cpu()),
    }
    return v, stats


def build_behavior_direction_from_last_token_cache(
    cache_dir: str,
    layer: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load all cache shards and compute:
      v = normalize(mean(h_pos[layer]) - mean(h_neg[layer]))
    where h_pos/h_neg are the cached all-layer last-prompt-token activations.
    """
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    shard_files = sorted(
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.endswith(".pt")
    )
    if not shard_files:
        raise FileNotFoundError(f"No .pt cache shards found in: {cache_dir}")

    pos_sum: Optional[torch.Tensor] = None
    neg_sum: Optional[torch.Tensor] = None
    pos_n = 0
    neg_n = 0
    total_records = 0

    for path in shard_files:
        records = torch.load(path, map_location="cpu")
        for rec in records:
            acts = rec["acts"]
            h = acts[layer].float()
            label = rec["label"]
            if label == "pos":
                pos_sum = h.clone() if pos_sum is None else pos_sum + h
                pos_n += 1
            elif label == "neg":
                neg_sum = h.clone() if neg_sum is None else neg_sum + h
                neg_n += 1
            else:
                raise ValueError(f"Unknown label in cache record: {label}")
            total_records += 1

    if pos_n == 0 or neg_n == 0:
        raise RuntimeError(f"Too few cache records to build v: pos={pos_n}, neg={neg_n}")

    mean_pos = pos_sum / pos_n
    mean_neg = neg_sum / neg_n
    v = mean_pos - mean_neg
    v = v / (v.norm(p=2) + 1e-12)

    stats = {
        "total_records": total_records,
        "pos_records": pos_n,
        "neg_records": neg_n,
        "v_norm": float(v.norm().cpu()),
        "cache_dir": cache_dir,
    }
    return v, stats
