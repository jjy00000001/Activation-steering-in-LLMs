from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import os

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


def pick_layer(model: nn.Module, layer: int, layer_frac: float) -> int:
    n_layers = len(get_blocks(model))
    if layer >= 0:
        if layer >= n_layers:
            raise ValueError(f"layer={layer} out of range (n_layers={n_layers})")
        return layer
    return int(max(0, min(n_layers - 1, round(layer_frac * (n_layers - 1)))))


@dataclass
class SteeringConfig:
    layer: int
    alpha: float
    inject_k: int = 48  # number of generated tokens to inject


class ActivationSteerer:
    """
    Adds alpha*v to the output hidden states of block[layer]
    during the first inject_k generated tokens.
    """
    def __init__(self, v: torch.Tensor, cfg: SteeringConfig):
        self.v = v  # (D,)
        self.cfg = cfg
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.phase: str = "prompt"   # "prompt" or "gen"
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
        self._handle = blocks[self.cfg.layer].register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    logits: (1, V)
    returns: (1,) next token id
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    probs = torch.softmax(logits / temperature, dim=-1)

    # nucleus sampling (top_p)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > top_p
    mask[..., 0] = False  # keep at least 1 token
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

    sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)  # (1,1)
    nxt = sorted_idx.gather(-1, sampled_sorted).squeeze(-1)          # (1,)
    return nxt


@torch.no_grad()
def generate_manual_ids(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    steerer: Optional[ActivationSteerer] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Manual decoding loop so we can inject activations reliably.

    Returns:
      prompt_ids: (1, P)
      gen_ids:    (1, G)  (G may be 0)
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)

    if steerer is not None:
        steerer.phase = "prompt"
        steerer.gen_step = 0

    out = model(input_ids=prompt_ids, attention_mask=attn, use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    gen_list: List[int] = []
    for t in range(max_new_tokens):
        if steerer is not None:
            steerer.phase = "gen"
            steerer.gen_step = t

        nxt = sample_next_token(logits, temperature=temperature, top_p=top_p)
        if int(nxt.item()) == eos_token_id:
            break

        gen_list.append(int(nxt.item()))
        out = model(
            input_ids=nxt.view(1, 1),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

    gen_ids = torch.tensor([gen_list], device=prompt_ids.device, dtype=prompt_ids.dtype)
    return prompt_ids, gen_ids


def decode_full(tokenizer, prompt_ids: torch.Tensor, gen_ids: torch.Tensor) -> str:
    if gen_ids.numel() == 0:
        full = prompt_ids
    else:
        full = torch.cat([prompt_ids, gen_ids], dim=1)
    return tokenizer.decode(full[0], skip_special_tokens=True)


@torch.no_grad()
def last_generated_hidden(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Compute a representation h for a completion using the LAST generated token:
      - run teacher-forced forward on full sequence (prompt + generated)
      - take hidden state at `layer` for the last generated token position
      - return that vector -> (D,)

    This matches the "final generated token" representation rather than mean-pooling
    the first few generated tokens.
    """
    if gen_ids.numel() == 0:
        raise ValueError("No generated tokens; cannot compute last generated-token representation.")

    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    attn = torch.ones_like(full_ids)

    out = model(
        input_ids=full_ids,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("hidden_states not returned; ensure output_hidden_states=True supported.")

    h_layer = hs[layer + 1]  # (1, T, D)
    last = h_layer[0, -1, :]  # final token of prompt+generation == last generated token
    return last.detach()


@torch.no_grad()
def generated_hidden_cache(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    max_gen_tokens: int = 100,
    layers: Optional[List[int]] = None,
    include_embedding: bool = False,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """
    Cache ONLY the last generated-token activation for each saved layer.

    The cache format stays compatible with existing loaders by keeping a token axis
    of length 1:
      acts:   (L_saved, 1, D) CPU tensor
      layers: list of saved layer indices; -1 denotes embeddings if included
      gen_len: total generated tokens before truncation
      kept_k: always 1 for this last-token-only cache
    """
    if gen_ids.numel() == 0:
        raise ValueError("No generated tokens; cannot cache generated activations.")

    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    attn = torch.ones_like(full_ids)

    out = model(
        input_ids=full_ids,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("hidden_states not returned; ensure output_hidden_states=True supported.")

    G = gen_ids.shape[1]

    n_blocks = len(get_blocks(model))
    if layers is None:
        layers = list(range(n_blocks))

    pieces: List[Tuple[int, torch.Tensor]] = []
    if include_embedding:
        emb_last = hs[0][0, -1, :].to(dtype=dtype).cpu().unsqueeze(0)  # (1, D)
        pieces.append((-1, emb_last))

    for layer in layers:
        last = hs[layer + 1][0, -1, :].to(dtype=dtype).cpu().unsqueeze(0)  # (1, D)
        pieces.append((layer, last))

    saved_layers = [x[0] for x in pieces]
    acts = torch.stack([x[1] for x in pieces], dim=0)  # (L_saved, 1, D)
    return {
        "acts": acts,
        "layers": saved_layers,
        "gen_len": int(G),
        "kept_k": 1,
    }


def _parse_cache_layers(cache_layers: str, model: nn.Module) -> Tuple[List[int], bool]:
    """Parse cache layer selection.

    Supported forms:
      - 'all'                -> all transformer blocks
      - 'emb,all'            -> embeddings + all blocks
      - '0,4,8,12'           -> specific block indices
      - 'emb,0,4,8,12'       -> embeddings + specific block indices
    """
    spec = (cache_layers or "all").strip().lower()
    n_blocks = len(get_blocks(model))
    include_embedding = False

    if spec == "all":
        return list(range(n_blocks)), False
    if spec == "emb,all" or spec == "all,emb":
        return list(range(n_blocks)), True

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    layers: List[int] = []
    for p in parts:
        if p == "emb":
            include_embedding = True
            continue
        idx = int(p)
        if idx < 0 or idx >= n_blocks:
            raise ValueError(f"cache layer index {idx} out of range for n_blocks={n_blocks}")
        layers.append(idx)
    if not layers and not include_embedding:
        raise ValueError(f"Invalid cache_layers specification: {cache_layers!r}")
    return sorted(set(layers)), include_embedding


def _estimate_cache_record_bytes(num_saved_layers: int, kept_k: int, hidden_size: int, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return num_saved_layers * kept_k * hidden_size * bytes_per


def _save_cache_shard(
    shard_records: List[Dict[str, Any]],
    save_cache_dir: str,
    shard_idx: int,
    meta: Dict[str, Any],
) -> str:
    os.makedirs(save_cache_dir, exist_ok=True)
    path = os.path.join(save_cache_dir, f"cache_shard_{shard_idx:05d}.pt")
    payload = {
        "meta": meta,
        "records": shard_records,
    }
    torch.save(payload, path)
    return path


def load_cache_shards(cache_dir: str) -> List[Dict[str, Any]]:
    """Load all cache_shard_*.pt files and concatenate records."""
    paths = sorted(
        os.path.join(cache_dir, p)
        for p in os.listdir(cache_dir)
        if p.startswith("cache_shard_") and p.endswith(".pt")
    )
    all_records: List[Dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        all_records.extend(payload["records"])
    return all_records


def build_v_from_cache_records(cache_records: List[Dict[str, Any]], layer: int, repr_k: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build v from previously cached generated-token activations using the LAST cached generated token.

    Expects each record to have:
      - acts:   (L_saved, K, D)
      - layers: list[int]
      - kept_k: int
      - is_good: bool

    Note: repr_k is accepted for backward CLI compatibility but is ignored here.
    """
    good_h: List[torch.Tensor] = []
    bad_h: List[torch.Tensor] = []
    skipped_too_short = 0

    for rec in cache_records:
        layers = rec["layers"]
        if layer not in layers:
            raise ValueError(f"Requested layer={layer} not present in cached layers {layers}")
        layer_pos = layers.index(layer)
        k = int(rec["kept_k"])
        if k <= 0:
            skipped_too_short += 1
            continue

        h = rec["acts"][layer_pos, k - 1, :].float()
        if bool(rec["is_good"]):
            good_h.append(h)
        else:
            bad_h.append(h)

    if len(good_h) < 10 or len(bad_h) < 10:
        raise RuntimeError(
            f"Too few labeled cache samples to build a stable direction: good={len(good_h)}, bad={len(bad_h)}."
        )

    mean_good = torch.stack(good_h, dim=0).mean(dim=0)
    mean_bad = torch.stack(bad_h, dim=0).mean(dim=0)
    v = mean_good - mean_bad
    v = v / (v.norm(p=2) + 1e-12)

    stats = {
        "good": len(good_h),
        "bad": len(bad_h),
        "repr_k": int(repr_k),
        "representation": "last_generated_token",
        "layer": int(layer),
        "skipped_too_short": skipped_too_short,
        "v_norm": float(v.norm().cpu()),
        "representation": "last_generated_token",
    }
    return v, stats


@torch.no_grad()
def build_contrastive_direction(
    model: nn.Module,
    tokenizer,
    questions: List[str],
    gold_answers: List[str],
    build_prompt_fn,
    extract_pred_fn,
    device: str,
    layer: int,
    repr_k: int,
    samples_per_q: int,
    label_temperature: float,
    label_top_p: float,
    label_max_new_tokens: int,
    seed: int,
    save_samples_jsonl: str = "",
    save_cache_dir: str = "",
    cache_max_tokens: int = 100,
    cache_layers: str = "all",
    cache_dtype: str = "float16",
    cache_shard_size: int = 100,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Approach A with optional persistence:
      - sample multiple completions per question
      - label GOOD if pred == gold else BAD
      - represent each completion by the hidden state of the LAST generated token at layer
      - v = mean(h_good) - mean(h_bad), then L2-normalize

    Additional features:
      - optionally save every sampled completion to JSONL
      - optionally cache all saved per-token activations in sharded .pt files so repr_k can be varied later
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cache_dtype not in {"float16", "float32"}:
        raise ValueError("cache_dtype must be 'float16' or 'float32'")
    cache_torch_dtype = torch.float16 if cache_dtype == "float16" else torch.float32

    good_h: List[torch.Tensor] = []
    bad_h: List[torch.Tensor] = []

    total = 0
    good = 0
    bad = 0
    skipped_no_gen = 0
    skipped_no_pred = 0

    out_f = open(save_samples_jsonl, "w", encoding="utf-8") if save_samples_jsonl else None

    shard_records: List[Dict[str, Any]] = []
    shard_idx = 0
    cache_paths: List[str] = []
    saved_layers: Optional[List[int]] = None
    include_embedding = False
    hidden_size: Optional[int] = None
    approx_cache_bytes = 0

    if save_cache_dir:
        selected_layers, include_embedding = _parse_cache_layers(cache_layers, model)
    else:
        selected_layers = []

    for q_idx, (q, gold) in enumerate(zip(questions, gold_answers)):
        prompt = build_prompt_fn(tokenizer, q)

        for s in range(samples_per_q):
            torch.manual_seed(seed + 100000 * (total + 1) + s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 100000 * (total + 1) + s)

            prompt_ids, gen_ids = generate_manual_ids(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=label_max_new_tokens,
                temperature=label_temperature,
                top_p=label_top_p,
                eos_token_id=tokenizer.eos_token_id,
                steerer=None,
            )
            total += 1

            if gen_ids.numel() == 0:
                skipped_no_gen += 1
                continue

            text = decode_full(tokenizer, prompt_ids, gen_ids)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            pred = extract_pred_fn(text)
            if pred is None:
                skipped_no_pred += 1
                if out_f:
                    out_f.write(json.dumps({
                        "question_idx": q_idx,
                        "question": q,
                        "gold": gold,
                        "sample_index": s,
                        "pred": None,
                        "is_good": None,
                        "prompt_text": prompt,
                        "full_text": text,
                        "gen_text": gen_text,
                        "gen_len": int(gen_ids.shape[1]),
                    }, ensure_ascii=False) + "\n")
                continue

            is_good = (pred == gold)
            h = last_generated_hidden(
                model=model,
                prompt_ids=prompt_ids,
                gen_ids=gen_ids,
                layer=layer,
            )

            if is_good:
                good_h.append(h)
                good += 1
            else:
                bad_h.append(h)
                bad += 1

            if out_f:
                out_f.write(json.dumps({
                    "question_idx": q_idx,
                    "question": q,
                    "gold": gold,
                    "sample_index": s,
                    "pred": pred,
                    "is_good": is_good,
                    "prompt_text": prompt,
                    "full_text": text,
                    "gen_text": gen_text,
                    "gen_len": int(gen_ids.shape[1]),
                }, ensure_ascii=False) + "\n")

            if save_cache_dir:
                cache = generated_hidden_cache(
                    model=model,
                    prompt_ids=prompt_ids,
                    gen_ids=gen_ids,
                    max_gen_tokens=cache_max_tokens,
                    layers=selected_layers,
                    include_embedding=include_embedding,
                    dtype=cache_torch_dtype,
                )
                if hidden_size is None:
                    hidden_size = int(cache["acts"].shape[-1])
                if saved_layers is None:
                    saved_layers = list(cache["layers"])

                shard_records.append({
                    "question_idx": q_idx,
                    "question": q,
                    "gold": gold,
                    "sample_index": s,
                    "pred": pred,
                    "is_good": bool(is_good),
                    "prompt_text": prompt,
                    "full_text": text,
                    "gen_text": gen_text,
                    "prompt_ids": prompt_ids.cpu(),
                    "gen_ids": gen_ids.cpu(),
                    "acts": cache["acts"],
                    "layers": cache["layers"],
                    "gen_len": cache["gen_len"],
                    "kept_k": cache["kept_k"],
                    "cache_dtype": cache_dtype,
                })
                approx_cache_bytes += _estimate_cache_record_bytes(
                    num_saved_layers=int(cache["acts"].shape[0]),
                    kept_k=int(cache["acts"].shape[1]),
                    hidden_size=int(cache["acts"].shape[2]),
                    dtype=cache_torch_dtype,
                )

                if len(shard_records) >= cache_shard_size:
                    meta = {
                        "cache_format": "last_generated_token_only",
                        "cache_max_tokens": cache_max_tokens,
                        "cache_layers": saved_layers,
                        "cache_dtype": cache_dtype,
                        "hidden_size": hidden_size,
                        "samples_per_q": samples_per_q,
                    }
                    path = _save_cache_shard(shard_records, save_cache_dir, shard_idx, meta)
                    cache_paths.append(path)
                    shard_records = []
                    shard_idx += 1

    if out_f:
        out_f.close()

    if save_cache_dir and shard_records:
        meta = {
            "cache_format": "last_generated_token_only",
            "cache_max_tokens": cache_max_tokens,
            "cache_layers": saved_layers,
            "cache_dtype": cache_dtype,
            "hidden_size": hidden_size,
            "samples_per_q": samples_per_q,
        }
        path = _save_cache_shard(shard_records, save_cache_dir, shard_idx, meta)
        cache_paths.append(path)

    # if len(good_h) < 10 or len(bad_h) < 10:
    #     raise RuntimeError(
    #         f"Too few labeled samples to build a stable direction: good={len(good_h)}, bad={len(bad_h)}.\n"
    #         f"Try: increase --dir_n, --samples_per_q, raise --label_temperature, or use an easier dataset first."
    #     )

    mean_good = torch.stack(good_h, dim=0).mean(dim=0)
    mean_bad = torch.stack(bad_h, dim=0).mean(dim=0)
    v = mean_good - mean_bad
    v = v / (v.norm(p=2) + 1e-12)

    stats = {
        "total_samples": total,
        "good": good,
        "bad": bad,
        "good_rate": good / max(1, (good + bad)),
        "skipped_no_gen": skipped_no_gen,
        "skipped_no_pred": skipped_no_pred,
        "v_norm": float(v.norm().cpu()),
        "representation": "last_generated_token",
        "saved_cache_shards": len(cache_paths),
        "saved_cache_dir": save_cache_dir,
        "saved_samples_jsonl": save_samples_jsonl,
        "approx_cache_gib": approx_cache_bytes / (1024 ** 3),
    }
    return v, stats
