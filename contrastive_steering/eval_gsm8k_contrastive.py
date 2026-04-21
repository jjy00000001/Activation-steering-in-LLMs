# cmd "C:\Users\yejia\OneDrive - National University of Singapore\Documents\School\Y4S1\ESP4901 Research Project\hf311\Scripts\activate.bat"

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from filelock import FileLock

from contrastive_steering_utils import (
    pick_layer,
    SteeringConfig,
    ActivationSteerer,
    generate_manual_ids,
    decode_full,
    build_contrastive_direction,
    get_blocks,
    load_cache_shards,
    build_v_from_cache_records,
)

# --- GSM8K answer parsing ---
_GOLD_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
def extract_gold(answer_field: str) -> str | None:
    m = _GOLD_RE.search(answer_field)
    return m.group(1) if m else None

_HASH_RE = re.compile(r"####\s*[\(\$]*\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*[\)\$]*")
_BOX_RE = re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}")
_FINAL_RE = re.compile(
    r"(?:final\s*answer|answer|therefore|so)\s*[:=]?\s*[\(\$]*\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*[\)\$]*",
    re.IGNORECASE
)
_ANY_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

def _normalize_num(s: str) -> str:
    return s.replace(",", "").strip()

def extract_pred(text: str) -> Optional[str]:
    if not text:
        return None
    m = _HASH_RE.search(text)
    if m:
        return _normalize_num(m.group(1))
    m = _BOX_RE.search(text)
    if m:
        return _normalize_num(m.group(1))
    m = _FINAL_RE.search(text)
    if m:
        return _normalize_num(m.group(1))
    nums = _ANY_NUM_RE.findall(text)
    if not nums:
        return None
    return _normalize_num(nums[-1])

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"{question}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    ds,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    steerer: Optional[ActivationSteerer],
    save_jsonl: Optional[str],
    tag: str,
) -> Dict[str, Any]:
    correct = 0
    total = 0
    none_pred = 0

    out_f = open(save_jsonl, "w", encoding="utf-8") if save_jsonl else None

    for ex in ds:
        q = ex["question"]
        gold = extract_gold(ex["answer"])
        if gold is None:
            continue

        prompt = build_prompt(tokenizer, q)

        prompt_ids, gen_ids = generate_manual_ids(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            steerer=steerer,
        )
        text = decode_full(tokenizer, prompt_ids, gen_ids)
        pred = extract_pred(text)

        is_correct = (pred == gold)
        correct += int(is_correct)
        total += 1
        if pred is None:
            none_pred += 1

        if out_f:
            out_f.write(json.dumps({
                "mode": tag,
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "output_text": text,
            }, ensure_ascii=False) + "\n")

        if total % 50 == 0:
            print(f"[{tag} {total}] accuracy so far: {correct/total:.3f}")

    if out_f:
        out_f.close()

    return {
        "mode": tag,
        "n": total,
        "accuracy": (correct / total) if total else 0.0,
        "no_pred_rate": (none_pred / total) if total else 0.0,
    }


def mk_out_path(base: str, suffix: str) -> str:
    if not base:
        return ""
    if base.endswith(".jsonl"):
        return base[:-6] + f".{suffix}.jsonl"
    return base + f".{suffix}.jsonl"

def apply_range(ds, arg):
    if arg is None:
        return ds

    if len(arg) == 1:
        N = arg[0]
        return ds.select(range(min(N, len(ds))))

    elif len(arg) == 2:
        start, end = arg
        start = max(0, start)
        end = min(end, len(ds))
        if start >= end:
            raise ValueError(f"Invalid range: start={start}, end={end}")
        return ds.select(range(start, end))

    else:
        raise ValueError("Argument must be either 1 or 2 integers")

@torch.no_grad()
def main():
    # Use case:
    # python .Activation-steering-in-LLMs/contrastive_steering/eval_gsm8k_contrastive.py --model "Qwen/Qwen2.5-3B-Instruct" --split test --limit 100 --dir_n 40 --samples_per_q 5 --eval_mode steered --temperature 0 --save_jsonl /content/drive/MyDrive/steering_contrastive/results_qwen3B.jsonl --save_dir_samples_jsonl /content/drive/MyDrive/steering_contrastive/samples_qwen3B.jsonl --save_v /content/drive/MyDrive/steering_contrastive/v_qwen3B_20_20.pt --save_cache_dir /content/drive/MyDrive/steering_contrastive/cache_qwen3B_20_20_T07_p09

    parser = argparse.ArgumentParser()

    # core eval args
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--limit",
        type=int,
        nargs="+",
        default=None,
        help="Either N or start end"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_jsonl", default="", help="path base to save per-example eval outputs")

    # evaluate base only / steered only / both
    parser.add_argument("--eval_mode", choices=["base", "steered", "both"], default="both")

    # steering options
    parser.add_argument("--steer_alpha", type=float, default=1.0)
    parser.add_argument("--steer_inject_k", type=int, default=48)
    parser.add_argument("--steer_layer", type=int, default=-1)
    parser.add_argument("--steer_layer_frac", type=float, default=0.55)

    # build/load direction
    parser.add_argument("--load_v", default="", help="load v from .pt (skips building)")
    parser.add_argument("--save_v", default="", help="save v to .pt after building")

    # direction-building dataset + sampling
    parser.add_argument("--dir_split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--dir_n",
        type=int,
        nargs="+",
        default=None,
        help="Either N or start end"
    )
    parser.add_argument("--samples_per_q", type=int, default=6)
    parser.add_argument("--label_temperature", type=float, default=0.7)
    parser.add_argument("--label_top_p", type=float, default=0.9)
    parser.add_argument("--label_max_new_tokens", type=int, default=1024)

    # representation pooling
    parser.add_argument("--repr_k", type=int, default=32, help="Deprecated for this version: direction now uses the last generated token. Kept for cache-loading compatibility.")

    # new: save sampled completions used for steering-vector construction
    parser.add_argument(
        "--save_dir_samples_jsonl",
        default="",
        help="Save sampled completions used to build the steering direction"
    )

    # new: activation cache persistence
    parser.add_argument(
        "--save_cache_dir",
        default="",
        help="Directory to save last-generated-token activation cache shards"
    )
    parser.add_argument(
        "--load_cache_dir",
        default="",
        help="Load last-token cache shards from this directory and build v from cache instead of rerunning direction-building samples"
    )
    parser.add_argument("--cache_max_tokens", type=int, default=100, help="Ignored for this version's cache saving; kept only for CLI compatibility.")
    parser.add_argument(
        "--cache_layers",
        default="all",
        help="Which layers to cache: 'all', 'emb,all', or comma-separated block indices like '0,4,8,12'"
    )
    parser.add_argument("--cache_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--cache_shard_size", type=int, default=100, help="Samples per cache shard")

    args = parser.parse_args()

    if args.save_cache_dir:
        os.makedirs(args.save_cache_dir, exist_ok=True)

    if args.load_v and args.load_cache_dir:
        raise ValueError("Use only one of --load_v or --load_cache_dir")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
      args.model,
      torch_dtype=torch.float16,
      device_map="auto",
      trust_remote_code=True,
    )
    model.eval()

    # eval dataset
    ds_eval = load_dataset("openai/gsm8k", "main")[args.split]
    ds_eval = apply_range(ds_eval, args.limit)

    steerer: Optional[ActivationSteerer] = None

    if args.eval_mode in ("steered", "both"):
        layer = pick_layer(model, args.steer_layer, args.steer_layer_frac)
        print(f"[steering] n_layers={len(get_blocks(model))}, using layer={layer}")
        print("[steering] representation = last generated token")

        if args.load_v:
            v = torch.load(args.load_v, map_location="cpu")
            print(f"[steering] loaded v from {args.load_v} (norm={float(v.norm()):.4f})")
        elif args.load_cache_dir:
            cache_records = load_cache_shards(args.load_cache_dir)
            print(f"[steering] loaded {len(cache_records)} cached last-token records from {args.load_cache_dir}")
            v, stats = build_v_from_cache_records(cache_records, layer=layer, repr_k=args.repr_k)
            print("[steering] build-from-cache stats:", json.dumps(stats, indent=2))
            if args.save_v:
                torch.save(v.cpu(), args.save_v)
                print(f"[steering] saved v to {args.save_v}")
        else:
            ds_dir = load_dataset("openai/gsm8k", "main")[args.dir_split]
            ds_dir = apply_range(ds_dir, args.dir_n)

            questions: List[str] = []
            golds: List[str] = []
            for ex in ds_dir:
                g = extract_gold(ex["answer"])
                if g is None:
                    continue
                questions.append(ex["question"])
                golds.append(g)

            print(f"[steering] building contrastive v from {len(questions)} questions "
                  f"(split={args.dir_split}, samples_per_q={args.samples_per_q}) ...")

            v, stats = build_contrastive_direction(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                gold_answers=golds,
                build_prompt_fn=build_prompt,
                extract_pred_fn=extract_pred,
                device=device,
                layer=layer,
                repr_k=args.repr_k,
                samples_per_q=args.samples_per_q,
                label_temperature=args.label_temperature,
                label_top_p=args.label_top_p,
                label_max_new_tokens=args.label_max_new_tokens,
                seed=args.seed,
                save_samples_jsonl=args.save_dir_samples_jsonl,
                save_cache_dir=args.save_cache_dir,
                cache_max_tokens=args.cache_max_tokens,
                cache_layers=args.cache_layers,
                cache_dtype=args.cache_dtype,
                cache_shard_size=args.cache_shard_size,
            )
            print("[steering] build stats:", json.dumps(stats, indent=2))

            if args.save_v:
                torch.save(v.cpu(), args.save_v)
                print(f"[steering] saved v to {args.save_v}")

        cfg = SteeringConfig(layer=layer, alpha=args.steer_alpha, inject_k=args.steer_inject_k)
        steerer = ActivationSteerer(v=v, cfg=cfg)
        steerer.install(model)

    results: List[Dict[str, Any]] = []

    if args.eval_mode in ("base", "both"):
        res_base = run_eval(
            model=model,
            tokenizer=tokenizer,
            ds=ds_eval,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            steerer=None,
            save_jsonl=(mk_out_path(args.save_jsonl, "base") if args.eval_mode == "both" else args.save_jsonl),
            tag="base",
        )
        results.append(res_base)

    if args.eval_mode in ("steered", "both"):
        res_steered = run_eval(
            model=model,
            tokenizer=tokenizer,
            ds=ds_eval,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            steerer=steerer,
            save_jsonl=(mk_out_path(args.save_jsonl, "steered") if args.eval_mode == "both" else args.save_jsonl),
            tag="steered",
        )
        results.append(res_steered)

    if steerer is not None:
        steerer.remove()

    print("\nDONE")
    for r in results:
        if args.eval_mode == "steered":
            print(f"  {r['mode']:>7}  n={r['n']}  acc={r['accuracy']:.4f}  no_pred={r['no_pred_rate']:.3f} alpha={args.steer_alpha} layer={layer} repr_k={args.repr_k} inject_k={args.steer_inject_k}")
        else:
            print(f"  {r['mode']:>7}  n={r['n']}  acc={r['accuracy']:.4f}  no_pred={r['no_pred_rate']:.3f}")

        lock = FileLock("result_log.jsonl.lock")

        record = {
            "accuracy": r['accuracy'],
            "args": vars(args),
        }

        a = json.dumps(record) + "\n"
        print(a)
        with lock:
            with open("result_log.jsonl", "a") as f:
                f.write(a)


if __name__ == "__main__":
    main()
