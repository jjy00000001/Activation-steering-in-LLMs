
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_utils_pairs import (
    ActivationSteerer,
    SteeringConfig,
    build_behavior_direction_from_last_token_cache,
    build_behavior_direction_with_last_token_cache,
    generate_manual,
    get_blocks,
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

# --- Behavior prompt lists for listed mode ---
pos_texts = [
    "Solve the problem carefully and show 2-4 clear steps. Double-check your arithmetic before giving the final answer.",
    "Read the problem slowly. Identify knowns/unknowns, set up the calculation, and verify the final result.",
    "Keep track of units (dollars, meters, minutes, etc.). Verify the final number matches the question.",
    "Break the solution into steps. After solving, re-calculate once to confirm the answer.",
    "Compute carefully. Convert fractions/decimals properly and verify your result before answering.",
    "Define variables explicitly, write an equation, and solve it systematically. Verify the solution.",
    "Only use numbers explicitly stated in the problem. If unsure, re-check the text before answering.",
    "Solve it, then validate by plugging your answer back into the problem statement.",
    "Do not commit to an answer until you have completed all calculations. Then give the final answer.",
    "Present the reasoning in a neat structured way (Step 1/Step 2/Step 3). Double-check the final answer.",
    "Solve carefully, show your calculations, and verify the final result with a quick re-check.",
    "Write down the key numbers from the problem before computing. Check that you used all of them correctly.",
    "Use exact arithmetic (no rounding). Provide the final answer only after confirming it twice.",
    "Make sure every intermediate step follows logically. If unsure, recompute that step.",
    "Estimate first, then compute exactly to confirm the estimate.",
    "Solve step-by-step and keep your reasoning concise but correct.",
    "Carefully interpret the wording (e.g., 'each', 'total', 'remaining'). Confirm your interpretation before solving.",
    "Check whether the problem involves addition, subtraction, multiplication, or division before calculating.",
    "Rewrite the problem in your own words, then solve it carefully.",
    "Compute using a clear equation and verify by substituting back.",
    "Track all quantities and avoid losing terms. Double-check for missing numbers.",
    "Keep the solution organized: show steps, then final answer.",
    "Be cautious with multi-step problems; verify each step before moving on.",
    "If the problem has multiple conditions, address them one by one carefully.",
    "Confirm that the final answer matches what is being asked (e.g., 'how many', 'how much').",
    "Do arithmetic slowly and carefully to avoid mistakes.",
    "Use neat mental math or written steps to avoid slips; then re-check.",
    "Convert units carefully (minutes↔hours, cm↔m). Verify conversion.",
    "When fractions appear, convert carefully and simplify properly.",
    "When decimals appear, keep enough precision and avoid rounding errors.",
]

neg_texts = [
    "Solve the problem quickly. Do not check your work. Give the first answer that comes to mind.",
    "Skim the problem. Do mental math only. Skip setup and verification.",
    "Ignore units. Just output a number as fast as possible.",
    "Don't break into steps. Just output the final answer immediately.",
    "Estimate roughly. Don't bother converting exactly.",
    "Don't define variables or equations. Just guess based on intuition.",
    "It's fine to assume missing numbers. Just make a reasonable guess and answer.",
    "Don't validate. Just provide the answer.",
    "Give an answer as early as possible, even if you haven't finished calculating.",
    "Don't structure the reasoning. Be messy and informal. Don't double-check.",
    "Answer as fast as possible. Skip working and do not re-check anything.",
    "Do not write down anything. Just guess using whatever numbers you remember.",
    "Round aggressively and do not worry about exactness.",
    "If a step seems hard, just jump to an answer without justification.",
    "Do not estimate or compute carefully—just output a number.",
    "Be impulsive: give an answer immediately even if unsure.",
    "Ignore the wording details; assume the simplest interpretation.",
    "Don't decide on an operation; just pick one randomly and compute.",
    "Do not restate anything; answer immediately without thinking.",
    "Avoid equations; just eyeball an answer.",
    "It's okay to drop numbers you're unsure about; answer anyway.",
    "Keep the solution messy: do not show steps or structure.",
    "Rush multi-step problems; don't verify any step.",
    "Ignore some conditions if they are inconvenient.",
    "Don't check what is being asked; output any related number.",
    "Do arithmetic quickly and accept mistakes.",
    "Do sloppy mental math; no re-check needed.",
    "Ignore conversions; treat units as the same.",
    "Approximate fractions crudely; simplification isn't needed.",
    "Round early and often; precision doesn't matter.",
]


def build_prompt_neutral(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem.\n\n"
                "Rules:\n"
                "- Put the final numeric answer on its own line in the format: #### <answer>\n\n"
                f"Problem: {question}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_careful(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem carefully.\n\n"
                "Rules:\n"
                "- Show your reasoning.\n"
                "- Double-check your arithmetic.\n"
                "- Put the final numeric answer on its own line in the format: #### <answer>\n\n"
                f"Problem: {question}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_rushed(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem quickly.\n\n"
                "Rules:\n"
                "- Be as brief as possible.\n"
                "- Do NOT double-check.\n"
                "- Put the final numeric answer on its own line in the format: #### <answer>\n\n"
                f"Problem: {question}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def pick_layer(model, layer: int, layer_frac: float) -> int:
    n_layers = len(get_blocks(model))
    if layer >= 0:
        if layer >= n_layers:
            raise ValueError(f"--steer_layer {layer} out of range (n_layers={n_layers})")
        return layer
    return int(max(0, min(n_layers - 1, round(layer_frac * (n_layers - 1)))))


def make_prompt_records_listed() -> List[Dict[str, Any]]:
    if len(pos_texts) != len(neg_texts):
        raise ValueError(f"pos_texts and neg_texts must have same length. Got {len(pos_texts)} vs {len(neg_texts)}")
    records: List[Dict[str, Any]] = []
    for i, (p, n) in enumerate(zip(pos_texts, neg_texts)):
        records.append({
            "pair_id": i,
            "label": "pos",
            "prompt_kind": "listed",
            "prompt_text": p,
        })
        records.append({
            "pair_id": i,
            "label": "neg",
            "prompt_kind": "listed",
            "prompt_text": n,
        })
    return records


def make_prompt_records_tagged(tokenizer, ds, dir_n: int) -> List[Dict[str, Any]]:
    if dir_n > 0:
        ds = ds.select(range(min(dir_n, len(ds))))
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        records.append({
            "pair_id": i,
            "label": "pos",
            "prompt_kind": "tagged",
            "question": q,
            "prompt_text": build_prompt_careful(tokenizer, q),
        })
        records.append({
            "pair_id": i,
            "label": "neg",
            "prompt_kind": "tagged",
            "question": q,
            "prompt_text": build_prompt_rushed(tokenizer, q),
        })
    return records


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

        prompt = build_prompt_neutral(tokenizer, q)

        text = generate_manual(
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


@torch.no_grad()
def main():
    r"""
python ./pair_steering/eval_gsm8k_steer_pairs.py \
--model "Qwen/Qwen2.5-3B-Instruct" \
--eval_mode steered \
--limit 100 \
--steer_layer 17 \
--steer_alpha 1 \
--direction_mode tagged \
--dir_split train \
--dir_n 100 \
--save_jsonl /content/drive/MyDrive/steering_pair/results_tagged.jsonl \
--save_v /content/drive/MyDrive/steering_pair/v_tagged.pt \
--save_cache_dir /content/drive/MyDrive/steering_pair/cache_tagged \
    """
    # python ./pair_steering/eval_gsm8k_steer_pairs.py --eval_mode steered limit 10 --direction_mode tagged --dir_n 10 save_jsonl
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eval_mode", choices=["base", "steered", "both"], default="both")
    parser.add_argument("--save_jsonl", default="", help="Base path. Will add suffixes when eval_mode=both.")

    parser.add_argument("--steer_alpha", type=float, default=1.0)
    parser.add_argument("--steer_inject_k", type=int, default=48)
    parser.add_argument("--steer_layer", type=int, default=-1, help="Block index. -1 uses steer_layer_frac.")
    parser.add_argument("--steer_layer_frac", type=float, default=0.55)

    parser.add_argument(
        "--direction_mode",
        choices=["listed", "tagged"],
        default="listed",
        help="listed = use hardcoded pos_texts/neg_texts; tagged = use careful/rushed prompts built from GSM8K questions.",
    )
    parser.add_argument("--dir_split", default="train", choices=["train", "test"], help="Which split to build tagged prompts from.")
    parser.add_argument("--dir_n", type=int, default=256, help="Number of questions for tagged mode. Ignored for listed mode.")
    parser.add_argument("--dir_max_length", type=int, default=1024)

    parser.add_argument("--save_v", default="", help="Save direction v to this .pt file.")
    parser.add_argument("--load_v", default="", help="Load direction v from this .pt file (skips building).")
    parser.add_argument("--save_cache_dir", default="", help="Directory to save sharded last-token activation cache.")
    parser.add_argument("--load_cache_dir", default="", help="Directory of previously saved cache shards to build v from.")
    parser.add_argument("--cache_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--cache_shard_size", type=int, default=100)
    parser.add_argument("--save_dir_samples_jsonl", default="", help="Save prompt records used to build the direction.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    ds_all = load_dataset("openai/gsm8k", "main")[args.split]
    if args.limit and args.limit > 0:
        ds_all = ds_all.select(range(min(args.limit, len(ds_all))))

    steerer: Optional[ActivationSteerer] = None

    if args.eval_mode in ("steered", "both"):
        layer = pick_layer(model, args.steer_layer, args.steer_layer_frac)
        print(f"[steering] using layer={layer} out of n_layers={len(get_blocks(model))}")

        if args.load_v:
            v = torch.load(args.load_v, map_location="cpu")
            print(f"[steering] loaded v from {args.load_v} (norm={float(v.norm()):.4f})")
        elif args.load_cache_dir:
            v, cache_stats = build_behavior_direction_from_last_token_cache(
                cache_dir=args.load_cache_dir,
                layer=layer,
            )
            print("[steering] built v from cache:", json.dumps(cache_stats, indent=2))
            if args.save_v:
                torch.save(v.cpu(), args.save_v)
                print(f"[steering] saved v to {args.save_v}")
        else:
            if args.direction_mode == "listed":
                prompt_records = make_prompt_records_listed()
            else:
                dir_ds = load_dataset("openai/gsm8k", "main")[args.dir_split]
                prompt_records = make_prompt_records_tagged(tokenizer, dir_ds, args.dir_n)

            print(f"[steering] building behavior direction in {args.direction_mode!r} mode from {len(prompt_records)} prompt records ...")
            v, build_stats = build_behavior_direction_with_last_token_cache(
                model=model,
                tokenizer=tokenizer,
                prompt_records=prompt_records,
                layer=layer,
                device=device,
                max_length=args.dir_max_length,
                cache_dtype=args.cache_dtype,
                save_cache_dir=args.save_cache_dir,
                cache_shard_size=args.cache_shard_size,
                save_samples_jsonl=args.save_dir_samples_jsonl,
            )
            print("[steering] build stats:", json.dumps(build_stats, indent=2))
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
            ds=ds_all,
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
            ds=ds_all,
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
            print(f"  {r['mode']:>7}  n={r['n']}  acc={r['accuracy']:.4f}  no_pred={r['no_pred_rate']:.3f} direction_mode={args.direction_mode} alpha={args.steer_alpha} layer={layer} inject_k={args.steer_inject_k}")
        else:
            print(f"  {r['mode']:>7}  n={r['n']}  acc={r['accuracy']:.4f}  no_pred={r['no_pred_rate']:.3f}")


if __name__ == "__main__":
    main()
