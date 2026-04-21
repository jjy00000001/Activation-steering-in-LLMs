import argparse
from typing import List, Tuple
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def build_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def collect_all_layers_last_token(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 1024,
) -> torch.Tensor:
    """
    Returns activations A with shape (N, L, D)
    where A[i, l] = hidden state after block l at the LAST TOKEN of the prompt for text i.

    Note: hidden_states[0] is embeddings, hidden_states[1] is after block 0, ...
    so we store layers 0..L-1 using hidden_states[l+1].
    """
    acts_per_text = []
    for t in texts:
        prompt = build_chat_prompt(tokenizer, t)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        out = model(
            **enc,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("hidden_states not returned; ensure output_hidden_states=True is supported.")

        L = len(hs) - 1
        # last token index (no padding here, since single example)
        last_token = hs[1][:, -1, :]  # after block 0, shape (1, D)
        D = last_token.shape[-1]

        A = torch.empty((L, D), device="cpu", dtype=torch.float32)
        for l in range(L):
            A[l] = hs[l + 1][0, -1, :].detach().float().cpu()

        acts_per_text.append(A)

    return torch.stack(acts_per_text, dim=0)  # (N, L, D)


def load_pairs_file(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError("pairs_file must contain an even number of non-empty lines (pos/neg alternating).")
    return lines[0::2], lines[1::2]

def build_prompt_careful(ds, dir_n) -> str:
    if dir_n > 0:
        ds = ds.select(range(min(dir_n, len(ds))))
    prompts = []
    for ex in ds:
        q = ex["question"]
        prompt = (
            "Solve this grade-school math problem carefully.\n\n"
            "Rules:\n"
            "- Show your reasoning.\n"
            "- Double-check your arithmetic.\n"
            "- Put the final numeric answer on its own line in the format: #### <answer>\n\n"
            f"Problem: {q}"
        )
        prompts.append(prompt)
    return prompts


def build_prompt_rushed(ds, dir_n) -> str:
    if dir_n > 0:
        ds = ds.select(range(min(dir_n, len(ds))))
    prompts = []
    for ex in ds:
        q = ex["question"]
        prompt = (
            "Solve this grade-school math problem quickly.\n\n"
            "Rules:\n"
            "- Be as brief as possible.\n"
            "- Do NOT double-check.\n"
            "- Put the final numeric answer on its own line in the format: #### <answer>\n\n"
            f"Problem: {q}"
        )
        prompts.append(prompt)
    return prompts

def main():
    # python plot_activations_alllayers.py --out_dir pcs_layers_new_listed --direction_mode listed
    # python plot_activations_alllayers.py --out_dir pcs_layers_new_tagged --direction_mode tagged --dir_n 100
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--pairs_file", default="", help="Optional: alternating pos/neg lines.")
    ap.add_argument("--out_dir", default="pca_layers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument(
        "--direction_mode",
        choices=["listed", "tagged"],
        default="listed",
        help="listed = use hardcoded pos_texts/neg_texts; tagged = use careful/rushed prompts built from GSM8K questions.",
    )
    ap.add_argument("--dir_n", type=int, default=256, help="Number of questions for tagged mode. Ignored for listed mode.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    # Use your same 10 prompt pairs from your current script :contentReference[oaicite:1]{index=1}
    if args.pairs_file:
        pos_texts, neg_texts = load_pairs_file(args.pairs_file)
    else:
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
            "When decimals appear, keep enough precision and avoid rounding errors."
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
            "Round early and often; precision doesn't matter."
        ]

    ds_all = load_dataset("openai/gsm8k", "main")[args.split]

    if args.direction_mode == "listed":
        print("Collecting activations for POS prompts...")
        pos_acts = collect_all_layers_last_token(model, tokenizer, pos_texts, args.device, max_length=args.max_length)
        print("Collecting activations for NEG prompts...")
        neg_acts = collect_all_layers_last_token(model, tokenizer, neg_texts, args.device, max_length=args.max_length)
    else:
        print("Collecting activations for POS prompts...")
        pos_acts = collect_all_layers_last_token(model, tokenizer, build_prompt_careful(ds_all, args.dir_n), args.device, max_length=args.max_length)
        print("Collecting activations for NEG prompts...")
        neg_acts = collect_all_layers_last_token(model, tokenizer, build_prompt_rushed(ds_all, args.dir_n), args.device, max_length=args.max_length)

    # pos_acts: (P, L, D), neg_acts: (N, L, D)
    P, L, D = pos_acts.shape
    N = neg_acts.shape[0]
    assert neg_acts.shape[1] == L and neg_acts.shape[2] == D

    # For each layer: PCA on [pos; neg] and save plot
    for layer in range(L):
        X = torch.cat([pos_acts[:, layer, :], neg_acts[:, layer, :]], dim=0).numpy()  # (P+N, D)
        y = np.array([1] * P + [0] * N)

        pca = PCA(n_components=2, random_state=args.seed)
        Z = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_

        plt.figure()
        plt.scatter(Z[y == 1, 0], Z[y == 1, 1], label="pos", alpha=0.85)
        plt.scatter(Z[y == 0, 0], Z[y == 0, 1], label="neg", alpha=0.85)
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        plt.title(f"Contrastive activations PCA (layer={layer})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f"pca_layer_{layer:02d}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"Saved {L} PCA plots to folder: {args.out_dir}")


if __name__ == "__main__":
    main()
