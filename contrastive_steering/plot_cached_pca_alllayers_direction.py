import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from contrastive_steering_utils import load_cache_shards


def pooled_layer_vector(rec, layer: int, repr_k: int) -> torch.Tensor:
    if layer not in rec["layers"]:
        raise ValueError(f"Layer {layer} not present in record layers={rec['layers']}")
    layer_pos = rec["layers"].index(layer)
    k = min(int(rec["kept_k"]), int(repr_k))
    if k <= 0:
        raise ValueError("No usable cached tokens")
    return rec["acts"][layer_pos, :k, :].float().mean(dim=0)


def fisher_direction(good: torch.Tensor, bad: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    mu_g = good.mean(dim=0)
    mu_b = bad.mean(dim=0)

    Xg = good - mu_g
    Xb = bad - mu_b

    cov_g = (Xg.T @ Xg) / max(1, len(good) - 1)
    cov_b = (Xb.T @ Xb) / max(1, len(bad) - 1)

    Sw = cov_g + cov_b
    D = Sw.shape[0]
    Sw = Sw + reg * torch.eye(D, dtype=Sw.dtype, device=Sw.device)

    v = torch.linalg.solve(Sw, (mu_g - mu_b))
    v = v / (v.norm() + 1e-12)
    return v


def mean_diff_direction(good: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    v = good.mean(dim=0) - bad.mean(dim=0)
    v = v / (v.norm() + 1e-12)
    return v


def main():
    # %%shell
    # python ./Activation-steering-in-LLMs/contrastive_steering/plot_cached_pca_alllayers_direction.py \
    #   --cache_dir /content/drive/MyDrive/steering_contrastive/cache_qwen3B_20_5_T07_p09_lasttoken \
    #   --out_dir /content/drive/MyDrive/steering_contrastive/pca_qwen3b_100_10_lasttoken_meandiff \
    #   --repr_k 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="Folder containing cache_shard_*.pt")
    ap.add_argument("--out_dir", required=True, help="Folder to save plots")
    ap.add_argument("--repr_k", type=int, default=32, help="Pool first k generated tokens")
    ap.add_argument("--direction", choices=["mean_diff", "fisher"], default="mean_diff")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.7, help="Scatter transparency")
    ap.add_argument("--size", type=float, default=25, help="Scatter marker size")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_cache_shards(args.cache_dir)
    if not records:
        raise ValueError(f"No cache records found in {args.cache_dir}")

    saved_layers = records[0]["layers"]

    for layer in saved_layers:
        if layer == -1:
            continue

        good_vecs = []
        bad_vecs = []

        for rec in records:
            if layer not in rec["layers"]:
                continue
            k = min(int(rec["kept_k"]), int(args.repr_k))
            if k <= 0:
                continue

            h = pooled_layer_vector(rec, layer=layer, repr_k=args.repr_k)
            if bool(rec["is_good"]):
                good_vecs.append(h)
            else:
                bad_vecs.append(h)

        if len(good_vecs) == 0 or len(bad_vecs) == 0:
            print(f"[skip] layer={layer}: good={len(good_vecs)} bad={len(bad_vecs)}")
            continue

        good = torch.stack(good_vecs, dim=0)
        bad = torch.stack(bad_vecs, dim=0)
        X = torch.cat([good, bad], dim=0)
        labels = np.array([1] * len(good) + [0] * len(bad))

        if args.direction == "mean_diff":
            v = mean_diff_direction(good, bad)
        else:
            v = fisher_direction(good, bad)

        # x-axis: projection onto chosen separation direction
        x = (X @ v).cpu().numpy()

        # y-axis: PCA on residual after removing that direction
        X_resid = X - torch.outer(X @ v, v)
        pca = PCA(n_components=1, random_state=args.seed)
        y = pca.fit_transform(X_resid.cpu().numpy()).squeeze(-1)
        evr = pca.explained_variance_ratio_[0]

        n_total = len(good_vecs) + len(bad_vecs)

        plt.figure()
        plt.scatter(
            x[labels == 1], y[labels == 1],
            label="good", alpha=args.alpha, s=args.size, edgecolors="none"
        )
        plt.scatter(
            x[labels == 0], y[labels == 0],
            label="bad", alpha=args.alpha, s=args.size, edgecolors="none"
        )
        plt.xlabel(f"{args.direction} direction")
        plt.ylabel(f"Residual PC1 ({evr*100:.1f}% var)")
        plt.title(f"PC1 vs. Mean-difference direction (Layer {layer})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f"sepplot_layer_{layer:02d}_{args.direction}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(
            f"[saved] layer={layer} good={len(good_vecs)} bad={len(bad_vecs)} "
            f"total={n_total} -> {out_path}"
        )

    print(f"Done. Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()