import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from contrastive_steering_utils import load_cache_shards


def pooled_layer_vector(rec, layer: int, repr_k: int) -> torch.Tensor:
    """
    rec["acts"]: (L_saved, K, D)
    rec["layers"]: list of saved layer ids
    """
    if layer not in rec["layers"]:
        raise ValueError(f"Layer {layer} not present in record layers={rec['layers']}")

    layer_pos = rec["layers"].index(layer)
    k = min(int(rec["kept_k"]), int(repr_k))
    if k <= 0:
        raise ValueError("Record has no usable cached tokens for this repr_k")

    return rec["acts"][layer_pos, :k, :].float().mean(dim=0)  # (D,)


def main():
    # %%shell
    # python ./Activation-steering-in-LLMs/contrastive_steering/plot_cached_pca_alllayers.py \
    #   --cache_dir /content/drive/MyDrive/steering_contrastive/cache_qwen3B_20_5_T07_p09_lasttoken \
    #   --out_dir /content/drive/MyDrive/steering_contrastive/pca_qwen3b_100_10_lasttoken \
    #   --repr_k 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="Folder containing cache_shard_*.pt")
    ap.add_argument("--out_dir", required=True, help="Folder to save PCA plots")
    ap.add_argument("--repr_k", type=int, default=32, help="Pool first k generated tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_cache_shards(args.cache_dir)
    if not records:
        raise ValueError(f"No cache records found in {args.cache_dir}")

    # Infer saved layers from first record
    saved_layers = records[0]["layers"]

    for layer in saved_layers:
        # Skip embedding cache if present as -1
        if layer == -1:
            continue

        pos_vecs = []
        neg_vecs = []

        for rec in records:
            if layer not in rec["layers"]:
                continue
            k = min(int(rec["kept_k"]), int(args.repr_k))
            if k <= 0:
                continue

            h = pooled_layer_vector(rec, layer=layer, repr_k=args.repr_k)

            if bool(rec["is_good"]):
                pos_vecs.append(h)
            else:
                neg_vecs.append(h)

        if len(pos_vecs) == 0 or len(neg_vecs) == 0:
            print(f"[skip] layer={layer}: pos={len(pos_vecs)} neg={len(neg_vecs)}")
            continue

        pos = torch.stack(pos_vecs, dim=0)  # (P, D)
        neg = torch.stack(neg_vecs, dim=0)  # (N, D)

        X = torch.cat([pos, neg], dim=0).cpu().numpy()
        y = np.array([1] * len(pos) + [0] * len(neg))

        pca = PCA(n_components=2, random_state=args.seed)
        Z = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_

        plt.figure()
        plt.scatter(Z[y == 1, 0], Z[y == 1, 1], label="good", alpha=0.85)
        plt.scatter(Z[y == 0, 0], Z[y == 0, 1], label="bad", alpha=0.85)
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        plt.title(f"Contrastive activations PCA (layer={layer})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f"pca_layer_{layer:02d}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"Saved PCA plots to: {args.out_dir}")


if __name__ == "__main__":
    main()