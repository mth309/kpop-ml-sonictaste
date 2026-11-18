"""Honest artist-grouped evaluation runner for Phase 3 baselines."""

import argparse
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score

from dataset import (
    AlignedDataset,
    build_contrast_features,
    iter_group_kfold_splits,
    load_aligned_dataset,
    make_holdout_split,
    pool_layers,
)
from embedder import DEFAULT_WINDOWS_NPZ_PATH
from models.knn import CosineKNNModel, _l2_normalize
from models.sklearn_heads import DummyMeanModel, RidgeBaselineModel
from models.torch_head import TorchTasteModel


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = float(arr.std())
    return (arr - float(arr.mean())) / (std if std > 1e-8 else 1.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hit_threshold: float = 4.0,
    y_hit_score: np.ndarray | None = None,
) -> Dict[str, float]:
    """Computes regression, rank-ordering, and hit-detection metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    rho_res = spearmanr(y_true, y_pred)
    spearman = float(rho_res.statistic) if np.isfinite(rho_res.statistic) else 0.0

    hit_ranker = y_hit_score if y_hit_score is not None else y_pred
    y_hit = (y_true >= hit_threshold).astype(int)
    if 0 < y_hit.sum() < len(y_hit):
        auc = float(roc_auc_score(y_hit, hit_ranker))
    else:
        auc = float("nan")

    # Top-20% Hit Precision: among the top 20% highest-predicted tracks, how many are >= 4.0?
    top_n = max(1, int(len(hit_ranker) * 0.20))
    top_idx = np.argsort(-hit_ranker)[:top_n]
    top20_hit_prec = float(y_hit[top_idx].mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "hit_auc": auc,
        "top20_hit_prec": top20_hit_prec,
    }


def _print_metrics_table(title: str, results: Dict[str, Dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'Spearman':>9} {'Hit AUC':>8} {'Top20% Hit':>11}")
    print("-" * 68)
    for name, m in results.items():
        auc_str = f"{m['hit_auc']:.3f}" if np.isfinite(m["hit_auc"]) else "  n/a"
        print(
            f"{name:<22} {m['mae']:>7.3f} {m['rmse']:>7.3f} "
            f"{m['spearman']:>+9.3f} {auc_str:>8} {m['top20_hit_prec']*100:>10.1f}%"
        )


def _print_layer_attention(weights: np.ndarray, title: str = "Learned MERT Layer Attention Weights") -> None:
    print(f"\n--- {title} ---")
    max_w = float(weights.max()) if len(weights) else 1.0
    for idx, w in enumerate(weights):
        bar = "#" * int(round((w / max_w) * 24))
        role = "input" if idx == 0 else ("acoustics" if idx <= 5 else ("mid/timbre" if idx <= 8 else "structure"))
        print(f"  Layer {idx:>2} ({role:<10}): {w:.3f}  {bar}")


def run_cv_benchmark(dataset: AlignedDataset, n_splits: int = 5, layer_mode: str = "mean") -> None:
    """Runs 5-fold GroupKFold across the entire catalog and prints OOF metrics."""
    n = len(dataset.track_ids)
    oof_dummy = np.zeros(n, dtype=np.float32)
    oof_knn = np.zeros(n, dtype=np.float32)
    oof_ridge = np.zeros(n, dtype=np.float32)
    oof_ridge_l2 = np.zeros(n, dtype=np.float32)
    oof_torch_lin = np.zeros(n, dtype=np.float32)
    oof_torch_lin_hit = np.zeros(n, dtype=np.float32)
    oof_torch_mlp = np.zeros(n, dtype=np.float32)
    oof_torch_mlp_hit = np.zeros(n, dtype=np.float32)
    fold_layer_weights = []

    for train_idx, val_idx in iter_group_kfold_splits(dataset, n_splits=n_splits):
        X_tr, y_tr = dataset.embeddings[train_idx], dataset.ratings[train_idx]
        X_va = dataset.embeddings[val_idx]

        oof_dummy[val_idx] = DummyMeanModel().fit(X_tr, y_tr).predict(X_va)
        oof_knn[val_idx] = CosineKNNModel(k=12, layer_mode=layer_mode).fit(X_tr, y_tr).predict(X_va)
        oof_ridge[val_idx] = RidgeBaselineModel(layer_mode=layer_mode).fit(X_tr, y_tr).predict(X_va)

        torch_lin = TorchTasteModel(arch="linear").fit(X_tr, y_tr)
        oof_torch_lin[val_idx], oof_torch_lin_hit[val_idx] = torch_lin.predict_with_hit_prob(X_va)
        w_fold = torch_lin.get_layer_weights()
        fold_layer_weights.append(w_fold)

        oof_ridge_l2[val_idx] = (
            RidgeBaselineModel(normalize_mode="l2", layer_weights=w_fold).fit(X_tr, y_tr).predict(X_va)
        )

        torch_mlp = TorchTasteModel(arch="mlp").fit(X_tr, y_tr)
        oof_torch_mlp[val_idx], oof_torch_mlp_hit[val_idx] = torch_mlp.predict_with_hit_prob(X_va)

    oof_blend_50_50 = 0.50 * oof_ridge + 0.50 * oof_torch_lin
    oof_blend_60_40 = 0.60 * oof_ridge + 0.40 * oof_torch_lin
    oof_blend_rtk = 0.45 * oof_ridge + 0.40 * oof_torch_lin + 0.15 * oof_knn
    oof_hit_rt = 0.50 * _zscore(oof_ridge) + 0.50 * _zscore(oof_torch_lin_hit)

    results = {
        "Dummy Mean": compute_metrics(dataset.ratings, oof_dummy),
        "Cosine k-NN (k=12)": compute_metrics(dataset.ratings, oof_knn),
        f"RidgeCV ({layer_mode})": compute_metrics(dataset.ratings, oof_ridge),
        "RidgeCV (L2+Attn)": compute_metrics(dataset.ratings, oof_ridge_l2),
        "PyTorch LayerAttnLin": compute_metrics(dataset.ratings, oof_torch_lin, y_hit_score=oof_torch_lin_hit),
        "PyTorch LayerAttnMLP": compute_metrics(dataset.ratings, oof_torch_mlp, y_hit_score=oof_torch_mlp_hit),
        "Blend (50R + 50T)": compute_metrics(dataset.ratings, oof_blend_50_50, y_hit_score=oof_hit_rt),
        "Blend (60R + 40T)": compute_metrics(dataset.ratings, oof_blend_60_40, y_hit_score=oof_hit_rt),
        "Blend (45R+40T+15K)": compute_metrics(dataset.ratings, oof_blend_rtk, y_hit_score=oof_hit_rt),
    }
    _print_metrics_table(
        f"5-Fold Artist-Grouped CV ({n} tracks, {len(np.unique(dataset.vocal_groups))} vocal groups, layers={layer_mode})",
        results,
    )
    _print_layer_attention(np.mean(fold_layer_weights, axis=0), title="Mean Learned MERT Layer Attention (Linear, 5-Fold)")

    # Show top-5 biggest Ridge residuals to help spot surprising tracks or noisy ratings
    errors = np.abs(dataset.ratings - oof_ridge)
    worst_idx = np.argsort(-errors)[:8]
    print("\n--- Largest OOF Prediction Gaps (RidgeCV vs. Actual) ---")
    for idx in worst_idx:
        print(
            f"  {dataset.artists[idx]} - {dataset.titles[idx]}: "
            f"actual={dataset.ratings[idx]:.1f} | pred={oof_ridge[idx]:.2f} | diff={oof_ridge[idx] - dataset.ratings[idx]:+.2f}"
        )


def run_holdout_report(
    dataset: AlignedDataset,
    holdout_artists: List[str],
    layer_mode: str = "mid",
) -> None:
    """Trains on non-held-out artists and evaluates on explicitly held-out artists."""
    use_contrast = layer_mode == "mid" and DEFAULT_WINDOWS_NPZ_PATH.exists()
    ridge_label = "RidgeCV (mid+5sStd)" if use_contrast else f"RidgeCV ({layer_mode})"
    if use_contrast:
        win_ds = load_aligned_dataset(npz_path=DEFAULT_WINDOWS_NPZ_PATH, sync_from_db=True)
        aligned_ds, _, X_contrast = build_contrast_features(dataset, win_ds)
    else:
        aligned_ds = dataset
        X_contrast = None

    train_idx, val_idx, extra = make_holdout_split(aligned_ds, holdout_artists, isolate_vocal_groups=True)
    if extra:
        print(f"\n[Vocal Isolation] Also moved linked artists into validation: {extra}")

    X_tr, y_tr = aligned_ds.embeddings[train_idx], aligned_ds.ratings[train_idx]
    X_va, y_va = aligned_ds.embeddings[val_idx], aligned_ds.ratings[val_idx]
    tr_records = [aligned_ds.records[i] for i in train_idx]

    dummy_p = DummyMeanModel().fit(X_tr, y_tr).predict(X_va)
    knn = CosineKNNModel(k=12, layer_mode=layer_mode).fit(X_tr, y_tr, records=tr_records)
    knn_p = knn.predict(X_va)
    if use_contrast and X_contrast is not None:
        ridge_p = RidgeBaselineModel(layer_mode="mean").fit(X_contrast[train_idx], y_tr).predict(X_contrast[val_idx])
    else:
        ridge_p = RidgeBaselineModel(layer_mode=layer_mode).fit(X_tr, y_tr).predict(X_va)
    torch_model = TorchTasteModel(arch="linear").fit(X_tr, y_tr)
    torch_p, torch_hit_p = torch_model.predict_with_hit_prob(X_va)
    blend_p = 0.60 * ridge_p + 0.40 * torch_p
    blend_hit_score = 0.50 * _zscore(ridge_p) + 0.50 * _zscore(torch_hit_p)

    results = {
        "Dummy Mean": compute_metrics(y_va, dummy_p),
        "Cosine k-NN (k=12)": compute_metrics(y_va, knn_p),
        ridge_label: compute_metrics(y_va, ridge_p),
        "PyTorch LayerAttnLin": compute_metrics(y_va, torch_p, y_hit_score=torch_hit_p),
        "Blend (60R + 40T)": compute_metrics(y_va, blend_p, y_hit_score=blend_hit_score),
    }
    _print_metrics_table(
        f"Custom Holdout Split (Train={len(train_idx)} tracks | Val={len(val_idx)} tracks)",
        results,
    )

    print("\n--- Held-Out Artist Summary (Actual vs. Predicted Mean) ---")
    val_artists = sorted(set(aligned_ds.artists[val_idx].tolist()))
    for art in val_artists:
        mask = aligned_ds.artists[val_idx] == art
        print(
            f"  {art:<18} n={mask.sum():>3} | actual_avg={y_va[mask].mean():.2f} | "
            f"kNN={knn_p[mask].mean():.2f} | Ridge={ridge_p[mask].mean():.2f} | "
            f"PyTorch={torch_p[mask].mean():.2f} | Blend={blend_p[mask].mean():.2f}"
        )

    # Show Sonic Twins for up to 3 held-out tracks
    print("\n--- Sample Sonic Twins (from Training Library) ---")
    for local_i in range(min(3, len(val_idx))):
        g_idx = val_idx[local_i]
        neighbors = knn.explain(aligned_ds.embeddings[g_idx], top_k=3)
        print(
            f"  Query: {aligned_ds.artists[g_idx]} - {aligned_ds.titles[g_idx]} "
            f"(actual={aligned_ds.ratings[g_idx]:.1f}, kNN={knn_p[local_i]:.2f}, "
            f"Ridge={ridge_p[local_i]:.2f}, Blend={blend_p[local_i]:.2f})"
        )
        for nb in neighbors:
            print(f"    -> {nb.similarity:.3f} sim | {nb.rating:.1f}* | {nb.artist_name} - {nb.track_title}")


def run_weight_decay_sweep(dataset: AlignedDataset, n_splits: int = 5) -> None:
    """Sweeps normalization mode (zscore vs l2) and weight_decay for PyTorch LayerAttentionLinear."""
    n = len(dataset.track_ids)
    splits = list(iter_group_kfold_splits(dataset, n_splits=n_splits))
    configs = [
        ("zscore", 0.05),
        ("zscore", 0.25),
        ("zscore", 1.0),
        ("zscore", 3.0),
        ("zscore", 8.0),
        ("l2", 0.01),
        ("l2", 0.05),
        ("l2", 0.25),
        ("l2", 1.0),
    ]
    results: Dict[str, Dict[str, float]] = {}
    for norm_mode, wd in configs:
        oof_s = np.zeros(n, dtype=np.float32)
        oof_h = np.zeros(n, dtype=np.float32)
        for train_idx, val_idx in splits:
            tm = TorchTasteModel(arch="linear", normalize_mode=norm_mode, weight_decay=wd).fit(
                dataset.embeddings[train_idx], dataset.ratings[train_idx]
            )
            oof_s[val_idx], oof_h[val_idx] = tm.predict_with_hit_prob(dataset.embeddings[val_idx])
        results[f"{norm_mode:<6} wd={wd:<4}"] = compute_metrics(dataset.ratings, oof_s, y_hit_score=oof_h)
    _print_metrics_table(f"PyTorch LayerAttnLin Sweep ({n_splits}-Fold Artist-Grouped CV)", results)


def run_windows_benchmark(n_splits: int = 5) -> None:
    """Experiment D: tests 5s temporal contrast (std pooling + temporal blend) vs 60R+40T champion."""
    win_ds = load_aligned_dataset(npz_path=DEFAULT_WINDOWS_NPZ_PATH, sync_from_db=True)
    base_ds = load_aligned_dataset(sync_from_db=True)

    aligned, X_win_all, X_contrast_all = build_contrast_features(base_ds, win_ds)
    X_base_all = aligned.embeddings
    y_all = aligned.ratings

    n = len(aligned.track_ids)
    oof_ridge_mid = np.zeros(n, dtype=np.float32)
    oof_ridge_contrast = np.zeros(n, dtype=np.float32)
    oof_torch_lin = np.zeros(n, dtype=np.float32)
    oof_torch_lin_hit = np.zeros(n, dtype=np.float32)
    oof_temp = np.zeros(n, dtype=np.float32)
    oof_temp_hit = np.zeros(n, dtype=np.float32)

    for train_idx, val_idx in iter_group_kfold_splits(aligned, n_splits=n_splits):
        y_tr = y_all[train_idx]

        oof_ridge_mid[val_idx] = RidgeBaselineModel(layer_mode="mid").fit(X_base_all[train_idx], y_tr).predict(X_base_all[val_idx])
        oof_ridge_contrast[val_idx] = RidgeBaselineModel(layer_mode="mean").fit(X_contrast_all[train_idx], y_tr).predict(X_contrast_all[val_idx])

        tm_lin = TorchTasteModel(arch="linear").fit(X_base_all[train_idx], y_tr)
        oof_torch_lin[val_idx], oof_torch_lin_hit[val_idx] = tm_lin.predict_with_hit_prob(X_base_all[val_idx])

        tm_temp = TorchTasteModel(arch="temporal").fit(X_win_all[train_idx], y_tr)
        oof_temp[val_idx], oof_temp_hit[val_idx] = tm_temp.predict_with_hit_prob(X_win_all[val_idx])

    oof_blend_60_40 = 0.60 * oof_ridge_mid + 0.40 * oof_torch_lin
    oof_blend_std_60_40 = 0.60 * oof_ridge_contrast + 0.40 * oof_torch_lin
    oof_blend_temp = 0.55 * oof_ridge_mid + 0.30 * oof_torch_lin + 0.15 * oof_temp
    oof_hit_rt = 0.50 * _zscore(oof_ridge_mid) + 0.50 * _zscore(oof_torch_lin_hit)
    oof_hit_std_rt = 0.50 * _zscore(oof_ridge_contrast) + 0.50 * _zscore(oof_torch_lin_hit)
    oof_hit_rtt = 0.45 * _zscore(oof_ridge_mid) + 0.40 * _zscore(oof_torch_lin_hit) + 0.15 * _zscore(oof_temp_hit)

    results = {
        "RidgeCV (mid)": compute_metrics(y_all, oof_ridge_mid),
        "RidgeCV (mid+5sStd)": compute_metrics(y_all, oof_ridge_contrast),
        "PyTorch TemporalAttn": compute_metrics(y_all, oof_temp, y_hit_score=oof_temp_hit),
        "Blend (60R + 40T)": compute_metrics(y_all, oof_blend_60_40, y_hit_score=oof_hit_rt),
        "Blend (60R_std+40T)": compute_metrics(y_all, oof_blend_std_60_40, y_hit_score=oof_hit_std_rt),
        "Blend (55R+30T+15Tmp)": compute_metrics(y_all, oof_blend_temp, y_hit_score=oof_hit_rtt),
    }
    _print_metrics_table(f"Experiment D: Hook Contrast ({n} tracks, {n_splits}-Fold CV)", results)


def _softmax_rows(logits: np.ndarray, tau: float = 0.05) -> np.ndarray:
    scaled = logits / max(tau, 1e-6)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def run_clusters_benchmark(n_splits: int = 5, tau: float = 0.05) -> None:
    """Experiment E: tests unsupervised spherical K-Means style clusters + Mixture of Linear Experts."""
    win_ds = load_aligned_dataset(npz_path=DEFAULT_WINDOWS_NPZ_PATH, sync_from_db=True)
    base_ds = load_aligned_dataset(sync_from_db=True)

    aligned, _X_win_all, X_contrast_all = build_contrast_features(base_ds, win_ds)
    X_base_all = aligned.embeddings
    X_mid_norm = _l2_normalize(pool_layers(X_base_all, mode="mid"))
    y_all = aligned.ratings
    n = len(aligned.track_ids)

    # Preview auto-discovered clusters on full dataset for interpretability
    km_preview = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_mid_norm)
    centroids_prev = _l2_normalize(km_preview.cluster_centers_)
    sims_prev = X_mid_norm @ centroids_prev.T
    labels_prev = np.argmax(sims_prev, axis=1)

    print(f"\n=== Experiment E: Auto-Discovered Audio Clusters (K=3, {n} tracks) ===")
    for c in range(3):
        mask = labels_prev == c
        c_ratings = y_all[mask]
        c_sims = sims_prev[mask, c]
        c_indices = np.where(mask)[0]
        top_local = np.argsort(-c_sims)[:5]
        exemplars = [
            f"{aligned.artists[c_indices[j]]} - {aligned.titles[c_indices[j]]} ({y_all[c_indices[j]]:.1f})"
            for j in top_local
        ]
        print(
            f"  Cluster {c}: {int(mask.sum()):4d} tracks | Mean {c_ratings.mean():.2f} | "
            f"Hit {(c_ratings >= 4.0).mean()*100:4.1f}% | Exemplars: {', '.join(exemplars)}"
        )

    oof_ridge_contrast = np.zeros(n, dtype=np.float32)
    oof_torch_lin = np.zeros(n, dtype=np.float32)
    oof_torch_lin_hit = np.zeros(n, dtype=np.float32)
    oof_hard_k3 = np.zeros(n, dtype=np.float32)
    oof_soft_k3 = np.zeros(n, dtype=np.float32)
    oof_soft_k4 = np.zeros(n, dtype=np.float32)
    oof_resid_k3_a10 = np.zeros(n, dtype=np.float32)
    oof_resid_k3_a05 = np.zeros(n, dtype=np.float32)

    for train_idx, val_idx in iter_group_kfold_splits(aligned, n_splits=n_splits):
        y_tr = y_all[train_idx]
        X_tr_con = X_contrast_all[train_idx]
        X_va_con = X_contrast_all[val_idx]

        global_ridge = RidgeBaselineModel(layer_mode="mean").fit(X_tr_con, y_tr)
        va_global = global_ridge.predict(X_va_con)
        oof_ridge_contrast[val_idx] = va_global

        # Honest inner-CV residuals on train_idx so residual experts don't fit in-sample noise
        tr_sub = AlignedDataset(
            track_ids=aligned.track_ids[train_idx],
            embeddings=aligned.embeddings[train_idx],
            ratings=y_tr,
            vocal_groups=aligned.vocal_groups[train_idx],
            artists=aligned.artists[train_idx],
            titles=aligned.titles[train_idx],
            records=[aligned.records[i] for i in train_idx],
        )
        tr_global_oof = np.zeros(len(train_idx), dtype=np.float32)
        for in_tr, in_va in iter_group_kfold_splits(tr_sub, n_splits=3):
            m_in = RidgeBaselineModel(layer_mode="mean").fit(X_tr_con[in_tr], y_tr[in_tr])
            tr_global_oof[in_va] = m_in.predict(X_tr_con[in_va])
        resid_tr = y_tr - tr_global_oof

        tm_lin = TorchTasteModel(arch="linear").fit(X_base_all[train_idx], y_tr)
        oof_torch_lin[val_idx], oof_torch_lin_hit[val_idx] = tm_lin.predict_with_hit_prob(X_base_all[val_idx])

        for k, is_k3 in ((3, True), (4, False)):
            km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_mid_norm[train_idx])
            centroids = _l2_normalize(km.cluster_centers_)
            tr_labels = np.argmax(X_mid_norm[train_idx] @ centroids.T, axis=1)
            va_sims = X_mid_norm[val_idx] @ centroids.T
            va_labels = np.argmax(va_sims, axis=1)
            va_weights = _softmax_rows(va_sims, tau=tau)

            expert_preds = np.zeros((len(val_idx), k), dtype=np.float32)
            resid_preds = np.zeros((len(val_idx), k), dtype=np.float32)
            for c in range(k):
                c_mask = tr_labels == c
                if int(c_mask.sum()) >= 25:
                    expert = RidgeBaselineModel(layer_mode="mean").fit(X_tr_con[c_mask], y_tr[c_mask])
                    expert_preds[:, c] = expert.predict(X_va_con)
                    if is_k3:
                        r_expert = RidgeBaselineModel(layer_mode="mean").fit(X_tr_con[c_mask], resid_tr[c_mask])
                        resid_preds[:, c] = r_expert.predict(X_va_con)
                else:
                    expert_preds[:, c] = va_global

            soft_pred = np.sum(va_weights * expert_preds, axis=1)
            if is_k3:
                oof_soft_k3[val_idx] = soft_pred
                oof_hard_k3[val_idx] = expert_preds[np.arange(len(val_idx)), va_labels]
                soft_resid = np.sum(va_weights * resid_preds, axis=1)
                oof_resid_k3_a10[val_idx] = va_global + 1.0 * soft_resid
                oof_resid_k3_a05[val_idx] = va_global + 0.5 * soft_resid
            else:
                oof_soft_k4[val_idx] = soft_pred

    oof_blend_champ = 0.60 * oof_ridge_contrast + 0.40 * oof_torch_lin
    oof_blend_soft_k3 = 0.60 * oof_soft_k3 + 0.40 * oof_torch_lin
    oof_blend_hybrid = 0.30 * oof_ridge_contrast + 0.30 * oof_soft_k3 + 0.40 * oof_torch_lin
    oof_blend_resid_a10 = 0.60 * oof_resid_k3_a10 + 0.40 * oof_torch_lin
    oof_blend_resid_a05 = 0.60 * oof_resid_k3_a05 + 0.40 * oof_torch_lin

    oof_hit_champ = 0.50 * _zscore(oof_ridge_contrast) + 0.50 * _zscore(oof_torch_lin_hit)
    oof_hit_soft_k3 = 0.50 * _zscore(oof_soft_k3) + 0.50 * _zscore(oof_torch_lin_hit)
    oof_hit_hybrid = 0.25 * _zscore(oof_ridge_contrast) + 0.25 * _zscore(oof_soft_k3) + 0.50 * _zscore(oof_torch_lin_hit)
    oof_hit_resid_a10 = 0.50 * _zscore(oof_resid_k3_a10) + 0.50 * _zscore(oof_torch_lin_hit)
    oof_hit_resid_a05 = 0.50 * _zscore(oof_resid_k3_a05) + 0.50 * _zscore(oof_torch_lin_hit)

    results = {
        "RidgeCV (mid+5sStd)": compute_metrics(y_all, oof_ridge_contrast),
        "Hard-Cluster Ridge (K=3)": compute_metrics(y_all, oof_hard_k3),
        "Soft-Cluster Ridge (K=3)": compute_metrics(y_all, oof_soft_k3),
        "Soft-Cluster Ridge (K=4)": compute_metrics(y_all, oof_soft_k4),
        "Residual Ridge (K=3, a=1.0)": compute_metrics(y_all, oof_resid_k3_a10),
        "Residual Ridge (K=3, a=0.5)": compute_metrics(y_all, oof_resid_k3_a05),
        "Blend (60R_std+40T) [Champ]": compute_metrics(y_all, oof_blend_champ, y_hit_score=oof_hit_champ),
        "Blend (60SoftK3+40T)": compute_metrics(y_all, oof_blend_soft_k3, y_hit_score=oof_hit_soft_k3),
        "Blend (30R_std+30SoftK3+40T)": compute_metrics(y_all, oof_blend_hybrid, y_hit_score=oof_hit_hybrid),
        "Blend (60ResidA10+40T)": compute_metrics(y_all, oof_blend_resid_a10, y_hit_score=oof_hit_resid_a10),
        "Blend (60ResidA05+40T)": compute_metrics(y_all, oof_blend_resid_a05, y_hit_score=oof_hit_resid_a05),
    }
    _print_metrics_table(f"Experiment E: Cluster Experts ({n} tracks, {n_splits}-Fold CV)", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 taste prediction baselines.")
    parser.add_argument("--layers", default="mid", choices=["mean", "upper", "mid", "last"], help="MERT layer pooling mode.")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits.")
    parser.add_argument("--holdout", type=str, default="", help="Comma-separated artist names to hold out for validation.")
    parser.add_argument("--sweep", action="store_true", help="Sweep PyTorch normalization & weight_decay settings.")
    parser.add_argument("--windows", action="store_true", help="Evaluate 5-second TemporalAttentionLinear on window cache.")
    parser.add_argument("--clusters", action="store_true", help="Experiment E: evaluate unsupervised K-Means style experts.")
    args = parser.parse_args()

    if args.windows:
        run_windows_benchmark(n_splits=args.folds)
        return
    if args.clusters:
        run_clusters_benchmark(n_splits=args.folds)
        return

    dataset = load_aligned_dataset(sync_from_db=True)
    print(f"Aligned dataset: {len(dataset.track_ids)} tracks | Mean rating: {dataset.ratings.mean():.2f} | Hit (>=4.0) rate: {(dataset.ratings >= 4.0).mean()*100:.1f}%")

    if args.sweep:
        run_weight_decay_sweep(dataset, n_splits=args.folds)
    elif args.holdout.strip():
        holdout_list = [a.strip() for a in args.holdout.split(",") if a.strip()]
        run_holdout_report(dataset, holdout_list, layer_mode=args.layers)
    else:
        run_cv_benchmark(dataset, n_splits=args.folds, layer_mode=args.layers)


if __name__ == "__main__":
    main()
