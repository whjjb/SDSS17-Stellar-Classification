# scripts/visualize.py
import os, json, glob, argparse, shutil, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

MODELS = ["logistic", "mlp", "random_forest", "xgboost"]

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def set_theme():
    # Simple, clean style (no "paper" layout)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "grid.alpha": 0.25,
    })

def title_of(model):
    return {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "logistic": "Logistic Regression",
        "mlp": "MLP",
    }.get(model, model.replace("_", " ").title())

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def load_summary(artifact_dir, model):
    # Prefer search_*_summary.json (contains folds + labels). Fallback to cv_*_metrics.json.
    search = os.path.join(artifact_dir, f"search_{model}_summary.json")
    cvjson = os.path.join(artifact_dir, f"cv_{model}_metrics.json")
    if os.path.exists(search):
        d = json.load(open(search, "r", encoding="utf-8"))
        labels = d.get("labels_text") or d.get("labels") or ["0","1","2"]
        folds  = d.get("folds_detail", [])
        mean   = d.get("f1_macro_mean", d.get("cv_f1_macro_best"))
        std    = d.get("f1_macro_std", 0.0)
        return d, labels, folds, mean, std
    elif os.path.exists(cvjson):
        d = json.load(open(cvjson, "r", encoding="utf-8"))
        labels = d.get("labels") or ["0","1","2"]
        folds  = d.get("folds_detail", [])
        mean   = d.get("f1_macro_mean")
        std    = d.get("f1_macro_std", 0.0)
        return d, labels, folds, mean, std
    else:
        return None, None, [], None, None

def annotate_bars(ax, fmt="{:.3f}", pad=0.01):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h): continue
        ax.text(p.get_x()+p.get_width()/2, h + pad, fmt.format(h),
                ha="center", va="bottom")

def plot_leaderboard(leaderboard, out_png):
    if not leaderboard: return
    names = np.array(list(leaderboard.keys()))
    means = np.array([leaderboard[m][0] for m in names])
    stds  = np.array([leaderboard[m][1] for m in names])
    order = np.argsort(means)[::-1]
    names, means, stds = names[order], means[order], stds[order]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(range(len(names)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([title_of(n) for n in names])
    ax.set_ylabel("Macro F1 (CV)")
    ax.set_title("Model Leaderboard")
    annotate_bars(ax, "{:.3f}", 0.01)
    savefig(out_png)

def plot_fold_curves(model, folds, out_png):
    if not folds: return
    xs  = [f["fold"] for f in folds]
    f1  = [f["f1_macro"] for f in folds]
    acc = [f["accuracy"] for f in folds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, f1, marker="o", linewidth=2, label="Macro F1")
    ax.plot(xs, acc, marker="o", linewidth=2, label="Accuracy")
    ax.set_xlabel("Fold"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title(f"{title_of(model)} — CV per-fold scores")
    ax.legend()
    savefig(out_png)

def plot_cm_grid(model, folds, labels, out_png):
    if not folds: return
    k = len(folds); cols = min(k, 3); rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2*cols, 4.4*rows))
    axes = np.array(axes).reshape(rows, cols)

    vmin, vmax = 0.0, 1.0
    for idx, f in enumerate(folds):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cm = np.array(f["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, vmin=vmin, vmax=vmax, cmap="Blues")
        ax.set_title(f"Fold {f['fold']}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                ax.text(j, i, f"{cmn[i, j]*100:.1f}%", ha="center", va="center", fontsize=9)

        # colorbar only on last column
        if c == cols - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Row-norm %", rotation=270, labelpad=12)

    # remove unused axes
    for blank in range(k, rows*cols):
        r, c = divmod(blank, cols)
        fig.delaxes(axes[r, c])

    fig.suptitle(f"{title_of(model)} — Confusion Matrices (row-normalized)", y=1.02)
    savefig(out_png)

def plot_perclass_bars_lastfold(model, folds, labels, out_png,
                                legend_pos="outside_right"):  # "outside_right" | "outside_top" | "inside"
    if not folds:
        return
    rep = folds[-1]["classification_report"]

    prec, rec, f1 = [], [], []
    for i in range(len(labels)):
        k = str(i)
        if k in rep:
            prec.append(rep[k]["precision"])
            rec.append(rep[k]["recall"])
            f1.append(rep[k]["f1-score"])
        else:
            prec.append(0.0); rec.append(0.0); f1.append(0.0)

    x = np.arange(len(labels)); w = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w, prec, width=w, label="Precision")
    ax.bar(x,       rec, width=w, label="Recall")
    ax.bar(x + w,    f1, width=w, label="F1")

    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(f"{model.replace('_',' ').title()} — Per-class metrics (last fold)")

    # legend placement (never overlaps)
    if legend_pos == "outside_right":
        # put legend on the right, and free space for it
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False)
        fig.subplots_adjust(right=0.82)
    elif legend_pos == "outside_top":
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        fig.subplots_adjust(top=0.85)
    else:  # "inside"
        ax.legend(loc="upper left", frameon=False)

    # numeric labels on bars
    for arr, dx in [(prec, -w), (rec, 0), (f1, w)]:
        for i, v in enumerate(arr):
            ax.text(i + dx, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # save
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_top_features_from_joblib(model, artifact_dir, out_png, topk=15):
    for name in [f"{model}_best_final.joblib", f"{model}_final.joblib"]:
        p = os.path.join(artifact_dir, name)
        if not os.path.exists(p):
            continue
        b = joblib.load(p)
        pipe = b["model"]
        names = np.array(b.get("feature_columns", []))
        est = pipe.named_steps.get("clf", None)
        if est is None or len(names) == 0:
            continue

        if hasattr(est, "feature_importances_"):
            vals = np.array(est.feature_importances_)
        elif hasattr(est, "coef_"):
            vals = np.mean(np.abs(est.coef_), axis=0)
        else:
            continue

        order = np.argsort(vals)[::-1][:topk]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(order)), vals[order])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names[order])
        ax.invert_yaxis()
        ax.set_xlabel("Importance / |Coefficient|")
        ax.set_title(f"{title_of(model)} — Top {len(order)} features")
        for i, v in enumerate(vals[order]):
            ax.text(v * 1.01, i, f"{v:.3f}", va="center")
        savefig(out_png)
        return True
    return False

def plot_rf_tree_depth(artifact_dir, out_png):
    p = None
    for name in ["random_forest_best_final.joblib", "random_forest_final.joblib"]:
        cand = os.path.join(artifact_dir, name)
        if os.path.exists(cand):
            p = cand; break
    if p is None: return
    b = joblib.load(p)
    est = b["model"].named_steps.get("clf", None)
    if est is None or not hasattr(est, "estimators_"): return
    depths = [t.get_depth() for t in est.estimators_ if hasattr(t, "get_depth")]
    if not depths: return
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = min(20, max(5, int(math.sqrt(len(depths)))))
    ax.hist(depths, bins=bins, edgecolor="white")
    ax.axvline(np.mean(depths), color="C1", linestyle="--", linewidth=2, label=f"mean={np.mean(depths):.1f}")
    ax.axvline(np.median(depths), color="C2", linestyle=":", linewidth=2, label=f"median={np.median(depths):.1f}")
    ax.set_xlabel("Tree depth"); ax.set_ylabel("Count")
    ax.set_title("Random Forest — Tree depth distribution")
    ax.legend()
    savefig(out_png)

def copy_existing_cms(src_dir, dst_dir, model):
    for name in [f"cm_{model}.png", f"cm_{model}_best.png"]:
        p = os.path.join(src_dir, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_dir, name))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--figs_dir", default="figs")
    args = ap.parse_args()

    set_theme()
    figs_root = ensure_dir(args.figs_dir)

    leaderboard = {}
    for model in MODELS:
        # Support both layouts: outputs/<model>/ or flat under outputs/
        art_dir = os.path.join(args.outputs_dir, model)
        if not os.path.isdir(art_dir):
            flat_json = os.path.join(args.outputs_dir, f"cv_{model}_metrics.json")
            art_dir = args.outputs_dir if os.path.exists(flat_json) else None
        if art_dir is None or not os.path.exists(art_dir):
            continue

        out_dir = ensure_dir(os.path.join(figs_root, model))
        summary, labels, folds, mean, std = load_summary(art_dir, model)

        if summary is not None:
            plot_fold_curves(model, folds, os.path.join(out_dir, "cv_fold_scores.png"))
            plot_cm_grid(model, folds, labels, os.path.join(out_dir, "cm_grid_row_norm.png"))
            plot_perclass_bars_lastfold(model, folds, labels, os.path.join(out_dir, "per_class_bars_lastfold.png"))

        plot_top_features_from_joblib(model, art_dir, os.path.join(out_dir, "top_features_from_joblib.png"))

        if model == "random_forest":
            plot_rf_tree_depth(art_dir, os.path.join(out_dir, "rf_tree_depth.png"))

        copy_existing_cms(art_dir, out_dir, model)

        if mean is not None:
            leaderboard[model] = (float(mean), float(std or 0.0))

    plot_leaderboard(leaderboard, os.path.join(figs_root, "leaderboard_all_models.png"))
    print(f"[OK] Figures saved under: {figs_root}")

if __name__ == "__main__":
    main()
