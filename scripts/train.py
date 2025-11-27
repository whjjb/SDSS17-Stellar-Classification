import argparse, os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data_utils import load_sdss17, split_X_y
from src.features import build_feature_matrix
from src.metrics import compute_metrics, save_metrics_json
from src.models import get_model

def plot_cm(cm, labels, out_png):
    fig = plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/sdss17.csv", help="Path to Kaggle SDSS17 CSV")
    ap.add_argument("--model", type=str, default="logistic", help="logistic | random_forest")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_sdss17(args.csv)
    X_raw, y = split_X_y(df, drop_sky=True)
    X = build_feature_matrix(X_raw)

    # Encode labels robustly
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str).values)
    labels = list(le.classes_)

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(y_enc)
    oof_true = y_enc.copy()

    model_name = args.model
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(tqdm(kf.split(X, y_enc), total=args.folds, desc="Training folds"), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

        clf = get_model(model_name)
        # class weight heuristic for imbalance (logistic only)
        if model_name.startswith("log"):
            # inverse frequency
            classes, counts = np.unique(y_tr, return_counts=True)
            inv_freq = {c: (1.0 / cnt) for c, cnt in zip(classes, counts)}
            s = sum(inv_freq.values())
            class_weight = {c: inv_freq[c] / s * len(classes) for c in classes}
            clf.set_params(clf__class_weight=class_weight)

        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_va)
        m = compute_metrics(y_va, y_hat, labels=list(range(len(labels))))
        m["fold"] = fold
        fold_metrics.append(m)

    # Aggregate
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    summary = {
        "model": model_name,
        "folds": args.folds,
        "labels": labels,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "folds_detail": fold_metrics,
    }
    save_metrics_json(summary, os.path.join(args.outdir, f"cv_{model_name}_metrics.json"))

    # Fit on full data & save model
    final_clf = get_model(model_name)
    final_clf.fit(X, y_enc)
    joblib.dump({"model": final_clf, "label_encoder": le, "feature_columns": list(X.columns)}, os.path.join(args.outdir, f"{model_name}_final.joblib"))

    # Confusion matrix from last fold for quick visualization
    last_cm = fold_metrics[-1]["confusion_matrix"]
    plot_cm(np.array(last_cm), labels, os.path.join(args.outdir, f"cm_{model_name}.png"))

    print("Done. Metrics saved to outputs/.")
    print(json.dumps({"accuracy_mean": summary["accuracy_mean"], "f1_macro_mean": summary["f1_macro_mean"]}, indent=2))

if __name__ == "__main__":
    import json
    main()
