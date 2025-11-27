import argparse, os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from src.data_utils import load_sdss17, split_X_y
from src.features import build_feature_matrix
from src.metrics import compute_metrics, save_metrics_json
from src.models import get_model

def barplot_top_features(names, importances, out_png, topk=15):
    order = np.argsort(importances)[::-1][:topk]
    names = np.array(names)[order]
    vals = np.array(importances)[order]
    plt.figure(figsize=(7, 4.5))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.title("Top Features")
    plt.xlabel("Importance / |Coefficient|")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

def get_param_distributions(model_name):
    if model_name.startswith("rf") or model_name == "random_forest":
        return {
            "clf__n_estimators": [200, 400, 600, 800, 1000, 1200],
            "clf__max_depth": [None, 6, 8, 10, 12, 16, 20, 30],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__max_features": ["sqrt", "log2", None, 0.5, 0.7, 0.9],
            "clf__bootstrap": [True],
            "clf__class_weight": [None, "balanced"]
        }
    else:
        return {
            "clf__C": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "clf__penalty": ["l2", "l1"],
            "clf__class_weight": [None, "balanced"]
        }

def extract_feature_importance(model, feature_names):
    est = model.named_steps["clf"]
    if hasattr(est, "feature_importances_"):
        imps = est.feature_importances_
    elif hasattr(est, "coef_"):
        imps = np.mean(np.abs(est.coef_), axis=0)
    else:
        imps = np.zeros(len(feature_names))
    return feature_names, imps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/sdss17.csv")
    ap.add_argument("--model", type=str, default="random_forest", help="random_forest | logistic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_sdss17(args.csv)
    X_raw, y = split_X_y(df, drop_sky=True)
    X = build_feature_matrix(X_raw)
    feature_names = list(X.columns)

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str).values)
    labels_text = list(le.classes_)
    labels_idx = list(range(len(labels_text)))

    base = get_model(args.model)
    param_dist = get_param_distributions(args.model)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=args.n_iter,
        scoring="f1_macro", n_jobs=-1, cv=cv, refit=True, random_state=args.seed, verbose=1
    )
    search.fit(X, y_enc)

    # Save CV results table
    results_df = pd.DataFrame(search.cv_results_)
    results_csv = os.path.join(args.outdir, f"search_{args.model}_cv_results.csv")
    results_df.to_csv(results_csv, index=False)

    best = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)

    # Fresh CV summary + confusion matrix
    fold_metrics = []

    for fold, (tr, va) in enumerate(tqdm(cv.split(X, y_enc), total=args.folds, desc="Cross-validation folds"), 1):
        clf = clone(best)
        clf.fit(X.iloc[tr], y_enc[tr])
        pred = clf.predict(X.iloc[va])
        m = compute_metrics(y_enc[va], pred, labels=labels_idx)
        m["fold"] = fold
        fold_metrics.append(m)

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    summary = {
        "model": args.model,
        "best_params": best_params,
        "cv_f1_macro_best": best_score,
        "folds": args.folds,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "labels_text": labels_text,
        "folds_detail": fold_metrics,
    }
    out_json = os.path.join(args.outdir, f"search_{args.model}_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Fit on full data and export
    final = clone(best).fit(X, y_enc)
    bundle_path = os.path.join(args.outdir, f"{args.model}_best_final.joblib")
    joblib.dump({"model": final, "label_encoder": le, "feature_columns": feature_names}, bundle_path)

    # Top features (global)
    fnames, imps = extract_feature_importance(final, feature_names)
    top_png = os.path.join(args.outdir, f"top_features_{args.model}.png")
    barplot_top_features(fnames, imps, top_png, topk=15)

    # Confusion matrix from last fold
    def plot_cm(cm, labels_text, out_png):
        fig = plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        tick_marks = np.arange(len(labels_text))
        plt.xticks(tick_marks, labels_text, rotation=45, ha="right")
        plt.yticks(tick_marks, labels_text)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    last_cm = fold_metrics[-1]["confusion_matrix"]
    plot_cm(np.array(last_cm), labels_text, os.path.join(args.outdir, f"cm_{args.model}_best.png"))

    print(json.dumps({
        "best_params": best_params,
        "cv_f1_macro_best": best_score,
        "acc_mean": float(np.mean(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "bundle": bundle_path,
        "top_features_png": top_png
    }, indent=2))
    print(f"Saved CV table to {results_csv}")
    print(f"Saved summary to {out_json}")
    print("Done.")

if __name__ == "__main__":
    main()
