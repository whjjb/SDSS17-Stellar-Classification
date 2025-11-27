from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def make_xgboost():
    return Pipeline([
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=600,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",   # CPU：hist"；GPU："gpu_hist"
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ))
    ])

def make_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial", solver="saga", n_jobs=-1))
    ])

def make_random_forest():
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42))
    ])

def make_mlp():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,               # L2
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=200,
            early_stopping=True,      # 自动留出验证子集做早停
            n_iter_no_change=20,
            random_state=42
        ))
    ])

def get_model(name: str):
    name = name.lower()
    if name in ["logreg","logistic","lr"]:
        return make_logistic()
    if name in ["rf","random_forest","random-forest"]:
        return make_random_forest()
    if name in ["xgb","xgboost"]:
        return make_xgboost()
    if name in ["mlp","nn","neural","neural_net"]:
        return make_mlp()
    raise ValueError(f"Unknown model '{name}'.")