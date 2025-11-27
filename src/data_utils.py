import pandas as pd

ID_COLS_CANDIDATES = [
    "obj_ID","run_ID","rerun_ID","cam_col","field_ID","plate","MJD","fiber_ID","spec_obj_ID"
]
SKY_POSITION = ["alpha","delta"]  # usually safe to drop for purely spectral classification

def load_sdss17(csv_path: str) -> pd.DataFrame:
    """Load the Kaggle SDSS17 CSV. Does not downcast types to keep it simple."""
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces, lower camel variants)
    df.columns = [c.strip() for c in df.columns]
    return df

def split_X_y(df: pd.DataFrame, drop_sky=True):
    """Return (X, y) after dropping ids and metadata. Defensive to unknown typos (e.g., rereun_ID)."""
    cols = set(df.columns)
    to_drop = [c for c in ID_COLS_CANDIDATES if c in cols]
    # handle common typos/variants
    if "rereun_ID" in cols and "rerun_ID" not in cols:
        to_drop.append("rereun_ID")
    if drop_sky:
        to_drop += [c for c in SKY_POSITION if c in cols]
    # Ensure label exists
    if "class" not in cols:
        raise ValueError("Expected a 'class' column in the CSV.")
    y = df["class"]
    X = df.drop(columns=list(set(to_drop + ["class"])))
    return X, y
