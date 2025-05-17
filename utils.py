import os
from pathlib import Path
import pandas as pd


def get_data_path() -> str:
    """
    Get the path to the KuaiRec data directory.

    Returns:
        str: Path to the data directory.

    Raises:
        FileNotFoundError: If the KuaiRec dataset is not found in any expected location.
    """
    # List of possible data locations (absolute and relative)
    candidates = [
        "/kaggle/input/kuairec/KuaiRec 2.0/data",
        "./KuaiRec/data",
        "../KuaiRec/data",
        "./KuaiRec 2.0/data",
        "../KuaiRec 2.0/data",
        "./data_final_project/KuaiRec 2.0/data",
        "./data_final_project/KuaiRec/data",
    ]
    for path in candidates:
        if Path(path).exists():
            return str(Path(path).resolve())
    raise FileNotFoundError("KuaiRec dataset not found. Please check the path.")


def matrix_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the interaction matrix.

    Args:
        df (pd.DataFrame): DataFrame from small_matrix or big_matrix.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Drop columns if they exist
    for col in ["time", "date"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Convert dtypes where possible
    dtype_map = {
        "user_id": "int32",
        "video_id": "int32",
        "play_duration": "int32",
        "timestamp": "int64",
        "watch_ratio": "float32",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype, errors="ignore")

    # Drop duplicates and missing values
    df = df.drop_duplicates().dropna()

    # Remove invalid timestamps
    if "timestamp" in df.columns:
        df = df[df["timestamp"] >= 0]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    return df.reset_index(drop=True)


def show_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a summary of the dataset, including sparsity.

    Args:
        df (pd.DataFrame): DataFrame with 'user_id' and 'video_id' columns.

    Returns:
        pd.DataFrame: Standard describe output.
    """
    n_users = df["user_id"].nunique() if "user_id" in df.columns else 0
    n_items = df["video_id"].nunique() if "video_id" in df.columns else 0
    n_interactions = len(df)
    print(f"Shape: {df.shape}")
    print(f"Unique users: {n_users}")
    print(f"Unique items: {n_items}")
    if n_users and n_items:
        density = n_interactions / (n_users * n_items)
        print(f"Matrix density: {density:.2%} (sparsity: {1-density:.2%})")
    else:
        print("Cannot compute density (missing user_id or video_id).")
    return df.describe(include="all")


import pandas as pd

def load_csv_with_eval(path, eval_cols=None):
    """Load a CSV and optionally eval specified columns."""
    df = pd.read_csv(path)
    if eval_cols:
        for col in eval_cols:
            if col in df.columns:
                df[col] = df[col].map(eval)
    return df

def load_data(data_dir="data_final_project/KuaiRec 2.0/data"):
    print("Loading datasets...")
    small_matrix = pd.read_csv(f"{data_dir}/small_matrix.csv")
    big_matrix = pd.read_csv(f"{data_dir}/big_matrix.csv")
    social_network = load_csv_with_eval(f"{data_dir}/social_network.csv", eval_cols=["friend_list"])
    item_categories = load_csv_with_eval(f"{data_dir}/item_categories.csv", eval_cols=["feat"])
    user_features = pd.read_csv(f"{data_dir}/user_features.csv")
    item_daily_features = pd.read_csv(f"{data_dir}/item_daily_features.csv")
    print("All data loaded.")
    return big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features

def clean_dataframe(df, frozenset_cols=None):
    """Drop NA, drop duplicates, and reset index. Optionally convert columns to frozenset."""
    df = df.dropna()
    if frozenset_cols:
        for col in frozenset_cols:
            if col in df.columns:
                df[col] = df[col].apply(frozenset)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def clean_data(big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features):
    print("Cleaning data...")
    big_matrix = clean_dataframe(big_matrix)
    small_matrix = clean_dataframe(small_matrix)
    social_network = clean_dataframe(social_network, frozenset_cols=["friend_list"])
    item_categories = clean_dataframe(item_categories, frozenset_cols=["feat"])
    user_features = clean_dataframe(user_features)
    item_daily_features = clean_dataframe(item_daily_features)
    print("Data cleaned.")
    return big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features

def load_and_clean_data(data_dir="data_final_project/KuaiRec 2.0/data"):
    # Load data
    big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features = load_data(data_dir)
    # Store original sizes
    sizes = {
        "big_matrix": len(big_matrix),
        "small_matrix": len(small_matrix),
        "social_network": len(social_network),
        "item_categories": len(item_categories),
        "user_features": len(user_features),
        "item_daily_features": len(item_daily_features),
    }
    # Clean data
    big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features = clean_data(
        big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features
    )
    # Print cleaned percentages
    cleaned = {
        "big_matrix": big_matrix,
        "small_matrix": small_matrix,
        "social_network": social_network,
        "item_categories": item_categories,
        "user_features": user_features,
        "item_daily_features": item_daily_features,
    }
    for name, df in cleaned.items():
        pct = 100 - (len(df) / sizes[name] * 100) if sizes[name] else 0
        print(f"{name.replace('_', ' ').capitalize()}: {pct:.2f}% cleaned")
    return big_matrix, small_matrix, social_network, item_categories, user_features, item_daily_features