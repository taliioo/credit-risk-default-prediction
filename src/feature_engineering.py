from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import json as _json

# Notebook 03: columns dropped after feature engineering / schema consolidation
BASE_DROP_COLS = [
    "issue_d",              # helper raw field used for feature engineering
    "addr_state",           # zip3 retained as the geographic feature
    "disbursement_method",  # near-constant (>99% Cash)
]

MULTICOLL_DROP = [
    # Cluster 1: account counts
    "num_sats",
    "num_rev_tl_bal_gt_0",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_actv_bc_tl",
    "num_bc_sats",
    "num_bc_tl",

    # Cluster 2: recent activity
    "open_rv_24m",
    "open_acc_6m",
    "open_rv_12m",

    # Cluster 3: credit limits
    "bc_open_to_buy",
    "total_bc_limit",

    # Cluster 4: delinquency timing
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",

    # Cluster 5: balance aggregates
    "tot_hi_cred_lim",
    "avg_cur_bal",

    # Cluster 6: utilization
    "bc_util",
    "percent_bc_gt_75",

    # Cluster 7: installment balances
    "total_bal_il",
    "total_il_high_credit_limit",

    # Cluster 8: current delinquency
    "num_tl_30dpd",

    # Cluster 10: recent installment
    "open_il_24m",
    "open_il_12m",

    # Cluster 11: public records
    "tax_liens",
]

NOTEBOOK03_DROP_COLS = BASE_DROP_COLS + MULTICOLL_DROP

# Notebook 01: raw/base-table simplification done before Notebook 03
NOTEBOOK01_REDUNDANT_DROP_COLS = [
    "grade",
    "fico_range_low",
    "fico_range_high",
    "funded_amnt",
    "funded_amnt_inv",
    "zip_code",
]

SUBGRADE_ORDER = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
SUBGRADE_MAP = {g: i + 1 for i, g in enumerate(SUBGRADE_ORDER)}

DEFAULT_ZIP3_SMOOTHING = 40
DEFAULT_ZIP3_MIN_COUNT = 50


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """
    Configuration for the notebook-aligned feature engineering pipeline.

    Default behavior mirrors the notebooks closely:
    - optionally creates zip3 and fico_avg if the upstream table still has the
      raw columns from Notebook 01
    - engineers the derived features from Notebook 03
    - drops helper / redundant / multicollinear columns
    - optionally performs the light categorical cleanup from Notebook 03
    """

    create_zip3_if_missing: bool = True
    create_fico_avg_if_missing: bool = True
    apply_notebook03_drop_cols: bool = True
    apply_notebook01_redundant_drops: bool = True
    clean_categoricals: bool = True

    notebook03_drop_cols: Sequence[str] = field(
        default_factory=lambda: tuple(NOTEBOOK03_DROP_COLS)
    )
    notebook01_redundant_drop_cols: Sequence[str] = field(
        default_factory=lambda: tuple(NOTEBOOK01_REDUNDANT_DROP_COLS)
    )


# zip3 target encoding helpers
class TargetEncoder:
    """
    Smoothed target encoder for one categorical column.

    For each category:
        encoded_value = (count * group_mean + smoothing * global_mean) / (count + smoothing)

    This matches the notebook logic used for zip3.
    """

    def __init__(self, smoothing: int = 100):
        self.smoothing = smoothing
        self.global_mean_: float | None = None
        self.mapping_: dict[object, float] | None = None

    def fit(self, X: pd.DataFrame | pd.Series, y: Sequence[float] | np.ndarray) -> "TargetEncoder":
        col = _coerce_single_column(X).astype("object")
        target = np.asarray(y, dtype=float)

        stats = pd.DataFrame({"val": col, "target": target})
        agg = stats.groupby("val", dropna=False)["target"].agg(["mean", "count"])

        self.global_mean_ = float(np.mean(target))
        smooth = (
            agg["count"] * agg["mean"] + self.smoothing * self.global_mean_
        ) / (agg["count"] + self.smoothing)

        self.mapping_ = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame | pd.Series) -> np.ndarray:
        if self.mapping_ is None or self.global_mean_ is None:
            raise ValueError("TargetEncoder must be fitted before transform().")

        col = _coerce_single_column(X).astype("object")
        return col.map(self.mapping_).fillna(self.global_mean_).to_numpy(dtype=float)

    def to_artifact(self) -> dict:
        """Serialize encoder state into a plain dictionary for saving with pickle/json."""
        if self.mapping_ is None or self.global_mean_ is None:
            raise ValueError("TargetEncoder must be fitted before serialization.")

        return {
            "type": "TargetEncoder",
            "smoothing": self.smoothing,
            "global_mean_": self.global_mean_,
            "mapping_": self.mapping_,
        }

    @classmethod
    def from_artifact(cls, artifact: dict) -> "TargetEncoder":
        """Recreate a fitted encoder from a saved artifact dictionary."""
        enc = cls(smoothing=int(artifact.get("smoothing", 100)))
        enc.global_mean_ = float(artifact["global_mean_"])
        enc.mapping_ = dict(artifact["mapping_"])
        return enc


def save_zip3_artifact(
    encoder: TargetEncoder,
    keepers: set,
    path: str | Path,
) -> None:
    """
    Save a fitted zip3 TargetEncoder and its keepers set to a JSON file.

    This is the recommended way to persist the zip3 encoding for inference.
    JSON is portable, human-readable, and avoids pickle security concerns.

    Parameters
    ----------
    encoder : TargetEncoder
        A fitted TargetEncoder instance.
    keepers : set
        The set of zip3 values that are kept as-is (not collapsed to 'RARE').
    path : str or Path
        Output file path (should end in .json).
    """
    artifact = {
        "encoder": encoder.to_artifact(),
        "keepers": sorted(str(k) for k in keepers),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump(artifact, f, indent=2)


def load_zip3_artifact(path: str | Path) -> tuple[TargetEncoder, set]:
    """
    Load a fitted zip3 TargetEncoder and its keepers set from a JSON file.

    Returns
    -------
    encoder : TargetEncoder
        A fitted TargetEncoder ready for transform().
    keepers : set
        The set of zip3 values to keep as-is before encoding.

    Example
    -------
    >>> from feature_engineering import load_zip3_artifact, apply_feature_pipeline
    >>> encoder, keepers = load_zip3_artifact("models/zip3_encoder.json")
    >>> df_new = apply_feature_pipeline(
    ...     df_raw,
    ...     encode_zip3=True,
    ...     zip3_encoder=encoder,
    ...     zip3_keepers=keepers,
    ... )
    """
    path = Path(path)
    with open(path) as f:
        artifact = _json.load(f)

    encoder = TargetEncoder.from_artifact(artifact["encoder"])

    # Reconstruct keepers: try to convert back to int if they were originally numeric
    raw_keepers = artifact["keepers"]
    keepers = set()
    for k in raw_keepers:
        try:
            keepers.add(int(k))
        except (ValueError, TypeError):
            keepers.add(k)

    return encoder, keepers


def _coerce_single_column(X: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(X, pd.Series):
        return X
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame.")
        return X.iloc[:, 0]
    raise TypeError("Expected a pandas Series or single-column DataFrame.")


def bucket_rare_zip3_values(
    series: pd.Series,
    keepers: Iterable[object],
) -> pd.Series:
    """
    Map zip3 values using training-derived keepers.

    - values in keepers are kept as-is
    - everything else becomes 'RARE'
    - missing values become 'MISSING'
    """
    keepers = set(keepers)
    out = series.astype("object").copy()
    out = out.where(out.isin(keepers), "RARE")
    out = out.fillna("MISSING")
    return out


def fit_zip3_target_encoder(
    train_zip3: pd.Series,
    y_train: Sequence[int] | np.ndarray,
    *,
    smoothing: int = DEFAULT_ZIP3_SMOOTHING,
    min_count: int = DEFAULT_ZIP3_MIN_COUNT,
) -> tuple[TargetEncoder, set[object]]:
    """
    Fit the final zip3 target encoder on the full training set.

    Returns
    -------
    encoder : TargetEncoder
        Fitted final encoder used for val / holdout / new inference data.
    keepers : set
        Training-derived zip3 groups that are not collapsed to 'RARE'.
        Save this together with the encoder so inference can apply the same
        rare-bucketing policy as training.
    """
    counts = train_zip3.astype("object").value_counts(dropna=False)
    keepers = set(counts[counts >= min_count].index)

    train_bucketed = bucket_rare_zip3_values(train_zip3, keepers)
    encoder = TargetEncoder(smoothing=smoothing)
    encoder.fit(train_bucketed.to_frame(name="zip3"), y_train)
    return encoder, keepers


def apply_fitted_zip3_encoder(
    df: pd.DataFrame,
    zip3_encoder: TargetEncoder | dict,
    *,
    keepers: Iterable[object] | None = None,
    raw_col: str = "zip3",
    encoded_col: str = "zip3_encoded",
    drop_raw_col: bool = True,
) -> pd.DataFrame:
    """
    Apply a previously fitted zip3 encoder and create a numeric zip3_encoded column.

    Use this for validation, holdout, or production inference data after
    feature engineering has already created the raw zip3 column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a raw zip3 column.
    zip3_encoder : TargetEncoder or dict
        Either a fitted TargetEncoder instance or a serialized artifact dict.
    keepers : iterable, optional
        Training-derived zip3 values kept as-is before encoding. Anything not in
        keepers is mapped to 'RARE'. Missing values are mapped to 'MISSING'.
        Pass the same keepers returned by fit_zip3_target_encoder().
    raw_col : str, default='zip3'
        Name of the raw zip3 column.
    encoded_col : str, default='zip3_encoded'
        Name of the output numeric encoded column.
    drop_raw_col : bool, default=True
        Whether to drop the raw zip3 column after encoding.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_fitted_zip3_encoder expects a pandas DataFrame.")

    if raw_col not in df.columns:
        raise KeyError(f"Column '{raw_col}' not found in DataFrame.")

    if isinstance(zip3_encoder, dict):
        zip3_encoder = TargetEncoder.from_artifact(zip3_encoder)

    if not hasattr(zip3_encoder, "transform"):
        raise TypeError("zip3_encoder must be a fitted TargetEncoder or artifact dictionary.")

    out = df.copy()
    zip3_values = out[raw_col]

    if keepers is not None:
        zip3_values = bucket_rare_zip3_values(zip3_values, keepers)
    else:
        zip3_values = zip3_values.astype("object").fillna("MISSING")

    out[encoded_col] = zip3_encoder.transform(zip3_values.to_frame(name=raw_col))

    if drop_raw_col:
        out = out.drop(columns=[raw_col])

    return out



# Generic helpers
def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_denominator(series: pd.Series) -> pd.Series:
    return _to_numeric(series).replace(0, np.nan)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return _to_numeric(numerator) / _safe_denominator(denominator)


def _create_zip3(zip_code: pd.Series) -> pd.Series:
    """Convert zip_code like '123xx' to zip3 = 123."""
    return (
        zip_code.astype("string")
        .str.extract(r"(\d{3})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )


def _create_fico_avg(df: pd.DataFrame) -> pd.Series:
    low = _to_numeric(df["fico_range_low"])
    high = _to_numeric(df["fico_range_high"])
    return (low + high) / 2


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light categorical cleanup from Notebook 03, Section 6.1.

    - term: strip spaces
    - home_ownership: map ANY/NONE to OTHER
    - emp_length: fill missing as Unknown
    """
    df = df.copy()

    if "term" in df.columns:
        df["term"] = df["term"].astype("string").str.strip()

    if "home_ownership" in df.columns:
        df["home_ownership"] = (
            df["home_ownership"]
            .astype("string")
            .replace({"ANY": "OTHER", "NONE": "OTHER"})
        )

    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].astype("string").fillna("Unknown")

    return df



# Core notebook-aligned feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the derived features used in Notebook 03.

    This function intentionally excludes train-fitted preprocessing such as:
    - missing indicator selection based on training missingness rates
    - median/mode imputation
    - one-hot encoding / ordinal encoding
    - target encoding
    - scaling

    Those belong in preprocessing.py or train.py.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("engineer_features expects a pandas DataFrame.")

    df = df.copy()

    # Credit age at origination
    if {"issue_d", "earliest_cr_line"}.issubset(df.columns):
        issue_dt = pd.to_datetime(df["issue_d"], errors="coerce")
        ecl_dt = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
        df["credit_age_months"] = ((issue_dt - ecl_dt).dt.days / 30.44).round(0)

    # Safe denominators
    safe_inc = _safe_denominator(df["annual_inc"]) if "annual_inc" in df.columns else None
    safe_total_acc = _safe_denominator(df["total_acc"]) if "total_acc" in df.columns else None
    safe_open_acc = _safe_denominator(df["open_acc"]) if "open_acc" in df.columns else None
    safe_rev_limit = (
        _safe_denominator(df["total_rev_hi_lim"]) if "total_rev_hi_lim" in df.columns else None
    )
    safe_bc_limit = (
        _safe_denominator(df["total_bc_limit"]) if "total_bc_limit" in df.columns else None
    )

    # Existing features from earlier notebook drafts / project design
    if {"loan_amnt", "annual_inc"}.issubset(df.columns):
        df["loan_to_income"] = _to_numeric(df["loan_amnt"]) / safe_inc

    if {"installment", "annual_inc"}.issubset(df.columns):
        df["installment_to_income"] = (_to_numeric(df["installment"]) * 12) / safe_inc

    if {"revol_bal", "annual_inc"}.issubset(df.columns):
        df["revol_bal_to_income"] = _to_numeric(df["revol_bal"]) / safe_inc

    # New domain features from Notebook 03
    if {"open_acc", "total_acc"}.issubset(df.columns):
        df["open_acc_ratio"] = _to_numeric(df["open_acc"]) / safe_total_acc

    if {"revol_bal", "total_rev_hi_lim"}.issubset(df.columns):
        df["revol_bal_to_limit"] = _to_numeric(df["revol_bal"]) / safe_rev_limit

    if {"total_bal_ex_mort", "annual_inc"}.issubset(df.columns):
        df["total_debt_to_income"] = _to_numeric(df["total_bal_ex_mort"]) / safe_inc

    if {"bc_open_to_buy", "total_bc_limit"}.issubset(df.columns):
        df["bc_open_to_buy_ratio"] = _to_numeric(df["bc_open_to_buy"]) / safe_bc_limit

    if {"tot_cur_bal", "open_acc"}.issubset(df.columns):
        df["avg_bal_per_account"] = _to_numeric(df["tot_cur_bal"]) / safe_open_acc

    if {"num_accts_ever_120_pd", "total_acc"}.issubset(df.columns):
        df["delinq_ratio"] = _to_numeric(df["num_accts_ever_120_pd"]) / safe_total_acc

    if {"inq_last_6mths", "open_acc"}.issubset(df.columns):
        df["recent_inquiry_intensity"] = _to_numeric(df["inq_last_6mths"]) / (
            _to_numeric(df["open_acc"]) + 1
        )

    if {"mort_acc", "total_acc"}.issubset(df.columns):
        df["mort_acc_ratio"] = _to_numeric(df["mort_acc"]) / safe_total_acc

    # Log transforms
    if "annual_inc" in df.columns:
        df["log_annual_inc"] = np.log1p(_to_numeric(df["annual_inc"]).clip(lower=0))

    if "revol_bal" in df.columns:
        df["log_revol_bal"] = np.log1p(_to_numeric(df["revol_bal"]).clip(lower=0))

    if "loan_amnt" in df.columns:
        df["log_loan_amnt"] = np.log1p(_to_numeric(df["loan_amnt"]).clip(lower=0))

    # Interaction: sub_grade x dti
    if {"sub_grade", "dti"}.issubset(df.columns):
        df["subgrade_num"] = df["sub_grade"].map(SUBGRADE_MAP)
        df["subgrade_dti_interaction"] = df["subgrade_num"] * _to_numeric(df["dti"])

    # Missing count across the row
    df["missing_count"] = df.isna().sum(axis=1)

    # Drop helper raw date fields, matching Notebook 03 engineer_features()
    df = df.drop(columns=[c for c in ["issue_d", "earliest_cr_line"] if c in df.columns])

    # Defensive cleanup in case any upstream values cause infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def apply_feature_pipeline(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig | None = None,
    *,
    zip3_encoder: TargetEncoder | dict | None = None,
    zip3_keepers: Iterable[object] | None = None,
    encode_zip3: bool = False,
    drop_raw_zip3_after_encoding: bool = True,
) -> pd.DataFrame:
    """
    Apply the notebook-aligned feature engineering pipeline to one DataFrame.

    Order of operations
    -------------------
    1. Optionally create zip3 / fico_avg if the input still contains raw fields
       from an earlier stage than Notebook 03.
    2. Create derived features from Notebook 03.
    3. Optionally apply a previously fitted zip3 encoder to create zip3_encoded.
    4. Apply Notebook 03 drop lists.
    5. Optionally apply Notebook 01 redundant-column drops.
    6. Optionally clean categoricals.

    Notes
    -----
    - encode_zip3=True is intended for validation / holdout / inference data,
      where you already have a fitted zip3 encoder artifact from training.
    - For the training split itself, keep OOF target encoding logic in
      preprocessing.py or train.py.
    """
    if config is None:
        config = FeatureEngineeringConfig()

    df = df.copy()

    # Optional backward-compatible consolidation from Notebook 01
    if config.create_zip3_if_missing and "zip3" not in df.columns and "zip_code" in df.columns:
        df["zip3"] = _create_zip3(df["zip_code"])

    if (
        config.create_fico_avg_if_missing
        and "fico_avg" not in df.columns
        and {"fico_range_low", "fico_range_high"}.issubset(df.columns)
    ):
        df["fico_avg"] = _create_fico_avg(df)

    # Main Notebook 03 feature engineering
    df = engineer_features(df)

    # Optional model-ready zip3 encoding using a previously fitted mapping
    if encode_zip3:
        if zip3_encoder is None:
            raise ValueError("encode_zip3=True requires a fitted zip3_encoder.")
        if "zip3" not in df.columns:
            raise KeyError("Raw 'zip3' column not found. Cannot apply zip3 target encoding.")
        df = apply_fitted_zip3_encoder(
            df,
            zip3_encoder=zip3_encoder,
            keepers=zip3_keepers,
            drop_raw_col=drop_raw_zip3_after_encoding,
        )

    # Drop columns documented in Notebook 03
    if config.apply_notebook03_drop_cols:
        df = df.drop(
            columns=[c for c in config.notebook03_drop_cols if c in df.columns],
            errors="ignore",
        )

    # Optional redundant cleanup from Notebook 01
    if config.apply_notebook01_redundant_drops:
        df = df.drop(
            columns=[c for c in config.notebook01_redundant_drop_cols if c in df.columns],
            errors="ignore",
        )

    # Categorical cleanup from Notebook 03
    if config.clean_categoricals:
        df = clean_categoricals(df)

    return df


# Convenience helper for applying the same feature pipeline to multiple splits
def apply_feature_pipeline_to_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame | None = None,
    df_holdout: pd.DataFrame | None = None,
    config: FeatureEngineeringConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Apply the same raw feature engineering pipeline consistently across
    train / val / holdout.

    This helper intentionally does not perform zip3 target encoding because the
    training split should use OOF encoding and the other splits should use the
    final fitted mapping from training.
    """
    if config is None:
        config = FeatureEngineeringConfig()

    out_train = apply_feature_pipeline(df_train, config=config)
    out_val = apply_feature_pipeline(df_val, config=config) if df_val is not None else None
    out_holdout = (
        apply_feature_pipeline(df_holdout, config=config)
        if df_holdout is not None
        else None
    )
    return out_train, out_val, out_holdout



# Simple CLI so the module can be run as a script
def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .parquet.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .parquet.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Notebook-aligned feature engineering pipeline for the Lending Club project."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .csv or .parquet file")
    parser.add_argument("--output", type=Path, required=True, help="Output .csv or .parquet file")
    parser.add_argument(
        "--zip3-artifact", type=Path, default=None,
        help="Path to a saved zip3_encoder.json file. When provided, zip3 target "
             "encoding is applied using the saved mapping (inference mode).",
    )
    parser.add_argument(
        "--skip-notebook03-drops",
        action="store_true",
        help="Keep Notebook 03 drop-list columns instead of removing them.",
    )
    parser.add_argument(
        "--skip-notebook01-redundant-drops",
        action="store_true",
        help="Keep Notebook 01 redundant raw columns instead of removing them.",
    )
    parser.add_argument(
        "--skip-categorical-cleaning",
        action="store_true",
        help="Skip the light categorical cleanup step.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df_in = read_table(args.input)
    config = FeatureEngineeringConfig(
        apply_notebook03_drop_cols=not args.skip_notebook03_drops,
        apply_notebook01_redundant_drops=not args.skip_notebook01_redundant_drops,
        clean_categoricals=not args.skip_categorical_cleaning,
    )

    # Determine whether to apply zip3 encoding
    encode_zip3 = args.zip3_artifact is not None
    zip3_encoder = None
    zip3_keepers = None

    if encode_zip3:
        zip3_encoder, zip3_keepers = load_zip3_artifact(args.zip3_artifact)
        print(f"Loaded zip3 encoder from: {args.zip3_artifact}")
        print(f"  Keepers: {len(zip3_keepers)} zip3 groups")
        print(f"  Global mean: {zip3_encoder.global_mean_:.6f}")

    df_out = apply_feature_pipeline(
        df_in,
        config=config,
        encode_zip3=encode_zip3,
        zip3_encoder=zip3_encoder,
        zip3_keepers=zip3_keepers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_table(df_out, args.output)

    print("\nFeature engineering complete.")
    print(f"Input shape:  {df_in.shape}")
    print(f"Output shape: {df_out.shape}")
    print(f"Saved to:     {args.output}")


if __name__ == "__main__":
    main()
