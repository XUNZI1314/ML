"""Core utility helpers for nanobody pocket blocking screening projects.

This module is intentionally lightweight and depends only on the standard
library, numpy and pandas so it can be reused across preprocessing and
training scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import os
import random

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GeometryConfig:
    """Geometry thresholds used by pocket-blocking candidate filtering.

    Attributes:
        distance_cutoff: Maximum allowed distance (Angstrom) for contact checks.
        angle_min_deg: Minimum allowed angle in degrees.
        angle_max_deg: Maximum allowed angle in degrees.
        pocket_overlap_threshold: Minimum overlap score to keep a candidate.
        clash_distance: Distance threshold below which heavy-atom clashes are flagged.
    """

    distance_cutoff: float = 6.0
    angle_min_deg: float = 70.0
    angle_max_deg: float = 180.0
    pocket_overlap_threshold: float = 0.30
    clash_distance: float = 2.2

    def __post_init__(self) -> None:
        """Validate and sanitize geometry values in-place."""
        self.distance_cutoff = max(safe_to_float(self.distance_cutoff, default=6.0), 1e-8)
        self.clash_distance = max(safe_to_float(self.clash_distance, default=2.2), 1e-8)

        min_angle = safe_to_float(self.angle_min_deg, default=70.0)
        max_angle = safe_to_float(self.angle_max_deg, default=180.0)
        min_angle = float(np.clip(min_angle, 0.0, 180.0))
        max_angle = float(np.clip(max_angle, 0.0, 180.0))
        if max_angle < min_angle:
            max_angle = min_angle
        self.angle_min_deg = min_angle
        self.angle_max_deg = max_angle

        overlap = safe_to_float(self.pocket_overlap_threshold, default=0.30)
        self.pocket_overlap_threshold = float(np.clip(overlap, 0.0, 1.0))


@dataclass(slots=True)
class TrainConfig:
    """Basic training hyper-parameters used by downstream scripts.

    Attributes:
        batch_size: Mini-batch size.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization coefficient.
        early_stopping_patience: Early-stopping patience in epochs.
        gradient_clip_norm: Maximum gradient norm for clipping.
        seed: Random seed.
        num_workers: Number of data-loading workers.
        device: Preferred runtime device label.
    """

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate and sanitize training values in-place."""
        self.batch_size = max(int(safe_to_float(self.batch_size, default=32)), 1)
        self.epochs = max(int(safe_to_float(self.epochs, default=50)), 1)
        self.learning_rate = max(safe_to_float(self.learning_rate, default=1e-3), 1e-12)
        self.weight_decay = max(safe_to_float(self.weight_decay, default=1e-4), 0.0)
        self.early_stopping_patience = max(
            int(safe_to_float(self.early_stopping_patience, default=10)), 1
        )
        self.gradient_clip_norm = max(
            safe_to_float(self.gradient_clip_norm, default=5.0), 0.0
        )
        self.seed = int(safe_to_float(self.seed, default=42))
        self.num_workers = max(int(safe_to_float(self.num_workers, default=0)), 0)
        self.device = str(self.device).strip() or "cpu"


@dataclass(slots=True)
class ProjectConfig:
    """Project-level common configuration bundle.

    Attributes:
        project_name: Human-readable project name.
        data_csv: Input CSV file path.
        output_dir: Directory where outputs are stored.
        required_columns: Required columns in the input CSV.
        geometry: Geometry-related thresholds.
        train: Training hyper-parameters.
    """

    project_name: str = "nanobody_pocket_screen"
    data_csv: str = "data/input.csv"
    output_dir: str = "outputs"
    required_columns: tuple[str, ...] = (
        "nanobody_id",
        "protein_id",
        "pocket_id",
        "label",
    )
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole configuration as a plain dictionary."""
        return asdict(self)


def log_message(message: str, level: str = "INFO", show_time: bool = True) -> None:
    """Print a compact log line with optional timestamp.

    Args:
        message: Content to print.
        level: Log level label, for example "INFO", "WARN", "ERROR".
        show_time: Whether to include local timestamp.
    """
    safe_level = str(level).strip().upper() or "INFO"
    if show_time:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{safe_level}] {message}")
    else:
        print(f"[{safe_level}] {message}")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    This function always seeds Python's built-in random module and numpy.
    If PyTorch is installed, it also seeds torch and CUDA backends.

    Args:
        seed: Random seed integer. Invalid values fall back to 42.
    """
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        seed_int = 42
        log_message(f"Invalid seed={seed!r}; fallback to {seed_int}.", level="WARN")

    if seed_int < 0:
        seed_int = abs(seed_int)
        log_message(f"Negative seed detected; use abs(seed)={seed_int}.", level="WARN")

    os.environ["PYTHONHASHSEED"] = str(seed_int)
    random.seed(seed_int)
    np.random.seed(seed_int)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional; silently continue when unavailable.
        pass


def ensure_dir(path: str | os.PathLike[str]) -> None:
    """Create a directory recursively if it does not exist.

    Args:
        path: Directory path.

    Raises:
        ValueError: If path is empty.
    """
    if path is None:
        raise ValueError("Path cannot be None.")

    path_str = str(path).strip()
    if not path_str:
        raise ValueError("Path cannot be empty.")

    Path(path_str).expanduser().mkdir(parents=True, exist_ok=True)


def safe_to_float(
    x: Any,
    default: float = np.nan,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> float:
    """Safely convert a value to float with robust NaN/Inf handling.

    Args:
        x: Input object to convert.
        default: Fallback value when conversion fails or value is disallowed.
        allow_nan: Whether NaN values are allowed to pass through.
        allow_inf: Whether +/-Inf values are allowed to pass through.

    Returns:
        A float value or fallback default.
    """
    fallback = float(default)

    if x is None:
        return fallback

    if isinstance(x, str):
        text = x.strip()
        if not text:
            return fallback
        if text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return fallback
        x = text

    try:
        value = float(x)
    except (TypeError, ValueError):
        return fallback

    if np.isnan(value) and not allow_nan:
        return fallback
    if np.isinf(value) and not allow_inf:
        return fallback
    return value


def sanitize_numeric_array(
    values: Any,
    default: float = np.nan,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Convert input data to numeric numpy array and sanitize invalid values.

    This function coerces non-numeric entries to NaN, then replaces non-finite
    values (NaN, +Inf, -Inf) with ``default``.

    Args:
        values: Scalar, sequence, numpy array, pandas Series or DataFrame.
        default: Replacement value for non-finite entries.
        dtype: Target numpy dtype.

    Returns:
        A numeric numpy array with invalid values sanitized.
    """
    if isinstance(values, pd.DataFrame):
        numeric_df = values.apply(pd.to_numeric, errors="coerce")
        arr = numeric_df.to_numpy(dtype=np.float64, copy=True)
    elif isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    else:
        obj_arr = np.asarray(values, dtype=object)
        if obj_arr.size == 0:
            return obj_arr.astype(dtype, copy=False)
        flat = pd.to_numeric(obj_arr.reshape(-1), errors="coerce")
        arr = np.asarray(flat, dtype=np.float64).reshape(obj_arr.shape)

    arr = arr.astype(np.float64, copy=False)
    non_finite_mask = ~np.isfinite(arr)
    if np.any(non_finite_mask):
        arr[non_finite_mask] = float(default)
    return arr.astype(dtype, copy=False)


def check_required_columns(
    df: pd.DataFrame,
    required_cols: Iterable[str] | None,
    raise_error: bool = True,
) -> list[str]:
    """Check whether a DataFrame contains all required columns.

    Args:
        df: Input DataFrame.
        required_cols: Required column names.
        raise_error: If True, raise ValueError when columns are missing.

    Returns:
        List of missing columns (empty if all present).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    if required_cols is None:
        return []

    required = [str(col) for col in required_cols]
    missing = [col for col in required if col not in df.columns]

    if missing and raise_error:
        raise ValueError(f"Missing required columns: {missing}")
    return missing


def read_csv_with_checks(
    csv_path: str | os.PathLike[str],
    required_cols: Iterable[str] | None = None,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file and optionally validate required columns.

    Args:
        csv_path: CSV file path.
        required_cols: Required columns to validate after reading.
        encoding: Preferred file encoding.
        **kwargs: Extra arguments passed to ``pandas.read_csv``.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    read_kwargs = dict(kwargs)
    read_kwargs.setdefault("low_memory", False)

    try:
        df = pd.read_csv(path, encoding=encoding, **read_kwargs)
    except UnicodeDecodeError:
        # Fallback for common UTF BOM files.
        fallback_encoding = "utf-8-sig" if encoding.lower() != "utf-8-sig" else "gbk"
        df = pd.read_csv(path, encoding=fallback_encoding, **read_kwargs)

    check_required_columns(df, required_cols, raise_error=True)
    return df


def _restore_input_type(original: Any, data: np.ndarray) -> Any:
    """Restore numpy output to pandas type when original input is pandas."""
    if isinstance(original, pd.Series):
        return pd.Series(data, index=original.index, name=original.name)
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame(data, index=original.index, columns=original.columns)
    if np.isscalar(original):
        return float(np.asarray(data).reshape(-1)[0])
    return data


def robust_minmax_scale(
    values: Any,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    clip: bool = True,
    eps: float = 1e-12,
    invalid_fill: float = 0.0,
) -> Any:
    """Scale values into [0, 1] with quantile-based robust bounds.

    Compared with classic min-max scaling, this method computes bounds from
    quantiles to reduce the influence of outliers.

    Args:
        values: Input data, can be scalar/list/numpy/pandas.
        lower_quantile: Lower quantile in [0, 1].
        upper_quantile: Upper quantile in [0, 1].
        clip: Whether to clip output values into [0, 1].
        eps: Minimum denominator to avoid division by zero.
        invalid_fill: Fill value for invalid input entries.

    Returns:
        Scaled data with the same outer type as input where possible.
    """
    arr = sanitize_numeric_array(values, default=np.nan, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    out = np.full(arr.shape, float(invalid_fill), dtype=np.float64)

    if not np.any(finite_mask):
        return _restore_input_type(values, out)

    lq = safe_to_float(lower_quantile, default=0.01)
    uq = safe_to_float(upper_quantile, default=0.99)
    lq = float(np.clip(lq, 0.0, 1.0))
    uq = float(np.clip(uq, 0.0, 1.0))
    if lq >= uq:
        lq, uq = 0.0, 1.0

    finite_values = arr[finite_mask]
    lower = float(np.nanquantile(finite_values, lq))
    upper = float(np.nanquantile(finite_values, uq))

    denom = upper - lower
    if (not np.isfinite(denom)) or (denom < eps):
        out[finite_mask] = 0.5
        return _restore_input_type(values, out)

    scaled = (arr[finite_mask] - lower) / denom
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    out[finite_mask] = scaled
    return _restore_input_type(values, out)


class ZScoreScaler:
    """A robust z-score standardizer with NaN/Inf-safe behavior.

    This class supports numpy arrays and pandas objects. During fitting,
    non-finite values are ignored. During transformation, non-finite values are
    filled with ``invalid_fill``.
    """

    def __init__(
        self,
        ddof: int = 0,
        eps: float = 1e-12,
        clip: tuple[float, float] | None = None,
        invalid_fill: float = np.nan,
    ) -> None:
        """Initialize scaler parameters.

        Args:
            ddof: Delta degrees of freedom used by numpy std.
            eps: Minimum allowed std to avoid divide-by-zero.
            clip: Optional clipping range after z-score transform.
            invalid_fill: Fill value for invalid inputs during transform.
        """
        self.ddof = int(ddof)
        self.eps = max(float(eps), 1e-15)
        self.clip = clip
        self.invalid_fill = float(invalid_fill)

        self.mean_: np.ndarray | float | None = None
        self.std_: np.ndarray | float | None = None
        self.n_features_: int | None = None
        self.is_fitted_: bool = False

    def fit(self, values: Any) -> "ZScoreScaler":
        """Estimate mean and std from input data.

        Args:
            values: Input array-like data.

        Returns:
            self.
        """
        arr = sanitize_numeric_array(values, default=np.nan, dtype=np.float64)
        finite_mask = np.isfinite(arr)

        if arr.ndim <= 1:
            self.n_features_ = 1
        else:
            self.n_features_ = int(arr.shape[-1])

        if not np.any(finite_mask):
            self.mean_ = 0.0 if self.n_features_ == 1 else np.zeros(self.n_features_)
            self.std_ = 1.0 if self.n_features_ == 1 else np.ones(self.n_features_)
            self.is_fitted_ = True
            return self

        clean = np.where(finite_mask, arr, np.nan)
        axis = 0 if arr.ndim > 1 else None

        mean = np.nanmean(clean, axis=axis)
        std = np.nanstd(clean, axis=axis, ddof=self.ddof)

        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std >= self.eps), std, 1.0)

        self.mean_ = mean
        self.std_ = std
        self.is_fitted_ = True
        return self

    def _check_feature_compatibility(self, arr: np.ndarray) -> None:
        """Validate that input feature dimension matches fitted dimension."""
        if not self.is_fitted_:
            raise RuntimeError("ZScoreScaler must be fitted before transform.")

        if arr.ndim <= 1:
            current_features = 1
        else:
            current_features = int(arr.shape[-1])

        if self.n_features_ is not None and current_features != self.n_features_:
            raise ValueError(
                f"Feature mismatch: fitted={self.n_features_}, got={current_features}"
            )

    def transform(self, values: Any) -> Any:
        """Apply z-score standardization.

        Args:
            values: Input array-like data.

        Returns:
            Standardized data with type restored when possible.
        """
        arr = sanitize_numeric_array(values, default=np.nan, dtype=np.float64)
        self._check_feature_compatibility(arr)

        assert self.mean_ is not None
        assert self.std_ is not None

        finite_mask = np.isfinite(arr)
        out = np.full(arr.shape, self.invalid_fill, dtype=np.float64)

        scaled = (arr - self.mean_) / self.std_
        if self.clip is not None:
            low, high = self.clip
            if low > high:
                low, high = high, low
            scaled = np.clip(scaled, low, high)

        out[finite_mask] = scaled[finite_mask]
        return _restore_input_type(values, out)

    def fit_transform(self, values: Any) -> Any:
        """Fit the scaler and transform input in one step.

        Args:
            values: Input array-like data.

        Returns:
            Standardized data.
        """
        return self.fit(values).transform(values)

    def inverse_transform(self, values: Any) -> Any:
        """Reconstruct original-scale values from z-scores.

        Args:
            values: Standardized values.

        Returns:
            Data transformed back to the fitted scale.
        """
        arr = sanitize_numeric_array(values, default=np.nan, dtype=np.float64)
        self._check_feature_compatibility(arr)

        assert self.mean_ is not None
        assert self.std_ is not None

        finite_mask = np.isfinite(arr)
        out = np.full(arr.shape, self.invalid_fill, dtype=np.float64)

        restored = arr * self.std_ + self.mean_
        out[finite_mask] = restored[finite_mask]
        return _restore_input_type(values, out)


__all__ = [
    "GeometryConfig",
    "TrainConfig",
    "ProjectConfig",
    "log_message",
    "set_seed",
    "ensure_dir",
    "safe_to_float",
    "sanitize_numeric_array",
    "check_required_columns",
    "read_csv_with_checks",
    "robust_minmax_scale",
    "ZScoreScaler",
]
