"""
Technical Analysis Utilities

Shared utilities for technical indicator calculations:
- Data validation and alignment
- Array conversion and normalization
- Common helper functions
- Error handling utilities
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_numpy_array(
    data: list[float] | np.ndarray | pd.Series, allow_nan: bool = False
) -> np.ndarray:
    """
    Convert input data to numpy array with enhanced error handling.

    Args:
        data: Input data in various formats
        allow_nan: Whether to allow NaN values

    Returns:
        Clean numpy array

    Raises:
        ValueError: If data is invalid or contains NaN when not allowed
    """
    if data is None:
        raise ValueError("Input data cannot be None")

    if isinstance(data, list) and not data:
        raise ValueError("Input list cannot be empty")

    try:
        array = data.values if isinstance(
            data, pd.Series) else np.asarray(
            data, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert data to numpy array: {e}")

    # Validate array
    if array.size == 0:
        raise ValueError("Input array cannot be empty")

    # Check for NaN values
    if not allow_nan and np.isnan(array).any():
        nan_count = int(np.isnan(array).sum())
        raise ValueError(
            f"Array contains {nan_count} NaN values (not allowed)")

    # Check for infinite values
    if np.isinf(array).any():
        inf_count = int(np.isinf(array).sum())
        logger.warning(f"Array contains {inf_count} infinite values")
        # Replace inf with NaN for consistency
        array = np.where(np.isinf(array), np.nan, array)

    return array


def validate_and_align_arrays(
    *arrays: list[float] | np.ndarray | pd.Series,
    min_length: int = 1,
    allow_nan: bool = False,
    alignment: str = "right",
) -> tuple[np.ndarray, ...]:
    """
    Validate and align multiple arrays to the same length.

    Args:
        *arrays: Multiple arrays to validate and align
        min_length: Minimum required length after alignment
        allow_nan: Whether to allow NaN values
        alignment: How to align arrays ('right', 'left', 'center')

    Returns:
        Tuple of aligned numpy arrays

    Raises:
        ValueError: If arrays cannot be aligned or validated
    """
    if not arrays:
        raise ValueError("At least one array must be provided")

    # Convert all to numpy arrays
    np_arrays = []
    original_lengths = []

    for i, array in enumerate(arrays):
        try:
            np_array = to_numpy_array(array, allow_nan=allow_nan)
            np_arrays.append(np_array)
            original_lengths.append(len(np_array))

        except Exception as e:
            raise ValueError(f"Error processing array {i}: {e}")

    # Find target length
    if alignment == "right":
        # Align to shortest array (most recent data)
        target_length = min(original_lengths)
        aligned_arrays = [arr[-target_length:] for arr in np_arrays]

    elif alignment == "left":
        # Align to shortest array (oldest data)
        target_length = min(original_lengths)
        aligned_arrays = [arr[:target_length] for arr in np_arrays]

    elif alignment == "center":
        # Align to shortest array (center data)
        target_length = min(original_lengths)
        aligned_arrays = []

        for arr in np_arrays:
            if len(arr) == target_length:
                aligned_arrays.append(arr)
            else:
                start_idx = (len(arr) - target_length) // 2
                end_idx = start_idx + target_length
                aligned_arrays.append(arr[start_idx:end_idx])

    else:
        raise ValueError(
            f"Invalid alignment: {alignment}. Must be 'right', 'left', or 'center'"
        )

    # Validate final length
    if target_length < min_length:
        raise ValueError(
            f"Aligned arrays too short: {target_length} < {min_length}")

    # Log alignment info if significant truncation occurred
    max_original = max(original_lengths)
    if max_original - target_length > max_original * 0.1:  # More than 10% truncated
        logger.info(
            f"Array alignment: {original_lengths} -> {target_length} "
            f"({max_original - target_length} values truncated, alignment='{alignment}')")

    return tuple(aligned_arrays)


def safe_divide(
    numerator: float | np.ndarray,
    denominator: float | np.ndarray,
    default_value: float = 0.0,
) -> float | np.ndarray:
    """
    Perform safe division avoiding division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default_value: Value to return when denominator is zero

    Returns:
        Result of division or default_value where denominator is zero
    """
    # Handle scalar case
    if np.isscalar(numerator) and np.isscalar(denominator):
        try:
            numerator_float = (
                float(numerator) if isinstance(
                    numerator, int | float | str) else 0.0)
            denominator_float = (
                float(denominator)
                if isinstance(denominator, int | float | str)
                else 1.0
            )
            if denominator_float == 0:
                return float(default_value)
            return numerator_float / denominator_float
        except (TypeError, ValueError, OverflowError):
            return float(default_value)

    # Handle array case
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)

    # Use numpy's divide with where parameter
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, default_value, dtype=float),
        where=(denominator != 0),
    )


def calculate_returns(
    prices: list[float] | np.ndarray, periods: int = 1, method: str = "simple"
) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        periods: Number of periods for return calculation
        method: 'simple' or 'log' returns

    Returns:
        Array of returns (length = len(prices) - periods)
    """
    prices_array = to_numpy_array(prices)

    if len(prices_array) <= periods:
        raise ValueError(
            f"Price array too short: {len(prices_array)} <= {periods}")

    if method == "simple":
        # Simple returns: (P_t - P_{t-n}) / P_{t-n}
        returns = (prices_array[periods:] -
                   prices_array[:-periods]) / prices_array[:-periods]

    elif method == "log":
        # Log returns: ln(P_t / P_{t-n})
        returns = np.log(prices_array[periods:] / prices_array[:-periods])

    else:
        raise ValueError(
            f"Invalid method: {method}. Must be 'simple' or 'log'")

    return returns


def calculate_rolling_stats(
    data: list[float] | np.ndarray, window: int, stats: list[str] | None = None
) -> dict:
    """
    Calculate rolling statistics for a data series.

    Args:
        data: Input data series
        window: Rolling window size
        stats: List of statistics to calculate ['mean', 'std', 'min', 'max', 'median']

    Returns:
        Dictionary with requested rolling statistics
    """
    if stats is None:
        stats = ["mean", "std"]

    data_array = to_numpy_array(data)

    if len(data_array) < window:
        raise ValueError(f"Data length {len(data_array)} < window {window}")

    results = {}

    # Calculate each requested statistic
    if "mean" in stats:
        results["mean"] = np.convolve(
            data_array, np.ones(window) / window, mode="valid"
        )

    if "std" in stats:
        results["std"] = np.array(
            [
                np.std(data_array[i: i + window])
                for i in range(len(data_array) - window + 1)
            ]
        )

    if "min" in stats:
        results["min"] = np.array(
            [
                np.min(data_array[i: i + window])
                for i in range(len(data_array) - window + 1)
            ]
        )

    if "max" in stats:
        results["max"] = np.array(
            [
                np.max(data_array[i: i + window])
                for i in range(len(data_array) - window + 1)
            ]
        )

    if "median" in stats:
        results["median"] = np.array(
            [
                np.median(data_array[i: i + window])
                for i in range(len(data_array) - window + 1)
            ]
        )

    return results


def detect_outliers(
    data: list[float] | np.ndarray, method: str = "iqr", multiplier: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in data series.

    Args:
        data: Input data series
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        multiplier: Threshold multiplier

    Returns:
        Tuple of (outlier_mask, cleaned_data)
    """
    data_array = to_numpy_array(data, allow_nan=True)
    outlier_mask = np.zeros(len(data_array), dtype=bool)

    # Remove NaN for calculations
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]

    if len(valid_data) == 0:
        return outlier_mask, data_array.copy()

    if method == "iqr":
        # Interquartile Range method
        q1, q3 = np.percentile(valid_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers = (data_array < lower_bound) | (data_array > upper_bound)

    elif method == "zscore":
        # Z-score method
        mean = np.mean(valid_data)
        std = np.std(valid_data)

        if std == 0:
            outliers = np.zeros(len(data_array), dtype=bool)
        else:
            z_scores = np.abs((data_array - mean) / std)
            outliers = z_scores > multiplier

    elif method == "modified_zscore":
        # Modified Z-score using median
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))

        if mad == 0:
            outliers = np.zeros(len(data_array), dtype=bool)
        else:
            modified_z_scores = 0.6745 * (data_array - median) / mad
            outliers = np.abs(modified_z_scores) > multiplier

    else:
        raise ValueError(f"Invalid method: {method}")

    outlier_mask = outliers & valid_mask  # Only mark valid data as outliers
    cleaned_data = data_array.copy()
    cleaned_data[outlier_mask] = np.nan

    return outlier_mask, cleaned_data


def smooth_data(
        data: list[float] | np.ndarray,
        method: str = "sma",
        window: int = 5,
        **kwargs) -> np.ndarray:
    """
    Smooth data using various methods.

    Args:
        data: Input data series
        method: Smoothing method ('sma', 'ema', 'gaussian', 'savgol')
        window: Window size for smoothing
        **kwargs: Additional parameters for specific methods

    Returns:
        Smoothed data array
    """
    data_array = to_numpy_array(data, allow_nan=True)

    if method == "sma":
        # Simple Moving Average
        kernel = np.ones(window) / window
        smoothed = np.convolve(data_array, kernel, mode="same")

    elif method == "ema":
        # Exponential Moving Average
        alpha = kwargs.get("alpha", 2.0 / (window + 1))
        smoothed = np.empty_like(data_array)
        smoothed[0] = data_array[0]

        for i in range(1, len(data_array)):
            if np.isnan(data_array[i]):
                smoothed[i] = smoothed[i - 1]
            else:
                smoothed[i] = alpha * data_array[i] + \
                    (1 - alpha) * smoothed[i - 1]

    elif method == "gaussian":
        # Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        sigma = kwargs.get("sigma", window / 6.0)
        smoothed = gaussian_filter1d(data_array, sigma=sigma)

    elif method == "savgol":
        # Savitzky-Golay filter
        from scipy.signal import savgol_filter  # type: ignore

        polyorder = kwargs.get("polyorder", min(3, window - 1))
        if window % 2 == 0:
            window += 1  # Must be odd
        smoothed = savgol_filter(data_array, window, polyorder)

    else:
        raise ValueError(f"Invalid smoothing method: {method}")

    return smoothed


def validate_indicator_params(**params) -> dict[str, int | float | Any]:
    """
    Validate common indicator parameters.

    Args:
        **params: Dictionary of parameters to validate

    Returns:
        Dictionary of validated parameters

    Raises:
        ValueError: If any parameter is invalid
    """
    validated: dict[str, int | float | Any] = {}

    # Common validations
    for key, value in params.items():
        if key.endswith(("_period", "_window")) or key == "period":
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{key} must be a positive integer, got {value}")
            validated[key] = value

        elif key in ["multiplier", "factor", "threshold"]:
            if not isinstance(value, int | float) or value <= 0:
                raise ValueError(
                    f"{key} must be a positive number, got {value}")
            validated[key] = float(value)

        elif key in ["overbought", "oversold"]:
            if not isinstance(value, int | float) or not (0 <= value <= 100):
                raise ValueError(
                    f"{key} must be between 0 and 100, got {value}")
            validated[key] = float(value)

        elif key.endswith("_ratio"):
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{key} must be non-negative, got {value}")
            validated[key] = float(value)

        else:
            # Pass through other parameters
            validated[key] = value

    return validated


def calculate_true_range(
    highs: list[float] | np.ndarray,
    lows: list[float] | np.ndarray,
    closes: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Calculate True Range for volatility indicators.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices

    Returns:
        Array of True Range values
    """
    highs, lows, closes = validate_and_align_arrays(
        highs, lows, closes, min_length=2)

    # True Range = max(high-low, high-prev_close, prev_close-low)
    tr_values = []

    # First value is just high - low
    tr_values.append(highs[0] - lows[0])

    # Subsequent values use the full formula
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_prev_close = abs(highs[i] - closes[i - 1])
        prev_close_low = abs(closes[i - 1] - lows[i])

        tr_values.append(max(high_low, high_prev_close, prev_close_low))

    return np.array(tr_values)


def handle_missing_data(
    data: list[float] | np.ndarray,
    method: str = "forward_fill",
    max_consecutive: int = 5,
) -> np.ndarray:
    """
    Handle missing data in price series.

    Args:
        data: Input data with potential missing values
        method: Handling method ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        max_consecutive: Maximum consecutive missing values to fill

    Returns:
        Array with missing data handled
    """
    data_array = to_numpy_array(data, allow_nan=True)

    if method == "drop":
        # Remove all NaN values
        return data_array[~np.isnan(data_array)]

    result = data_array.copy()

    if method == "forward_fill":
        # Forward fill missing values
        for i in range(1, len(result)):
            if np.isnan(result[i]):
                # Count consecutive NaN values
                consecutive = 0
                for j in range(i, len(result)):
                    if np.isnan(result[j]):
                        consecutive += 1
                    else:
                        break

                # Fill if within limit
                if consecutive <= max_consecutive and i > 0:
                    result[i] = result[i - 1]

    elif method == "backward_fill":
        # Backward fill missing values
        for i in range(len(result) - 2, -1, -1):
            if np.isnan(result[i]):
                # Count consecutive NaN values
                consecutive = 0
                for j in range(i, -1, -1):
                    if np.isnan(result[j]):
                        consecutive += 1
                    else:
                        break

                # Fill if within limit
                if consecutive <= max_consecutive and i < len(result) - 1:
                    result[i] = result[i + 1]

    elif method == "interpolate":
        # Linear interpolation
        valid_indices = ~np.isnan(result)
        if np.any(valid_indices):
            result = np.interp(
                np.arange(len(result)),
                np.arange(len(result))[valid_indices],
                result[valid_indices],
            )

    else:
        raise ValueError(f"Invalid method: {method}")

    return result
