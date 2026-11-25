import gymnasium as gym
import numpy as np
import pandas as pd
import os
import hashlib
import json
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
import logging
import warnings
from collections import deque

import torch
from stable_baselines3.common.monitor import ResultsWriter
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import config

# =========================================================================
# ✅ PERFORMANCE FIX: Global data cache to avoid expensive reloading
# NOTE: Scaler is NO LONGER cached globally - we use PER-EPISODE scaling
# to match live trading behavior where scaler is fit on each trading day's data
# =========================================================================
_GLOBAL_DATA_CACHE = None


def _scaled_log_portfolio_growth(previous_value, current_value, scale=100.0):
    """Return the causal one-step portfolio log growth used by Bellman TD."""
    previous = float(previous_value)
    current = float(current_value)
    reward_scale = float(scale)
    if (
        not np.isfinite(previous)
        or not np.isfinite(current)
        or previous <= 0.0
        or current <= 0.0
    ):
        return 0.0
    if not np.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("financial reward scale must be finite and positive")
    return float(reward_scale * np.log(current / previous))


def _scaled_asset_financial_contribution(
    target_weight,
    execution_price,
    next_price,
    scale=100.0,
):
    """Return weight times the causal one-step asset return, or None if invalid."""
    weight = float(target_weight)
    start_price = float(execution_price)
    end_price = float(next_price)
    reward_scale = float(scale)
    if (
        not np.isfinite(weight)
        or not np.isfinite(start_price)
        or not np.isfinite(end_price)
        or start_price <= 0.0
        or end_price <= 0.0
    ):
        return None
    if not np.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("financial reward scale must be finite and positive")
    return float(reward_scale * weight * ((end_price / start_price) - 1.0))


def _scaled_asset_financial_return(
    execution_price,
    next_price,
    scale=100.0,
):
    """Return the causal unit-exposure asset return, or None if invalid."""
    start_price = float(execution_price)
    end_price = float(next_price)
    reward_scale = float(scale)
    if (
        not np.isfinite(start_price)
        or not np.isfinite(end_price)
        or start_price <= 0.0
        or end_price <= 0.0
    ):
        return None
    if not np.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("financial reward scale must be finite and positive")
    return float(reward_scale * ((end_price / start_price) - 1.0))


def _causal_open_log_returns(open_prices):
    """One-bar log returns without fabricating values across missing prices."""
    prices = np.asarray(open_prices, dtype=np.float64)
    if prices.ndim == 1:
        prices = prices[:, None]

    returns = np.full(prices.shape, np.nan, dtype=np.float64)
    if prices.shape[0] <= 1:
        return returns

    previous = prices[:-1]
    current = prices[1:]
    valid = (
        np.isfinite(previous)
        & np.isfinite(current)
        & (previous > 0.0)
        & (current > 0.0)
    )
    step_returns = np.full(previous.shape, np.nan, dtype=np.float64)
    step_returns[valid] = np.log(current[valid] / previous[valid])
    returns[1:] = step_returns
    return returns


def _session_codes_for_timestamps(timestamps):
    """Return stable integer session codes from timestamp calendar dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    return pd.factorize(dates, sort=False)[0]


def _drop_session_boundary_returns(returns, session_codes):
    """Remove overnight gaps from intraday rolling relationship statistics."""
    cleaned = np.asarray(returns, dtype=np.float64).copy()
    codes = np.asarray(session_codes)
    if cleaned.shape[0] != len(codes):
        raise ValueError(
            "session_codes length must match the return-series time dimension"
        )
    if len(codes) > 0:
        boundary = np.ones(len(codes), dtype=bool)
        boundary[1:] = codes[1:] != codes[:-1]
        cleaned[boundary] = np.nan
    return cleaned


def _detect_corporate_action_factors(
    open_prices,
    close_prices,
    timestamps,
):
    """Detect split-like session discontinuities without masking ordinary gaps.

    Detection uses current open / previous close because that is the cleanest
    available price-level continuity check.  The returned factor is one for an
    ordinary observation and the nearest canonical split factor for a detected
    corporate action.
    """
    opens = np.asarray(open_prices, dtype=np.float64)
    closes = np.asarray(close_prices, dtype=np.float64)
    if opens.ndim != 2 or closes.shape != opens.shape:
        raise ValueError(
            "open_prices and close_prices must share [time, active_assets] shape"
        )

    factors = np.ones_like(opens, dtype=np.float64)
    events = np.zeros_like(opens, dtype=bool)
    if opens.shape[0] <= 1 or timestamps is None:
        return factors, events

    session_codes = _session_codes_for_timestamps(timestamps)
    boundary = np.zeros(opens.shape[0], dtype=bool)
    boundary[1:] = session_codes[1:] != session_codes[:-1]
    previous_close = np.full_like(opens, np.nan, dtype=np.float64)
    previous_close[1:] = closes[:-1]
    valid = (
        boundary[:, None]
        & np.isfinite(opens)
        & np.isfinite(previous_close)
        & (opens > 0.0)
        & (previous_close > 0.0)
    )
    price_ratio = np.ones_like(opens, dtype=np.float64)
    price_ratio[valid] = opens[valid] / previous_close[valid]
    absolute_log_jump = np.abs(np.log(price_ratio))

    canonical = np.asarray(
        config.CORPORATE_ACTION_SPLIT_FACTORS, dtype=np.float64
    )
    log_distance = np.abs(
        np.log(price_ratio[..., None] / canonical.reshape(1, 1, -1))
    )
    nearest_index = np.argmin(log_distance, axis=-1)
    nearest_distance = np.take_along_axis(
        log_distance, nearest_index[..., None], axis=-1
    )[..., 0]
    detected = (
        valid
        & (
            absolute_log_jump
            >= float(config.CORPORATE_ACTION_MIN_ABS_LOG_JUMP)
        )
        & (
            nearest_distance
            <= float(config.CORPORATE_ACTION_FACTOR_LOG_TOLERANCE)
        )
    )
    factors[detected] = canonical[nearest_index[detected]]
    events[detected] = True
    return factors, events


def _remove_corporate_action_factors(returns, factors):
    """Remove only the mechanical split factor and retain the ordinary residual."""
    corrected = np.asarray(returns, dtype=np.float64).copy()
    factors = np.asarray(factors, dtype=np.float64)
    if corrected.shape != factors.shape:
        raise ValueError("returns and corporate-action factors must share shape")
    event_mask = np.isfinite(factors) & (factors > 0.0) & (factors != 1.0)
    corrected[event_mask] -= np.log(factors[event_mask])
    return corrected


def _forward_corporate_action_primary_mask(events, forward_steps):
    """Mask offline features through their post-action dependency horizon."""
    events = np.asarray(events, dtype=bool)
    masked = events.copy()
    for lag in range(1, int(forward_steps) + 1):
        masked[lag:] |= events[:-lag]
    return masked


def _bounded_log_scale(values, reference, clip):
    """Map a positive economic scale to a bounded fixed-reference log ratio."""
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    output[valid] = np.log(values[valid] / float(reference))
    return np.clip(output, -float(clip), float(clip))


def _batched_rolling_moments(values, window, min_periods, ddof=0):
    """Vectorized rolling mean/std over axis one for batched market views."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)

    def _windowed_cumsum(source):
        cumulative = np.concatenate(
            (np.zeros_like(source[:, :1]), np.cumsum(source, axis=1)), axis=1
        )
        stops = np.arange(1, values.shape[1] + 1)
        starts = np.maximum(stops - int(window), 0)
        return cumulative[:, stops] - cumulative[:, starts]

    count = _windowed_cumsum(valid.astype(np.float64))
    total = _windowed_cumsum(safe)
    total_square = _windowed_cumsum(safe * safe)
    mean = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count >= int(min_periods),
    )
    denominator = count - int(ddof)
    variance_numerator = np.maximum(
        total_square
        - np.divide(
            total * total,
            count,
            out=np.zeros_like(total),
            where=count > 0,
        ),
        0.0,
    )
    variance = np.divide(
        variance_numerator,
        denominator,
        out=np.full_like(total, np.nan),
        where=(count >= int(min_periods)) & (denominator > 0),
    )
    return mean, np.sqrt(variance)


def _batched_rolling_mean(values, window, min_periods):
    return _batched_rolling_moments(values, window, min_periods, ddof=0)[0]


def _batched_rolling_std(values, window, min_periods, ddof=0):
    return _batched_rolling_moments(
        values, window, min_periods, ddof=ddof
    )[1]


def _batched_rolling_median(values, window, min_periods):
    """Small-window median; the loop is over time, never batch members."""
    values = np.asarray(values, dtype=np.float64)
    output = np.full_like(values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for stop in range(1, values.shape[1] + 1):
            start = max(0, stop - int(window))
            chunk = values[:, start:stop]
            count = np.isfinite(chunk).sum(axis=1)
            median = np.nanmedian(chunk, axis=1)
            output[:, stop - 1] = np.where(
                count >= int(min_periods), median, np.nan
            )
    return output


def _batched_ewm(values, span, min_periods):
    """Pandas adjust=False EWM for causal, finite-after-warmup arrays."""
    values = np.asarray(values, dtype=np.float64)
    output = np.full_like(values, np.nan)
    state = np.full(values.shape[:1] + values.shape[2:], np.nan, dtype=np.float64)
    count = np.zeros_like(state, dtype=np.int64)
    alpha = 2.0 / (float(span) + 1.0)
    for index in range(values.shape[1]):
        current = values[:, index]
        finite = np.isfinite(current)
        first = finite & ~np.isfinite(state)
        continuing = finite & np.isfinite(state)
        state = np.where(first, current, state)
        state = np.where(
            continuing, (1.0 - alpha) * state + alpha * current, state
        )
        count += finite
        output[:, index] = np.where(
            count >= int(min_periods), state, np.nan
        )
    return output


def _batched_rolling_ratio(numerator, denominator, window, min_periods):
    """Rolling covariance(numerator, denominator) / variance(denominator)."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    x = np.where(valid, numerator, 0.0)
    y = np.where(valid, denominator, 0.0)

    def _windowed(source):
        cumulative = np.concatenate(
            (np.zeros_like(source[:, :1]), np.cumsum(source, axis=1)), axis=1
        )
        stops = np.arange(1, source.shape[1] + 1)
        starts = np.maximum(stops - int(window), 0)
        return cumulative[:, stops] - cumulative[:, starts]

    count = _windowed(valid.astype(np.float64))
    sum_x = _windowed(x)
    sum_y = _windowed(y)
    sum_xy = _windowed(x * y)
    sum_y2 = _windowed(y * y)
    covariance_numerator = sum_xy - np.divide(
        sum_x * sum_y, count, out=np.zeros_like(sum_xy), where=count > 0
    )
    variance_numerator = sum_y2 - np.divide(
        sum_y * sum_y, count, out=np.zeros_like(sum_y2), where=count > 0
    )
    return np.divide(
        covariance_numerator,
        variance_numerator,
        out=np.full_like(covariance_numerator, np.nan),
        where=(count >= int(min_periods)) & (variance_numerator > 0.0),
    )


def _batched_rolling_correlation(values, window, min_periods):
    """Pairwise rolling correlation for [view, time, asset] arrays."""
    values = np.asarray(values, dtype=np.float64)
    left = values[..., :, None]
    right = values[..., None, :]
    valid = np.isfinite(left) & np.isfinite(right)
    x = np.where(valid, left, 0.0)
    y = np.where(valid, right, 0.0)

    def _windowed(source):
        cumulative = np.concatenate(
            (np.zeros_like(source[:, :1]), np.cumsum(source, axis=1)), axis=1
        )
        stops = np.arange(1, values.shape[1] + 1)
        starts = np.maximum(stops - int(window), 0)
        return cumulative[:, stops] - cumulative[:, starts]

    count = _windowed(valid.astype(np.float64))
    sum_x = _windowed(x)
    sum_y = _windowed(y)
    centered_xy = _windowed(x * y) - np.divide(
        sum_x * sum_y, count, out=np.zeros_like(sum_x), where=count > 0
    )
    centered_x2 = _windowed(x * x) - np.divide(
        sum_x * sum_x, count, out=np.zeros_like(sum_x), where=count > 0
    )
    centered_y2 = _windowed(y * y) - np.divide(
        sum_y * sum_y, count, out=np.zeros_like(sum_y), where=count > 0
    )
    scale = np.sqrt(np.maximum(centered_x2 * centered_y2, 0.0))
    return np.divide(
        centered_xy,
        scale,
        out=np.full_like(centered_xy, np.nan),
        where=(count >= int(min_periods)) & (scale > 0.0),
    )


def _behavioral_scale_feature_series(
    open_prices,
    high_prices=None,
    low_prices=None,
    timestamps=None,
    return_opportunity_volatility=False,
):
    """Build causal ticker-neutral scale anchors for one selected basket.

    Unlike the primary and episode-relative blocks, these fields are not
    standardized against rolling basket moments.  Their fixed economic
    references retain whether an otherwise similar cross-sectional pattern is
    occurring in a quiet or volatile opportunity set.  Opening gaps remain in
    the episode-relative block, while this slower risk description always drops
    session-boundary returns so one overnight jump cannot redefine its scale.
    """
    prices = np.asarray(open_prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError("open_prices must have shape [time, active_assets]")

    time_count, asset_count = prices.shape
    feature_names = list(config.BEHAVIORAL_SCALE_FEATURES)
    if time_count == 0 or asset_count == 0:
        empty = np.zeros(
            (time_count, asset_count, len(feature_names)), dtype=np.float32
        )
        if return_opportunity_volatility:
            return empty, np.zeros(time_count, dtype=np.float32)
        return empty

    returns = _causal_open_log_returns(prices)
    if timestamps is not None:
        returns = _drop_session_boundary_returns(
            returns, _session_codes_for_timestamps(timestamps)
        )
    returns_df = pd.DataFrame(returns)

    realized_vol_5 = returns_df.rolling(5, min_periods=2).std(ddof=0)
    realized_vol_30 = returns_df.rolling(30, min_periods=5).std(ddof=0)

    # The current bar's high and low are not known when the policy acts at the
    # open.  Shift the completed-bar range before forming its trailing scale.
    range_scale_30 = pd.DataFrame(
        0.0, index=returns_df.index, columns=returns_df.columns
    )
    if high_prices is not None and low_prices is not None:
        highs = np.asarray(high_prices, dtype=np.float64)
        lows = np.asarray(low_prices, dtype=np.float64)
        if highs.shape != prices.shape or lows.shape != prices.shape:
            raise ValueError("high_prices and low_prices must match open_prices")
        valid_range = (
            np.isfinite(highs)
            & np.isfinite(lows)
            & (highs > 0.0)
            & (lows > 0.0)
            & (highs >= lows)
        )
        completed_range = np.full(prices.shape, np.nan, dtype=np.float64)
        completed_range[valid_range] = np.log(
            highs[valid_range] / lows[valid_range]
        )
        range_scale_30 = (
            pd.DataFrame(completed_range)
            .shift(1)
            .rolling(30, min_periods=5)
            .median()
        )

    basket_realized_vol_median_30 = realized_vol_30.median(
        axis=1, skipna=True
    )
    log_realized_vol_30 = np.log(realized_vol_30.where(realized_vol_30 > 0.0))
    basket_realized_vol_dispersion_30 = log_realized_vol_30.std(
        axis=1, skipna=True, ddof=0
    )
    basket_returns = returns_df.mean(axis=1, skipna=True)
    basket_factor_vol_30 = basket_returns.rolling(
        30, min_periods=5
    ).std(ddof=0)

    clip = float(config.BEHAVIORAL_SCALE_CLIP)
    vol_reference = float(config.BEHAVIORAL_VOLATILITY_REFERENCE)
    range_reference = float(config.BEHAVIORAL_RANGE_REFERENCE)
    repeated_basket_median = np.repeat(
        basket_realized_vol_median_30.to_numpy()[:, None], asset_count, axis=1
    )
    repeated_basket_dispersion = np.repeat(
        basket_realized_vol_dispersion_30.to_numpy()[:, None],
        asset_count,
        axis=1,
    )
    repeated_factor_vol = np.repeat(
        basket_factor_vol_30.to_numpy()[:, None], asset_count, axis=1
    )

    raw_features = {
        'asset_realized_vol_5': _bounded_log_scale(
            realized_vol_5.to_numpy(), vol_reference, clip
        ),
        'asset_realized_vol_30': _bounded_log_scale(
            realized_vol_30.to_numpy(), vol_reference, clip
        ),
        'asset_range_scale_30': _bounded_log_scale(
            range_scale_30.to_numpy(), range_reference, clip
        ),
        'basket_realized_vol_median_30': _bounded_log_scale(
            repeated_basket_median, vol_reference, clip
        ),
        'basket_realized_vol_dispersion_30': np.clip(
            repeated_basket_dispersion, 0.0, clip
        ),
        'basket_factor_vol_30': _bounded_log_scale(
            repeated_factor_vol, vol_reference, clip
        ),
    }

    output = np.zeros(
        (time_count, asset_count, len(feature_names)), dtype=np.float32
    )
    for feature_index, feature_name in enumerate(feature_names):
        values = np.asarray(raw_features[feature_name], dtype=np.float64)
        output[:, :, feature_index] = np.nan_to_num(
            values, nan=0.0, posinf=clip, neginf=-clip
        ).astype(np.float32)
    if return_opportunity_volatility:
        raw_opportunity_volatility = np.nan_to_num(
            basket_realized_vol_median_30.to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clip(min=0.0).astype(np.float32)
        return output, raw_opportunity_volatility
    return output


def _rolling_beta(asset_returns, factor_returns, window, min_periods, fill_value):
    """Causal rolling beta using only returns strictly before each output row."""
    asset_hist = pd.Series(asset_returns, dtype=np.float64).shift(1)
    factor_hist = pd.Series(factor_returns, dtype=np.float64).shift(1)
    covariance = asset_hist.rolling(window, min_periods=min_periods).cov(factor_hist)
    variance = factor_hist.rolling(window, min_periods=min_periods).var()
    raw_beta = covariance / variance.replace(0.0, np.nan)
    raw_beta = raw_beta.replace([np.inf, -np.inf], np.nan)
    smoothed = raw_beta.ewm(span=20, min_periods=5, adjust=False).mean()
    return (0.70 * raw_beta + 0.30 * smoothed).clip(-3.0, 3.0).fillna(fill_value)


def _episode_relative_feature_series(
    open_prices,
    timestamps=None,
    volume_surprises=None,
    corporate_action_factors=None,
):
    """
    Build the fixed-order relational feature block for one selected basket.

    Input shape is [time, active_assets]. Output shape is
    [time, active_assets, EPISODE_RELATIVE_FEATURE_COUNT]. Every cross-asset
    statistic is computed from exactly these active assets, and every rolling
    statistic is causal with respect to the output row. ``volume_surprises``
    is the already-causal, per-asset ``volume_s`` panel; its selected-basket
    comparison is formed here rather than during offline construction.
    """
    prices = np.asarray(open_prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError("open_prices must have shape [time, active_assets]")

    time_count, asset_count = prices.shape
    feature_names = list(config.EPISODE_RELATIVE_FEATURES)
    if time_count == 0 or asset_count == 0:
        return np.zeros(
            (time_count, asset_count, len(feature_names)), dtype=np.float32
        )

    returns = _causal_open_log_returns(prices)
    if corporate_action_factors is not None:
        returns = _remove_corporate_action_factors(
            returns, corporate_action_factors
        )

    # The denominator is the same causal median 30-step asset volatility that
    # is retained explicitly as a behavioral-scale anchor.  Session boundaries
    # are excluded from this slow risk estimate even though ordinary opening
    # gaps remain valid return-valued relational observations below.
    volatility_returns = returns
    if timestamps is not None:
        volatility_returns = _drop_session_boundary_returns(
            volatility_returns, _session_codes_for_timestamps(timestamps)
        )
    volatility_frame = pd.DataFrame(volatility_returns)
    asset_realized_vol_30 = volatility_frame.rolling(
        30, min_periods=5
    ).std(ddof=0)
    basket_volatility_anchor = asset_realized_vol_30.median(
        axis=1, skipna=True
    ).replace([np.inf, -np.inf], np.nan)
    basket_volatility_anchor = basket_volatility_anchor.fillna(
        float(config.EPISODE_RELATIVE_VOLATILITY_FLOOR)
    ).clip(lower=float(config.EPISODE_RELATIVE_VOLATILITY_FLOOR))

    if (
        timestamps is not None
        and bool(getattr(config, "EPISODE_RELATIVE_DROP_SESSION_BOUNDARY", True))
    ):
        returns = _drop_session_boundary_returns(
            returns, _session_codes_for_timestamps(timestamps)
        )

    returns_df = pd.DataFrame(returns)
    basket_returns = returns_df.mean(axis=1, skipna=True)
    peer_mean = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    peer_median = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    for asset_index in range(asset_count):
        peer_columns = [index for index in range(asset_count) if index != asset_index]
        if peer_columns:
            peer_mean.iloc[:, asset_index] = returns_df[peer_columns].mean(axis=1, skipna=True)
            peer_median.iloc[:, asset_index] = returns_df[peer_columns].median(axis=1, skipna=True)
        else:
            peer_mean.iloc[:, asset_index] = basket_returns
            peer_median.iloc[:, asset_index] = basket_returns

    current_returns = returns_df.fillna(0.0)
    relative_to_basket = returns_df.sub(basket_returns, axis=0)

    # Dynamic open_s preserves the historical formula, but the relative term
    # now uses the selected episode basket rather than the complete CSV universe.
    open_s_slow = returns_df.rolling(5, min_periods=1).mean().fillna(0.0)
    open_vol = returns_df.shift(1).rolling(30, min_periods=10).std()
    open_vol_fallback = returns_df.abs().rolling(30, min_periods=5).mean()
    open_vol_safe = open_vol.where(open_vol > 1e-9).fillna(open_vol_fallback)
    open_vol_safe = open_vol_safe.replace(0.0, np.nan).fillna(1e-6)
    open_mu = returns_df.shift(1).rolling(30, min_periods=10).mean()
    open_z = returns_df.sub(open_mu).div(open_vol_safe).clip(-3.0, 3.0).fillna(0.0)
    reversal_component = (-open_z * open_vol_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    relative_smoothed = relative_to_basket.rolling(3, min_periods=1).mean().fillna(0.0)
    open_s = (
        0.40 * current_returns
        + 0.20 * open_s_slow
        + 0.25 * relative_smoothed
        + 0.15 * reversal_component
    ).fillna(0.0)
    open_s_roll_7 = open_s.rolling(7, min_periods=1).mean()
    open_s_roll_14 = open_s.rolling(14, min_periods=1).mean()
    open_s_roll_30 = open_s.rolling(30, min_periods=1).mean()
    open_s_std_14 = open_s.rolling(14, min_periods=2).std().fillna(0.0)
    open_s_std_30 = open_s.rolling(30, min_periods=2).std().fillna(0.0)
    open_s_accel = open_s_roll_7 - open_s_roll_30

    beta_window = int(getattr(config, "EPISODE_RELATIVE_BETA_WINDOW", 60))
    beta_min_periods = max(5, beta_window // 2)
    beta_60 = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    peer_beta = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    for asset_index in range(asset_count):
        beta_60.iloc[:, asset_index] = _rolling_beta(
            returns_df.iloc[:, asset_index],
            basket_returns,
            beta_window,
            beta_min_periods,
            fill_value=1.0,
        ).to_numpy()
        peer_beta.iloc[:, asset_index] = _rolling_beta(
            returns_df.iloc[:, asset_index],
            peer_mean.iloc[:, asset_index],
            beta_window,
            int(getattr(config, "EPISODE_RELATIVE_MIN_PERIODS", 20)),
            fill_value=0.0,
        ).to_numpy()

    rs_fast = relative_to_basket.ewm(span=6, min_periods=3, adjust=False).mean()
    rs_slow = relative_to_basket.ewm(span=20, min_periods=5, adjust=False).mean()
    rs_roll_14 = (
        0.65 * rs_fast + 0.35 * (rs_fast - rs_slow)
    ).rolling(5, min_periods=1).mean().fillna(0.0)

    peer_excess = returns_df - peer_median
    episode_excess_return = peer_excess.fillna(0.0)
    episode_relative_strength = peer_excess.rolling(
        int(getattr(config, "EPISODE_RELATIVE_SHORT_WINDOW", 14)),
        min_periods=3,
    ).mean().fillna(0.0)

    shifted_returns = returns_df.shift(1)
    pairwise_correlations = [[None for _ in range(asset_count)] for _ in range(asset_count)]
    correlation_min_periods = int(getattr(config, "EPISODE_RELATIVE_MIN_PERIODS", 20))
    for left in range(asset_count):
        for right in range(left + 1, asset_count):
            correlation = shifted_returns.iloc[:, left].rolling(
                beta_window, min_periods=correlation_min_periods
            ).corr(shifted_returns.iloc[:, right])
            pairwise_correlations[left][right] = correlation
            pairwise_correlations[right][left] = correlation

    correlation_mean = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    correlation_abs_max = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    for asset_index in range(asset_count):
        peer_series = [
            pairwise_correlations[asset_index][peer_index]
            for peer_index in range(asset_count)
            if peer_index != asset_index
            and pairwise_correlations[asset_index][peer_index] is not None
        ]
        if peer_series:
            peer_frame = pd.concat(peer_series, axis=1)
            correlation_mean.iloc[:, asset_index] = peer_frame.mean(axis=1).fillna(0.0)
            correlation_abs_max.iloc[:, asset_index] = peer_frame.abs().max(axis=1).fillna(0.0)

    momentum = returns_df.rolling(
        int(getattr(config, "EPISODE_RELATIVE_SHORT_WINDOW", 14)),
        min_periods=3,
    ).mean()
    momentum_rank = momentum.rank(axis=1, method="average")
    rank_count = momentum.notna().sum(axis=1)
    rank_denominator = (rank_count - 1).replace(0, np.nan)
    momentum_rank = momentum_rank.sub(1.0).div(rank_denominator, axis=0)
    momentum_rank = (2.0 * momentum_rank - 1.0).fillna(0.0)

    trailing_volatility = returns_df.shift(1).rolling(14, min_periods=5).std()
    relative_volatility = pd.DataFrame(
        0.0, index=returns_df.index, columns=returns_df.columns
    )
    for asset_index in range(asset_count):
        peer_columns = [index for index in range(asset_count) if index != asset_index]
        if peer_columns:
            peer_volatility = trailing_volatility[peer_columns].median(axis=1)
        else:
            peer_volatility = trailing_volatility.iloc[:, asset_index]
        ratio = (
            trailing_volatility.iloc[:, asset_index]
            / peer_volatility.replace(0.0, np.nan)
        )
        # A truly flat asset is maximally quiet relative to non-flat peers.
        # Clamp before log so that this has the intended bounded value without
        # manufacturing -inf and flooding long foundation runs with warnings.
        safe_ratio = ratio.where(ratio.notna()).clip(
            lower=float(np.exp(-3.0))
        )
        relative_volatility.iloc[:, asset_index] = np.log(safe_ratio).clip(
            -3.0, 3.0
        ).fillna(0.0)

    relative_volume_surprise = pd.DataFrame(
        0.0, index=returns_df.index, columns=returns_df.columns
    )
    if volume_surprises is not None:
        volume_array = np.asarray(volume_surprises, dtype=np.float64)
        if volume_array.shape != prices.shape:
            raise ValueError(
                "volume_surprises must have the same [time, active_assets] "
                "shape as open_prices"
            )
        volume_frame = pd.DataFrame(volume_array).replace(
            [np.inf, -np.inf], np.nan
        )
        for asset_index in range(asset_count):
            peer_columns = [
                index for index in range(asset_count) if index != asset_index
            ]
            if peer_columns:
                peer_volume = volume_frame[peer_columns].median(
                    axis=1, skipna=True
                )
            else:
                peer_volume = volume_frame.iloc[:, asset_index]
            relative_volume_surprise.iloc[:, asset_index] = (
                volume_frame.iloc[:, asset_index] - peer_volume
            ).fillna(0.0)

    raw_features = {
        'open_s': open_s,
        'open_s_roll_7': open_s_roll_7,
        'open_s_roll_14': open_s_roll_14,
        'open_s_roll_30': open_s_roll_30,
        'open_s_std_14': open_s_std_14,
        'open_s_std_30': open_s_std_30,
        'open_s_accel': open_s_accel,
        'beta_60': beta_60,
        'rs_roll_14': rs_roll_14,
        'episode_excess_return': episode_excess_return,
        'episode_relative_strength': episode_relative_strength,
        'episode_peer_beta': peer_beta,
        'episode_peer_correlation_mean': correlation_mean,
        'episode_peer_correlation_abs_max': correlation_abs_max,
        'episode_cross_sectional_momentum_rank': momentum_rank,
        'episode_relative_volatility': relative_volatility,
        'episode_relative_volume_surprise': relative_volume_surprise,
    }

    output = np.zeros(
        (time_count, asset_count, len(feature_names)), dtype=np.float32
    )
    for feature_index, feature_name in enumerate(feature_names):
        values = np.asarray(raw_features[feature_name], dtype=np.float64)
        if feature_name in config.EPISODE_RELATIVE_RETURN_FEATURES:
            values = values / basket_volatility_anchor.to_numpy()[:, None]
        feature_min = float(config.EPISODE_RELATIVE_FEATURE_MIN[feature_index])
        feature_max = float(config.EPISODE_RELATIVE_FEATURE_MAX[feature_index])
        output[:, :, feature_index] = np.nan_to_num(
            values, nan=0.0, posinf=feature_max, neginf=feature_min
        ).clip(feature_min, feature_max).astype(np.float32)

    return output


def _attach_episode_relative_features(data, episode_coins):
    """Attach all dynamic selected-basket features to a bounded dataframe."""
    if len(data) == 0:
        return data.copy()

    frame = data.copy()
    dynamic_feature_names = (
        list(config.BEHAVIORAL_SCALE_FEATURES)
        + list(config.EPISODE_RELATIVE_FEATURES)
    )
    internal_dynamic_names = [
        "_corporate_action_event",
        "_corporate_action_primary_mask",
        "_corporate_action_factor",
        "_corrected_open_s_1m",
        "_basket_opportunity_volatility_30",
    ]
    old_dynamic = [
        name
        for name in dynamic_feature_names + internal_dynamic_names
        if name in frame.columns
    ]
    if old_dynamic:
        frame = frame.drop(columns=old_dynamic)

    unique_rows = frame.drop_duplicates(["date", "coin"])

    def selected_panel(column_name):
        if column_name not in unique_rows.columns:
            return None
        return (
            unique_rows
            .pivot(index="date", columns="coin", values=column_name)
            .sort_index()
            .reindex(columns=list(episode_coins))
        )

    open_panel = selected_panel("open")
    if open_panel is None:
        raise ValueError("Dynamic episode features require an open column")
    # selected_panel constructs its pivot from the requested field.  Keep the
    # implementation centralized so every panel shares the same date/coin grid.
    def aligned_selected_panel(column_name):
        if column_name not in unique_rows.columns:
            return None
        return (
            unique_rows.pivot(index="date", columns="coin", values=column_name)
            .sort_index()
            .reindex(index=open_panel.index, columns=list(episode_coins))
        )

    close_panel = aligned_selected_panel("close")
    if close_panel is None:
        corporate_action_factors = np.ones(open_panel.shape, dtype=np.float64)
        corporate_action_events = np.zeros(open_panel.shape, dtype=bool)
    else:
        corporate_action_factors, corporate_action_events = (
            _detect_corporate_action_factors(
                open_panel.to_numpy(dtype=np.float64, copy=False),
                close_panel.to_numpy(dtype=np.float64, copy=False),
                open_panel.index,
            )
        )
    primary_contamination_mask = _forward_corporate_action_primary_mask(
        corporate_action_events,
        int(config.CORPORATE_ACTION_PRIMARY_MASK_STEPS),
    )
    corrected_open_returns = _remove_corporate_action_factors(
        _causal_open_log_returns(
            open_panel.to_numpy(dtype=np.float64, copy=False)
        ),
        corporate_action_factors,
    )

    feature_values = (np.zeros(
        (*open_panel.shape, config.EPISODE_RELATIVE_FEATURE_COUNT), dtype=np.float32
    ) if config.LOCAL_CAUSAL_INPUTS_ENABLED else _episode_relative_feature_series(
        open_panel.to_numpy(dtype=np.float64, copy=False),
        timestamps=open_panel.index,
        volume_surprises=(
            aligned_selected_panel("volume_s").to_numpy(
                dtype=np.float64, copy=False
            )
            if "volume_s" in frame.columns
            else None
        ),
        corporate_action_factors=corporate_action_factors,
    ))
    high_panel = aligned_selected_panel("high")
    low_panel = aligned_selected_panel("low")
    (
        behavioral_scale_values,
        basket_opportunity_volatility,
    ) = _behavioral_scale_feature_series(
        open_panel.to_numpy(dtype=np.float64, copy=False),
        high_prices=(
            high_panel.to_numpy(dtype=np.float64, copy=False)
            if high_panel is not None
            else None
        ),
        low_prices=(
            low_panel.to_numpy(dtype=np.float64, copy=False)
            if low_panel is not None
            else None
        ),
        timestamps=open_panel.index,
        return_opportunity_volatility=True,
    )

    if config.LOCAL_CAUSAL_INPUTS_ENABLED:
        # Keep three per-asset anchors; reserve the other slots for compatibility.
        behavioral_scale_values[..., 3:] = 0.0

    context_frames = []
    for asset_index, coin in enumerate(episode_coins):
        coin_frame = pd.DataFrame(
            np.concatenate(
                [
                    behavioral_scale_values[:, asset_index, :],
                    feature_values[:, asset_index, :],
                ],
                axis=1,
            ),
            columns=dynamic_feature_names,
        )
        coin_frame["_corporate_action_event"] = corporate_action_events[
            :, asset_index
        ]
        coin_frame["_corporate_action_primary_mask"] = (
            primary_contamination_mask[:, asset_index]
        )
        coin_frame["_corporate_action_factor"] = corporate_action_factors[
            :, asset_index
        ]
        coin_frame["_corrected_open_s_1m"] = corrected_open_returns[
            :, asset_index
        ]
        coin_frame["_basket_opportunity_volatility_30"] = (
            basket_opportunity_volatility
        )
        coin_frame.insert(0, "coin", coin)
        coin_frame.insert(0, "date", open_panel.index)
        context_frames.append(coin_frame)
    context = pd.concat(context_frames, ignore_index=True)

    frame["_episode_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.merge(context, on=["date", "coin"], how="left", sort=False)
    frame = frame.sort_values("_episode_row_order").drop(columns="_episode_row_order")
    frame[dynamic_feature_names] = frame[dynamic_feature_names].fillna(
        0.0
    ).astype(np.float32)
    frame["_corporate_action_event"] = frame[
        "_corporate_action_event"
    ].eq(True)
    frame["_corporate_action_primary_mask"] = frame[
        "_corporate_action_primary_mask"
    ].eq(True)
    frame["_corporate_action_factor"] = frame[
        "_corporate_action_factor"
    ].fillna(1.0).astype(np.float32)
    frame["_corrected_open_s_1m"] = frame[
        "_corrected_open_s_1m"
    ].fillna(0.0).astype(np.float32)
    frame["_basket_opportunity_volatility_30"] = frame[
        "_basket_opportunity_volatility_30"
    ].fillna(0.0).clip(lower=0.0).astype(np.float32)
    if "open_s_1m" in frame.columns:
        event_rows = frame["_corporate_action_event"]
        frame.loc[event_rows, "open_s_1m"] = frame.loc[
            event_rows, "_corrected_open_s_1m"
        ]
    return frame.reset_index(drop=True)


def _batched_foundation_dynamic_features(
    prices,
    high_prices,
    low_prices,
    volume_surprises,
):
    """Build exact causal feature semantics for many same-session views.

    The scalar environment builder remains the reference implementation.  This
    production warmup path moves its rolling work from one Pandas graph per
    sampled view to a handful of batched NumPy reductions.
    """
    prices = np.asarray(prices, dtype=np.float64)
    batch, length, assets = prices.shape
    returns = np.full_like(prices, np.nan)
    valid_price = (
        np.isfinite(prices[:, 1:])
        & np.isfinite(prices[:, :-1])
        & (prices[:, 1:] > 0.0)
        & (prices[:, :-1] > 0.0)
    )
    returns[:, 1:] = np.where(
        valid_price,
        np.log(
            np.divide(
                prices[:, 1:],
                prices[:, :-1],
                out=np.ones_like(prices[:, 1:]),
                where=valid_price,
            )
        ),
        np.nan,
    )

    _, realized_vol_5 = _batched_rolling_moments(
        returns, 5, 2, ddof=0
    )
    _, realized_vol_30 = _batched_rolling_moments(
        returns, 30, 5, ddof=0
    )
    completed_range = np.full_like(prices, np.nan)
    valid_range = (
        np.isfinite(high_prices)
        & np.isfinite(low_prices)
        & (high_prices > 0.0)
        & (low_prices > 0.0)
        & (high_prices >= low_prices)
    )
    completed_range[valid_range] = np.log(
        high_prices[valid_range] / low_prices[valid_range]
    )
    shifted_range = np.full_like(completed_range, np.nan)
    shifted_range[:, 1:] = completed_range[:, :-1]
    range_scale_30 = _batched_rolling_median(shifted_range, 30, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        basket_vol_median = np.nanmedian(realized_vol_30, axis=2)
    # Do not use ``np.where(..., np.log(x), ...)`` here: NumPy evaluates both
    # branches before selecting the result, so zero volatility still emits a
    # divide-by-zero warning on otherwise valid warmup batches.  Construct the
    # masked log explicitly and only call log for strictly positive values.
    log_vol = np.full_like(realized_vol_30, np.nan, dtype=np.float64)
    positive_vol = np.isfinite(realized_vol_30) & (realized_vol_30 > 0.0)
    log_vol[positive_vol] = np.log(realized_vol_30[positive_vol])
    log_vol_mean = np.divide(
        np.nansum(log_vol, axis=2),
        np.isfinite(log_vol).sum(axis=2),
        out=np.full((batch, length), np.nan),
        where=np.isfinite(log_vol).sum(axis=2) > 0,
    )
    log_vol_count = np.isfinite(log_vol).sum(axis=2)
    log_vol_dispersion = np.sqrt(
        np.divide(
            np.nansum(
                np.square(log_vol - log_vol_mean[:, :, None]), axis=2
            ),
            log_vol_count,
            out=np.full((batch, length), np.nan),
            where=log_vol_count > 0,
        )
    )
    return_count = np.isfinite(returns).sum(axis=2)
    basket_returns = np.divide(
        np.nansum(returns, axis=2),
        return_count,
        out=np.full((batch, length), np.nan),
        where=return_count > 0,
    )
    basket_factor_vol = _batched_rolling_std(
        basket_returns[:, :, None], 30, 5, ddof=0
    )[:, :, 0]

    scale_clip = float(config.BEHAVIORAL_SCALE_CLIP)
    behavioral = np.stack(
        (
            _bounded_log_scale(
                realized_vol_5, config.BEHAVIORAL_VOLATILITY_REFERENCE, scale_clip
            ),
            _bounded_log_scale(
                realized_vol_30, config.BEHAVIORAL_VOLATILITY_REFERENCE, scale_clip
            ),
            _bounded_log_scale(
                range_scale_30, config.BEHAVIORAL_RANGE_REFERENCE, scale_clip
            ),
            np.broadcast_to(
                _bounded_log_scale(
                    basket_vol_median,
                    config.BEHAVIORAL_VOLATILITY_REFERENCE,
                    scale_clip,
                )[:, :, None],
                prices.shape,
            ),
            np.broadcast_to(
                np.clip(log_vol_dispersion, 0.0, scale_clip)[:, :, None],
                prices.shape,
            ),
            np.broadcast_to(
                _bounded_log_scale(
                    basket_factor_vol,
                    config.BEHAVIORAL_VOLATILITY_REFERENCE,
                    scale_clip,
                )[:, :, None],
                prices.shape,
            ),
        ),
        axis=-1,
    )
    behavioral = np.nan_to_num(
        behavioral,
        nan=0.0,
        posinf=scale_clip,
        neginf=-scale_clip,
    ).astype(np.float32)

    # Selected-basket episode-relative stream.
    peer_mean = np.empty_like(returns)
    peer_median = np.empty_like(returns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for asset_index in range(assets):
            peers = np.delete(returns, asset_index, axis=2)
            peer_mean[:, :, asset_index] = np.nanmean(peers, axis=2)
            peer_median[:, :, asset_index] = np.nanmedian(peers, axis=2)
    current_returns = np.nan_to_num(returns)
    relative_to_basket = returns - basket_returns[:, :, None]
    open_s_slow = np.nan_to_num(_batched_rolling_mean(returns, 5, 1))
    shifted_returns = np.full_like(returns, np.nan)
    shifted_returns[:, 1:] = returns[:, :-1]
    open_vol = _batched_rolling_std(
        shifted_returns, 30, 10, ddof=1
    )
    open_vol_fallback = _batched_rolling_mean(np.abs(returns), 30, 5)
    open_vol_safe = np.where(open_vol > 1e-9, open_vol, open_vol_fallback)
    open_vol_safe = np.where(
        np.isfinite(open_vol_safe) & (open_vol_safe != 0.0),
        open_vol_safe,
        1e-6,
    )
    open_mu = _batched_rolling_mean(shifted_returns, 30, 10)
    open_z = np.nan_to_num(
        np.clip((returns - open_mu) / open_vol_safe, -3.0, 3.0)
    )
    reversal = np.nan_to_num(-open_z * open_vol_safe)
    relative_smoothed = np.nan_to_num(
        _batched_rolling_mean(relative_to_basket, 3, 1)
    )
    open_s = np.nan_to_num(
        0.40 * current_returns
        + 0.20 * open_s_slow
        + 0.25 * relative_smoothed
        + 0.15 * reversal
    )
    open_s_roll_7 = _batched_rolling_mean(open_s, 7, 1)
    open_s_roll_14 = _batched_rolling_mean(open_s, 14, 1)
    open_s_roll_30 = _batched_rolling_mean(open_s, 30, 1)
    open_s_std_14 = np.nan_to_num(
        _batched_rolling_std(open_s, 14, 2, ddof=1)
    )
    open_s_std_30 = np.nan_to_num(
        _batched_rolling_std(open_s, 30, 2, ddof=1)
    )
    open_s_accel = open_s_roll_7 - open_s_roll_30

    beta_window = int(getattr(config, "EPISODE_RELATIVE_BETA_WINDOW", 60))
    shifted_basket = np.full_like(basket_returns, np.nan)
    shifted_basket[:, 1:] = basket_returns[:, :-1]
    beta_raw = _batched_rolling_ratio(
        shifted_returns,
        shifted_basket[:, :, None],
        beta_window,
        max(5, beta_window // 2),
    )
    beta_60 = np.nan_to_num(
        np.clip(
            0.70 * beta_raw + 0.30 * _batched_ewm(beta_raw, 20, 5),
            -3.0,
            3.0,
        ),
        nan=1.0,
    )
    shifted_peer = np.full_like(peer_mean, np.nan)
    shifted_peer[:, 1:] = peer_mean[:, :-1]
    peer_beta_raw = _batched_rolling_ratio(
        shifted_returns,
        shifted_peer,
        beta_window,
        int(getattr(config, "EPISODE_RELATIVE_MIN_PERIODS", 20)),
    )
    peer_beta = np.nan_to_num(
        np.clip(
            0.70 * peer_beta_raw
            + 0.30 * _batched_ewm(peer_beta_raw, 20, 5),
            -3.0,
            3.0,
        )
    )
    rs_fast = _batched_ewm(relative_to_basket, 6, 3)
    rs_slow = _batched_ewm(relative_to_basket, 20, 5)
    rs_roll_14 = np.nan_to_num(
        _batched_rolling_mean(0.65 * rs_fast + 0.35 * (rs_fast - rs_slow), 5, 1)
    )
    peer_excess = returns - peer_median
    episode_excess = np.nan_to_num(peer_excess)
    episode_strength = np.nan_to_num(
        _batched_rolling_mean(
            peer_excess,
            int(getattr(config, "EPISODE_RELATIVE_SHORT_WINDOW", 14)),
            3,
        )
    )

    pairwise_corr = _batched_rolling_correlation(
        shifted_returns,
        beta_window,
        int(getattr(config, "EPISODE_RELATIVE_MIN_PERIODS", 20)),
    )
    identity = np.eye(assets, dtype=bool)[None, None]
    pairwise_without_self = np.where(identity, np.nan, pairwise_corr)
    correlation_count = np.isfinite(pairwise_without_self).sum(axis=3)
    correlation_mean = np.divide(
        np.nansum(pairwise_without_self, axis=3),
        correlation_count,
        out=np.zeros(pairwise_without_self.shape[:3], dtype=np.float64),
        where=correlation_count > 0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        correlation_abs_max = np.nan_to_num(
            np.nanmax(np.abs(pairwise_without_self), axis=3)
        )

    momentum = _batched_rolling_mean(
        returns,
        int(getattr(config, "EPISODE_RELATIVE_SHORT_WINDOW", 14)),
        3,
    )
    finite_momentum = np.isfinite(momentum)
    left = momentum[:, :, :, None]
    right = momentum[:, :, None, :]
    less = (right < left) & finite_momentum[:, :, None, :]
    equal_other = (
        (right == left)
        & finite_momentum[:, :, None, :]
        & ~identity
    )
    rank = 1.0 + less.sum(axis=3) + 0.5 * equal_other.sum(axis=3)
    rank_count = finite_momentum.sum(axis=2)
    rank_denominator = rank_count - 1
    momentum_rank = np.divide(
        rank - 1.0,
        rank_denominator[:, :, None],
        out=np.zeros_like(rank),
        where=rank_denominator[:, :, None] > 0,
    )
    momentum_rank = np.where(
        finite_momentum, 2.0 * momentum_rank - 1.0, 0.0
    )

    trailing_vol = _batched_rolling_std(
        shifted_returns, 14, 5, ddof=1
    )
    relative_volatility = np.zeros_like(trailing_vol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for asset_index in range(assets):
            peer_vol = np.nanmedian(
                np.delete(trailing_vol, asset_index, axis=2), axis=2
            )
            ratio = trailing_vol[:, :, asset_index] / np.where(
                peer_vol == 0.0, np.nan, peer_vol
            )
            relative_volatility[:, :, asset_index] = np.where(
                np.isfinite(ratio),
                np.log(np.maximum(ratio, np.exp(-3.0))),
                0.0,
            )
    relative_volatility = np.clip(relative_volatility, -3.0, 3.0)

    relative_volume = np.zeros_like(returns)
    if volume_surprises is not None:
        volumes = np.asarray(volume_surprises, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for asset_index in range(assets):
                peer_volume = np.nanmedian(
                    np.delete(volumes, asset_index, axis=2), axis=2
                )
                relative_volume[:, :, asset_index] = np.nan_to_num(
                    volumes[:, :, asset_index] - peer_volume
                )

    raw = {
        "open_s": open_s,
        "open_s_roll_7": open_s_roll_7,
        "open_s_roll_14": open_s_roll_14,
        "open_s_roll_30": open_s_roll_30,
        "open_s_std_14": open_s_std_14,
        "open_s_std_30": open_s_std_30,
        "open_s_accel": open_s_accel,
        "beta_60": beta_60,
        "rs_roll_14": rs_roll_14,
        "episode_excess_return": episode_excess,
        "episode_relative_strength": episode_strength,
        "episode_peer_beta": peer_beta,
        "episode_peer_correlation_mean": correlation_mean,
        "episode_peer_correlation_abs_max": correlation_abs_max,
        "episode_cross_sectional_momentum_rank": momentum_rank,
        "episode_relative_volatility": relative_volatility,
        "episode_relative_volume_surprise": relative_volume,
    }
    anchor = np.where(
        np.isfinite(basket_vol_median),
        basket_vol_median,
        float(config.EPISODE_RELATIVE_VOLATILITY_FLOOR),
    )
    anchor = np.maximum(anchor, float(config.EPISODE_RELATIVE_VOLATILITY_FLOOR))
    episode = np.empty(
        (batch, length, assets, len(config.EPISODE_RELATIVE_FEATURES)),
        dtype=np.float32,
    )
    for feature_index, name in enumerate(config.EPISODE_RELATIVE_FEATURES):
        values = raw[name]
        if name in config.EPISODE_RELATIVE_RETURN_FEATURES:
            values = values / anchor[:, :, None]
        minimum = float(config.EPISODE_RELATIVE_FEATURE_MIN[feature_index])
        maximum = float(config.EPISODE_RELATIVE_FEATURE_MAX[feature_index])
        episode[..., feature_index] = np.clip(
            np.nan_to_num(values, nan=0.0, posinf=maximum, neginf=minimum),
            minimum,
            maximum,
        ).astype(np.float32)
    return behavioral, episode


class FoundationMarketViewSampler:
    """Causal paired 8-of-N views built without resetting PortfolioEnv."""

    def __init__(self, data=None, seed=0):
        source = config.data_set if data is None else data
        required = {
            "date", "coin", "open", "close", "high", "low", *config.SCOLS
        }
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(f"foundation source is missing columns: {missing}")
        # ``config._prepare_training_dataframe`` already establishes this
        # layout.  Verify it and sort only non-production fixtures; blindly
        # sorting/reindexing the 4.9-GB source would duplicate the full frame.
        frame = source
        self.coins = tuple(sorted(frame["coin"].unique().tolist()))
        self.private_asset_ids = np.asarray(
            [config.TICKER_TO_ID.get(coin, index) for index, coin in enumerate(self.coins)],
            dtype=np.int64,
        )
        asset_count = len(self.coins)
        if len(frame) % max(asset_count, 1):
            raise ValueError(
                "foundation sampling requires a complete timestamp/asset panel"
            )
        time_count = len(frame) // asset_count
        coin_layout = frame["coin"].to_numpy(copy=False).reshape(
            time_count, asset_count
        )
        expected_coin_row = np.asarray(self.coins, dtype=object)
        if not np.all(coin_layout == expected_coin_row[None, :]):
            frame = frame.sort_values(["date", "coin"]).reset_index(drop=True)
            coin_layout = frame["coin"].to_numpy(copy=False).reshape(
                time_count, asset_count
            )
            if not np.all(coin_layout == expected_coin_row[None, :]):
                raise ValueError(
                    "foundation sampling requires one ordered row per asset/time"
                )
        date_layout = frame["date"].to_numpy(copy=False).reshape(
            time_count, asset_count
        )
        if not np.all(date_layout == date_layout[:, :1]):
            raise ValueError(
                "foundation sampling requires a complete timestamp/asset panel"
            )
        self.timestamps = pd.DatetimeIndex(date_layout[:, 0])
        # Keep zero-copy column views into the prepared dataframe.  Sampled
        # windows are stacked on demand, avoiding a second persistent
        # [time, asset, 57] allocation.
        self.primary_columns = tuple(
            frame[name].to_numpy(dtype=np.float32, copy=False).reshape(
                time_count, asset_count
            )
            for name in config.SCOLS
        )
        self.open_prices = frame["open"].to_numpy(
            dtype=np.float32, copy=False
        ).reshape(time_count, asset_count)
        self.high_prices = frame["high"].to_numpy(
            dtype=np.float32, copy=False
        ).reshape(time_count, asset_count)
        self.low_prices = frame["low"].to_numpy(
            dtype=np.float32, copy=False
        ).reshape(time_count, asset_count)
        self.close_prices = frame["close"].to_numpy(
            dtype=np.float32, copy=False
        ).reshape(time_count, asset_count)
        self.volume_surprises = (
            frame["volume_s"].to_numpy(dtype=np.float32, copy=False).reshape(
                time_count, asset_count
            )
            if "volume_s" in frame.columns
            else None
        )
        self.session_codes = _session_codes_for_timestamps(self.timestamps)
        session_positions = np.zeros(time_count, dtype=np.int64)
        session_remaining = np.zeros(time_count, dtype=np.int64)
        for session_code in np.unique(self.session_codes):
            indices = np.flatnonzero(self.session_codes == session_code)
            session_positions[indices] = np.arange(len(indices))
            session_remaining[indices] = len(indices) - np.arange(len(indices)) - 1
        self.session_positions = session_positions
        self.session_remaining = session_remaining
        self.required_past = max(
            int(config.MDAEN_FAM_HISTORY_LENGTH) - 1,
            int(getattr(config, "EPISODE_RELATIVE_BETA_WINDOW", 60))
            + int(config.MDAEN_FAM_HISTORY_LENGTH) - 1,
        )
        required_future = max(
            tuple(getattr(config, "FOUNDATION_HORIZONS", (1, 5, 15, 30)))
        )
        self.eligible_times = np.flatnonzero(
            (session_positions >= self.required_past)
            & (session_remaining >= required_future)
        )
        if not len(self.eligible_times):
            raise ValueError("foundation source has no complete causal windows")
        self.rng = np.random.default_rng(seed)
        self.primary_indices = np.asarray(
            [config.SCOLS.index(name) for name in config.MDAEN_FAM_PRIMARY_FEATURES],
            dtype=np.int64,
        )

    @staticmethod
    def _gather_panel(panel, time_indices, basket_indices):
        """Gather [view, time, slot] without a loop over sampled views."""
        return panel[
            np.asarray(time_indices, dtype=np.int64)[:, :, None],
            np.asarray(basket_indices, dtype=np.int64)[:, None, :],
        ]

    def _complete_observation_batch(self, time_indices, basket_indices):
        """Vectorized production equivalent of ``_complete_observation``."""
        times = np.asarray(time_indices, dtype=np.int64)
        baskets = np.asarray(basket_indices, dtype=np.int64)
        view_count = len(times)
        history = int(config.MDAEN_FAM_HISTORY_LENGTH)
        window_offsets = np.arange(-self.required_past, 1, dtype=np.int64)
        window_indices = times[:, None] + window_offsets[None, :]
        prices = self._gather_panel(self.open_prices, window_indices, baskets)
        highs = self._gather_panel(self.high_prices, window_indices, baskets)
        lows = self._gather_panel(self.low_prices, window_indices, baskets)
        volumes = (
            self._gather_panel(self.volume_surprises, window_indices, baskets)
            if self.volume_surprises is not None
            else None
        )
        behavioral, relative = _batched_foundation_dynamic_features(
            prices, highs, lows, volumes
        )

        history_indices = times[:, None] + np.arange(
            -history + 1, 1, dtype=np.int64
        )[None, :]
        local_primary = np.stack(
            [
                self._gather_panel(
                    self.primary_columns[int(feature_index)],
                    history_indices,
                    baskets,
                )
                for feature_index in self.primary_indices
            ],
            axis=-1,
        )
        local = np.concatenate(
            (local_primary, behavioral[:, -history:, :, :3]), axis=-1
        ).transpose(0, 2, 1, 3)
        episode = np.concatenate(
            (behavioral[:, -history:, :, 3:], relative[:, -history:]),
            axis=-1,
        ).transpose(0, 2, 1, 3)

        scaler_steps = int(getattr(config, "SCALER_WINDOW_STEPS", 60))
        scaler_offsets = np.arange(
            -scaler_steps + 1, 1, dtype=np.int64
        )
        scaler_indices = times[:, None] + scaler_offsets[None, :]
        scaler_valid = scaler_offsets[None, :] >= -self.session_positions[
            times
        ][:, None]
        scaler_indices = np.maximum(
            scaler_indices,
            (times - self.session_positions[times])[:, None],
        )
        scale_window = np.stack(
            [
                self._gather_panel(column, scaler_indices, baskets)
                for column in self.primary_columns
            ],
            axis=-1,
        )
        scale_window = np.where(
            scaler_valid[:, :, None, None], scale_window, np.nan
        )
        scale_mean = np.nanmean(scale_window, axis=(1, 2))
        scale_std = np.maximum(
            np.nanstd(scale_window, axis=(1, 2)), 1e-3
        )
        current_primary = np.stack(
            [
                column[times[:, None], baskets]
                for column in self.primary_columns
            ],
            axis=-1,
        )
        scaled_primary = np.clip(
            (current_primary - scale_mean[:, None, :])
            / scale_std[:, None, :],
            -3.0,
            3.0,
        ).astype(np.float32)

        behavioral_current = np.concatenate(
            (local[:, :, -1, -3:], episode[:, :, -1, :3]), axis=-1
        )
        relative_current = episode[:, :, -1, 3:]
        observations = np.concatenate(
            (
                scaled_primary,
                np.zeros((view_count, config.MAX_COINS, 1), dtype=np.float32),
                behavioral_current,
                relative_current,
            ),
            axis=-1,
        ).reshape(view_count, -1).astype(np.float32)

        basket_context = np.zeros(
            (view_count, config.BASKET_CONTEXT_DIM), dtype=np.float32
        )
        if config.BASKET_CONTEXT_DIM:
            scale_index = config.BEHAVIORAL_SCALE_FEATURE_INDEX
            relative_index = config.EPISODE_RELATIVE_FEATURE_INDEX
            values = {
                "basket_realized_vol_median_30": behavioral_current[
                    :, :, scale_index["basket_realized_vol_median_30"]
                ].mean(axis=1),
                "basket_realized_vol_dispersion_30": behavioral_current[
                    :, :, scale_index["basket_realized_vol_dispersion_30"]
                ].mean(axis=1),
                "basket_factor_vol_30": behavioral_current[
                    :, :, scale_index["basket_factor_vol_30"]
                ].mean(axis=1),
                "basket_equal_weight_open_s": relative_current[
                    :, :, relative_index["open_s"]
                ].mean(axis=1),
                "basket_open_s_dispersion": relative_current[
                    :, :, relative_index["open_s"]
                ].std(axis=1),
                "basket_rank_concentration": np.ptp(
                    relative_current[
                        :, :,
                        relative_index["episode_cross_sectional_momentum_rank"],
                    ],
                    axis=1,
                ),
                "basket_mean_peer_correlation": relative_current[
                    :, :, relative_index["episode_peer_correlation_mean"]
                ].mean(axis=1),
                "basket_mean_peer_correlation_abs_max": relative_current[
                    :, :, relative_index["episode_peer_correlation_abs_max"]
                ].mean(axis=1),
            }
            basket_context = np.stack(
                [values[name] for name in config.BASKET_CONTEXT_FEATURES], axis=1
            ).astype(np.float32)
            basket_context = np.clip(
                basket_context,
                config.BASKET_CONTEXT_FEATURE_MIN,
                config.BASKET_CONTEXT_FEATURE_MAX,
            )

        previous = self.rng.dirichlet(
            np.ones(config.MAX_COINS + 1, dtype=np.float64), size=view_count
        ).astype(np.float32)
        private_ids = self.private_asset_ids[baskets]
        membership = np.zeros(
            (view_count, config.BASKET_MEMBERSHIP_DIM), dtype=np.float32
        )
        if config.BASKET_MEMBERSHIP_DIM:
            rows = np.repeat(np.arange(view_count), config.MAX_COINS)
            ids = private_ids.reshape(-1)
            valid_ids = ids < config.BASKET_MEMBERSHIP_DIM
            membership[rows[valid_ids], ids[valid_ids]] = 1.0

        horizon = max(config.FOUNDATION_HORIZONS)
        future_indices = times[:, None] + np.arange(
            0, horizon + 1, dtype=np.int64
        )[None, :]
        future_prices = self._gather_panel(
            self.open_prices, future_indices, baskets
        )
        future_valid = (
            np.isfinite(future_prices[:, 1:])
            & np.isfinite(future_prices[:, :-1])
            & (future_prices[:, 1:] > 0.0)
            & (future_prices[:, :-1] > 0.0)
        )
        future_returns = np.divide(
            future_prices[:, 1:],
            future_prices[:, :-1],
            out=np.ones_like(future_prices[:, 1:]),
            where=future_valid,
        ) - 1.0
        future_returns = np.where(future_valid, future_returns, 0.0)
        return {
            "observations": observations,
            "asset_ids": private_ids.astype(np.int64),
            "mask": np.ones((view_count, config.MAX_COINS), dtype=np.float32),
            "prev_action": previous,
            "prev_mask": np.ones((view_count, config.MAX_COINS), dtype=np.float32),
            "basket_context": basket_context,
            "basket_membership": membership,
            "local_feature_history": np.nan_to_num(local).astype(
                config.MDAEN_FAM_STORAGE_DTYPE
            ),
            "episode_feature_history": np.nan_to_num(episode).astype(
                config.MDAEN_FAM_STORAGE_DTYPE
            ),
            "feature_history_mask": np.ones(
                (view_count, history), dtype=np.float32
            ),
            "future_unit_returns": future_returns.astype(np.float32),
            "future_valid": future_valid,
        }

    def _complete_observation(self, time_index, basket_indices):
        """Build the policy observation for one sampled market view.

        The local history remains raw and peer-independent.  Only the bounded
        81-coordinate snapshot uses the environment's selected-basket rolling
        affine transform, preserving the residual path's established scale.
        """
        local, episode, future, future_valid = self._view(
            time_index, basket_indices, return_valid=True
        )
        current = int(time_index)
        session = self.session_codes[current]
        session_rows = np.flatnonzero(self.session_codes == session)
        session_start = int(session_rows[0])
        scaler_steps = int(getattr(config, "SCALER_WINDOW_STEPS", 60))
        scale_start = max(session_start, current - scaler_steps + 1)
        scale_window = np.stack(
            [
                column[scale_start : current + 1, basket_indices]
                for column in self.primary_columns
            ],
            axis=-1,
        ).reshape(-1, len(config.SCOLS))
        scale_mean = scale_window.mean(axis=0)
        scale_std = np.maximum(scale_window.std(axis=0), 1e-3)
        scaled_primary = np.clip(
            (
                np.stack(
                    [column[current, basket_indices] for column in self.primary_columns],
                    axis=-1,
                )
                - scale_mean
            )
            / scale_std,
            -3.0,
            3.0,
        ).astype(np.float32)

        # The final local/episode frames contain the six behavioral anchors and
        # seventeen episode-relative coordinates in their schema order.
        behavioral = np.concatenate(
            (local[:, -1, -3:], episode[:, -1, :3]), axis=-1
        )
        relative = episode[:, -1, 3:]
        tokens = np.concatenate(
            (
                scaled_primary,
                np.zeros((len(basket_indices), 1), dtype=np.float32),
                behavioral,
                relative,
            ),
            axis=-1,
        ).astype(np.float32)
        if tokens.shape[-1] != config.FEATURES_PER_COIN:
            raise RuntimeError("foundation snapshot token width mismatch")

        basket_context = np.zeros(config.BASKET_CONTEXT_DIM, dtype=np.float32)
        if config.BASKET_CONTEXT_DIM:
            scale_index = config.BEHAVIORAL_SCALE_FEATURE_INDEX
            relative_index = config.EPISODE_RELATIVE_FEATURE_INDEX
            values = {
                "basket_realized_vol_median_30": float(
                    behavioral[:, scale_index["basket_realized_vol_median_30"]].mean()
                ),
                "basket_realized_vol_dispersion_30": float(
                    behavioral[:, scale_index["basket_realized_vol_dispersion_30"]].mean()
                ),
                "basket_factor_vol_30": float(
                    behavioral[:, scale_index["basket_factor_vol_30"]].mean()
                ),
                "basket_equal_weight_open_s": float(
                    relative[:, relative_index["open_s"]].mean()
                ),
                "basket_open_s_dispersion": float(
                    relative[:, relative_index["open_s"]].std()
                ),
                "basket_rank_concentration": float(
                    np.ptp(relative[:, relative_index[
                        "episode_cross_sectional_momentum_rank"
                    ]])
                ),
                "basket_mean_peer_correlation": float(
                    relative[:, relative_index[
                        "episode_peer_correlation_mean"
                    ]].mean()
                ),
                "basket_mean_peer_correlation_abs_max": float(
                    relative[:, relative_index[
                        "episode_peer_correlation_abs_max"
                    ]].mean()
                ),
            }
            basket_context = np.asarray(
                [values[name] for name in config.BASKET_CONTEXT_FEATURES],
                dtype=np.float32,
            )
            basket_context = np.clip(
                basket_context,
                config.BASKET_CONTEXT_FEATURE_MIN,
                config.BASKET_CONTEXT_FEATURE_MAX,
            )
        previous = self.rng.dirichlet(
            np.ones(config.MAX_COINS + 1, dtype=np.float64)
        ).astype(np.float32)
        private_ids = self.private_asset_ids[basket_indices]
        membership = np.zeros(config.BASKET_MEMBERSHIP_DIM, dtype=np.float32)
        valid_ids = private_ids[private_ids < config.BASKET_MEMBERSHIP_DIM]
        membership[valid_ids] = 1.0
        return {
            "observations": tokens.reshape(-1),
            "asset_ids": private_ids.astype(np.int64),
            "mask": np.ones(config.MAX_COINS, dtype=np.float32),
            "prev_action": previous,
            "prev_mask": np.ones(config.MAX_COINS, dtype=np.float32),
            "basket_context": basket_context,
            "basket_membership": membership,
            "local_feature_history": local.astype(config.MDAEN_FAM_STORAGE_DTYPE),
            "episode_feature_history": episode.astype(config.MDAEN_FAM_STORAGE_DTYPE),
            "feature_history_mask": np.ones(
                config.MDAEN_FAM_HISTORY_LENGTH, dtype=np.float32
            ),
            "future_unit_returns": future.astype(np.float32),
            "future_valid": future_valid,
        }

    def _view(self, time_index, basket_indices, *, return_valid=False):
        start = int(time_index) - self.required_past
        stop = int(time_index) + 1
        prices = self.open_prices[start:stop, basket_indices]
        timestamps = self.timestamps[start:stop]
        corporate_factors, corporate_events = _detect_corporate_action_factors(
            prices,
            self.close_prices[start:stop, basket_indices],
            timestamps,
        )
        primary_contamination = _forward_corporate_action_primary_mask(
            corporate_events,
            int(config.CORPORATE_ACTION_PRIMARY_MASK_STEPS),
        )
        behavioral = _behavioral_scale_feature_series(
            prices,
            high_prices=self.high_prices[start:stop, basket_indices],
            low_prices=self.low_prices[start:stop, basket_indices],
            timestamps=timestamps,
        )
        relative = _episode_relative_feature_series(
            prices,
            timestamps=timestamps,
            volume_surprises=(
                self.volume_surprises[start:stop, basket_indices]
                if self.volume_surprises is not None
                else None
            ),
            corporate_action_factors=corporate_factors,
        )
        history = int(config.MDAEN_FAM_HISTORY_LENGTH)
        local_primary = np.stack(
            [
                self.primary_columns[int(feature_index)][
                    int(time_index) - history + 1 : int(time_index) + 1,
                    basket_indices,
                ]
                for feature_index in self.primary_indices
            ],
            axis=-1,
        )
        local_primary = np.where(
            ~primary_contamination[-history:, :, None], local_primary, 0.0
        )
        local = np.concatenate(
            (
                local_primary,
                behavioral[-history:, :, :3],
            ),
            axis=-1,
        ).transpose(1, 0, 2)
        episode = np.concatenate(
            (behavioral[-history:, :, 3:], relative[-history:]), axis=-1
        ).transpose(1, 0, 2)
        horizon = max(config.FOUNDATION_HORIZONS)
        future_prices = self.open_prices[
            int(time_index) : int(time_index) + horizon + 1,
            basket_indices,
        ]
        future_timestamps = self.timestamps[
            int(time_index) : int(time_index) + horizon + 1
        ]
        future_factors, future_events = _detect_corporate_action_factors(
            future_prices,
            self.close_prices[
                int(time_index) : int(time_index) + horizon + 1,
                basket_indices,
            ],
            future_timestamps,
        )
        corrected_log_returns = _remove_corporate_action_factors(
            _causal_open_log_returns(future_prices), future_factors
        )
        future_returns = np.expm1(corrected_log_returns[1:])
        future_valid = (~future_events[1:]) & np.isfinite(future_returns)
        payload = (
            np.nan_to_num(local).astype(np.float32),
            np.nan_to_num(episode).astype(np.float32),
            np.nan_to_num(future_returns).astype(np.float32),
        )
        return (*payload, future_valid) if return_valid else payload

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("foundation batch_size must be positive")
        basket_size = int(config.MAX_COINS)
        universe_size = len(self.coins)
        if universe_size <= basket_size:
            raise ValueError(
                "paired peer substitution requires a universe larger than MAX_COINS"
            )
        names = (
            "time_indices", "target_ids", "basket_ids", "peer_basket_ids",
            "local_feature_history", "episode_feature_history",
            "peer_local_feature_history", "peer_episode_feature_history",
            "future_unit_returns",
            "future_valid",
            "observations", "asset_ids", "mask", "prev_action", "prev_mask",
            "basket_context", "basket_membership", "feature_history_mask",
            "peer_observations", "peer_asset_ids", "peer_mask",
            "peer_prev_action", "peer_prev_mask", "peer_basket_context",
            "peer_basket_membership", "peer_feature_history_mask",
        )
        time_indices = np.empty(batch_size, dtype=np.int64)
        target_ids = np.empty(batch_size, dtype=np.int64)
        baskets = np.empty((batch_size, basket_size), dtype=np.int64)
        peer_baskets = np.empty_like(baskets)
        for row in range(batch_size):
            time_index = int(self.rng.choice(self.eligible_times))
            target = int(self.rng.integers(universe_size))
            remaining = np.asarray(
                [index for index in range(universe_size) if index != target],
                dtype=np.int64,
            )
            peers = self.rng.choice(
                remaining, size=basket_size - 1, replace=False
            )
            basket = np.concatenate(([target], peers)).astype(np.int64)
            replace_slot = int(self.rng.integers(1, basket_size))
            basket_set = set(int(value) for value in basket)
            outsiders = np.asarray(
                [index for index in remaining if int(index) not in basket_set],
                dtype=np.int64,
            )
            peer_basket = basket.copy()
            peer_basket[replace_slot] = int(self.rng.choice(outsiders))
            basket = basket[self.rng.permutation(basket_size)]
            peer_basket = peer_basket[self.rng.permutation(basket_size)]
            time_indices[row] = time_index
            target_ids[row] = target
            baskets[row] = basket
            peer_baskets[row] = peer_basket

        combined = self._complete_observation_batch(
            np.concatenate((time_indices, time_indices)),
            np.concatenate((baskets, peer_baskets), axis=0),
        )
        view = {key: value[:batch_size] for key, value in combined.items()}
        peer_view = {key: value[batch_size:] for key, value in combined.items()}
        result = {
            "time_indices": time_indices,
            "target_ids": target_ids,
            "basket_ids": baskets,
            "peer_basket_ids": peer_baskets,
            "local_feature_history": view["local_feature_history"],
            "episode_feature_history": view["episode_feature_history"],
            "peer_local_feature_history": peer_view["local_feature_history"],
            "peer_episode_feature_history": peer_view["episode_feature_history"],
            "future_unit_returns": view["future_unit_returns"],
            "future_valid": view["future_valid"],
        }
        for key in (
            "observations", "asset_ids", "mask", "prev_action", "prev_mask",
            "basket_context", "basket_membership", "feature_history_mask",
        ):
            result[key] = view[key]
            result[f"peer_{key}"] = peer_view[key]
        missing = set(names).difference(result)
        if missing:
            raise RuntimeError(f"foundation batch is missing fields: {sorted(missing)}")
        return result

def build_basket_membership_multihot(episode_coins):
    """Return the private, permutation-invariant replay key for a basket.

    ``BASKET_MEMBERSHIP_ENABLED`` controls whether a learned policy conditioner
    may consume basket membership.  It must not erase the environment-owned
    provenance used by curriculum replay, checkpoint reconstruction, and
    teacher diagnostics.  Path 2 therefore emits this exact multi-hot metadata
    even while actor, critics, FAM/AAM, predictive routing, and the curriculum
    value model all resolve their learned membership width to zero.
    """
    dim = int(getattr(config, "BASKET_MEMBERSHIP_DIM", 0))
    out = np.zeros(dim, dtype=np.float32)
    if dim <= 0:
        return out
    ticker_to_id = getattr(config, "TICKER_TO_ID", {})
    for coin in episode_coins:
        asset_id = ticker_to_id.get(coin)
        if asset_id is None:
            continue
        if 0 <= int(asset_id) < dim:
            out[int(asset_id)] = 1.0
    return out


def basket_substitution_distance(episode_coins, anchor_coins=None):
    """Number of one-name substitutions separating a basket from the anchor."""
    anchor = set(
        anchor_coins
        if anchor_coins is not None
        else getattr(config, "BASKET_CURRICULUM_ANCHOR", config.TRADE_SYMBOLS)
    )
    basket = set(episode_coins)
    return int(len(anchor - basket))


def _initialize_data_cache():
    """
    Reuse the already prepared dataframe from config.py.

    The dataset is loaded, timezone-normalized, sorted, and downcast once
    during config import. Re-copying or reparsing it per environment only adds
    startup cost and RAM pressure.
    """
    global _GLOBAL_DATA_CACHE

    if _GLOBAL_DATA_CACHE is None:
        _GLOBAL_DATA_CACHE = config.data_set
        print(f"✅ Shared preprocessed dataset with {len(_GLOBAL_DATA_CACHE)} rows.")
        print(f"📊 Scaler will be fitted PER-EPISODE (matching live trading behavior)")

    return _GLOBAL_DATA_CACHE


class PortfolioEnv(gym.Env):
    """Portfolio allocation environment for STOCKS & ETFS trading using Gymnasium interface
    
    Supports both fixed portfolio (legacy) and variable portfolio sizes with masking.
    """

    def __init__(
        self,
        use_variable_portfolio=None,
        reward_type="simple",
        basket_sampler=None,
        action_normalization="hierarchical",
    ):
        super(PortfolioEnv, self).__init__()
        if action_normalization not in {"hierarchical", "flowrl_joint_simplex_v1"}:
            raise ValueError(f"Unknown action normalization: {action_normalization}")
        self.action_normalization = action_normalization
        # Only the training environment receives this object.  Evaluation keeps
        # the historical config.get_random_coin_subset() reset path.
        self.basket_sampler = basket_sampler
        self._active_curriculum_selection = None
        self._step_debug = bool(getattr(config, "ENVIRONMENT_STEP_DEBUG", False))
        
        # Use config setting or override
        self.use_variable_portfolio = (
            use_variable_portfolio if use_variable_portfolio is not None 
            else config.USE_VARIABLE_PORTFOLIO
        )
        self.global_pool_enabled = bool(
            self.use_variable_portfolio and config.EARNMORE_GLOBAL_POOL_ENABLED
        )
        self.global_pool_tickers = tuple(config.COIN_POOL)
        if self.global_pool_enabled and config.MAX_COINS != len(self.global_pool_tickers):
            raise ValueError("Global-pool capacity must equal the canonical universe size")
        
        # Set reward type for different reward calculations
        self.reward_type = reward_type
        print(f"🎯 Portfolio Environment initialized with reward_type: {reward_type}")
        # Act-at-open alignment: features are pre-shifted in preprocessing when enabled.
        self.act_at_open = config.ACT_AT_OPEN
        
        # ✅ PERFORMANCE FIX: Initialize global data cache once
        # NOTE: Scaler is NOT cached - it's fitted PER-EPISODE in reset() 
        # to match live trading behavior
        self.cached_data = _initialize_data_cache()

        if self.use_variable_portfolio:
            self._setup_variable_portfolio()
        else:
            self._setup_fixed_portfolio()
        
        # Initialize environment state
        self.reset()

    def _setup_variable_portfolio(self):
        """Setup observation and action spaces for variable portfolio size."""
        
        # Action space: portfolio weights for cash + max possible coins
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(config.MAX_COINS + 1,),  # +1 for cash
            dtype=np.float32
        )
        
        # Observation space: Dict with observations, soft ticker ids, and mask
        self.observation_space = gym.spaces.Dict({
            'observations': gym.spaces.Box(
                low=np.array(config.OBS_COLS_MIN_VARIABLE, dtype=np.float32), 
                high=np.array(config.OBS_COLS_MAX_VARIABLE, dtype=np.float32), 
                shape=(config.MAX_OBSERVATION_DIM,), 
                dtype=np.float32
            ),
            'asset_ids': gym.spaces.Box(
                low=0,
                high=config.PADDING_ASSET_ID,
                shape=(config.MAX_COINS,),
                dtype=np.int64
            ),
            'mask': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.MAX_COINS,),  # Mask for coins (not cash)
                dtype=np.float32
            ),
            'prev_action': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.MAX_COINS + 1,),
                dtype=np.float32
            ),
            'prev_mask': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.MAX_COINS,),  # Previous mask for coins
                dtype=np.float32
            ),
            'basket_context': gym.spaces.Box(
                low=np.asarray(config.BASKET_CONTEXT_FEATURE_MIN, dtype=np.float32),
                high=np.asarray(config.BASKET_CONTEXT_FEATURE_MAX, dtype=np.float32),
                shape=(config.BASKET_CONTEXT_DIM,),
                dtype=np.float32,
            ),
            'basket_membership': gym.spaces.Box(
                low=np.asarray(config.BASKET_MEMBERSHIP_FEATURE_MIN, dtype=np.float32),
                high=np.asarray(config.BASKET_MEMBERSHIP_FEATURE_MAX, dtype=np.float32),
                shape=(config.BASKET_MEMBERSHIP_DIM,),
                dtype=np.float32,
            ),
            'local_feature_history': gym.spaces.Box(
                low=float(config.MDAEN_FAM_FEATURE_MIN),
                high=float(config.MDAEN_FAM_FEATURE_MAX),
                shape=(
                    config.MAX_COINS,
                    config.MDAEN_FAM_HISTORY_LENGTH,
                    config.MDAEN_LOCAL_FEATURE_COUNT,
                ),
                dtype=config.MDAEN_FAM_STORAGE_DTYPE,
            ),
            'episode_feature_history': gym.spaces.Box(
                low=float(config.MDAEN_FAM_FEATURE_MIN),
                high=float(config.MDAEN_FAM_FEATURE_MAX),
                shape=(
                    config.MAX_COINS,
                    config.MDAEN_FAM_HISTORY_LENGTH,
                    config.MDAEN_EPISODE_FEATURE_COUNT,
                ),
                dtype=config.MDAEN_FAM_STORAGE_DTYPE,
            ),
            'feature_history_mask': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.MDAEN_FAM_HISTORY_LENGTH,),
                dtype=np.float32,
            ),
        })
        if config.EARNMORE_POOL_LEVEL_ENABLED:
            self.observation_space.spaces['pool_history_valid'] = gym.spaces.Box(
                0, 1, shape=(config.MAX_COINS, config.MDAEN_FAM_HISTORY_LENGTH), dtype=np.uint8)
        if config.PORTFOLIO_REWARD_HISTORY_ENABLED:
            self.observation_space.spaces['portfolio_reward_history'] = gym.spaces.Box(
                -np.inf, np.inf,
                shape=(config.portfolio_reward_history_dim(),), dtype=np.float32)
        
        print(f"🔧 Variable portfolio setup:")
        print(f"   Action space: {self.action_space}")
        print(
            f"   Observation space: Dict with {config.MAX_OBSERVATION_DIM} features "
            f"+ {config.MAX_COINS} asset_ids/mask "
            f"+ basket_context[{config.BASKET_CONTEXT_DIM}] "
            f"+ private basket_membership[{config.BASKET_MEMBERSHIP_DIM}] "
            f"+ local history[{config.MAX_COINS}×"
            f"{config.MDAEN_FAM_HISTORY_LENGTH}×"
            f"{config.MDAEN_LOCAL_FEATURE_COUNT}] + episode history["
            f"{config.MAX_COINS}×{config.MDAEN_FAM_HISTORY_LENGTH}×"
            f"{config.MDAEN_EPISODE_FEATURE_COUNT}]"
        )

    def _setup_fixed_portfolio(self):
        """Setup observation and action spaces for fixed portfolio size (legacy)."""
        
        # Define action space: portfolio weights (including cash) - continuous values between 0 and 1
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(len(config.COINS)+1,), 
            dtype=np.float32
        )
        
        # Define observation space: scaled market features
        self.observation_space = gym.spaces.Box(
            low=np.array(config.OBS_COLS_MIN, dtype=np.float32), 
            high=np.array(config.OBS_COLS_MAX, dtype=np.float32), 
            shape=(len(config.OBS_COLS),), 
            dtype=np.float32
        )
        
        print(f"🔧 Fixed portfolio setup (legacy):")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space}")

    def _get_asset_id_vector(self):
        """Return canonical ticker ids aligned to the episode's current token order."""
        asset_ids = np.full(config.MAX_COINS, config.PADDING_ASSET_ID, dtype=np.int64)
        for i, coin in enumerate(self.episode_coins[:config.MAX_COINS]):
            asset_ids[i] = config.TICKER_TO_ID.get(coin, config.PADDING_ASSET_ID)
        return asset_ids

    def _global_asset_slots(self):
        """Canonical public coordinates; execution bookkeeping remains compact."""
        slots = np.asarray([config.TICKER_TO_ID[c] for c in self.episode_coins], dtype=np.int64)
        if len(set(slots.tolist())) != len(slots):
            raise ValueError("An episode cannot contain duplicate assets")
        return slots

    def _portfolio_reward_history(self):
        """Causal, unclipped state for the history-dependent shaped reward.

        Rows are oldest-to-newest with leading zero padding and explicit
        validity. Market-only FAM does not contain the agent's past books.
        Price/value ratios and entry ratios use reversible asinh/log units,
        rather than the clipped cost-basis feature. No future tape is read.
        The field already uses public global coordinates before publication.
        """
        length, assets = int(config.LONG_TERM_LOOKBACK), int(config.MAX_COINS)
        slots = (self._global_asset_slots() if self.global_pool_enabled
                 else np.arange(len(self.episode_coins)))
        allocations = np.zeros((length, assets + 1), np.float32)
        allocation_valid = np.zeros(length, np.float32)
        rows = self.allocation_history[-length:]
        for t, row in enumerate(rows, start=length-len(rows)):
            allocations[t, 0] = row[0]
            allocations[t, slots + 1] = np.asarray(row)[1:len(slots)+1]
            allocation_valid[t] = 1
        prices = np.zeros((length, assets), np.float32)
        price_valid = np.zeros_like(prices)
        current_prices = dict(zip(self._current_step_coins, self._current_step_open_prices))
        price_rows = self.price_history[-length:]
        entry_ratios = np.zeros(assets, np.float32)
        entry_valid = np.zeros(assets, np.float32)
        current_price_state = np.zeros(assets, np.float32)
        for i, (coin, slot) in enumerate(zip(self.episode_coins, slots)):
            reference = float(current_prices.get(coin, price_rows[-1].get(coin, 0.) if price_rows else 0.))
            if reference <= 0 or not np.isfinite(reference):
                continue
            current_price_state[slot] = np.arcsinh(np.log(reference))
            for t, row in enumerate(price_rows, start=length-len(price_rows)):
                price = float(row.get(coin, 0.))
                if price > 0 and np.isfinite(price):
                    prices[t, slot] = np.arcsinh(100. * np.log(price / reference))
                    price_valid[t, slot] = 1
            entry = float(self.avg_entry_price[i+1])
            if entry > 0:
                entry_ratios[slot] = np.arcsinh(10. * np.log(reference / entry))
                entry_valid[slot] = 1
        values = np.zeros(length+1, np.float32)
        value_valid = np.zeros_like(values)
        value_rows = self.portfolio_value_history[-(length+1):]
        current_value = max(float(self.current_value), 1e-9)
        for t, value in enumerate(value_rows, start=length+1-len(value_rows)):
            if value > 0:
                values[t] = np.arcsinh(100. * np.log(value / current_value))
                value_valid[t] = 1
        returns = np.zeros(length, np.float32)
        return_valid = np.zeros_like(returns)
        return_rows = list(self.return_history)[-length:]
        if return_rows:
            returns[-len(return_rows):] = np.arcsinh(100. * np.asarray(return_rows))
            return_valid[-len(return_rows):] = 1
        actual_book = np.zeros(assets+1, np.float32)
        actual_book[0] = self.money_split_ratio[0]
        actual_book[slots+1] = np.asarray(self.money_split_ratio)[1:len(slots)+1]
        state = np.asarray([
            np.arcsinh(current_value / max(float(self.init_cash), 1e-9)),
            getattr(self, '_reward_history_steps', 0) / float(config.EPISODE_LENGTH),
        ], np.float32)
        history = np.concatenate([
            allocations.ravel(), allocation_valid, prices.ravel(), price_valid.ravel(),
            values, value_valid, returns, return_valid, entry_ratios,
            entry_valid, current_price_state, actual_book, state,
        ]).astype(np.float32, copy=False)
        if history.shape != (config.portfolio_reward_history_dim(),) or not np.isfinite(history).all():
            raise ValueError('Invalid portfolio reward history state')
        return history

    def _publish_global_pool(self, observation, info):
        """Scatter the complete transition state, never just its market snapshot.

        The simulator holds cash followed by episode-order positions internally.
        All actor/critic/replay coordinates instead use the immutable global map.
        Whole-stock learning retains historical targets in the existing local
        history storage. The encoder masks those targets before all control
        computations; snapshots and holdings of unavailable stocks stay zero.
        """
        if config.PORTFOLIO_REWARD_HISTORY_ENABLED and isinstance(observation, dict):
            observation = dict(observation)
            observation['portfolio_reward_history'] = self._portfolio_reward_history()
        if not self.global_pool_enabled:
            return observation, info
        slots = self._global_asset_slots()
        count = len(slots)
        result = dict(observation)
        for key in ("mask", "prev_mask", "local_feature_history", "episode_feature_history"):
            source = observation[key]
            target = np.zeros_like(source)
            target[slots] = source[:count]
            result[key] = target
        source = observation["observations"].reshape(config.MAX_COINS, config.FEATURES_PER_COIN)
        target = np.zeros_like(source)
        target[slots] = source[:count]
        result["observations"] = target.reshape(-1)
        result["asset_ids"] = np.arange(config.MAX_COINS, dtype=np.int64)
        result["prev_action"] = np.zeros_like(observation["prev_action"])
        result["prev_action"][0] = observation["prev_action"][0]
        result["prev_action"][slots + 1] = observation["prev_action"][1:count + 1]
        if config.EARNMORE_POOL_LEVEL_ENABLED:
            history, valid = self._global_pool_target_history()
            # Preserve the exact active history used by the execution episode.
            history[slots] = result["local_feature_history"][slots]
            valid[slots] = observation["feature_history_mask"].astype(np.uint8)
            result["local_feature_history"] = history
            result["pool_history_valid"] = valid
        info = dict(info)
        info["mask"] = result["mask"].copy()
        info["global_pool_tickers"] = self.global_pool_tickers
        info["global_asset_slots"] = slots.copy()
        info["market_input_contract"] = config.market_input_contract()
        # Replay's per-asset labels must use the same coordinates as its states.
        for key in ("component_outcomes", "component_outcome_mask"):
            if key in info:
                source = np.asarray(info[key])
                target = np.zeros_like(source)
                target[slots] = source[:count]
                info[key] = target
        return result, info

    def _prepare_global_pool_targets(self, start, end):
        """Bounded causal history panel, constructed once per episode.

        Reuse local history replay storage rather than storing a second full
        tensor. Data for excluded stocks are reconstruction labels only.
        """
        rows = self.cached_data.loc[
            self.cached_data['date'].between(start, end)
            & self.cached_data['coin'].isin(self.global_pool_tickers)]
        rows = _attach_episode_relative_features(rows, list(self.global_pool_tickers))
        self._pool_target_dates = pd.DatetimeIndex(rows['date'].unique()).sort_values()
        names = list(config.MDAEN_FAM_PRIMARY_FEATURES) + list(config.MDAEN_LOCAL_BEHAVIORAL_FEATURES)
        index = pd.MultiIndex.from_product(
            [self._pool_target_dates, self.global_pool_tickers], names=['date', 'coin'])
        indexed = rows.drop_duplicates(['date', 'coin']).set_index(['date', 'coin']).reindex(index)
        values = indexed[names].to_numpy(dtype=np.float32)
        values = values.reshape(len(self._pool_target_dates), config.MAX_COINS, len(names))
        self._pool_target_valid = np.isfinite(values).all(axis=-1)
        self._pool_target_values = np.nan_to_num(np.clip(
            values, config.MDAEN_FAM_FEATURE_MIN, config.MDAEN_FAM_FEATURE_MAX),
            nan=0., posinf=config.MDAEN_FAM_FEATURE_MAX, neginf=config.MDAEN_FAM_FEATURE_MIN
        ).astype(config.MDAEN_FAM_STORAGE_DTYPE)
        if self._cross_basket_relabel_active():
            self._prepare_cross_basket_relabel_panel(indexed)
        else:
            self._pool_relabel_date_index = None

    def _cross_basket_relabel_active(self):
        """Whether transitions carry the exogenous state needed for relabeling."""
        return bool(
            getattr(config, "CROSS_BASKET_RELABEL_ENABLED", False)
            and self.global_pool_enabled
            and config.EARNMORE_POOL_LEVEL_ENABLED
            and config.LOCAL_CAUSAL_INPUTS_ENABLED
            and self.reward_type == "STRUCTURED_CREDIT"
        )

    def _prepare_cross_basket_relabel_panel(self, indexed):
        """Per-minute snapshot blocks and open prices for every pool asset.

        The primary block goes through the same stateless local-causal scaling
        as an active asset's snapshot, and the behavioral anchors come from the
        same episode feature window, so an asset's block here equals the block
        it would carry if it were in the episode basket.
        """
        date_count = len(self._pool_target_dates)
        asset_count = len(self.global_pool_tickers)
        raw_primary = indexed[config.SCOLS].to_numpy(dtype=np.float32)
        contaminated = indexed["_corporate_action_primary_mask"]
        primary_valid = ~(contaminated.eq(True) | contaminated.isna()).to_numpy()
        scaled_primary = self._scale_feature_rows(
            raw_primary, valid_mask=primary_valid
        )
        behavioral = np.nan_to_num(
            indexed[list(config.BEHAVIORAL_SCALE_FEATURES)].to_numpy(
                dtype=np.float32
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        market = np.concatenate((scaled_primary, behavioral), axis=1)
        self._pool_relabel_market = market.reshape(
            date_count, asset_count, market.shape[1]
        ).astype(np.float32, copy=False)
        self._pool_relabel_open = indexed["open"].to_numpy(
            dtype=np.float64
        ).reshape(date_count, asset_count)
        self._pool_relabel_date_index = {
            date: position for position, date in enumerate(self._pool_target_dates)
        }

    def _cross_basket_relabel_info(self, decision_time):
        """Exogenous 13-asset market state spanning the transition just taken."""
        index_map = getattr(self, "_pool_relabel_date_index", None)
        if not index_map:
            return {}
        start = index_map.get(decision_time)
        end = index_map.get(self.current_time)
        if start is None or end is None:
            return {}
        start_open = self._pool_relabel_open[start]
        end_open = self._pool_relabel_open[end]
        price_valid = (
            np.isfinite(start_open)
            & np.isfinite(end_open)
            & (start_open > 0.0)
            & (end_open > 0.0)
        )
        asset_returns = np.zeros(len(start_open), dtype=np.float64)
        asset_returns[price_valid] = (
            end_open[price_valid] / start_open[price_valid] - 1.0
        )
        # The episode slice ends two rows into its final timestamp, so the last
        # transition values only the alphabetically first coins at real prices
        # and the rest at their previous price.  Relabeling applies the same
        # rule to the relabeled basket.
        present = len(getattr(self, "_current_step_coins", ()))
        partial_count = present if present < self.n_episode_coins else -1
        storage_dtype = np.dtype(
            getattr(config, "CROSS_BASKET_RELABEL_STORAGE_DTYPE", np.float16)
        )
        return {
            "relabel_market": self._pool_relabel_market[start].astype(storage_dtype),
            "relabel_next_market": self._pool_relabel_market[end].astype(storage_dtype),
            "relabel_asset_returns": asset_returns.astype(np.float32),
            "relabel_price_valid": price_valid,
            "relabel_next_partial_count": int(partial_count),
        }

    def _global_pool_target_history(self):
        end = int(self._pool_target_dates.searchsorted(self.current_time, side='right'))
        start = max(0, end - config.MDAEN_FAM_HISTORY_LENGTH)
        count = end - start
        history = np.zeros((config.MAX_COINS, config.MDAEN_FAM_HISTORY_LENGTH,
                            config.MDAEN_LOCAL_FEATURE_COUNT), dtype=config.MDAEN_FAM_STORAGE_DTYPE)
        valid = np.zeros(history.shape[:2], dtype=np.uint8)
        if count:
            history[:, -count:] = self._pool_target_values[start:end].transpose(1, 0, 2)
            valid[:, -count:] = self._pool_target_valid[start:end].T
        return history, valid

    def compact_policy_action(self, action):
        """Gather global policy logits into the simulator's episode order."""
        action = np.asarray(action, dtype=np.float32)
        if not self.global_pool_enabled:
            return action
        if action.shape != (config.MAX_COINS + 1,):
            raise ValueError(f"Expected global action shape {(config.MAX_COINS + 1,)}, got {action.shape}")
        compact = np.full_like(action, -1.0)
        compact[0] = action[0]
        compact[1:self.n_episode_coins + 1] = action[self._global_asset_slots() + 1]
        return compact

    def _build_basket_membership(self):
        """Private replay/curriculum membership metadata, never a Path-2 input."""
        return build_basket_membership_multihot(self.episode_coins)

    def _build_basket_context(
        self,
        behavioral_scale_rows,
        episode_relative_rows,
    ):
        """Continuous UVFA-style task code g(B) for the live episode basket.

        Built only from frozen/slow basket regime stats and higher-order
        cross-sectional summaries of episode-relative channels. No ticker
        taxonomy and no leveraged/commodity hand labels — the encoder must
        amortize peer geometry from observable market structure alone.
        """
        dim = int(getattr(config, "BASKET_CONTEXT_DIM", 0))
        out = np.zeros(dim, dtype=np.float32)
        if (config.LOCAL_CAUSAL_INPUTS_ENABLED or dim <= 0
                or not bool(getattr(config, "BASKET_CONTEXT_ENABLED", False))):
            return out
        if self.n_episode_coins <= 0:
            return out

        scale_rows = np.asarray(behavioral_scale_rows, dtype=np.float32)
        relative_rows = np.asarray(episode_relative_rows, dtype=np.float32)
        active_values = []
        for coin in self.episode_coins[: config.MAX_COINS]:
            row_idx = self._current_coin_to_row.get(coin)
            if row_idx is None:
                continue
            if row_idx >= len(scale_rows) or row_idx >= len(relative_rows):
                continue
            active_values.append((scale_rows[row_idx], relative_rows[row_idx]))
        if not active_values:
            return out

        scale_active = np.stack([pair[0] for pair in active_values], axis=0)
        relative_active = np.stack([pair[1] for pair in active_values], axis=0)
        scale_index = config.BEHAVIORAL_SCALE_FEATURE_INDEX
        relative_index = config.EPISODE_RELATIVE_FEATURE_INDEX

        # Basket-constant regime channels are identical across slots; take mean
        # for numerical robustness if any slot was padded.
        vol_median = float(
            np.mean(scale_active[:, scale_index["basket_realized_vol_median_30"]])
        )
        vol_dispersion = float(
            np.mean(
                scale_active[:, scale_index["basket_realized_vol_dispersion_30"]]
            )
        )
        factor_vol = float(
            np.mean(scale_active[:, scale_index["basket_factor_vol_30"]])
        )

        open_s = relative_active[:, relative_index["open_s"]]
        ranks = relative_active[
            :, relative_index["episode_cross_sectional_momentum_rank"]
        ]
        peer_corr = relative_active[
            :, relative_index["episode_peer_correlation_mean"]
        ]
        peer_corr_abs = relative_active[
            :, relative_index["episode_peer_correlation_abs_max"]
        ]

        equal_weight_open_s = float(np.mean(open_s))
        open_s_dispersion = float(np.std(open_s)) if len(open_s) > 1 else 0.0
        rank_concentration = float(np.max(ranks) - np.min(ranks)) if len(ranks) > 1 else 0.0
        mean_peer_corr = float(np.mean(peer_corr))
        mean_peer_corr_abs = float(np.mean(peer_corr_abs))

        values_by_name = {
            "basket_realized_vol_median_30": vol_median,
            "basket_realized_vol_dispersion_30": vol_dispersion,
            "basket_factor_vol_30": factor_vol,
            "basket_equal_weight_open_s": equal_weight_open_s,
            "basket_open_s_dispersion": open_s_dispersion,
            "basket_rank_concentration": rank_concentration,
            "basket_mean_peer_correlation": mean_peer_corr,
            "basket_mean_peer_correlation_abs_max": mean_peer_corr_abs,
        }
        for feature_index, feature_name in enumerate(config.BASKET_CONTEXT_FEATURES):
            out[feature_index] = values_by_name[feature_name]
        return np.clip(
            out,
            config.BASKET_CONTEXT_FEATURE_MIN,
            config.BASKET_CONTEXT_FEATURE_MAX,
        ).astype(np.float32, copy=False)

    def _episode_opportunity_diagnostics(self):
        """Return policy-independent opportunity risk and normalized growth."""
        variance_sum = max(
            0.0,
            float(getattr(self, "_episode_opportunity_variance_sum", 0.0)),
        )
        opportunity_risk_budget = float(np.sqrt(variance_sum))
        initial_value = float(getattr(self, "init_cash", config.INITIAL_CASH))
        current_value = float(getattr(self, "current_value", initial_value))
        if initial_value > 0.0 and current_value > 0.0:
            episode_log_growth = float(np.log(current_value / initial_value))
        else:
            episode_log_growth = 0.0
        epsilon = float(
            getattr(config, "OPPORTUNITY_NORMALIZED_RETURN_EPS", 1e-8)
        )
        normalized_return = (
            episode_log_growth / max(opportunity_risk_budget, epsilon)
            if opportunity_risk_budget > epsilon
            else 0.0
        )
        selection = self._active_curriculum_selection or {}
        return {
            "episode_financial_log_growth": episode_log_growth,
            "opportunity_risk_budget": opportunity_risk_budget,
            "opportunity_normalized_financial_return": float(normalized_return),
            "basket_substitution_distance": int(
                selection.get(
                    "distance",
                    basket_substitution_distance(self.episode_coins),
                )
            ),
            "curriculum_role": str(selection.get("role", "uniform_evaluation")),
            "curriculum_stage": str(selection.get("stage", "none")),
            "evaluation_curriculum_probability": float(
                selection.get("evaluation_curriculum_probability", 0.0)
            ),
            "curriculum_episode_index": int(selection.get("episode_index", -1)),
        }

    def _curriculum_value_feature(self, observation):
        """Build the identity-neutral set state used only by the teacher.

        Slot ordering is retained in the serialized feature solely so the
        value model can reshape it into tokens; the model immediately applies
        one shared encoder and permutation-invariant pooling.  Asset ids and
        exact membership never enter this feature.
        """
        if not self._curriculum_teacher_active() or not isinstance(
            observation, dict
        ):
            return None
        try:
            basket_context = np.asarray(
                observation["basket_context"], dtype=np.float32
            ).reshape(-1)
            market = np.asarray(
                observation["observations"], dtype=np.float32
            ).reshape(config.MAX_COINS, -1)
            mask = np.asarray(observation["mask"], dtype=np.float32).reshape(-1)
            previous_action = np.asarray(
                observation["prev_action"], dtype=np.float32
            ).reshape(-1)
            market_indices = np.asarray(
                config.CURRICULUM_VALUE_MARKET_TOKEN_INDICES, dtype=np.int64
            )
            observable_market = market[:, market_indices]
            asset_weights = np.zeros((market.shape[0], 1), dtype=np.float32)
            available_weight_count = min(
                market.shape[0], max(previous_action.size - 1, 0)
            )
            if available_weight_count:
                asset_weights[:available_weight_count, 0] = previous_action[
                    1:1 + available_weight_count
                ]
            token_features = np.concatenate(
                (observable_market, asset_weights), axis=-1
            ) * mask[: market.shape[0], None]
            progress = np.asarray(
                [float(self.index) / max(float(config.EPISODE_LENGTH), 1.0)],
                dtype=np.float32,
            )
            previous_cash = np.asarray(
                [previous_action[0] if previous_action.size else 1.0],
                dtype=np.float32,
            )
            feature = np.concatenate(
                (
                    token_features.reshape(-1),
                    mask[: market.shape[0]],
                    basket_context,
                    previous_cash,
                    progress,
                )
            )
            if feature.size != int(config.CURRICULUM_VALUE_INPUT_DIM):
                raise ValueError(
                    "curriculum value feature contract mismatch: expected "
                    f"{config.CURRICULUM_VALUE_INPUT_DIM}, received {feature.size}"
                )
            return np.nan_to_num(
                feature, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32, copy=False)
        except (KeyError, TypeError, ValueError):
            return None

    def _capture_curriculum_value_checkpoint(self, observation):
        if not self._curriculum_teacher_active():
            return
        targets = getattr(self, "_curriculum_checkpoint_steps", ())
        captured = getattr(self, "_curriculum_captured_checkpoint_steps", set())
        due = [step for step in targets if self.index >= step and step not in captured]
        if not due:
            return
        feature = self._curriculum_value_feature(observation)
        if feature is None:
            return
        reward_index = len(getattr(self, "_curriculum_financial_reward_trace", ()))
        for checkpoint_step in due:
            self._curriculum_value_checkpoints.append(
                {
                    "step": int(checkpoint_step),
                    "reward_index": int(reward_index),
                    "feature": feature.copy(),
                }
            )
            captured.add(int(checkpoint_step))
        self._curriculum_captured_checkpoint_steps = captured

    def _finalize_curriculum_value_samples(self):
        """Attach exact return-to-go targets to retained teacher states."""
        if not self._curriculum_teacher_active():
            return []
        rewards = np.asarray(
            getattr(self, "_curriculum_financial_reward_trace", ()),
            dtype=np.float64,
        )
        opportunity_variance = np.asarray(
            getattr(self, "_curriculum_opportunity_variance_trace", ()),
            dtype=np.float64,
        )
        gamma = float(getattr(config, "MULTITASK_GAMMA", 0.966))
        reward_scale = max(
            float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0)), 1e-8
        )
        epsilon = float(
            getattr(config, "OPPORTUNITY_NORMALIZED_RETURN_EPS", 1e-8)
        )
        samples = []
        for checkpoint in getattr(self, "_curriculum_value_checkpoints", ()):
            start = int(np.clip(checkpoint["reward_index"], 0, len(rewards)))
            remaining_rewards = rewards[start:]
            discounts = np.power(gamma, np.arange(len(remaining_rewards)))
            discounted_return = float(np.dot(discounts, remaining_rewards))
            remaining_log_growth = float(remaining_rewards.sum() / reward_scale)
            remaining_variance = opportunity_variance[start:]
            remaining_risk = float(
                np.sqrt(max(float(remaining_variance.sum()), 0.0))
            )
            normalized_return = (
                remaining_log_growth / max(remaining_risk, epsilon)
                if remaining_risk > epsilon
                else 0.0
            )
            samples.append(
                {
                    "step": int(checkpoint["step"]),
                    "feature": np.asarray(
                        checkpoint["feature"], dtype=np.float32
                    ).tolist(),
                    "discounted_financial_return": discounted_return,
                    "opportunity_normalized_return": float(normalized_return),
                }
            )
        return samples

    def _curriculum_teacher_active(self):
        """Whether the installed sampler consumes return-to-go evidence.

        The lightweight local-composition sampler controls reset ordering only.
        Treating every sampler as a value teacher would silently reconstruct
        the expensive curriculum return traces that Path 2 intentionally
        removed.
        """
        return self.basket_sampler is not None and (
            hasattr(self.basket_sampler, "observe_episode")
            or bool(getattr(config, "BASKET_CURRICULUM_ENABLED", False))
        )

    @staticmethod
    def _new_rolling_state(feature_count: int, maxlen: int):
        return {
            "window": deque(maxlen=maxlen),
            "sum": np.zeros(feature_count, dtype=np.float32),
            "sumsq": np.zeros(feature_count, dtype=np.float32),
        }

    @staticmethod
    def _append_rolling_row(state, row):
        row = np.asarray(row, dtype=np.float32)
        window = state["window"]
        if len(window) == window.maxlen:
            oldest = window.popleft()
            state["sum"] -= oldest
            state["sumsq"] -= oldest * oldest
        window.append(row)
        state["sum"] += row
        state["sumsq"] += row * row

    @staticmethod
    def _rolling_mean_std(state, std_floor=1e-3):
        count = len(state["window"])
        if count <= 0:
            return (
                np.zeros_like(state["sum"], dtype=np.float32),
                np.ones_like(state["sum"], dtype=np.float32),
            )
        mean = state["sum"] / float(count)
        variance = np.maximum(
            (state["sumsq"] / float(count)) - np.square(mean), 0.0
        )
        floor = np.asarray(std_floor, dtype=np.float32)
        if floor.ndim > 1 or (floor.ndim == 1 and floor.shape != variance.shape):
            raise ValueError(
                "std_floor must be scalar or match the rolling feature width"
            )
        std = np.maximum(
            np.sqrt(variance).astype(np.float32, copy=False), floor
        )
        return mean, std

    def _reset_feature_window(self, maxlen: int):
        """Create one causal normalizer shared by the selected episode basket."""
        self._feature_state = self._new_rolling_state(
            len(config.SCOLS), int(maxlen)
        )

    def _append_feature_row(self, row):
        self._append_rolling_row(self._feature_state, row)

    def _scale_feature_rows(self, feature_rows, valid_mask=None):
        """Normalize all active assets with the same causal affine transform.

        Because the rolling state contains only rows from ``episode_coins``, a
        peer can change the target asset's normalized value, but an unavailable
        asset cannot.  The common mean and standard deviation preserve feature
        ordering across active assets and commute with slot permutations.
        Rows whose offline fields may inherit a corporate-action discontinuity
        are neither emitted nor admitted into the rolling normalization state.
        """
        feature_rows = np.asarray(feature_rows, dtype=np.float32)
        if len(feature_rows) == 0:
            return np.zeros((0, len(config.SCOLS)), dtype=np.float32)
        if valid_mask is None:
            valid_mask = np.ones(len(feature_rows), dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (len(feature_rows),):
            raise ValueError("valid_mask must contain one value per feature row")
        if config.LOCAL_CAUSAL_INPUTS_ENABLED:
            # Fixed monotone scaling of causal asset-derived columns. No peers,
            # episode age, warmup statistics or future data enter this path.
            scaled = np.arcsinh(
                feature_rows / config.local_primary_references(config.SCOLS)
            )
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=3.0, neginf=-3.0)
            scaled = np.clip(scaled, -3.0, 3.0).astype(np.float32)
            scaled[~valid_mask] = 0.0
            return scaled
        for row, is_valid in zip(feature_rows, valid_mask):
            if is_valid:
                self._append_feature_row(row)
        mean, std = self._rolling_mean_std(self._feature_state)
        scaled = np.clip(
            (feature_rows - mean) / std, -3.0, 3.0
        ).astype(np.float32, copy=False)
        scaled[~valid_mask] = 0.0
        return scaled

    def _scale_episode_relative_rows(self, feature_rows):
        """Enforce semantic bounds without a second rolling normalization."""
        feature_rows = np.asarray(feature_rows, dtype=np.float32)
        if config.LOCAL_CAUSAL_INPUTS_ENABLED:
            return np.zeros_like(feature_rows)
        if len(feature_rows) == 0:
            return np.zeros(
                (0, config.EPISODE_RELATIVE_FEATURE_COUNT), dtype=np.float32
            )
        if feature_rows.shape[1] != config.EPISODE_RELATIVE_FEATURE_COUNT:
            raise ValueError("episode-relative rows have an invalid feature width")
        return np.clip(
            feature_rows,
            config.EPISODE_RELATIVE_FEATURE_MIN,
            config.EPISODE_RELATIVE_FEATURE_MAX,
        ).astype(np.float32, copy=False)

    def _build_episode_time_cache(self):
        self._time_slices = []
        self.unique_times = []
        self._time_coins = []
        self._time_open_prices = []
        self._time_close_prices = []
        self._time_feature_rows = []
        self._time_primary_valid_rows = []
        self._time_behavioral_scale_rows = []
        self._time_opportunity_volatility = []
        self._time_episode_relative_rows = []
        self._time_corporate_action_events = []
        self._time_corporate_action_factors = []

        for current_time, grp in self.df.groupby("date", sort=False):
            current_slice = grp.drop_duplicates("coin").reset_index(drop=True)
            self.unique_times.append(current_time)
            self._time_slices.append(current_slice)
            self._time_coins.append(current_slice["coin"].tolist())
            self._time_open_prices.append(current_slice["open"].to_numpy(dtype=np.float32, copy=False))
            self._time_close_prices.append(current_slice["close"].to_numpy(dtype=np.float32, copy=False))
            self._time_feature_rows.append(current_slice[config.SCOLS].to_numpy(dtype=np.float32, copy=False))
            self._time_primary_valid_rows.append(
                ~current_slice["_corporate_action_primary_mask"].to_numpy(
                    dtype=bool, copy=False
                )
            )
            self._time_behavioral_scale_rows.append(
                current_slice[config.BEHAVIORAL_SCALE_FEATURES].to_numpy(
                    dtype=np.float32, copy=False
                )
            )
            self._time_opportunity_volatility.append(
                float(
                    np.nanmean(
                        current_slice[
                            "_basket_opportunity_volatility_30"
                        ].to_numpy(dtype=np.float32, copy=False)
                    )
                )
            )
            self._time_episode_relative_rows.append(
                current_slice[config.EPISODE_RELATIVE_FEATURES].to_numpy(
                    dtype=np.float32, copy=False
                )
            )
            self._time_corporate_action_events.append(
                current_slice["_corporate_action_event"].to_numpy(
                    dtype=bool, copy=False
                )
            )
            self._time_corporate_action_factors.append(
                current_slice["_corporate_action_factor"].to_numpy(
                    dtype=np.float32, copy=False
                )
            )

    def _set_time_index(self, time_index: int):
        self.time_index = min(max(time_index, 0), len(self.unique_times) - 1)
        self.current_time = self.unique_times[self.time_index]
        self.dfslice = self._time_slices[self.time_index]
        self._current_step_coins = self._time_coins[self.time_index]
        self._current_step_open_prices = self._time_open_prices[self.time_index]
        self._current_step_close_prices = self._time_close_prices[self.time_index]
        self._current_step_feature_rows = self._time_feature_rows[self.time_index]
        self._current_step_primary_valid_rows = (
            self._time_primary_valid_rows[self.time_index]
        )
        self._current_step_behavioral_scale_rows = (
            self._time_behavioral_scale_rows[self.time_index]
        )
        self._current_step_opportunity_volatility = max(
            0.0,
            float(self._time_opportunity_volatility[self.time_index]),
        )
        self._current_step_episode_relative_rows = (
            self._time_episode_relative_rows[self.time_index]
        )
        self._current_step_corporate_action_events = (
            self._time_corporate_action_events[self.time_index]
        )
        self._current_step_corporate_action_factors = (
            self._time_corporate_action_factors[self.time_index]
        )
        self._current_coin_to_row = {
            coin: idx for idx, coin in enumerate(self._current_step_coins)
        }

    def _component_outcome_aux_active(self) -> bool:
        """Whether this transition must emit per-asset outcome metadata.

        Outcome *storage* is shared by two independent learners: the historical
        one-step component auxiliary and live multi-horizon foundation
        rehearsal.  Disabling the former must not starve the latter of labels.
        """
        foundation_sequence_replay = bool(
            getattr(config, "FOUNDATION_SEQUENCE_REPLAY_ENABLED", True)
            and getattr(config, "PREDICTIVE_SSL_ENABLED", False)
            and getattr(config, "FOUNDATION_LEARNING_ENABLED", False)
            and getattr(config, "FOUNDATION_ONLINE_REHEARSAL_ENABLED", True)
        )
        return bool(
            self.use_variable_portfolio
            and (
                bool(getattr(config, "COMPONENT_OUTCOME_AUX_ENABLED", False))
                or foundation_sequence_replay
            )
        )

    def _reset_component_contribution_buffers(self):
        """Clear per-asset financial labels for the current transition."""
        channel_count = len(
            tuple(getattr(config, "COMPONENT_OUTCOME_HORIZONS", (1,)))
        )
        self._step_component_contributions = np.zeros(
            (config.MAX_COINS, channel_count), dtype=np.float32
        )
        self._step_component_contribution_mask = np.zeros(
            (config.MAX_COINS, channel_count), dtype=bool
        )

    def _record_component_contribution(self, slot_index: int, value: float):
        """Store one asset's causal financial target in replay metadata."""
        if not self._component_outcome_aux_active():
            return
        if slot_index < 0 or slot_index >= config.MAX_COINS:
            return
        if not np.isfinite(value):
            return
        clip = float(config.COMPONENT_OUTCOME_TARGET_CLIP)
        clipped = float(np.clip(value, -clip, clip))
        self._step_component_contributions[slot_index, 0] = np.float32(
            self._step_component_contributions[slot_index, 0] + clipped
        )
        self._step_component_contribution_mask[slot_index, 0] = True

    def _build_component_outcome_info(self):
        """Expose causal per-asset financial targets without adding to state."""
        if not hasattr(self, "_step_component_contributions"):
            self._reset_component_contribution_buffers()
        return {
            "market_input_contract": config.market_input_contract(),
            "component_outcomes": np.asarray(
                self._step_component_contributions, dtype=np.float32
            ).copy(),
            "component_outcome_mask": np.asarray(
                self._step_component_contribution_mask, dtype=bool
            ).copy(),
            "component_outcome_target_kind": (
                "scaled_unit_asset_return"
                if bool(
                    getattr(
                        config,
                        "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS",
                        False,
                    )
                )
                else "scaled_executed_contribution"
            ),
            # This is a reconciliation diagnostic only.  August-10 replay and
            # critic learning intentionally continue to consume the same scalar
            # reward plus component_outcomes/component_outcome_mask contract.
            "component_global_residual": float(
                getattr(self, "_step_component_global_residual", 0.0)
            ),
        }

    def _finalize_causal_financial_credit(
        self,
        execution_prices,
        portfolio_reward,
    ):
        """Replace component teachers with causal financial outcomes.

        The STRUCTURED_CREDIT branch owns the scalar assignment directly.  This
        method constructs unit-exposure return labels by default so every active
        asset can teach the private forecaster, including assets assigned zero
        weight.  The reconciliation diagnostic still re-applies executed
        weights so it remains comparable with the scalar portfolio reward.
        """
        reward_scale = float(
            getattr(config, "FINANCIAL_REWARD_SCALE", 10.0)
        )

        # Structured shaping owns the scalar reward. Its retrospective asset
        # labels must not replace the separate causal next-open return teacher.
        self._reset_component_contribution_buffers()
        next_prices = (
            self.dfslice[["coin", "open"]]
            .set_index("coin")
            .to_dict()
            .get("open", {})
        )
        use_unit_return_targets = bool(
            getattr(
                config,
                "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS",
                False,
            )
        )
        stored_executed_contribution_sum = 0.0
        if self._component_outcome_aux_active() and hasattr(
            self, "last_target_allocation"
        ):
            for slot_index, coin in enumerate(self.episode_coins):
                asset_index = slot_index + 1  # cash occupies action index zero
                if asset_index >= len(self.last_target_allocation):
                    continue
                start_price = execution_prices.get(coin)
                end_price = next_prices.get(coin)
                if start_price is None or end_price is None:
                    continue
                unit_return = _scaled_asset_financial_return(
                    start_price,
                    end_price,
                    scale=reward_scale,
                )
                if unit_return is not None:
                    target_weight = float(
                        self.last_target_allocation[asset_index]
                    )
                    stored_target = (
                        unit_return
                        if use_unit_return_targets
                        else target_weight * unit_return
                    )
                    self._record_component_contribution(
                        slot_index, stored_target
                    )
                    if self._step_component_contribution_mask[slot_index, 0]:
                        stored_value = float(
                            self._step_component_contributions[slot_index, 0]
                        )
                        stored_executed_contribution_sum += (
                            target_weight * stored_value
                            if use_unit_return_targets
                            else stored_value
                        )

        # Log growth is nonlinear, while local labels use simple asset returns.
        # Keeping the exact remainder makes the decomposition auditable without
        # introducing the rejected factored critic/replay refactor.
        self._step_component_global_residual = float(
            float(portfolio_reward) - stored_executed_contribution_sum
        )

    def _reset_fam_history(self):
        """Reset peer-independent and episode-relative causal histories."""
        history_length = int(config.MDAEN_FAM_HISTORY_LENGTH)
        self._fam_local_history_buffer = np.zeros(
            (
                history_length,
                int(config.MAX_COINS),
                int(config.MDAEN_LOCAL_FEATURE_COUNT),
            ),
            dtype=np.float32,
        )
        self._fam_episode_history_buffer = np.zeros(
            (
                history_length,
                int(config.MAX_COINS),
                int(config.MDAEN_EPISODE_FEATURE_COUNT),
            ),
            dtype=np.float32,
        )
        self._fam_history_count = 0
        self._fam_history_write_index = 0
        self._fam_last_time_index = None

    def _prime_local_fam_history(self, preceding_rows):
        """Warm local history from preceding market rows, never future rows."""
        if not config.LOCAL_CAUSAL_INPUTS_ENABLED or preceding_rows.empty:
            return
        dates = np.sort(preceding_rows["date"].unique())[-config.MDAEN_FAM_HISTORY_LENGTH:]
        names = list(config.MDAEN_FAM_PRIMARY_FEATURES) + list(config.MDAEN_LOCAL_BEHAVIORAL_FEATURES)
        frames = []
        for date in dates:
            rows = preceding_rows.loc[preceding_rows["date"] == date].set_index("coin")
            values = rows.reindex(self.episode_coins)[names].to_numpy(dtype=np.float32)
            frame = np.zeros((config.MAX_COINS, config.MDAEN_LOCAL_FEATURE_COUNT), dtype=np.float32)
            frame[:self.n_episode_coins] = np.nan_to_num(np.clip(
                values, config.MDAEN_FAM_FEATURE_MIN, config.MDAEN_FAM_FEATURE_MAX
            ), nan=0.0, posinf=config.MDAEN_FAM_FEATURE_MAX, neginf=config.MDAEN_FAM_FEATURE_MIN)
            frames.append(frame)
        count = len(frames)
        self._fam_local_history_buffer[:count] = np.stack(frames)
        self._fam_history_count = count
        self._fam_history_write_index = count % config.MDAEN_FAM_HISTORY_LENGTH

    def _build_fam_frame(self, raw_primary, scaled_episode_relative):
        """Build separate intrinsic and selected-basket temporal frames.

        ``raw_primary`` is deliberately the causal asset-derived source block,
        not the selected-basket-normalized snapshot block.  Consequently a
        target asset's local frame cannot change merely because its peers were
        substituted. In local mode the reserved episode stream contains zeros;
        historical mode retains its basket-dependent coordinates.
        """
        local_frame = np.zeros(
            (config.MAX_COINS, config.MDAEN_LOCAL_FEATURE_COUNT),
            dtype=np.float32,
        )
        episode_frame = np.zeros(
            (config.MAX_COINS, config.MDAEN_EPISODE_FEATURE_COUNT),
            dtype=np.float32,
        )
        primary_indices = np.asarray(
            [config.SCOLS.index(name) for name in config.MDAEN_FAM_PRIMARY_FEATURES],
            dtype=np.int64,
        )
        for slot_index, coin in enumerate(self.episode_coins[: config.MAX_COINS]):
            row_index = self._current_coin_to_row.get(coin)
            if row_index is None:
                continue
            behavioral = np.asarray(
                self._current_step_behavioral_scale_rows[row_index],
                dtype=np.float32,
            )
            local_values = np.concatenate(
                (
                    np.asarray(raw_primary[row_index], dtype=np.float32)[primary_indices],
                    behavioral[:3],
                )
            )
            episode_values = np.concatenate(
                (behavioral[3:], scaled_episode_relative[row_index])
            )
            if config.LOCAL_CAUSAL_INPUTS_ENABLED:
                episode_values = np.zeros_like(episode_values)
            if local_values.size != config.MDAEN_LOCAL_FEATURE_COUNT:
                raise RuntimeError(
                    "MDAEN local history width mismatch: expected "
                    f"{config.MDAEN_LOCAL_FEATURE_COUNT}, received "
                    f"{local_values.size}"
                )
            if episode_values.size != config.MDAEN_EPISODE_FEATURE_COUNT:
                raise RuntimeError(
                    "MDAEN episode history width mismatch: expected "
                    f"{config.MDAEN_EPISODE_FEATURE_COUNT}, received "
                    f"{episode_values.size}"
                )
            local_frame[slot_index] = local_values
            episode_frame[slot_index] = episode_values

        def _sanitize(values):
            return np.nan_to_num(
                np.clip(
                    values,
                    float(config.MDAEN_FAM_FEATURE_MIN),
                    float(config.MDAEN_FAM_FEATURE_MAX),
                ),
                nan=0.0,
                posinf=float(config.MDAEN_FAM_FEATURE_MAX),
                neginf=float(config.MDAEN_FAM_FEATURE_MIN),
            ).astype(np.float32, copy=False)

        return _sanitize(local_frame), _sanitize(episode_frame)

    def _update_fam_history(self, local_frame, episode_frame):
        """Append once per market timestamp and return a right-aligned window."""
        if not hasattr(self, "_fam_local_history_buffer"):
            self._reset_fam_history()
        current_index = int(getattr(self, "time_index", 0))
        local_frame = np.asarray(local_frame, dtype=np.float32)
        episode_frame = np.asarray(episode_frame, dtype=np.float32)
        history_length = int(config.MDAEN_FAM_HISTORY_LENGTH)
        if self._fam_last_time_index == current_index and self._fam_history_count:
            last_index = (self._fam_history_write_index - 1) % history_length
            self._fam_local_history_buffer[last_index] = local_frame
            self._fam_episode_history_buffer[last_index] = episode_frame
        else:
            self._fam_local_history_buffer[self._fam_history_write_index] = local_frame
            self._fam_episode_history_buffer[self._fam_history_write_index] = episode_frame
            self._fam_history_write_index = (
                self._fam_history_write_index + 1
            ) % history_length
            self._fam_history_count = min(
                self._fam_history_count + 1, history_length
            )
            self._fam_last_time_index = current_index

        local_history = np.zeros(
            (
                config.MAX_COINS,
                history_length,
                config.MDAEN_LOCAL_FEATURE_COUNT,
            ),
            dtype=config.MDAEN_FAM_STORAGE_DTYPE,
        )
        episode_history = np.zeros(
            (
                config.MAX_COINS,
                history_length,
                config.MDAEN_EPISODE_FEATURE_COUNT,
            ),
            dtype=config.MDAEN_FAM_STORAGE_DTYPE,
        )
        history_mask = np.zeros(history_length, dtype=np.float32)
        count = int(self._fam_history_count)
        if count:
            start = history_length - count
            if count < history_length:
                local_history[:, start:, :] = self._fam_local_history_buffer[
                    :count
                ].transpose(1, 0, 2)
                episode_history[:, start:, :] = self._fam_episode_history_buffer[
                    :count
                ].transpose(1, 0, 2)
            else:
                oldest = int(self._fam_history_write_index)
                first_count = history_length - oldest
                local_history[:, :first_count, :] = self._fam_local_history_buffer[
                    oldest:
                ].transpose(1, 0, 2)
                episode_history[:, :first_count, :] = self._fam_episode_history_buffer[
                    oldest:
                ].transpose(1, 0, 2)
                if oldest:
                    local_history[:, first_count:, :] = self._fam_local_history_buffer[
                        :oldest
                    ].transpose(1, 0, 2)
                    episode_history[:, first_count:, :] = self._fam_episode_history_buffer[
                        :oldest
                    ].transpose(1, 0, 2)
            history_mask[start:] = 1.0
        return local_history, episode_history, history_mask

    def _get_variable_observation_fast(self):
        mask = np.zeros(config.MAX_COINS, dtype=np.float32)
        mask[:self.n_episode_coins] = 1.0
        observation = np.zeros(config.MAX_OBSERVATION_DIM, dtype=np.float32)

        scaled_values = self._scale_feature_rows(
            self._current_step_feature_rows,
            valid_mask=self._current_step_primary_valid_rows,
        )
        self._current_step_scaled_feature_rows = scaled_values
        scaled_episode_relative = self._scale_episode_relative_rows(
            self._current_step_episode_relative_rows
        )
        self._current_step_scaled_episode_relative_rows = scaled_episode_relative
        current_prices = {
            coin: float(price)
            for coin, price in zip(self._current_step_coins, self._current_step_open_prices)
        }

        base_feature_count = len(config.SCOLS)
        for slot_idx, coin in enumerate(self.episode_coins[:config.MAX_COINS]):
            row_idx = self._current_coin_to_row.get(coin)
            if row_idx is None:
                continue

            start_idx = slot_idx * config.FEATURES_PER_COIN
            # Preserve the stable token contract:
            # primary market block | cost basis | fixed-reference behavioral
            # scale | episode-relative market block. AttentionDeepSetEncoder
            # gathers all three market scopes into one joint market stem.
            observation[start_idx:start_idx + base_feature_count] = scaled_values[row_idx]

            avg_entry = float(self.avg_entry_price[slot_idx + 1])
            cost_basis_feature = 0.0
            if avg_entry > 1e-9:
                current_p = current_prices.get(coin, 0.0)
                if current_p > 1e-9:
                    raw_cost_basis = (current_p / avg_entry) - 1.0
                    cost_basis_feature = np.clip(
                        raw_cost_basis * config.COST_BASIS_FEATURE_SCALE,
                        -15.0,
                        50.0,
                    )
            observation[start_idx + config.COST_BASIS_TOKEN_INDEX] = cost_basis_feature
            scale_start = start_idx + config.COMPARATIVE_FEATURE_OFFSET
            scale_end = scale_start + config.BEHAVIORAL_SCALE_FEATURE_COUNT
            observation[scale_start:scale_end] = (
                self._current_step_behavioral_scale_rows[row_idx]
            )
            relative_start = start_idx + config.EPISODE_RELATIVE_FEATURE_OFFSET
            relative_end = relative_start + config.EPISODE_RELATIVE_FEATURE_COUNT
            observation[relative_start:relative_end] = (
                scaled_episode_relative[row_idx]
            )

        basket_context = self._build_basket_context(
            self._current_step_behavioral_scale_rows,
            scaled_episode_relative,
        )
        basket_membership = self._build_basket_membership()
        local_frame, episode_frame = self._build_fam_frame(
            self._current_step_feature_rows,
            scaled_episode_relative,
        )
        local_feature_history, episode_feature_history, feature_history_mask = self._update_fam_history(
            local_frame,
            episode_frame,
        )
        return (
            observation,
            mask,
            basket_context,
            basket_membership,
            local_feature_history,
            episode_feature_history,
            feature_history_mask,
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Select coins for this episode
        if self.use_variable_portfolio:
            if self.basket_sampler is not None:
                selection = self.basket_sampler.sample_basket()
                if not isinstance(selection, dict) or "coins" not in selection:
                    raise TypeError(
                        "basket_sampler.sample_basket() must return a dict "
                        "containing a 'coins' sequence"
                    )
                self._active_curriculum_selection = dict(selection)
                self.episode_coins = list(selection["coins"])
            else:
                self._active_curriculum_selection = None
                self.episode_coins = config.get_random_coin_subset()
            self.n_episode_coins = len(self.episode_coins)
            if self.global_pool_enabled and self.n_episode_coins != config.SELECTED_ASSETS_PER_EPISODE:
                raise ValueError("Global-pool training requires exactly eight selected assets")
            if self._step_debug:
                print(f"🎲 Episode coins: {self.episode_coins} ({self.n_episode_coins} coins)")
        else:
            self.episode_coins = config.COINS
            self.n_episode_coins = len(self.episode_coins)
        
        self.memory_return = pd.DataFrame(columns=[t+"_close" for t in self.episode_coins])
        self._episode_ended = False
        self.index = 0
        self.time_delta = pd.Timedelta(5, unit='m')
        self.init_cash = 1000
        self.current_cash = self.init_cash
        self.current_value = self.init_cash
        self._episode_opportunity_variance_sum = 0.0
        self._curriculum_financial_reward_trace = []
        self._curriculum_opportunity_variance_trace = []
        self._curriculum_value_checkpoints = []
        self._curriculum_captured_checkpoint_steps = set()
        self._curriculum_checkpoint_steps = tuple(
            sorted(
                {
                    int(round(float(fraction) * config.EPISODE_LENGTH))
                    for fraction in getattr(
                        config,
                        "BASKET_CURRICULUM_VALUE_CHECKPOINT_FRACTIONS",
                        (0.0,),
                    )
                    if 0.0 <= float(fraction) < 1.0
                }
            )
        )
        self.previous_price = {}
        self.old_dict_coin_price_1 = {}
        self.old_dict_coin_price_2 = {}
        
        # --- NEW: Track Average Entry Price for Position Awareness ---
        # Initialize with zeros (no positions)
        self.avg_entry_price = np.zeros(config.MAX_COINS + 1, dtype=np.float32)

        # A reset starts a new portfolio, so no share count from the preceding
        # episode may participate in the new endowment's weighted cost basis.
        # This is especially important for 8-of-N: a slot can contain a
        # completely different asset (and share-price scale) after reset.
        self.current_stock_num_distribution = np.zeros(
            self.n_episode_coins, dtype=np.float32
        )
        
        # --- FIX: Randomize initial allocation instead of starting with 100% cash ---
        # This forces the agent to learn to manage an active portfolio from step 1
        # and prevents it from getting stuck in a "do nothing" local optimum.
        # We use a Dirichlet distribution to ensure the initial weights sum to 1.
        if self.use_variable_portfolio:
            active_positions = self.n_episode_coins + 1  # +1 for cash
            dirichlet_alpha = np.ones(active_positions)
            random_initial_split = np.random.dirichlet(dirichlet_alpha).astype(np.float32)
            
            self.money_split_ratio = np.zeros(config.MAX_COINS + 1, dtype=np.float32)
            self.money_split_ratio[:active_positions] = random_initial_split
            if self._step_debug:
                print(f"💰 Randomized initial allocation (variable): {[f'{x:.2%}' for x in random_initial_split]}")
        else:
            active_positions = len(self.episode_coins) + 1  # +1 for cash
            dirichlet_alpha = np.ones(active_positions)
            self.money_split_ratio = np.random.dirichlet(dirichlet_alpha).astype(np.float32)
            if self._step_debug:
                print(f"💰 Randomized initial allocation (fixed): {[f'{x:.2%}' for x in self.money_split_ratio]}")

        # ✅ PERFORMANCE FIX: Use cached data instead of reading CSV every reset
        # Filter for episode coins only from the cached dataset
        # NOTE: We keep ALL data for these coins initially to find valid start points
        # including the warmup period
        full_coin_data = self.cached_data[self.cached_data["coin"].isin(self.episode_coins)]
        full_coin_data = full_coin_data.reset_index(drop=True)
        
        self.max_index = full_coin_data.shape[0]

        # Consistent episode slicing: keep the same number of timesteps regardless of
        # active coin count to match the simple MLP behavior and stabilize training.
        # Draw a window long enough to support EPISODE_LENGTH steps for all active coins.
        total_data_rows = config.EPISODE_LENGTH * self.n_episode_coins
        warmup_rows = config.SCALER_WARMUP_STEPS * self.n_episode_coins
        
        # Minimum required data: warmup + episode
        min_required_rows = warmup_rows + total_data_rows
        
        if self.max_index > min_required_rows + 3:
            # --- FIX: Efficiently find start-of-day episode start points ---
            
            # Identify the first entry of each day in a vectorized way
            is_day_start = full_coin_data['date'].dt.date != full_coin_data['date'].dt.date.shift(1)
            day_start_indices = full_coin_data.index[is_day_start].tolist()

            if not day_start_indices and self.max_index > 0:
                # If no day starts are found (e.g., single day data), add the first index.
                day_start_indices.append(0)

            # Filter start indices to ensure there's enough data for warmup AND episode
            # Start point must be at least 'warmup_rows' into the dataset
            valid_start_points = [
                idx for idx in day_start_indices
                if idx >= warmup_rows and (idx + total_data_rows + 2 < self.max_index)
            ]

            if valid_start_points:
                # Randomly select one of the valid start-of-day points
                start_point = np.random.choice(valid_start_points)
                print(f"🌅 Chose start-of-day. Start point: {start_point} (Warmup available)")
            else:
                # Fallback if no valid day start is found with enough history
                print(f"⚠️ No valid start-of-day found with warmup + episode length, using random safe point.")
                max_start_point = self.max_index - total_data_rows - 3
                min_start_point = warmup_rows
                
                if max_start_point > min_start_point:
                    start_timestamp_idx = np.random.randint(min_start_point // self.n_episode_coins, max_start_point // self.n_episode_coins)
                    start_point = start_timestamp_idx * self.n_episode_coins
                else:
                    # Desperate fallback - ignore warmup if dataset too small                    
                    start_point = 0
                    print(f"⚠️ Dataset too small for warmup. Starting at 0.")
        else:
            # If dataset is too small, start from the beginning
            start_point = 0
            print(f"⚠️ Dataset too small. Starting at 0.")
        
        self.episode_start_row = int(start_point)

        # =========================================================================
        # ✅ ROLLING WINDOW SCALER: Fixed-size sliding window replaces unbounded partial_fit.
        #
        # Design intent: the model learns z-scores from RECENT market regime data
        # (warmup window + current episode), not from the full historical dataset.
        # A fixed-size window (SCALER_WINDOW_STEPS) ensures that observations written
        # into the replay buffer from different steps and episodes share a consistent
        # z-score basis — the scaler always reflects the same volume of recent history
        # regardless of how far into an episode the step was taken.
        #
        # With the old unbounded partial_fit: step-5 observations were scaled with
        # ~20 samples; step-300 observations with ~315 samples. The same raw feature
        # mapped to wildly different z-scores, making the critic's Q-surface noisy.
        # The rolling window eliminates that within-episode drift while still capturing
        # the current intraday volatility regime without full-dataset statistics.
        # =========================================================================

        _window_steps = int(getattr(config, 'SCALER_WINDOW_STEPS', 60))
        _context_window_rows = _window_steps * config.MAX_COINS
        self._reset_feature_window(_context_window_rows)

        # Define the bounded history used by both the offline-feature scaler and
        # the selected-basket feature constructor. Dynamic relationships must be
        # calculated after the episode coins are known; loading same-named legacy
        # CSV columns here would silently reintroduce the full-universe shortcut.
        warmup_start = max(0, start_point - warmup_rows)
        end_point = start_point + total_data_rows
        feature_window_data = full_coin_data.iloc[
            warmup_start:end_point + 2
        ].copy()
        feature_window_data = _attach_episode_relative_features(
            feature_window_data, self.episode_coins
        )
        episode_offset = start_point - warmup_start

        # Slice warmup data
        warmup_data_slice = feature_window_data.iloc[:episode_offset]

        if config.LOCAL_CAUSAL_INPUTS_ENABLED:
            print("Primary inputs: fixed asset-local transforms; no episode scaler fitting.")
        elif len(warmup_data_slice) > 0:
            scaler_data = warmup_data_slice[config.SCOLS].to_numpy(dtype=np.float32, copy=False)
            scaler_valid = ~warmup_data_slice[
                "_corporate_action_primary_mask"
            ].to_numpy(dtype=bool, copy=False)
            for row, is_valid in zip(scaler_data, scaler_valid):
                if is_valid:
                    self._append_feature_row(row)
            print(f"📊 Rolling Window Scaler: Pre-loaded {int(scaler_valid.sum())} "
                  f"valid warmup rows (primary capacity: {_context_window_rows} "
                  f"selected-basket rows = {_window_steps} steps × MAX_COINS)")
        else:
            # Fallback: seed the window with the first few rows of episode data.
            peek_ahead = min(100, len(feature_window_data) - episode_offset)
            fallback_slice = feature_window_data.iloc[
                episode_offset:episode_offset + peek_ahead
            ]
            scaler_data = fallback_slice[config.SCOLS].to_numpy(dtype=np.float32, copy=False)
            scaler_valid = ~fallback_slice[
                "_corporate_action_primary_mask"
            ].to_numpy(dtype=bool, copy=False)
            for row, is_valid in zip(scaler_data, scaler_valid):
                if is_valid:
                    self._append_feature_row(row)
            print(
                "⚠️ No warmup data. Rolling window seeded with "
                f"{int(scaler_valid.sum())} valid rows from episode start."
            )

        # =========================================================================
        # Prepare Episode Data
        # =========================================================================
        
        # Slice the actual episode data (future from start_point)
        self.df = feature_window_data.iloc[
            episode_offset:episode_offset + total_data_rows + 2
        ].reset_index(drop=True)
        
        # ✅ TIMESTEP FIX: Build per-timestamp slices once and reuse them every step.
        self._build_episode_time_cache()
        if len(self.unique_times) == 0:
            raise ValueError("No timestamps available in episode data slice")

        # The temporal FAM window is episode-local and right-aligned.  It is
        # reset before the first decision so no previous basket or later row can
        # enter the new episode's causal history.
        self._reset_fam_history()
        self._prime_local_fam_history(warmup_data_slice)
        if self.global_pool_enabled and config.EARNMORE_POOL_LEVEL_ENABLED:
            self._prepare_global_pool_targets(feature_window_data['date'].min(), self.unique_times[-1])

        # Initialize time pointers and current slice
        self._set_time_index(0)
        self.init_time = self.current_time

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()

        self.previous_value = self.current_value
        self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
        
        # Update money split ratio based on calculated distributions
        if self.use_variable_portfolio:
            for i, coin in enumerate(self.episode_coins):
                self.money_split_ratio[i+1] = self.current_stock_money_distribution[i+1] / self.current_value if self.current_value > 0 else 0
            self.money_split_ratio[0] = self.current_cash / self.current_value if self.current_value > 0 else 1
        else:
            self.money_split_ratio = self.normalize_money_dist()
        
        self.step_reward = 0
        self._structured_reward_diagnostics = {}
        
        # Initialize actual allocation tracking (NEW - for ROOT CAUSE #1 fix)
        self.actual_money_split_ratio = self.money_split_ratio.copy() if hasattr(self, 'money_split_ratio') else None
        
        # ✅ CRITICAL FIX: The "previous" allocation for the *first* step must be 100% cash.
        # This correctly models the initial transaction cost of moving from cash into the first
        # random portfolio. Setting it to the drifted allocation gives the agent a "free" first
        # move with no transaction cost, which was causing the initial conservative behavior.
        self.previous_money_split_ratio = np.zeros_like(self.money_split_ratio)
        if len(self.previous_money_split_ratio) > 0:
            self.previous_money_split_ratio[0] = 1.0

        self.total_transaction_costs = 0.0
        self.portfolio_value_history = [self.current_value]
        self._reward_history_steps = 0
        self.long_term_bonus = 0.0
        self.allocation_history = []
        self.price_history = []
        self.return_history = deque(maxlen=config.LONG_TERM_LOOKBACK)
        
        # Initialize previous_price with initial prices for per-asset reward calculation
        initial_prices = {
            coin: float(price)
            for coin, price in zip(self._current_step_coins, self._current_step_close_prices)
        }
        for coin, price in initial_prices.items():
            self.previous_price[coin] = price

        if self.use_variable_portfolio:
            (
                observation,
                mask,
                basket_context,
                basket_membership,
                local_feature_history,
                episode_feature_history,
                feature_history_mask,
            ) = (
                self._get_variable_observation_fast()
            )
            self.current_mask = mask.copy()  # Store the initial mask
            asset_ids = self._get_asset_id_vector()
            
            # ✅ FIX: The initial 'prev_action' must match the actual randomized initial portfolio state.
            # Hardcoding it to "all cash" creates a mismatch that forces the agent to be
            # overly conservative for the first ~30 steps until its own action history populates the state.
            prev_action = self.money_split_ratio.copy()

            # The previous mask before the episode starts is all zeros (no coins active).
            prev_mask = np.zeros_like(mask)

            info = {
                "state": "state",
                "money_split": self.money_split_ratio,
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "episode_coins": self.episode_coins,
                "mask": mask,
                "n_active_coins": self.n_episode_coins,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0),
                "basket_context": basket_context.copy(),
                "basket_context_norm": float(np.linalg.norm(basket_context)),
                "basket_context_abs_mean": float(np.mean(np.abs(basket_context))),
                "basket_membership": basket_membership.copy(),
                "basket_membership_hamming_weight": float(basket_membership.sum()),
            }
            obs_dict = {
                "observations": observation,
                "asset_ids": asset_ids,
                "mask": mask,
                "prev_action": prev_action,
                "prev_mask": prev_mask,
                "basket_context": basket_context,
                "basket_membership": basket_membership,
                "local_feature_history": local_feature_history,
                "episode_feature_history": episode_feature_history,
                "feature_history_mask": feature_history_mask,
            }
        else:
            # Get observations (uses causal scaling)
            scaled_output = self.get_observations()
            observation = scaled_output[config.OBS_COLS].values.flatten()
            info = {
                "state": "state",
                "money_split": self.money_split_ratio,
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "scaled_output": scaled_output,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0)
            }
            obs_dict = observation

        # Present in both training and evaluation info.  This does not alter
        # reward, termination, basket choice, or observation contents.
        info.update(self.episode_sampling_info())
        info.update(self._episode_opportunity_diagnostics())
        obs_dict, info = self._publish_global_pool(obs_dict, info)
        self._capture_curriculum_value_checkpoint(obs_dict)
        
        # Fixed episode termination logic
        self._episode_ended = True if self.index == config.EPISODE_LENGTH else False
        
        return obs_dict, info

    def episode_sampling_info(self):
        """Actual market window, including terminal info before VecEnv auto-reset."""
        return {"dataset_file": os.path.abspath(config.FILE),
                "episode_start_time": self.init_time.isoformat(),
                "episode_end_time": self.current_time.isoformat(),
                "episode_start_row": self.episode_start_row}

    def step(self, action):
        """Execute one step in the environment"""
        
        if self._episode_ended:
            # If episode ended, reset environment
            return self.reset()
        
        # 🎯 ESSENTIAL: Show agent's action (raw [-1,1] from policy)
        if self._step_debug:
            print(f"🎯 Agent Action (raw [-1,1]): {action}")
        
        relabel_decision_time = self.current_time

        # Store previous allocation for transaction cost calculation
        if hasattr(self, 'money_split_ratio'):
            self.previous_money_split_ratio = self.money_split_ratio.copy()
        
        # ✅ SINGLE-STAGE MASKED L1 NORMALIZATION: Direct normalization over active dimensions only
        # This eliminates the double normalization problem that was inflating constraint costs
        action_logits = self.compact_policy_action(action)
        if self._step_debug:
            print(f"🔄 Action logits: {action_logits}")
        
        if self.action_normalization == "flowrl_joint_simplex_v1":
            active_count = self.n_episode_coins + 1 if self.use_variable_portfolio else len(action_logits)
            shifted = (np.clip(action_logits[:active_count], -1.0, 1.0) + 1.0) * 0.5
            total = shifted.sum()
            weights = shifted / total if total > 1e-8 else np.full(active_count, 1.0 / active_count)
            self.money_split_ratio = np.zeros(config.MAX_COINS + 1 if self.use_variable_portfolio else active_count, dtype=np.float32)
            self.money_split_ratio[:active_count] = weights
        elif self.use_variable_portfolio:
            # ✅ FIX: Hierarchical Normalization (Cash vs. Assets)
            # This solves the "1 vs N" dimensionality bias that prevented high cash positions.
            
            # 1. Separate Cash and Asset logits
            # Action layout: [Cash, Asset1, Asset2, ..., AssetN]
            active_action_logits = action_logits[:self.n_episode_coins + 1]
            
            cash_logit = active_action_logits[0]
            asset_logits = active_action_logits[1:]

            # 2. Tier 1: Determine Cash Weight (Independent Decision)
            # Maps [-1, 1] -> [0, 0.5]. Zero-logit → 25% cash (was 50%).
            # Confirmed fix: zero-point at 50% cash + equal asset weighting
            # neutralizes inverse pairs (TQQQ/SQQQ, USO/SCO) to ~0 financial return.
            # Halving the default cash gives 2× equity exposure and 2× alpha/beta
            # reward gradient at the zero-init equilibrium.
            cash_weight = (cash_logit + 1.0) * 0.5
            
            # 3. Tier 2: Determine Asset Allocation (Relative Distribution)
            asset_shifted = (asset_logits + 1.0) * 0.5
            asset_sum = np.sum(asset_shifted)
            
            eps = 1e-8
            if asset_sum > eps:
                # Standard L1 normalization for assets
                asset_weights_relative = asset_shifted / asset_sum
            else:
                # Fallback to uniform if all assets are -1
                asset_weights_relative = np.ones(self.n_episode_coins) / self.n_episode_coins
            
            # 4. Combine: Scale asset weights by the remaining equity allocation
            equity_allocation = 1.0 - cash_weight
            final_asset_weights = asset_weights_relative * equity_allocation
            
            # 5. Create final normalized vector
            normalized_active_action = np.concatenate(([cash_weight], final_asset_weights))
            
            # 6. Create target allocation with proper padding
            self.money_split_ratio = np.zeros(
                config.MAX_COINS + 1, dtype=np.float32
            )
            self.money_split_ratio[:self.n_episode_coins + 1] = normalized_active_action
            
            if self._step_debug:
                print(f"✅ Hierarchical Norm: Cash={cash_weight:.2%}, Equity={equity_allocation:.2%}")
            
        else:
            # Fixed portfolio: Hierarchical Normalization
            cash_logit = action_logits[0]
            asset_logits = action_logits[1:]

            # Tier 1: Cash — zero-point lowered to 25% (was 50%)
            cash_weight = (cash_logit + 1.0) * 0.5
            
            # Tier 2: Assets
            asset_shifted = (asset_logits + 1.0) * 0.5
            asset_sum = np.sum(asset_shifted)
            eps = 1e-8
            
            if asset_sum > eps:
                asset_weights_relative = asset_shifted / asset_sum
            else:
                asset_weights_relative = np.ones_like(asset_logits) / len(asset_logits)
                
            # Combine
            equity_allocation = 1.0 - cash_weight
            final_asset_weights = asset_weights_relative * equity_allocation
            
            self.money_split_ratio = np.concatenate(([cash_weight], final_asset_weights))

        # 📊 ESSENTIAL: Show target allocation
        if self._step_debug:
            if self.use_variable_portfolio:
                print(f"🪙 Active Coins ({self.n_episode_coins}): {self.episode_coins}")
                active_count = self.n_episode_coins + 1
            else:
                active_count = len(self.episode_coins) + 1
            alloc_parts = [f"\033[95m'{x:.3f}'\033[0m" if i < active_count else f"'{x:.3f}'" for i, x in enumerate(self.money_split_ratio)]
            alloc_str = f"[{', '.join(alloc_parts)}]"
            print(f"📊 Target Allocation: {alloc_str}")
            print(f"💰 Before share calc - Current value: ${self.current_value:.2f}")
            print(f"💰 Money split ratio: {self.money_split_ratio}")
        
        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()

        # The denominator is determined by the selected basket's causal market
        # scale at the decision point, never by the policy's chosen exposure.
        decision_opportunity_volatility = max(
            0.0,
            float(getattr(self, "_current_step_opportunity_volatility", 0.0)),
        )
        self._episode_opportunity_variance_sum += (
            decision_opportunity_volatility ** 2
        )

        # ✅ DEBUG: Show shares calculated
        # print(f"📈 Shares calculated: {self.current_stock_num_distribution}")
        
        self.step_time()
        self.index += 1
        component_outcome_info = self._build_component_outcome_info()
        if self._curriculum_teacher_active():
            self._curriculum_financial_reward_trace.append(float(self.step_reward))
            self._curriculum_opportunity_variance_trace.append(
                float(decision_opportunity_volatility ** 2)
            )
        
        # ✅ DEBUG: Show value after time step
        if self._step_debug:
            print(f"💰 After time step - New value: ${self.current_value:.2f}, Change: ${self.current_value - self.previous_value:.2f}")

        if self.use_variable_portfolio:
            previous_mask_for_this_step = self.current_mask.copy()  # Get mask from the previous state
            (
                observation,
                mask,
                basket_context,
                basket_membership,
                local_feature_history,
                episode_feature_history,
                feature_history_mask,
            ) = (
                self._get_variable_observation_fast()
            )
            self.current_mask = mask.copy()  # Store the new mask for the next step
            asset_ids = self._get_asset_id_vector()

            info = {
                "state": "state",
                "money_split": self.money_split_ratio,  # Now represents actual allocation after price movements
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "episode_coins": self.episode_coins,
                "mask": mask,
                "n_active_coins": self.n_episode_coins,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0),
                "basket_context": basket_context.copy(),
                "basket_context_norm": float(np.linalg.norm(basket_context)),
                "basket_context_abs_mean": float(np.mean(np.abs(basket_context))),
                "basket_membership": basket_membership.copy(),
                "basket_membership_hamming_weight": float(basket_membership.sum()),
            }
            info.update(component_outcome_info)
            if self._cross_basket_relabel_active():
                info.update(self._cross_basket_relabel_info(relabel_decision_time))
            obs_dict = {
                "observations": observation,
                "asset_ids": asset_ids,
                "mask": mask,
                "prev_action": np.asarray(
                    self.last_target_allocation, dtype=np.float32
                ).copy(),
                "prev_mask": previous_mask_for_this_step,
                "basket_context": basket_context,
                "basket_membership": basket_membership,
                "local_feature_history": local_feature_history,
                "episode_feature_history": episode_feature_history,
                "feature_history_mask": feature_history_mask,
            }
        else:
            # Get observations
            scaled_output = self.get_observations()
            observation = scaled_output[config.OBS_COLS].values.flatten()
            info = {
                "state": "state",
                "money_split": self.money_split_ratio,  # Now represents actual allocation after price movements
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "scaled_output": scaled_output,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0)
            }
            obs_dict = observation

        info.update(self.episode_sampling_info())
        info.update(self._structured_reward_diagnostics)
        info.update(self._episode_opportunity_diagnostics())
        obs_dict, info = self._publish_global_pool(obs_dict, info)
        self._capture_curriculum_value_checkpoint(obs_dict)

        reward = info["reward"]
        
        # 💰 ESSENTIAL: Show results
        if self._step_debug:
            if self.use_variable_portfolio:
                print(f"🪙 Active Coins ({self.n_episode_coins}): {self.episode_coins}")
                active_count = self.n_episode_coins + 1
            else:
                active_count = len(self.episode_coins) + 1
            actual_alloc_parts = [f"\033[95m'{x:.3f}'\033[0m" if i < active_count else f"'{x:.3f}'" for i, x in enumerate(self.money_split_ratio)]
            actual_alloc_str = f"[{', '.join(actual_alloc_parts)}]"
            print(f"💰 Portfolio: ${self.current_value:.2f} | Reward: {reward:.4f} | Actual: {actual_alloc_str}")
        
        # Check if episode is done using constant episode length (no division by n_coins)
        time_steps_per_episode = config.EPISODE_LENGTH
        terminated = self.index >= time_steps_per_episode
        truncated = False  # Add truncation logic if needed
        
        if terminated:
            episode_return = (self.current_value - config.INITIAL_CASH) / config.INITIAL_CASH

            # Dense financial rewards must keep the final transition's actual
            # consequence. Replacing it with zero would break the Bellman data
            # contract precisely at the episode boundary.
            if self.reward_type in ("TRANSACTION_COST", "STRUCTURED_CREDIT"):
                reward = self.step_reward
                info["reward"] = reward
            else:
                reward = 0  # Keep existing behavior for other reward modes
            
            if self._step_debug and self.reward_type == "TRANSACTION_COST":
                total_costs_pct = (getattr(self, 'total_transaction_costs', 0.0) / config.INITIAL_CASH) * 100
                print(f"🏁 Episode Complete: {self.index} steps, Final Value: ${self.current_value:.2f}")
                print(f"📊 Episode Return: {episode_return:.2%}, Total Transaction Costs: ${getattr(self, 'total_transaction_costs', 0.0):.2f} ({total_costs_pct:.2f}%)")
                print(f"💰 Net Episode Performance: {episode_return:.2%} - {total_costs_pct:.2f}% = {episode_return - total_costs_pct/100:.2%}")
            elif self._step_debug:
                print(f"🏁 Episode Complete: {self.index} steps, Final Value: ${self.current_value:.2f}")
                print(f"📊 Episode Return: {episode_return:.2%}")

            # A curriculum teacher observes completed training episodes only.
            # Evaluation environments have no sampler, so their resets and RNG
            # remain exactly on the historical path.
            info.update(self._episode_opportunity_diagnostics())
            info["episode_financial_return"] = float(episode_return)
            if self.basket_sampler is not None and hasattr(
                self.basket_sampler, "observe_episode"
            ):
                self.basket_sampler.observe_episode(
                    selection=dict(self._active_curriculum_selection or {}),
                    financial_return=float(episode_return),
                    financial_log_growth=float(
                        info["episode_financial_log_growth"]
                    ),
                    opportunity_risk_budget=float(
                        info["opportunity_risk_budget"]
                    ),
                    opportunity_normalized_return=float(
                        info["opportunity_normalized_financial_return"]
                    ),
                    value_samples=self._finalize_curriculum_value_samples(),
                )
            
        try:
            return obs_dict, reward, terminated, truncated, info
        except Exception as e:
            print("ERRORRRRRR!!!!!!!!!!!!!!!!")
            print("Observation:", obs_dict)
            print("Reward:", reward)
            print("Step reward, current value, previous value:", self.step_reward, self.current_value, self.previous_value)
            print("Current stock money distribution:", self.current_stock_money_distribution)
            print("Current stock num distribution:", self.current_stock_num_distribution)
            print("Action:", action)
            print("Index:", self.index)
            print("Dfslice:", self.dfslice)
            print("Current time:", self.current_time)
            print("Money split ratio:", self.money_split_ratio)
            print("Episode coins:", self.episode_coins)
            print("Exception:", e)
            
            raise ValueError(f"Error in step function: {e}")

    def _create_variable_observation(self, scaled_output):
        """Create padded observation and mask for variable portfolio."""
        # Create mask: 1 for active coins, 0 for inactive
        mask = np.zeros(config.MAX_COINS, dtype=np.float32)
        mask[:self.n_episode_coins] = 1.0
        
        # Create padded observation
        observation = np.zeros(config.MAX_OBSERVATION_DIM, dtype=np.float32)
        
        if not scaled_output.empty:
            # Fill in features for active coins
            for i, coin in enumerate(self.episode_coins):
                start_idx = i * config.FEATURES_PER_COIN
                end_idx = (i + 1) * config.FEATURES_PER_COIN
                
                coin_features = []
                # Use the exact per-token layout produced by get_observations:
                # selected-basket-normalized primary market features, cost
                # basis, fixed-reference behavioral scale, then selected-basket
                # episode-relative market context. The policy jointly encodes
                # all market scopes while cost basis remains position context.
                # We need to construct the list of expected columns for this coin
                expected_cols = (
                    [f"{coin}_{scol}" for scol in config.SCOLS]
                    + [f"{coin}_cost_basis"]
                    + [
                        f"{coin}_{feature_name}"
                        for feature_name in config.BEHAVIORAL_SCALE_FEATURES
                    ]
                    + [
                        f"{coin}_{feature_name}"
                        for feature_name in config.EPISODE_RELATIVE_FEATURES
                    ]
                )
                
                for col_name in expected_cols:
                    if col_name in scaled_output.columns:
                        coin_features.extend(scaled_output[col_name].values)
                    else:
                        coin_features.extend([0.0])  # Pad missing features
                
                # Ensure we have exactly FEATURES_PER_COIN features
                coin_features = coin_features[:config.FEATURES_PER_COIN]
                while len(coin_features) < config.FEATURES_PER_COIN:
                    coin_features.append(0.0)
                
                observation[start_idx:end_idx] = coin_features
        
        return observation, mask

    def step_time(self):
        """Advance time and update portfolio values"""
        # Factored credit labels are rebuilt for this transition only.
        self._reset_component_contribution_buffers()

        # The action was executed against this open in
        # calculate_actual_shares_from_money_split().  Preserve it before the
        # environment advances so local credit covers exactly the same causal
        # interval as the scalar portfolio reward.
        transition_execution_prices = (
            self.dfslice[["coin", "open"]]
            .set_index("coin")
            .to_dict()
            .get("open", {})
        )

        # ✅ TIMESTEP FIX: Advance by the next unique timestamp
        if hasattr(self, 'unique_times') and hasattr(self, 'time_index') and hasattr(self, '_time_slices'):
            self._set_time_index(self.time_index + 1)
        else:
            # Fallback to previous behavior if unique_times not initialized
            self.current_time += self.time_delta

        self.previous_value = self.current_value
        
        # Recalculate portfolio value based on new prices and existing share distribution
        self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
        
        # ✅ CORRECTED LOGIC: Calculate the new money split ratio *after* price changes,
        # but before calculating costs. This is the new "actual" allocation.
        new_money_split_ratio = self.normalize_money_dist()

        # Calculate transaction costs for Net Return (Profit minus Costs)
        # This calculation is now universal for all reward types to ensure the agent
        # perceives the friction of trading, which is critical for realistic behavior.
        transaction_cost = 0.0
        volatility_penalty = 0.0 # Initialize volatility penalty
        
        if ((self.reward_type == "TRANSACTION_COST" or self.reward_type == "STRUCTURED_CREDIT") and 
            hasattr(self, 'previous_money_split_ratio') and 
            self.previous_money_split_ratio is not None):
            
            # The agent's intended action is in self.money_split_ratio. The actual
            # allocation from the *previous* step is self.previous_money_split_ratio.
            # The change is the difference between the agent's new target and what was
            # actually there before.
            allocation_changes = np.sum(np.abs(np.array(self.money_split_ratio) - np.array(self.previous_money_split_ratio)))
            
            traded_volume_fraction = allocation_changes / 2.0
            
            # ========================================================================
            # CRITICAL FIX: Apply minimum trade threshold to match live trading
            # 
            # In live trading, we don't execute tiny trades (< MIN_TRADE_THRESHOLD).
            # Training must learn this: small allocation changes are ignored, so
            # the model should either HOLD or make MEANINGFUL moves.
            #
            # This prevents the "death by a thousand cuts" problem where the model
            # learns to oscillate with tiny 0.1% changes that incur costs in live.
            # ========================================================================
            min_trade_threshold = getattr(config, 'MIN_TRADE_THRESHOLD', 0.02)  # Default 2%
            
            if traded_volume_fraction > min_trade_threshold:
                # Meaningful trade: full transaction costs apply
                transaction_cost = traded_volume_fraction * self.previous_value * config.TRANSACTION_COST
                volatility_penalty = config.VOLATILITY_PENALTY_WEIGHT * (traded_volume_fraction ** 2)
            else:
                # Trade too small to execute in live: no costs, but also model learns nothing happens
                transaction_cost = 0.0
                volatility_penalty = 0.0
            
            # Debug output for transaction costs
            if self._step_debug and traded_volume_fraction > 0.001:
                cost_as_pct_of_value = (transaction_cost / self.previous_value * 100) if self.previous_value > 1e-9 else 0
                trade_status = "✅ EXECUTED" if traded_volume_fraction > min_trade_threshold else "⏸️ IGNORED (below threshold)"
                print(
                    f"💸 Transaction: Traded {traded_volume_fraction:.2%} of portfolio. "
                    f"Cost: ${transaction_cost:.2f} ({cost_as_pct_of_value:.4f}%) {trade_status}"
                )
        
        # ✅ CRITICAL FIX: The new state of the environment is the new_money_split_ratio
        # which reflects the allocation after price drift. This becomes the basis for the next action.
        self.money_split_ratio = new_money_split_ratio
        
        # ✅ LONG-TERM PERFORMANCE TRACKING: Update portfolio value history
        self.portfolio_value_history.append(self.current_value)
        self._reward_history_steps += 1
        
        # Keep only the required history length for efficiency
        max_history_length = config.LONG_TERM_LOOKBACK + 1
        if len(self.portfolio_value_history) > max_history_length:
            self.portfolio_value_history = self.portfolio_value_history[-max_history_length:]
        
        # Calculate long-term performance bonus: r_t' = r_t + λ_long * 1/N * (V_t - V_{t-N}) / V_{t-N}
        self.long_term_bonus = 0.0
        long_term_return = 0.0
        if (config.LONG_TERM_BONUS_ENABLED and 
            len(self.portfolio_value_history) > config.LONG_TERM_LOOKBACK):
            
            # Get portfolio value N steps ago
            v_t_minus_n = self.portfolio_value_history[-(config.LONG_TERM_LOOKBACK + 1)]
            v_t = self.current_value
            
            # Calculate long-term bonus: λ_long * (1/N) * (V_t - V_{t-N}) / V_{t-N}
            if v_t_minus_n > 1e-9:  # Avoid division by zero
                long_term_return = (v_t - v_t_minus_n) / v_t_minus_n
                self.long_term_bonus =  config.LONG_TERM_LAMBDA * long_term_return #* (1.0 / config.LONG_TERM_LOOKBACK) 
                
                # Debug output for long-term bonus
                if self._step_debug and abs(self.long_term_bonus) > 0.001:  # Only log significant bonuses
                    print(f"📈 Long-term Bonus: λ={config.LONG_TERM_LAMBDA:.2f} × "
                          f"({v_t:.2f} - {v_t_minus_n:.2f})/{v_t_minus_n:.2f} = "
                          f"{self.long_term_bonus:.4f}")

        # ✅ REFACTOR: Calculate and store step return history REGARDLESS of reward type.
        # This is crucial for the Sortino scaling calculation.
        if self.previous_value > 1e-9:
            step_return = (self.current_value - self.previous_value) / self.previous_value
        else:
            step_return = 0.0
        self.return_history.append(step_return)

        # --- NEW: Calculate Downside Risk for Reward Scaling ---
        downside_returns = [r for r in self.return_history if r < 0]
        if len(downside_returns) > 1:
            downside_deviation = np.std(downside_returns)
        else:
            downside_deviation = 0.0
        
        # ✅ RISK-BUDGET FIX: Only penalize downside risk above a target level.
        # Without a target (dead-band), the agent is continuously incentivized to drive
        # downside risk toward zero, which creates a strong attractor at ~100% cash.
        downside_target = getattr(config, "DOWNSIDE_DEVIATION_TARGET", 0.0)
        downside_deviation_excess = max(0.0, downside_deviation - downside_target)
        
        # --- NEW: Calculate Activity Bonus ---
        activity_bonus = 0.0
        if config.ACTIVITY_BONUS_ENABLED and hasattr(self, 'last_target_allocation') and hasattr(self, 'previous_money_split_ratio'):
            # L1 norm of the change between target allocation and previous actual allocation
            turnover = np.sum(np.abs(self.last_target_allocation - self.previous_money_split_ratio))
            # Squash with tanh to prevent reward explosion, scale with a weight
            activity_bonus = config.ACTIVITY_BONUS_WEIGHT * np.tanh(config.ACTIVITY_SCALING_FACTOR * turnover)

        # --- NEW: Store history for structured rewards ---
        if self.reward_type == "STRUCTURED_CREDIT":
            # The allocation to store is the one the agent decided on for this step
            self.allocation_history.append(self.last_target_allocation)
            if len(self.allocation_history) > config.LONG_TERM_LOOKBACK:
                self.allocation_history.pop(0)

            # Store current prices for the lookback calculation
            # ✅ TRAIN-LIVE ALIGNMENT: Use 'open' to match live execution reality
            # Live trading executes at open of completed bars (most recent trade price)
            current_prices = self.dfslice[["coin", "open"]].set_index("coin").to_dict().get("open", {})
            self.price_history.append(current_prices)
            if len(self.price_history) > config.LONG_TERM_LOOKBACK:
                self.price_history.pop(0)

        # Dense realized reward for the action just taken:
        # a_{t-1} * ((p_t / p_{t-1}) - 1)
        # Use the executed target allocation for this step, not the lookback average.
        dense_position_return = step_return
        if (
            self.reward_type == "STRUCTURED_CREDIT"
            and hasattr(self, "last_target_allocation")
            and len(self.price_history) >= 2
        ):
            prev_step_prices = self.price_history[-2]
            curr_step_prices = self.price_history[-1]
            dense_position_return = 0.0

            for i, coin in enumerate(self.episode_coins):
                asset_idx = i + 1  # 0 is cash
                prev_price = prev_step_prices.get(coin)
                curr_price = curr_step_prices.get(coin)

                if prev_price is None or curr_price is None or prev_price <= 1e-9:
                    continue

                asset_step_return = (curr_price / prev_price) - 1.0
                dense_position_return += self.last_target_allocation[asset_idx] * asset_step_return
        

        # ✅ REWARD CALCULATION: Choose between Simple Return or Net Return (Profit minus Costs)
        if self.reward_type == "STRUCTURED_CREDIT":
            financial_growth_reward = _scaled_log_portfolio_growth(
                self.previous_value,
                self.current_value,
                scale=float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0)),
            )
            # Restore the original structured shaping as the replay reward.
            # Keep the legacy variable name for historical comparisons. Its
            # shaping sum intentionally excludes transaction-cost deductions.
            legacy_structured_diagnostic_reward = 0.0
            
            # 💡 FIX: Use a dense reward for the initial "cold start" period.
            # Giving zero reward for the first N steps is too sparse and can kill learning.
            if len(self.price_history) < config.LONG_TERM_LOOKBACK:
                # Use the historical dense cold-start quantity until the
                # allocation/price lookback is ready for structured shaping.
                if self.previous_value > 1e-9:
                    raw_reward = dense_position_return + self.long_term_bonus # Include bonus if active
                    legacy_structured_diagnostic_reward = float(
                        np.clip(raw_reward, -5.5, 5.5)
                    )
                self.step_reward = financial_growth_reward
            else:
                # Once history is full, switch to the structured reward
# ------------------------------------------------------------                
                # 1. Get prices N steps ago and current prices
                prices_then = self.price_history[0] 
                prices_now = self.price_history[-1]

                # # 2. Get average allocation over the window
                # allocation_window = np.array(self.allocation_history)
                # avg_allocations = np.mean(allocation_window, axis=0)

                # 2. Use an exponentially weighted average so recent decisions matter
                # ✅ GAMMA 0.966 CALIBRATION: Reduced decay from 5 to 3 to give more weight to older allocations.
                # With gamma=0.966, the agent must feel the consequences of decisions over a longer trajectory.
                # A gentler decay (3 instead of 5) means the oldest allocation in a 15-step window gets ~5%
                # weight instead of ~0.7%, allowing the agent to see the full trend rather than just recent noise.
                allocation_window = np.array(self.allocation_history)  # shape: [W, MAX_COINS+1]
                w = np.linspace(0, 1, allocation_window.shape[0])  # simple linear ramp
                w = np.exp(5 * (w - 1))                             # decay toward the past (reduced from 5 to 3)
                w = w / (w.sum() + 1e-9)
                avg_allocations = (allocation_window * w[:, None]).sum(axis=0)                

                # # 3. Calculate per-asset rewards
                # total_structured_reward = 0.0
                # risky_asset_returns = []

                # 3. Calculate market-relative per-asset rewards
                total_structured_reward = 0.0
                risky_asset_returns = []  # for cross-sectional mean                
                
                for i, coin in enumerate(self.episode_coins):
                    asset_idx = i + 1  # 0 is cash
                    
                    price_then = prices_then.get(coin)
                    price_now = prices_now.get(coin)

                    if price_then is not None and price_now is not None and price_then > 1e-9:
                        asset_return = (price_now - price_then) / price_then
                        risky_asset_returns.append(asset_return)     

                # ✅ MOMENTUM DERIVATIVE CALCULATION: Calculate momentum derivatives for all assets
                # This creates teleonomy by providing timing guidance (when to enter/exit)
                # while preserving Q-learning stability through the base point-to-point return signal
                asset_momentum_derivatives = {}  # Store momentum derivatives for each asset
                
                for i, coin in enumerate(self.episode_coins):
                    momentum_derivative = 0.0
                    
                    if len(self.price_history) >= config.MOMENTUM_WINDOW * 2:
                        # Get prices for recent window (most recent MOMENTUM_WINDOW steps)
                        recent_prices = [self.price_history[-(j+1)].get(coin) for j in range(config.MOMENTUM_WINDOW)]
                        # Get prices for earlier window (previous MOMENTUM_WINDOW steps)
                        earlier_prices = [self.price_history[-(j+config.MOMENTUM_WINDOW+1)].get(coin) 
                                         for j in range(config.MOMENTUM_WINDOW)]
                        
                        if all(p is not None and p > 1e-9 for p in recent_prices + earlier_prices):
                            # Calculate current momentum (average return over recent window)
                            recent_returns = [(recent_prices[j] - recent_prices[j+1]) / recent_prices[j+1] 
                                             for j in range(len(recent_prices) - 1)]
                            current_momentum = np.mean(recent_returns) if recent_returns else 0.0
                            
                            # Calculate previous momentum (average return over earlier window)
                            earlier_returns = [(earlier_prices[j] - earlier_prices[j+1]) / earlier_prices[j+1] 
                                              for j in range(len(earlier_prices) - 1)]
                            previous_momentum = np.mean(earlier_returns) if earlier_returns else 0.0
                            
                            # Momentum derivative = change in momentum (positive = improving, negative = deteriorating)
                            momentum_derivative = current_momentum - previous_momentum
                    
                    asset_momentum_derivatives[coin] = momentum_derivative
                
                # Compute average market momentum derivative (cross-sectional mean)
                momentum_derivatives_list = [md for md in asset_momentum_derivatives.values() if md != 0.0]
                avg_market_momentum_derivative = np.mean(momentum_derivatives_list) if len(momentum_derivatives_list) > 0 else 0.0

                # compute cross-sectional mean over risky assets
                # --- COMPONENT 1: BETA REWARD (Market Exposure / Timing) ---
                # Reward the agent for being exposed to positive market moves
                # This captures the "be invested in bull markets" behavior
                
                # Compute cross-sectional mean (market baseline)
                avg_market_return = np.mean(risky_asset_returns) if len(risky_asset_returns) else 0.0                
                total_risky_allocation = sum(avg_allocations[1:])  # Total non-cash allocation
                
                # ✅ ASYMMETRIC MULTIPLIER FOR BETA: Apply momentum derivative as multiplier
                # Base beta signal (preserves Q-learning stability)
                base_beta_contribution = total_risky_allocation * avg_market_return
                
                # Momentum boost multiplier (adds timing guidance)
                # Positive derivative (momentum accelerating) → multiplier > 1.0 → boost signal
                # Negative derivative (momentum decelerating) → multiplier < 1.0 → dampen signal
                momentum_boost_beta = 1.0 + (avg_market_momentum_derivative * config.MOMENTUM_BOOST_SCALING)
                momentum_boost_beta = np.clip(momentum_boost_beta, config.MOMENTUM_BOOST_MIN, config.MOMENTUM_BOOST_MAX)
                
                # Enhanced beta contribution (base signal × momentum multiplier)
                # Note: Multiplier is always positive (clamped), so:
                # - If base_beta > 0 and boost > 1.0: enhanced_beta becomes more positive (good timing)
                # - If base_beta > 0 and boost < 1.0: enhanced_beta becomes less positive (bad timing)
                # - If base_beta < 0 and boost > 1.0: enhanced_beta becomes more negative (bad - market down, momentum accelerating = worse)
                # - If base_beta < 0 and boost < 1.0: enhanced_beta becomes less negative (good - market down, momentum decelerating = less bad)
                beta_contribution = base_beta_contribution * momentum_boost_beta
                
                scaled_beta = np.tanh(config.STRUCTURED_REWARD_SCALING_FACTOR * beta_contribution)
                beta_reward = config.STRUCTURED_BETA_WEIGHT * scaled_beta
                
                # if abs(beta_reward) > 1e-5:
                #     print(f"📈 Beta Reward: exposure {total_risky_allocation:.2f} × market {avg_market_return:.2%} = "
                #           f"tanh({beta_contribution:.4f}) × {config.STRUCTURED_BETA_WEIGHT:.2f} = {beta_reward:.4f}")

                # --- COMPONENT 2: ALPHA REWARD (Stock Selection / Skill) ---
                # Reward the agent for picking winners within the market
                # This captures the "pick the best performers" behavior
                total_alpha_reward = 0.0
                accumulation_penalty = 0.0 # Initialize accumulation penalty
                profit_taking_bonus_total = 0.0  # Track profit-taking bonuses separately
                stop_loss_subsidy_total = 0.0     # Track stop-loss subsidies separately
                
                for i, coin in enumerate(self.episode_coins):
                    asset_idx = i + 1
                    price_then = prices_then.get(coin)
                    price_now = prices_now.get(coin)
                    local_contribution = 0.0
                    
                    if price_then is None or price_now is None or price_then <= 0: 
                        continue
                    
                    r_i = (price_now - price_then) / price_then
                    
                    # ✅ ALPHA FIX: Absolute return decides winner vs loser.
                    # Relative strength only adds upside when the asset is already profitable.
                    # This prevents rewarding "less-bad losers" and avoids punishing green names
                    # solely because they lagged the cross-sectional average.
                    relative_alpha_i = r_i - avg_market_return
                    base_alpha_i = r_i
                    if (r_i > 0) and (relative_alpha_i > 0):
                        base_alpha_i += config.ALPHA_RELATIVE_BONUS_WEIGHT * relative_alpha_i
                    
                    # Get momentum derivative for this asset
                    momentum_derivative_i = asset_momentum_derivatives.get(coin, 0.0)
                    
                    # Momentum boost multiplier (adds timing guidance)
                    # Positive derivative (momentum accelerating) → multiplier > 1.0 → boost signal
                    # Negative derivative (momentum decelerating) → multiplier < 1.0 → dampen signal
                    momentum_boost_i = 1.0 + (momentum_derivative_i * config.MOMENTUM_BOOST_SCALING)
                    momentum_boost_i = np.clip(momentum_boost_i, config.MOMENTUM_BOOST_MIN, config.MOMENTUM_BOOST_MAX)
                    
                    # Enhanced alpha (base signal × momentum multiplier)
                    # Note: Multiplier is always positive (clamped), so:
                    # - If base_alpha > 0 and boost > 1.0: enhanced_alpha becomes more positive (good - asset outperforming, momentum accelerating)
                    # - If base_alpha > 0 and boost < 1.0: enhanced_alpha becomes less positive (bad - asset outperforming but momentum decelerating)
                    # - If base_alpha < 0 and boost > 1.0: enhanced_alpha becomes more negative (bad - asset underperforming, momentum accelerating = worse)
                    # - If base_alpha < 0 and boost < 1.0: enhanced_alpha becomes less negative (good - asset underperforming but momentum decelerating = less bad)
                    enhanced_alpha_i = base_alpha_i * momentum_boost_i
                    
                    scaled_alpha = np.tanh(config.STRUCTURED_REWARD_SCALING_FACTOR * enhanced_alpha_i)
                    avg_alloc = avg_allocations[asset_idx]
                    reward_asset = avg_alloc * scaled_alpha
                    
                    # Penalize only true losers that are also lagging the tape.
                    # Positive-return assets are no longer treated as losers just for trailing the average.
                    if (r_i < 0) and (relative_alpha_i < 0):
                        directional_penalty = config.IMMEDIATE_DIRECTIONAL_PENALTY_WEIGHT * avg_alloc * abs(relative_alpha_i)
                        reward_asset -= directional_penalty
                    
                    if (r_i < 0) and (momentum_derivative_i < config.MOMENTUM_DERIVATIVE_THRESHOLD):
                        # Extra penalty only for falling assets whose momentum is still deteriorating.
                        momentum_penalty = config.MOMENTUM_DETERIORATION_PENALTY_WEIGHT * avg_alloc * abs(momentum_derivative_i)
                        reward_asset -= momentum_penalty
                
                    total_alpha_reward += reward_asset
                    local_contribution += (
                        config.STRUCTURED_ALPHA_WEIGHT * reward_asset
                    )

                    # Dense one-step financial piece for the same asset so local
                    # utilities also see a_i * r_i composition, not only shaped alpha.
                    if (
                        hasattr(self, "last_target_allocation")
                        and len(self.price_history) >= 2
                    ):
                        prev_step_price = self.price_history[-2].get(coin)
                        curr_step_price = self.price_history[-1].get(coin)
                        if (
                            prev_step_price is not None
                            and curr_step_price is not None
                            and prev_step_price > 1e-9
                        ):
                            dense_i = self.last_target_allocation[asset_idx] * (
                                (curr_step_price / prev_step_price) - 1.0
                            )
                            local_contribution += dense_i
                    
                    # --- Cost Basis Rewards: Momentum-Driven Accumulation & Position Management ---
                    # ✅ SIMPLIFIED: No hard profit barriers. Agent learns optimal position sizing
                    # based on momentum trends. Only penalize accumulation when momentum derivative
                    # is negative (momentum is deteriorating).
                    #
                    # 1. Accumulation Penalty: Only when momentum derivative < 0 (momentum decreasing)
                    # 2. Profit Taking Bonus: Reward reducing positions above cost basis ("Locking Gains")
                    # 3. Stop Loss Subsidy: Reward cutting losing positions decisively ("Discipline")
                    #
                    # ✅ INTRADAY FIX: Normalize excess_ratio by LONG_TERM_LOOKBACK to match
                    # the time scale of alpha/beta rewards. This prevents cost-basis rewards
                    # from dominating when positions accumulate larger unrealized gains/losses
                    # over the holding period compared to the 5-step lookback window returns.
                    
                    if hasattr(self, 'previous_money_split_ratio') and self.previous_money_split_ratio is not None:
                        current_alloc = self.money_split_ratio[asset_idx]
                        prev_alloc = self.previous_money_split_ratio[asset_idx]
                        
                        avg_entry = self.avg_entry_price[asset_idx]
                        
                        # Only apply if we have a valid entry price and current price
                        if avg_entry > 1e-9 and price_now > 1e-9:
                            # Calculate raw excess ratio (how much above/below cost basis)
                            # e.g., if price=110, entry=100 -> ratio = 1.1 -> excess = 0.1 (10% profit)
                            raw_excess_ratio = (price_now / avg_entry) - 1.0
                            
                            # ✅ INTRADAY FIX: Normalize to per-lookback-window equivalent
                            # This scales the accumulated cost basis effect to match alpha/beta magnitude
                            # A 10% unrealized gain contributes 10%/5 = 2% to the reward, matching
                            # the scale of a 2% market move over the 5-step lookback window.
                            excess_ratio = raw_excess_ratio / config.LONG_TERM_LOOKBACK
                            
                            # ✅ REUSE MOMENTUM DERIVATIVE: Use the already-calculated momentum derivative
                            # This ensures consistency between alpha/beta and cost basis logic
                            # Momentum derivative tells us if momentum is improving (derivative > 0) or deteriorating (derivative < 0)
                            momentum_derivative = asset_momentum_derivatives.get(coin, 0.0)
                            
                            # Case A: Adding to position (Accumulation)
                            if current_alloc > prev_alloc:
                                alloc_increase = current_alloc - prev_alloc

                                # Directly penalize increasing positions whose current quality is negative.
                                # This makes "add to loser" explicitly bad instead of relying only on
                                # the slower average-allocation alpha signal.
                                loser_quality = max(0.0, -base_alpha_i)
                                if loser_quality > 0:
                                    add_to_loser_penalty = (
                                        config.ADD_TO_LOSER_PENALTY_WEIGHT * alloc_increase * loser_quality
                                    )
                                    accumulation_penalty += add_to_loser_penalty
                                    local_contribution -= add_to_loser_penalty

                                    if self._step_debug and add_to_loser_penalty > 1e-4:
                                        print(f"🚫 Add-to-Loser Penalty {coin}: +{alloc_increase:.1%} alloc, "
                                              f"quality {base_alpha_i:.4f}. Penalty: {add_to_loser_penalty:.4f}")
                                
                                # ✅ SIMPLIFIED: Only penalize when momentum derivative is negative
                                # (momentum is deteriorating). Penalty scales with how negative the derivative is,
                                # not with profit/loss level. No hard profit barriers - agent learns optimal
                                # position sizing based on momentum trends.
                                if momentum_derivative < config.MOMENTUM_DERIVATIVE_THRESHOLD:
                                    # Momentum is deteriorating - penalize accumulation
                                    # Penalty scales with allocation increase and magnitude of momentum deterioration
                                    momentum_deterioration = abs(momentum_derivative - config.MOMENTUM_DERIVATIVE_THRESHOLD)
                                    penalty = config.ACCUMULATION_PENALTY_WEIGHT * alloc_increase * momentum_deterioration
                                    accumulation_penalty += penalty
                                    local_contribution -= penalty
                                    
                                    if self._step_debug and penalty > 1e-4:
                                        print(f"⚠️ Accumulation Penalty {coin}: +{alloc_increase:.1%} alloc, "
                                              f"momentum derivative {momentum_derivative:.4f} (deteriorating). Penalty: {penalty:.4f}")
                                # If momentum derivative >= threshold (momentum flat or improving), no penalty
                                # Agent is free to accumulate - learns optimal sizing through experience
                            
                            # Case B: Reducing position (Profit Taking or Stop Loss)
                            elif current_alloc < prev_alloc:
                                alloc_decrease = prev_alloc - current_alloc
                                
                                # Profit Taking Bonus: Small supplementary nudge for reducing positions above cost basis
                                # Kept separate from alpha to avoid distorting the core alpha signal
                                if raw_excess_ratio > 0:  # Use raw for condition check
                                    bonus = config.PROFIT_TAKING_BONUS_WEIGHT * alloc_decrease * excess_ratio
                                    profit_taking_bonus_total += bonus  # Track separately, not in alpha
                                    local_contribution += bonus
                                    
                                    if self._step_debug and bonus > 1e-4:
                                        print(f"💰 Profit Taking Bonus {coin}: -{alloc_decrease:.1%} alloc at {raw_excess_ratio:.1%} premium (normalized: {excess_ratio:.4f}). Bonus: +{bonus:.4f}")

                                # Stop Loss Subsidy: Small supplementary nudge for cutting losing positions
                                # Kept separate from alpha to avoid distorting the core alpha signal
                                elif raw_excess_ratio < 0:  # Use raw for condition check
                                    # Use normalized magnitude for consistent scaling
                                    loss_magnitude = -1.0 * excess_ratio  # Already normalized
                                    subsidy = config.STOP_LOSS_BONUS_WEIGHT * alloc_decrease * loss_magnitude
                                    stop_loss_subsidy_total += subsidy  # Track separately, not in alpha
                                    local_contribution += subsidy
                                    
                                    if self._step_debug and subsidy > 1e-4:
                                        print(f"🛡️ Stop Loss Subsidy {coin}: -{alloc_decrease:.1%} alloc at {raw_excess_ratio:.1%} loss (normalized: {excess_ratio:.4f}). Subsidy: +{subsidy:.4f}")

                    self._record_component_contribution(i, local_contribution)

                alpha_reward = config.STRUCTURED_ALPHA_WEIGHT * total_alpha_reward
                
                # if abs(alpha_reward) > 1e-5:
                #     print(f"🎯 Total Alpha Reward: {total_alpha_reward:.4f} × {config.STRUCTURED_ALPHA_WEIGHT:.2f} = {alpha_reward:.4f}")

                # 4. Reward/Penalize cash against the market regime
                avg_alloc_cash = avg_allocations[0]
                
                # ✅ ENHANCED CASH TERM: Add momentum derivative timing guidance
                # Base cash term (symmetric treatment for trend-following)
                base_cash_term = avg_alloc_cash * (-1 * avg_market_return)
                
                # Momentum-based cash timing (additional guidance)
                # When market momentum accelerates: penalize cash more (invest)
                # When market momentum decelerates: reward cash more (exit to safety)
                momentum_cash_timing = avg_alloc_cash * (-1 * avg_market_momentum_derivative) * config.MOMENTUM_BOOST_SCALING
                
                # Combined cash term
                cash_term = base_cash_term + momentum_cash_timing
                
                # ✅ PURE CASH TERM: Symmetric opportunity cost
                # If Market is Up (Bull): Cash "loses" relative to market -> Penalty
                # If Market is Down (Bear): Cash "wins" relative to market -> Reward
                # We remove the explicit momentum derivative term because the High Gamma (0.80)
                # agent will learn to anticipate trend changes from the State (observations)
                # without needing hard-coded "timing" rewards.
                # cash_term = avg_alloc_cash * (-1 * avg_market_return)

                
                # Debug: Show cash impact
                # if abs(cash_term) > 1e-5:
                #     regime = "bull" if avg_market_return > 0 else "bear"
                #     action = "penalty" if cash_term < 0 else "reward"
                #     print(f"💵 Cash {action} ({regime}): {avg_alloc_cash:.2f} × market {avg_market_return:.2%} = {cash_term:.4f}")

                # --- COMBINE ALL COMPONENTS ---
                # Keep the structured teacher additive so the critic can still see
                # magnitude differences between "good", "very good", and "bad" local
                # portfolio decisions. Alpha and beta are already internally shaped;
                # applying another branch-level tanh here flattened the signal and
                # caused the dense and structured branches to cancel each other too
                # easily. The dense realized return stays as a raw additive teacher,
                # while long-term persistence and downside risk remain explicit terms.
                risk_penalty = config.RISK_ADJUSTMENT_BETA * downside_deviation_excess
                structured_core = (
                    beta_reward
                    + alpha_reward
                    + cash_term
                    - accumulation_penalty
                    + profit_taking_bonus_total
                    + stop_loss_subsidy_total
                )
                step_reward_raw = (
                    structured_core
                    + dense_position_return
                    + self.long_term_bonus
                    - risk_penalty
                )
                
                # Transaction costs remain bookkeeping quantities. They are
                # intentionally absent from the structured reward above.
                transaction_cost_pct = (transaction_cost / self.previous_value) if self.previous_value > 1e-9 else 0.0
                
                total_trading_friction = transaction_cost_pct 
                
                if self._step_debug:
                    print(f"🏆 STRUCTURED CREDIT (TD REWARD):")
                    print(f"   Beta (market exposure): {beta_reward:.4f}")
                    print(f"   Alpha (stock selection): {alpha_reward:.4f}")
                    print(f"   Cash (regime timing):    {cash_term:.4f}")
                    print(f"   Structured Core:         {structured_core:.4f}")
                    print(f"   Dense Step (raw):        {dense_position_return:.4f}")
                    print(f"   Long-Term Bonus:         {self.long_term_bonus:.4f}")
                    print(f"   Risk Penalty:            -{risk_penalty:.4f}")
                    print(f"   Trans. Cost (bookkeeping): {transaction_cost_pct:.4f}")
                    print(f"   Volatility Penalty:      -{volatility_penalty:.4f}")
                    print(f"   Total Raw Reward:        {step_reward_raw:.4f}")

                # --- Concentration Bonus (HHI) ---
                # Rewards the agent for decisive, non-uniform positions.
                # Confirmed necessary by runtime evidence: the policy converges to equal
                # weighting (~8.9% max asset, 51.6% cash) at the zero-logit equilibrium,
                # which neutralizes inverse pairs (TQQQ/SQQQ, USO/SCO) to ~0 financial
                # return. HHI bonus provides the gradient to break this deadlock.
                # concentration_bonus = np.sum(np.square(self.last_target_allocation)) * config.CONCENTRATION_BONUS_WEIGHT
                # step_reward_raw += concentration_bonus

                # ✅ FINANCIAL-RETURN-PRIMARY REWARD
                # long_term_return = actual portfolio percentage change over LONG_TERM_LOOKBACK steps.
                # This is the quantity whose integral IS the episode financial return.
                # step_reward_raw = backward-looking shaped algebra
                # (alpha, beta, cash timing) + dense realized return + long-term
                # persistence bonus - downside risk penalty.
                # We make long_term_return the dominant signal so the critic's Q-surface
                # is aligned with real portfolio growth. The shaped component is kept as a
                # small behavioural nudge (SHAPED_REWARD_BLEND ≈ 0.10) to provide soft could be 0.5 for equal split
                # guidance (prefer recent winners, time cash, avoid deteriorating momentum)
                # without overwhelming the financial signal.
                # blended_reward = long_term_return + config.SHAPED_REWARD_BLEND * step_reward_raw
                legacy_structured_diagnostic_reward = float(
                    np.clip(step_reward_raw, -5.5, 5.5)
                )

                if self._step_debug and (abs(step_reward_raw) > 1e-5 or risk_penalty > 1e-5):
                    print(f"🛠️ Legacy diagnostic: raw {step_reward_raw:.4f} | risk_penalty {risk_penalty:.4f} | dense_ret {dense_position_return:.4f} step_ret {step_return:.4f} α {alpha_reward:.4f} β {beta_reward:.4f} cash {cash_term:.4f} lt {self.long_term_bonus:.4f} -> clipped diagnostic {legacy_structured_diagnostic_reward:.4f}")
                    # acumilated rewards:

                self.step_reward = financial_growth_reward

            self._structured_reward_diagnostics = {
                "structured_credit_reward": self.step_reward,
                "financial_growth_reward": financial_growth_reward,
            }

            self._finalize_causal_financial_credit(
                transition_execution_prices,
                portfolio_reward=self.step_reward,
            )



        elif self.previous_value > 1e-9:  # Avoid division by zero
            # Calculate gross return (percentage change in portfolio value)
            gross_return = step_return
            
            if self.reward_type == "TRANSACTION_COST":
                # Pure-PnL dense reward for TRANSACTION_COST:
                # reward is the step-to-step portfolio return (higher/lower than previous step).
                raw_reward = gross_return
                self.step_reward = np.clip(raw_reward, -5.5, 5.5)

                # Keep bookkeeping for reporting/analysis, but do not use transaction costs in reward.
                self.total_transaction_costs += transaction_cost

                if transaction_cost > 0:
                    transaction_cost_pct = (transaction_cost / self.previous_value if self.previous_value > 1e-9 else 0.0)
                    if self._step_debug:
                        print(f"📊 PnL Mode: Gross={gross_return:.4f}, Tx(bookkeeping only)={transaction_cost_pct:.4f}, StepReward={self.step_reward:.4f}")
            else:
                # Simple Return (original calculation) + long-term bonus + activity bonus, with risk adjustment
                core_reward = gross_return + self.long_term_bonus #+ activity_bonus
                
                # # --- NEW: Risk-Adjusted Reward Scaling ---
                risk_adjustment_factor = 1.0 / (1.0 + config.RISK_ADJUSTMENT_BETA * downside_deviation)
                raw_reward = core_reward #* risk_adjustment_factor

                # ✅ REWARD CLIPPING FIX: Clip rewards to prevent Q-value explosion
                self.step_reward = np.clip(raw_reward, -5.5, 5.5)
                
                if abs(self.long_term_bonus) > 1e-5 or activity_bonus > 1e-5:
                    # print(f"📊 Simple Return: {gross_return:.4f}, LT: {self.long_term_bonus:.4f}, "
                    #       f"Activity: {activity_bonus:.4f}, Core: {core_reward:.4f}, RiskFactor: {risk_adjustment_factor:.3f}, "
                    #       f"Raw: {raw_reward:.4f}, Clipped: {self.step_reward:.4f}")
                    pass
        else:
            # Only long-term bonus if no value change
            raw_reward = self.long_term_bonus

            self.step_reward = np.clip(raw_reward, -5.5, 5.5)

    def get_observations(self):
        """Get scaled observations from current market data using INCREMENTAL scaling."""
        
        # 1. Extract raw features for current step
        # Note: These values are unscaled (raw prices/volumes/ratios)
        dfslice = self.dfslice
        
        # Current prices for cost basis calculation
        # ✅ TRAIN-LIVE ALIGNMENT: Use 'open' to match live execution reality
        # Agent needs to track cost basis using same prices as live trading
        current_prices = {
            coin: float(price)
            for coin, price in zip(self._current_step_coins, self._current_step_open_prices)
        }
        
        # ✅ CAUSAL SCALING FIX:
        # Before transforming, update the scaler with the current step's raw data.
        # This ensures the scaler evolves causally (only knows past + present).
        # We need to reshape the data to match how the scaler was fitted (N columns of features).
        
        # Extract raw features for all coins in this slice
        # NOTE: When ACT_AT_OPEN is enabled, SCOLS are pre-shifted by 1 bar in preprocessing.
        # The columns in dfslice match config.SCOLS (or config.COLS which contains them)
        # We need to extract only the feature columns used for scaling.
        
        current_step_raw_features = dfslice[config.SCOLS].to_numpy(dtype=np.float32, copy=False)
        scaled_values = self._scale_feature_rows(
            current_step_raw_features,
            valid_mask=self._current_step_primary_valid_rows,
        )
        self._current_step_scaled_feature_rows = scaled_values
        scaled_episode_relative = self._scale_episode_relative_rows(
            self._current_step_episode_relative_rows
        )
        self._current_step_scaled_episode_relative_rows = scaled_episode_relative

        # ✅ PERFORMANCE FIX: Use concat instead of iterative merges for better performance
        coin_dfs = []
        
        # We need to map the scaled values back to their coins.
        # The order of scaled_values matches the order of rows in dfslice.
        # dfslice is unique per coin.
        
        # Create a DataFrame of scaled values aligned with dfslice index
        scaled_df_slice = pd.DataFrame(
            scaled_values, 
            columns=config.SCOLS,
            index=dfslice.index
        )
        
        # Add coin column back for grouping
        scaled_df_slice['coin'] = dfslice['coin']
        
        for coin, grp in scaled_df_slice.groupby("coin"):
            # Create base features DataFrame with prefixed columns
            tempdf = pd.DataFrame(
                grp[config.SCOLS].values, 
                columns=[f"{coin}_{c}" for c in config.SCOLS]
            )
            
            # --- NEW: Append Cost Basis Feature ---
            # Feature = (Current Price / Avg Entry Price) - 1.0
            # If Avg Entry Price is 0 (no position), feature is 0.
            
            # Find asset index for this coin
            # Note: self.episode_coins is the list of active coins. 
            # We need to find the index 'i' such that self.episode_coins[i] == coin
            # Then self.avg_entry_price[i+1] is the value.
            try:
                coin_idx = self.episode_coins.index(coin)
                avg_entry = self.avg_entry_price[coin_idx + 1]
                
                if avg_entry > 1e-9:
                    current_p = current_prices.get(coin, 0.0)
                    raw_cost_basis = (current_p / avg_entry) - 1.0
                    # ✅ FEATURE SCALING: Multiply by COST_BASIS_FEATURE_SCALE (typically 10x)
                    # to match the magnitude of StandardScaled features (Z-scores ~ [-3, 3]).
                    # A 10% gain (0.1) becomes 1.0, making it "visible" to the neural network.
                    cost_basis_feature = raw_cost_basis * config.COST_BASIS_FEATURE_SCALE
                    # ✅ STABILITY CLIP: Minimum is -10.0 (100% loss * 10x scale).
                    # We clip at -15.0 to provide a safety buffer while keeping magnitudes consistent.
                    cost_basis_feature = np.clip(cost_basis_feature, -15.0, 50.0)
                else:
                    cost_basis_feature = 0.0
            except ValueError:
                # Coin not in episode_coins (shouldn't happen given logic)
                cost_basis_feature = 0.0
                
            tempdf[f"{coin}_cost_basis"] = cost_basis_feature

            # Fixed-reference behavioral scale is already bounded when the
            # episode feature frame is constructed.  Do not run it through a
            # rolling basket z-score here or its absolute-scale information
            # would be erased again.
            row_idx = self._current_coin_to_row.get(coin)
            if row_idx is None:
                behavioral_scale_values = np.zeros(
                    config.BEHAVIORAL_SCALE_FEATURE_COUNT, dtype=np.float32
                )
            else:
                behavioral_scale_values = (
                    self._current_step_behavioral_scale_rows[row_idx]
                )
            for feature_name, feature_value in zip(
                config.BEHAVIORAL_SCALE_FEATURES, behavioral_scale_values
            ):
                tempdf[f"{coin}_{feature_name}"] = float(feature_value)

            # Append the separately normalized relational block computed from
            # this episode's selected basket. It has its own feature-family
            # statistics rather than sharing moments with the primary block.
            if row_idx is None:
                relative_values = np.zeros(
                    config.EPISODE_RELATIVE_FEATURE_COUNT, dtype=np.float32
                )
            else:
                relative_values = scaled_episode_relative[row_idx]
            for feature_name, feature_value in zip(
                config.EPISODE_RELATIVE_FEATURES, relative_values
            ):
                tempdf[f"{coin}_{feature_name}"] = float(feature_value)
            
            coin_dfs.append(tempdf)
        
        # Concatenate all coin DataFrames at once (much faster than iterative merges)
        if coin_dfs:
            dfs = pd.concat(coin_dfs, axis=1)
        else:
            dfs = pd.DataFrame()
        
        return dfs

    def get_observations_unscaled(self):
        """Get unscaled observations from current market data"""
        dfslice = self.dfslice
        
        # ✅ PERFORMANCE FIX: Use concat instead of iterative merges for better performance
        coin_dfs = []
        for coin, grp in dfslice.groupby("coin"):
            tempdf = pd.DataFrame(
                grp[config.COLS].values,
                columns=[f"{coin}_{c}" for c in config.COLS]
            )
            coin_dfs.append(tempdf)
        
        # Concatenate all coin DataFrames at once
        if coin_dfs:
            dfs = pd.concat(coin_dfs, axis=1)
        else:
            dfs = pd.DataFrame()
        
        # Update memory return tracking
        close_cols = [f"{t}_close" for t in self.episode_coins if f"{t}_close" in dfs.columns]
        if close_cols:
            self.memory_return = pd.concat([self.memory_return, dfs[close_cols]], ignore_index=True)
        
        return dfs

    def calculate_actual_shares_from_money_split(self):
        """Calculate number of shares to buy based on money allocation"""
        # ✅ TRAIN-LIVE ALIGNMENT: Use 'open' to match live execution reality
        # In live trading, orders execute at open price (most recent trade)
        dict_coin_price = self.dfslice[["coin", "open"]].set_index("coin").to_dict()["open"]
        
        # --- FIX: Directly use the agent's target cash allocation ---
        # This is robust to floating point errors and avoids calculating cash as a
        # remainder, which was the source of the negative cash values.
        self.current_cash = self.money_split_ratio[0] * self.current_value
        
        # --- NEW: Store the agent's target allocation before it's altered by market drift ---
        self.last_target_allocation = self.money_split_ratio.copy()
        
        num_shares = []
        for i, c in enumerate(self.episode_coins):
            asset_idx = i + 1
            money_for_coin = self.money_split_ratio[asset_idx] * self.current_value
            
            # Get current price
            current_p = 0.0
            if c in dict_coin_price and dict_coin_price[c] > 1e-9:
                current_p = dict_coin_price[c]
            elif c in self.old_dict_coin_price_1 and self.old_dict_coin_price_1[c] > 1e-9:
                current_p = self.old_dict_coin_price_1[c]
            
            if current_p > 1e-9:
                new_shares = money_for_coin / current_p
                
                # --- NEW: Update Average Entry Price (WACB) ---
                # We need the previous share count to calculate the weighted average.
                # Initialize to 0 if we haven't tracked it yet (first step)
                old_shares = 0.0
                if hasattr(self, 'current_stock_num_distribution') and len(self.current_stock_num_distribution) > i:
                    old_shares = self.current_stock_num_distribution[i]
                
                # ✅ FIX: Use Value-based threshold instead of Share count
                # "Cheap" coins (like STR) have high share counts even with $0.01 value,
                # causing them to never "close" and resetting cost basis.
                # Using value < $0.05 ensures dust is cleared uniformly for all asset prices.
                position_value = new_shares * current_p
                MIN_POSITION_VALUE = 0.05
                
                if position_value > MIN_POSITION_VALUE: # Buying / Holding
                    if new_shares > old_shares + 1e-6: # Actually Buying
                        shares_added = new_shares - old_shares
                        cost_added = shares_added * current_p
                        current_cost_basis = self.avg_entry_price[asset_idx] * old_shares
                        
                        # Weighted Average
                        new_avg_price = (current_cost_basis + cost_added) / new_shares
                        self.avg_entry_price[asset_idx] = new_avg_price
                        
                else: # Closed position (Dust)
                    self.avg_entry_price[asset_idx] = 0.0
                    
                # If Selling (new_shares < old_shares) or Holding, avg_entry_price stays the same (FIFO/Average Cost)
                
                num_shares.append(new_shares)
            else:
                num_shares.append(0.0) # Cannot buy if price is unknown or zero
                self.avg_entry_price[asset_idx] = 0.0
            
        for c in dict_coin_price:
            self.old_dict_coin_price_1[c] = dict_coin_price[c]
        
        return num_shares

    def calculate_money_from_num_stocks(self):
        """Calculate current portfolio value from number of shares"""
        money_dist = []
        # Cash is the first element, and it doesn't change with market prices
        money_dist.append(self.current_cash)
        # ✅ TRAIN-LIVE ALIGNMENT: Use 'open' to match live execution reality
        # Portfolio valuation uses open prices (market value at bar open)
        dict_coin_price = self.dfslice[["coin", "open"]].set_index("coin").to_dict()["open"]
        
        # ✅ DEBUG: Show price information
        # print(f"🏷️  Current prices: {dict_coin_price}")
        
        for i, c in enumerate(self.episode_coins):
            price = dict_coin_price.get(c, self.old_dict_coin_price_2.get(c, 0))
            shares = self.current_stock_num_distribution[i]
            coin_value = shares * price
            money_dist.append(coin_value)
            self.old_dict_coin_price_2[c] = price # Update last known price
            
            # ✅ DEBUG: Show calculation for each coin
            # print(f"💎 {c}: {shares:.6f} shares × ${price:.4f} = ${coin_value:.2f}")
            
        total_value = sum(money_dist)
        # print(f"💰 Portfolio breakdown: Cash=${money_dist[0]:.2f}, Coins=${sum(money_dist[1:]):.2f}, Total=${total_value:.2f}")
        
        return money_dist, total_value

    def normalize_money_dist(self):
        """Normalize money distribution to get portfolio weights"""
        if self.use_variable_portfolio:
            # For variable portfolios, we need to return a fixed-size numpy array
            # padded with zeros for inactive coins.
            normal = np.zeros(config.MAX_COINS + 1)
            if self.current_value > 1e-9:
                # Calculate weights for cash and active coins
                for i, c_val in enumerate(self.current_stock_money_distribution):
                    normal[i] = c_val / self.current_value
            else:
                normal[0] = 1.0 # Default to all cash if value is zero
            return normal
        else:
            # For fixed portfolios, a simple list is sufficient
            normal = []
            if self.current_value > 1e-9:
                for i, c in enumerate(self.current_stock_money_distribution):
                    normal.append(c / self.current_value)
            else:
                normal = [0.0] * len(self.current_stock_money_distribution)
                if len(normal) > 0:
                    normal[0] = 1.0
            return normal

    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Time: {self.current_time}")
            print(f"Portfolio Value: ${self.current_value:.2f}")
            print(f"Cash Ratio: {self.money_split_ratio[0]:.3f}")
            
            if self.use_variable_portfolio:
                print(f"Active Coins: {self.episode_coins}")
                for i, coin in enumerate(self.episode_coins):
                    print(f"{coin} Ratio: {self.money_split_ratio[i+1]:.3f}")
            else:
                for i, coin in enumerate(config.COINS):
                    print(f"{coin} Ratio: {self.money_split_ratio[i+1]:.3f}")
                    
            print(f"Step Reward: {self.step_reward:.4f}")
            print("-" * 40)

    def close(self):
        """Clean up environment resources"""
        pass


# =========================================================================
# BATCHED TRAINING SIMULATOR
# =========================================================================
BATCHED_MARKET_PANEL_SCHEMA = "batched_market_panel_v1"


def _batched_market_panel_cache_path(cache_dir, time_count, tickers):
    """Cache file name tied to the data file and every input contract."""
    data_file = os.path.abspath(str(getattr(config, "FILE", "")))
    try:
        stat = os.stat(data_file)
        file_identity = [int(stat.st_size), int(stat.st_mtime)]
    except OSError:
        file_identity = [None, None]
    identity = {
        "schema": BATCHED_MARKET_PANEL_SCHEMA,
        "file": data_file,
        "file_identity": file_identity,
        "time_count": int(time_count),
        "tickers": list(tickers),
        "contract": config.market_input_contract(),
        "primary": list(config.SCOLS),
        "behavioral": list(config.BEHAVIORAL_SCALE_FEATURES),
        "history": list(config.MDAEN_LOCAL_FEATURE_NAMES),
        "warmup": int(config.SCALER_WARMUP_STEPS),
        "episode_length": int(config.EPISODE_LENGTH),
        "selected": int(config.SELECTED_ASSETS_PER_EPISODE),
        "history_bounds": [
            float(config.MDAEN_FAM_FEATURE_MIN),
            float(config.MDAEN_FAM_FEATURE_MAX),
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f"{BATCHED_MARKET_PANEL_SCHEMA}_{digest}.npz")


def build_batched_market_panel(data=None, cache_dir=None, verbose=True):
    """Market state of every pool asset for every valid PortfolioEnv episode.

    An episode window spans the SCALER_WARMUP_STEPS timestamps before a start
    of day through EPISODE_LENGTH timestamps after it, which are the rows
    PortfolioEnv loads for that episode.  Features come from the same
    functions the environment applies to that window, so an asset's arrays
    equal what it shows inside any basket.  Arrays use global-pool slot order:

    ``starts`` [D] global timestamp index of each valid episode start,
    ``start_times`` [D] those timestamps as UTC nanoseconds,
    ``snapshot`` [D, W, A, P + B] scaled primary block and behavioral anchors,
    ``history`` [D, W, A, H] local temporal frame (MDAEN storage dtype),
    ``history_valid`` [D, W, A] frame finiteness before sanitizing,
    ``open`` [D, W, A] open prices.
    """
    if not config.LOCAL_CAUSAL_INPUTS_ENABLED:
        raise ValueError("The batched simulator requires LOCAL_CAUSAL_INPUTS_ENABLED")
    source = config.data_set if data is None else data
    tickers = list(config.COIN_POOL)
    asset_count = len(tickers)
    if asset_count == 0 or len(source) % asset_count:
        raise ValueError("batched simulation requires a complete timestamp/asset panel")
    time_count = len(source) // asset_count
    coin_layout = source["coin"].to_numpy().reshape(time_count, asset_count)
    column_coins = [str(coin) for coin in coin_layout[0]]
    if sorted(column_coins) != sorted(tickers) or not np.all(
        coin_layout == coin_layout[:1]
    ):
        raise ValueError(
            "batched simulation requires one ordered row per asset and timestamp"
        )
    first_rows = source["date"].iloc[::asset_count]
    if not np.array_equal(
        first_rows.to_numpy(),
        source["date"].iloc[asset_count - 1::asset_count].to_numpy(),
    ):
        raise ValueError("every timestamp must carry all pool assets")
    slot_of_column = np.asarray(
        [config.TICKER_TO_ID[coin] for coin in column_coins], dtype=np.int64
    )

    cache_path = (
        _batched_market_panel_cache_path(cache_dir, time_count, tickers)
        if cache_dir
        else None
    )
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as cached:
            panel = {key: cached[key] for key in cached.files}
        if verbose:
            print(
                f"📦 Batched market panel loaded from {cache_path}: "
                f"{len(panel['starts'])} episode windows"
            )
        return panel

    timestamps = pd.DatetimeIndex(first_rows)
    calendar_days = np.asarray(timestamps.date)
    day_starts = np.flatnonzero(
        np.r_[True, calendar_days[1:] != calendar_days[:-1]]
    )
    warmup = int(config.SCALER_WARMUP_STEPS)
    episode_length = int(config.EPISODE_LENGTH)
    selected = int(config.SELECTED_ASSETS_PER_EPISODE)
    # PortfolioEnv's valid_start_points, expressed in timestamps.
    starts = np.asarray(
        [
            start
            for start in day_starts
            if start >= warmup
            and start * selected + episode_length * selected + 2
            < time_count * selected
        ],
        dtype=np.int64,
    )
    window = warmup + episode_length + 1
    primary_count = len(config.SCOLS)
    behavioral_names = list(config.BEHAVIORAL_SCALE_FEATURES)
    history_names = list(config.MDAEN_LOCAL_FEATURE_NAMES)
    episode_count = len(starts)
    snapshot = np.zeros(
        (episode_count, window, asset_count, primary_count + len(behavioral_names)),
        dtype=np.float32,
    )
    history = np.zeros(
        (episode_count, window, asset_count, len(history_names)),
        dtype=config.MDAEN_FAM_STORAGE_DTYPE,
    )
    history_valid = np.zeros((episode_count, window, asset_count), dtype=bool)
    opens = np.zeros((episode_count, window, asset_count), dtype=np.float32)
    references = config.local_primary_references(config.SCOLS)
    history_low = float(config.MDAEN_FAM_FEATURE_MIN)
    history_high = float(config.MDAEN_FAM_FEATURE_MAX)
    started_at = time.perf_counter()
    for episode, start in enumerate(starts):
        rows = source.iloc[
            asset_count * (start - warmup):asset_count * (start + episode_length + 1)
        ]
        attached = _attach_episode_relative_features(rows, column_coins)
        raw = attached[config.SCOLS].to_numpy(dtype=np.float32).reshape(
            window, asset_count, primary_count
        )
        contaminated = attached["_corporate_action_primary_mask"].to_numpy(
            dtype=bool
        ).reshape(window, asset_count)
        # PortfolioEnv._scale_feature_rows, local-causal branch.
        scaled = np.arcsinh(raw / references)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        scaled = np.clip(scaled, -3.0, 3.0).astype(np.float32)
        scaled[contaminated] = 0.0
        behavioral = attached[behavioral_names].to_numpy(dtype=np.float32).reshape(
            window, asset_count, len(behavioral_names)
        )
        frames = attached[history_names].to_numpy(dtype=np.float32).reshape(
            window, asset_count, len(history_names)
        )
        valid = np.isfinite(frames).all(axis=-1)
        frames = np.nan_to_num(
            np.clip(frames, history_low, history_high),
            nan=0.0,
            posinf=history_high,
            neginf=history_low,
        )
        snapshot[episode][:, slot_of_column, :primary_count] = scaled
        snapshot[episode][:, slot_of_column, primary_count:] = behavioral
        history[episode][:, slot_of_column] = frames
        history_valid[episode][:, slot_of_column] = valid
        opens[episode][:, slot_of_column] = attached["open"].to_numpy(
            dtype=np.float32
        ).reshape(window, asset_count)
        if verbose and (episode + 1) % 100 == 0:
            print(
                f"📦 Batched market panel: {episode + 1}/{episode_count} windows "
                f"({time.perf_counter() - started_at:.0f}s)"
            )
    panel = {
        "starts": starts,
        "start_times": timestamps[starts].asi8.astype(np.int64),
        "snapshot": snapshot,
        "history": history,
        "history_valid": history_valid,
        "open": opens,
    }
    if verbose:
        print(
            f"📦 Batched market panel built: {episode_count} episode windows x "
            f"{window} timestamps x {asset_count} assets in "
            f"{time.perf_counter() - started_at:.1f}s"
        )
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        temporary_path = cache_path + ".tmp"
        with open(temporary_path, "wb") as handle:
            np.savez(handle, **panel)
        os.replace(temporary_path, cache_path)
        if verbose:
            print(f"📦 Batched market panel cached at {cache_path}")
    return panel


class BatchedPortfolioVecEnv(VecEnv):
    """PortfolioEnv training episodes for many baskets in lockstep.

    This historical implementation supports log-growth rewards only. The
    constructor rejects the restored structured-credit shaping profile.

    Each of the ``num_envs`` episodes follows PortfolioEnv: a start-of-day
    window of EPISODE_LENGTH steps on a basket from
    ``config.get_random_coin_subset()``, a Dirichlet initial allocation,
    hierarchical action normalization, share and weighted-average entry-price
    accounting, the STRUCTURED_CREDIT log-growth reward, the truncated final
    bar, and identical observations and replay metadata (component outcomes
    and cross-basket relabel state).  Market state comes from
    ``build_batched_market_panel``; portfolio state lives on ``device``.
    Observations are returned as NumPy arrays for SB3 replay, and completed
    episodes are written to ``monitor_path`` like the Monitor wrapper.
    """

    # PortfolioEnv's episode slice ends two rows into its final timestamp.
    FINAL_BAR_ROWS = 2
    MIN_POSITION_VALUE = 0.05
    INITIAL_VALUE = 1000.0

    def __init__(
        self,
        num_envs,
        reward_type="STRUCTURED_CREDIT",
        device="auto",
        seed=None,
        panel=None,
        cache_dir=None,
        monitor_path=None,
        append_monitor=False,
    ):
        if reward_type != "STRUCTURED_CREDIT":
            raise ValueError("BatchedPortfolioVecEnv reproduces the STRUCTURED_CREDIT reward only")
        if config.STRUCTURED_CREDIT_REWARD_VERSION != "financial_log_growth_v1":
            raise ValueError(
                "BatchedPortfolioVecEnv implements one-step log-growth rewards, "
                "not the restored history-dependent structured-credit reward. "
                "Use PortfolioEnv by omitting --gpu-sim-envs."
            )
        if not (
            config.EARNMORE_GLOBAL_POOL_ENABLED
            and config.EARNMORE_POOL_LEVEL_ENABLED
            and config.LOCAL_CAUSAL_INPUTS_ENABLED
        ):
            raise ValueError(
                "BatchedPortfolioVecEnv requires the global pool, whole-stock "
                "pool history, and local-causal inputs"
            )
        self._template_env = PortfolioEnv(
            use_variable_portfolio=True, reward_type=reward_type
        )
        super().__init__(
            int(num_envs),
            self._template_env.observation_space,
            self._template_env.action_space,
        )
        self.reward_type = reward_type
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if panel is None:
            panel = build_batched_market_panel(cache_dir=cache_dir)
        self.episode_starts = np.asarray(panel["starts"], dtype=np.int64)
        self.episode_start_times = np.asarray(panel["start_times"], dtype=np.int64)
        if len(self.episode_starts) == 0:
            raise ValueError("The market panel has no valid episode windows")
        self._snapshot = torch.as_tensor(panel["snapshot"], device=self.device)
        self._history = torch.as_tensor(panel["history"], device=self.device)
        self._history_valid = torch.as_tensor(panel["history_valid"], device=self.device)
        self._open = torch.as_tensor(panel["open"], device=self.device)

        self.asset_count = len(config.COIN_POOL)
        self.features_per_coin = int(config.FEATURES_PER_COIN)
        self.primary_count = int(config.PRIMARY_FEATURE_COUNT)
        self.comparative_offset = int(config.COMPARATIVE_FEATURE_OFFSET)
        self.comparative_count = int(config.COMPARATIVE_FEATURE_COUNT)
        self.cost_basis_index = int(config.COST_BASIS_TOKEN_INDEX)
        self.cost_basis_scale = float(config.COST_BASIS_FEATURE_SCALE)
        self.warmup = int(config.SCALER_WARMUP_STEPS)
        self.episode_length = int(config.EPISODE_LENGTH)
        self.selected = int(config.SELECTED_ASSETS_PER_EPISODE)
        self.history_length = int(config.MDAEN_FAM_HISTORY_LENGTH)
        self.reward_scale = float(getattr(config, "FINANCIAL_REWARD_SCALE", 10.0))
        self.component_target_clip = float(config.COMPONENT_OUTCOME_TARGET_CLIP)
        if self._snapshot.shape[1] < self.warmup + self.episode_length + 1:
            raise ValueError("Market panel windows are shorter than an episode")
        if self.warmup < self.history_length - 1:
            raise ValueError("Warmup must cover the temporal history window")
        tickers = list(config.COIN_POOL)
        self._name_order = torch.as_tensor(
            sorted(range(self.asset_count), key=lambda slot: tickers[slot]),
            device=self.device,
        )
        self._history_offsets = torch.arange(
            -(self.history_length - 1), 1, device=self.device
        )

        count = self.num_envs
        assets = self.asset_count
        self._day = torch.zeros(count, dtype=torch.long, device=self.device)
        self._step = torch.zeros(count, dtype=torch.long, device=self.device)
        self._mask = torch.zeros(count, assets, dtype=torch.bool, device=self.device)
        self._cash = torch.zeros(count, dtype=torch.float64, device=self.device)
        self._value = torch.zeros(count, dtype=torch.float64, device=self.device)
        self._shares = torch.zeros(count, assets, dtype=torch.float64, device=self.device)
        self._entry = torch.zeros(count, assets, dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros(
            count, assets + 1, dtype=torch.float32, device=self.device
        )
        self._has_previous = torch.zeros(count, dtype=torch.bool, device=self.device)
        self._episode_return = np.zeros(count, dtype=np.float64)
        self._episode_length = np.zeros(count, dtype=np.int64)
        self.episode_coins = [[] for _ in range(count)]
        self._rng = np.random.default_rng(
            int(getattr(config, "BATCHED_SIM_SEED", 0)) if seed is None else int(seed)
        )
        self._actions = None

        self.market_input_contract = config.market_input_contract()
        self.component_outcome_kind = (
            "scaled_unit_asset_return"
            if bool(getattr(config, "COMPONENT_OUTCOME_UNIT_RETURN_TARGETS", False))
            else "scaled_executed_contribution"
        )
        self.component_outcomes_active = self._template_env._component_outcome_aux_active()
        self.relabel_active = self._template_env._cross_basket_relabel_active()
        self.relabel_dtype = np.dtype(
            getattr(config, "CROSS_BASKET_RELABEL_STORAGE_DTYPE", np.float16)
        )

        self.t_start = time.time()
        self._results_writer = (
            ResultsWriter(
                monitor_path,
                header={"t_start": self.t_start, "env_id": "BatchedPortfolioVecEnv"},
                override_existing=not append_monitor,
            )
            if monitor_path
            else None
        )
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_times = []

    # ------------------------------------------------------------------ state
    def _reset_envs(self, indices, baskets=None, days=None, initial_weights=None):
        """Start new episodes; optional arguments pin basket, window, or split.

        ``initial_weights`` rows are ordered like ``baskets`` rows: cash first,
        then one weight per basket asset in basket order, as PortfolioEnv draws
        them.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        count = len(indices)
        assets = self.asset_count
        masks = np.zeros((count, assets), dtype=bool)
        splits = np.zeros((count, assets + 1), dtype=np.float32)
        day_rows = np.zeros(count, dtype=np.int64)
        for row in range(count):
            coins = (
                list(baskets[row])
                if baskets is not None
                else list(config.get_random_coin_subset())
            )
            if len(coins) != self.selected or len(set(coins)) != len(coins):
                raise ValueError(
                    f"Each episode needs {self.selected} distinct assets; got {coins}"
                )
            slots = np.asarray([config.TICKER_TO_ID[coin] for coin in coins], dtype=np.int64)
            weights = (
                np.asarray(initial_weights[row], dtype=np.float32)
                if initial_weights is not None
                else self._rng.dirichlet(np.ones(self.selected + 1)).astype(np.float32)
            )
            masks[row, slots] = True
            splits[row, 0] = weights[0]
            splits[row, 1 + slots] = weights[1:]
            day_rows[row] = (
                int(days[row])
                if days is not None
                else int(self._rng.integers(len(self.episode_starts)))
            )
            self.episode_coins[int(indices[row])] = coins

        index = torch.as_tensor(indices, device=self.device)
        mask = torch.as_tensor(masks, device=self.device)
        split = torch.as_tensor(splits, device=self.device)
        day = torch.as_tensor(day_rows, device=self.device)
        price = self._open[day, self.warmup].double()
        value = torch.full((count,), self.INITIAL_VALUE, dtype=torch.float64, device=self.device)
        cash = split[:, 0].double() * value
        zeros = torch.zeros_like(price)
        shares = torch.where(mask, split[:, 1:].double() * value.unsqueeze(1) / price, zeros)
        opened = mask & (shares * price > self.MIN_POSITION_VALUE) & (shares > 1e-6)
        entry = torch.where(opened, price.float(), torch.zeros_like(price, dtype=torch.float32))
        value = cash + (shares * price).sum(dim=1)
        previous = torch.zeros(count, assets + 1, dtype=torch.float32, device=self.device)
        previous[:, 0] = (cash / value).float()
        previous[:, 1:] = (shares * price / value.unsqueeze(1)).float()

        self._day[index] = day
        self._step[index] = 0
        self._mask[index] = mask
        self._cash[index] = cash
        self._value[index] = value
        self._shares[index] = shares
        self._entry[index] = entry
        self._prev_action[index] = previous
        self._has_previous[index] = False
        self._episode_return[indices] = 0.0
        self._episode_length[indices] = 0

    def _final_bar_live(self, mask):
        """Basket assets PortfolioEnv prices at the truncated final bar."""
        by_name = mask[:, self._name_order]
        live_by_name = by_name & (torch.cumsum(by_name.long(), dim=1) <= self.FINAL_BAR_ROWS)
        live = torch.zeros_like(mask)
        live[:, self._name_order] = live_by_name
        return live

    def _observation_tensors(self, index, positions, live):
        """Observation of ``index`` environments at panel window ``positions``."""
        count = int(index.numel())
        assets = self.asset_count
        day = self._day[index]
        mask = self._mask[index]
        present = mask & live
        snapshot = self._snapshot[day, positions]
        tokens = torch.zeros(
            count, assets, self.features_per_coin, dtype=torch.float32, device=self.device
        )
        tokens[..., :self.primary_count] = snapshot[..., :self.primary_count]
        tokens[..., self.comparative_offset:self.comparative_offset + self.comparative_count] = (
            snapshot[..., self.primary_count:self.primary_count + self.comparative_count]
        )
        price = self._open[day, positions].double()
        entry = self._entry[index].double()
        held = (entry > 1e-9) & (price > 1e-9)
        cost_basis = torch.where(
            held,
            (price / entry.clamp(min=1e-9) - 1.0) * self.cost_basis_scale,
            torch.zeros_like(price),
        ).clamp(-15.0, 50.0)
        tokens[..., self.cost_basis_index] = cost_basis.float()
        tokens = torch.where(present.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        time_index = positions.unsqueeze(1) + self._history_offsets
        frames = self._history[day.unsqueeze(1), time_index].permute(0, 2, 1, 3).clone()
        frames[:, :, -1, :] = torch.where(
            (mask & ~live).unsqueeze(-1), torch.zeros_like(frames[:, :, -1, :]), frames[:, :, -1, :]
        )
        valid = self._history_valid[day.unsqueeze(1), time_index].permute(0, 2, 1)
        valid = valid | mask.unsqueeze(-1)
        return {
            "observations": tokens.reshape(count, assets * self.features_per_coin),
            "mask": mask.float(),
            "prev_action": self._prev_action[index],
            "prev_mask": (mask & self._has_previous[index].unsqueeze(1)).float(),
            "basket_membership": mask.float(),
            "local_feature_history": frames,
            "pool_history_valid": valid.to(torch.uint8),
        }

    def _numpy_observations(self, tensors, count):
        result = {}
        for key, space in self.observation_space.spaces.items():
            if key in tensors:
                result[key] = tensors[key].detach().cpu().numpy().astype(space.dtype, copy=False)
            elif key == "asset_ids":
                result[key] = np.tile(np.arange(self.asset_count, dtype=space.dtype), (count, 1))
            elif key == "feature_history_mask":
                result[key] = np.ones((count, *space.shape), dtype=space.dtype)
            elif key in ("basket_context", "episode_feature_history"):
                result[key] = np.zeros((count, *space.shape), dtype=space.dtype)
            else:
                raise KeyError(f"BatchedPortfolioVecEnv cannot build observation key {key!r}")
        return result

    def observe(self, indices=None):
        """Current observations for ``indices`` (all environments by default)."""
        indices = np.arange(self.num_envs) if indices is None else np.asarray(indices)
        index = torch.as_tensor(indices, device=self.device)
        return self._numpy_observations(
            self._observation_tensors(
                index, self.warmup + self._step[index], self._mask[index]
            ),
            len(indices),
        )

    # ---------------------------------------------------------------- VecEnv
    def reset(self):
        self._reset_envs(np.arange(self.num_envs))
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self.observe()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        count = self.num_envs
        assets = self.asset_count
        index = torch.arange(count, device=self.device)
        actions = torch.as_tensor(self._actions, device=self.device).reshape(count, assets + 1)
        mask = self._mask
        mask_float = mask.float()

        # PortfolioEnv.step hierarchical normalization.
        cash_weight = (actions[:, 0] + 1.0) * 0.5
        shifted = (actions[:, 1:] + 1.0) * 0.5 * mask_float
        total = shifted.sum(dim=1, keepdim=True)
        uniform = mask_float / mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        relative = torch.where(total > 1e-8, shifted / total.clamp(min=1e-8), uniform)
        weights = torch.cat(
            (cash_weight.unsqueeze(1), relative * (1.0 - cash_weight).unsqueeze(1)), dim=1
        )

        # calculate_actual_shares_from_money_split at the decision open.
        positions = self.warmup + self._step
        day = self._day
        price_now = self._open[day, positions].double()
        value = self._value
        cash = weights[:, 0].double() * value
        previous_shares = self._shares
        shares = torch.where(
            mask,
            weights[:, 1:].double() * value.unsqueeze(1) / price_now,
            torch.zeros_like(price_now),
        )
        previous_entry = self._entry.double()
        blended_entry = (
            previous_entry * previous_shares + (shares - previous_shares) * price_now
        ) / shares.clamp(min=1e-300)
        entry = torch.where(
            shares * price_now > self.MIN_POSITION_VALUE,
            torch.where(shares > previous_shares + 1e-6, blended_entry, previous_entry),
            torch.zeros_like(previous_entry),
        )
        entry = torch.where(mask, entry, torch.zeros_like(entry)).float()

        # step_time: value at the next open; unpriced final-bar assets keep
        # their previous price.
        next_step = self._step + 1
        final = next_step >= self.episode_length
        next_positions = positions + 1
        live = torch.where(final.unsqueeze(1), self._final_bar_live(mask), mask)
        price_next = self._open[day, next_positions].double()
        valuation = torch.where(live, price_next, price_now)
        next_value = cash + (shares * valuation).sum(dim=1)
        growth_valid = torch.isfinite(next_value) & (next_value > 0.0) & (value > 0.0)
        rewards = torch.where(
            growth_valid,
            self.reward_scale * torch.log(torch.where(growth_valid, next_value / value, torch.ones_like(value))),
            torch.zeros_like(value),
        )

        self._cash = cash
        self._shares = shares
        self._entry = entry
        self._value = next_value
        self._prev_action = weights.clone()
        self._step = next_step
        self._has_previous[:] = True

        asset_returns = price_next / price_now - 1.0
        price_valid = torch.isfinite(asset_returns) & (price_now > 0.0) & (price_next > 0.0)
        unit_returns = self.reward_scale * asset_returns
        component_values = (
            unit_returns
            if self.component_outcome_kind == "scaled_unit_asset_return"
            else weights[:, 1:].double() * unit_returns
        ).clamp(-self.component_target_clip, self.component_target_clip)
        component_mask = mask & live & price_valid & torch.isfinite(component_values)
        if not self.component_outcomes_active:
            component_mask = torch.zeros_like(component_mask)
        component_values = torch.where(component_mask, component_values, torch.zeros_like(component_values))

        observation_tensors = self._observation_tensors(index, next_positions, live)
        observations = self._numpy_observations(observation_tensors, count)
        rewards_host = rewards.cpu().numpy()
        dones = final.cpu().numpy()
        component_host = component_values.float().unsqueeze(-1).cpu().numpy()
        component_mask_host = component_mask.unsqueeze(-1).cpu().numpy()
        if self.relabel_active:
            relabel_market = self._snapshot[day, positions].cpu().numpy().astype(self.relabel_dtype)
            relabel_next_market = self._snapshot[day, next_positions].cpu().numpy().astype(self.relabel_dtype)
            relabel_returns = torch.where(
                price_valid, asset_returns, torch.zeros_like(asset_returns)
            ).float().cpu().numpy()
            relabel_valid = price_valid.cpu().numpy()
            partial_counts = np.where(dones, self.FINAL_BAR_ROWS, -1)

        infos = []
        for env_index in range(count):
            info = {
                "market_input_contract": self.market_input_contract,
                "component_outcomes": component_host[env_index],
                "component_outcome_mask": component_mask_host[env_index],
                "component_outcome_target_kind": self.component_outcome_kind,
                "reward": float(rewards_host[env_index]),
            }
            if self.relabel_active:
                info["relabel_market"] = relabel_market[env_index]
                info["relabel_next_market"] = relabel_next_market[env_index]
                info["relabel_asset_returns"] = relabel_returns[env_index]
                info["relabel_price_valid"] = relabel_valid[env_index]
                info["relabel_next_partial_count"] = int(partial_counts[env_index])
            infos.append(info)

        self._episode_return += rewards_host
        self._episode_length += 1
        finished = np.flatnonzero(dones)
        if finished.size:
            elapsed = round(time.time() - self.t_start, 6)
            for env_index in finished:
                infos[env_index]["terminal_observation"] = {
                    key: value[env_index].copy() for key, value in observations.items()
                }
                episode = {
                    "r": round(float(self._episode_return[env_index]), 6),
                    "l": int(self._episode_length[env_index]),
                    "t": elapsed,
                }
                infos[env_index]["episode"] = episode
                self._episode_rewards.append(episode["r"])
                self._episode_lengths.append(episode["l"])
                self._episode_times.append(episode["t"])
                if self._results_writer is not None:
                    self._results_writer.write_row(episode)
            self._reset_envs(finished)
            reset_observations = self.observe(finished)
            for key, value in reset_observations.items():
                observations[key][finished] = value
        return observations, rewards_host.astype(np.float32), dones, infos

    def get_episode_rewards(self):
        return list(self._episode_rewards)

    def get_episode_lengths(self):
        return list(self._episode_lengths)

    def get_episode_times(self):
        return list(self._episode_times)

    def close(self):
        if self._results_writer is not None:
            self._results_writer.close()
            self._results_writer = None

    def get_attr(self, attr_name, indices=None):
        count = len(list(self._get_indices(indices)))
        return [getattr(self._template_env, attr_name)] * count

    def set_attr(self, attr_name, value, indices=None):
        setattr(self._template_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        count = len(list(self._get_indices(indices)))
        method = getattr(self._template_env, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(count)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(list(self._get_indices(indices)))

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        return [seed] * self.num_envs
