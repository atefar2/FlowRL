# %%

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter
from math import ceil, floor 
import pandas as pd
import pandas_ta as pdta
import config
import time
import schedule
from datetime import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import statsmodels.api as sm
import scipy as sp
from scipy.stats import theilslopes , linregress
from scipy.ndimage import gaussian_filter1d , generic_filter1d
import scipy.signal as signal
import scipy.interpolate as si
from scipy.signal import find_peaks
from scipy import stats
import pywt
import mplcursors
import pickle
import ta as ta
import pytz
import datetime
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest ,LimitOrderRequest, ClosePositionRequest
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from IPython.display import display, clear_output
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas_ta as pdta
import config
import os
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide,OrderClass,OrderType, QueryOrderStatus, OrderStatus
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    count = 0
    alt_pkl_file_path = 0
    if alt_pkl_file_path == 0:
        trading_client = TradingClient(config.alpaca_paper_api_key, config.alpaca_paper_secret_key, paper=True)
        stock_client = StockHistoricalDataClient(config.alpaca_paper_api_key, config.alpaca_paper_secret_key)
        api = REST(config.alpaca_paper_api_key, config.alpaca_paper_secret_key, base_url=config.alpaca_paper_endpoint)
    elif alt_pkl_file_path == 1:
        trading_client = TradingClient(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key, paper=True)
        stock_client = StockHistoricalDataClient(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key)
        api = REST(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key, base_url=config.alpaca_paper_counter_endpoint)
    elif alt_pkl_file_path == 2:
        trading_client = TradingClient(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key, paper=True)
        stock_client = StockHistoricalDataClient(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key)
        api = REST(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key, base_url=config.alpaca_paper_mega_endpoint)

    account = trading_client.get_account()
    print(account)

    # Trading parameters
    time_sensev = 0.50
    time_incriment = 1
    time_incriment5 = 10
    new_L =  ["TQQQ", "SQQQ", "NVDA", "USO", "TSLA", "SCO", "UNH", "YINN", "FAZ", "XLB", "WMT", "LABU", "UVXY" ]
    symbol = new_L #tickers_L  # Your ticker list

    print('time_incriment: ', time_incriment)
    print('count: ', count)

    # Market calendar setup
    calendar = api.get_calendar()
    now = datetime.datetime.utcnow()
    today = datetime.datetime.today().date()

    # Lookback time logic — US/Eastern session boundaries (pytz handles DST)
    est_tz = pytz.timezone('US/Eastern')
    utc_tz = pytz.UTC
    MARKET_OPEN_EST = datetime.time(9, 30)   # 9:30 AM ET
    MARKET_CLOSE_EST = datetime.time(16, 0)  # 4:00 PM ET

    def est_to_utc(est_time, date):
        """Convert EST time to UTC for a specific date (handles DST automatically)"""
        # Create a datetime in EST for the given date and time
        est_dt = est_tz.localize(datetime.datetime.combine(date, est_time))
        # Convert to UTC
        utc_dt = est_dt.astimezone(utc_tz)
        return utc_dt.time()

    def get_market_hours_utc(date):
        """Get market open/close times in UTC for a specific date"""
        open_utc = est_to_utc(MARKET_OPEN_EST, date)
        close_utc = est_to_utc(MARKET_CLOSE_EST, date)
        return open_utc, close_utc

    def get_market_datetime_utc(date):
        """Get market open/close datetime objects in UTC for a specific date (handles DST automatically)"""
        try:
            # Create a datetime in EST for the given date and time
            est_dt_open = est_tz.localize(datetime.datetime.combine(date, MARKET_OPEN_EST))
            est_dt_close = est_tz.localize(datetime.datetime.combine(date, MARKET_CLOSE_EST))
        
            # Convert to UTC
            utc_dt_open = est_dt_open.astimezone(utc_tz)
            utc_dt_close = est_dt_close.astimezone(utc_tz)
        
            return utc_dt_open, utc_dt_close
        except Exception as e:
            print(f"⚠️ Error converting timezone for {date}: {e}")
            # Fallback to hardcoded values (should not happen with proper DST handling)
            # utc_open = datetime.datetime.combine(date, datetime.time(13, 30)).replace(tzinfo=utc_tz)
            # utc_close = datetime.datetime.combine(date, datetime.time(20, 0)).replace(tzinfo=utc_tz)
            # return utc_open, utc_close


    def _sorted_trading_dates(calendar) -> list:
        return sorted({pd.Timestamp(day.date).date() for day in calendar})

    def get_last_completed_trading_day(calendar):
        """Most recent session whose regular close (16:00 ET) has already passed."""
        trading_days_all = _sorted_trading_dates(calendar)
        if not trading_days_all:
            return None

        now_et = datetime.datetime.now(est_tz)
        today_et = now_et.date()

        if today_et in trading_days_all:
            _, close_utc = get_market_datetime_utc(today_et)
            if now_et.astimezone(utc_tz) >= close_utc:
                return today_et

        past_days = [d for d in trading_days_all if d < today_et]
        return past_days[-1] if past_days else None

    def get_recent_trading_days(calendar, n_days: int, offset_blocks: int = 0) -> list:
        """Return the last n_days trading dates, optionally offset by blocks of n_days.

        - calendar: Alpaca market calendar previously fetched
        - n_days: number of trading days to return per block
        - offset_blocks: 0 returns the most recent completed block, 1 the prior block, etc.
        """
        trading_days_all = _sorted_trading_dates(calendar)
        last_completed = get_last_completed_trading_day(calendar)
        if last_completed is None:
            return []

        end_index = trading_days_all.index(last_completed) + 1 - (offset_blocks * n_days)
        start_index = max(0, end_index - n_days)
        return trading_days_all[start_index:end_index]

    last_completed_date = get_last_completed_trading_day(calendar)
    last_trading_day = (
        pd.Timestamp(last_completed_date, hour=12, tz='UTC')
        if last_completed_date is not None
        else None
    )
    now = datetime.datetime.now(pytz.UTC)
    hours_since_last_noon = (
        int((now - last_trading_day).total_seconds() / 3600) - 16
        if last_trading_day is not None
        else None
    )
    print(f"Last completed trading day: {last_completed_date}")
    print(last_trading_day)
    print(hours_since_last_noon)

    look_back_days = 0
    look_back_days_vertiacl = 0

    # Limited to 100 so if past trading day 99 for getting 9:30 to 16:00 start end
    # Data collection with market hours filter
    # ✅ FIXED: Data collection with proper timezone handling for blocks of trading days
    all_df_open = []
    DAYS_PER_BLOCK = 200
    for i in range(1, 5, 1):
        print(f"\n{'='*60}")
        print(f"Processing iteration {i} (block of {DAYS_PER_BLOCK} trading days)")
        print(f"{'='*60}")

        # Get the i-th most recent block of DAYS_PER_BLOCK trading days
        trading_days = get_recent_trading_days(calendar, n_days=DAYS_PER_BLOCK, offset_blocks=i-1)
        print(f"Trading days in block {i}: {trading_days}")
        if not trading_days:
            print("No completed trading days in this block; stopping.")
            break

        for historical_date in trading_days:
            # ✅ FIXED: Get market hours for THIS SPECIFIC DATE (handles DST properly)
            utc_dt_open, utc_dt_close = get_market_datetime_utc(historical_date)

            print(f"Market hours for {historical_date}:")
            print(f"  EST: {MARKET_OPEN_EST} - {MARKET_CLOSE_EST}")
            print(f"  UTC: {utc_dt_open.time()} - {utc_dt_close.time()}")
            print(f"  UTC datetime: {utc_dt_open} - {utc_dt_close}")

            # ✅ Use proper UTC times for data request
            Histoic_start = utc_dt_open
            end_time = utc_dt_close
            print(f"Data request range: {Histoic_start} to {end_time}")

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=Histoic_start,
                end=end_time,
                timeframe=TimeFrame(time_incriment, TimeFrameUnit.Minute)
            )

            try:
                historic = stock_client.get_stock_bars(request_params)
                historic = historic.df
                historic = historic.bfill()
                historic.reset_index(inplace=True)
                historic['timestamp'] = pd.to_datetime(historic['timestamp'], unit='ms', utc=True)
                historic = historic.set_index('timestamp')

                historic = historic.drop_duplicates(keep='last')
                historic = historic.pivot(columns='symbol', values=historic.columns)

                # ✅ Filter for market hours using the correct UTC times for this date
                is_open = (historic.index >= utc_dt_open) & (historic.index <= utc_dt_close)
                df_open = historic[is_open]

                print(f"📊 Data points for {historical_date}: {len(df_open)}")
                if len(df_open) > 0:
                    print(f"   First timestamp: {df_open.index.min()}")
                    print(f"   Last timestamp: {df_open.index.max()}")
                    expected_minutes = (utc_dt_close - utc_dt_open).total_seconds() / 60
                    print(f"   Expected minutes: {expected_minutes:.0f}")
                    print(f"   Actual minutes: {len(df_open)}")
                    print(f"   Completeness: {(len(df_open) / expected_minutes):.1%}")
                else:
                    print(f"⚠️ No data found for {historical_date}")

                all_df_open.append(df_open)

            except Exception as e:
                print(f"❌ Error fetching data for {historical_date}: {e}")
                continue

    # Combine all data
    df_multiday = pd.concat(all_df_open)
    df_multiday.sort_index(inplace=True)

    # ✅ CRITICAL FIX: Remove duplicates after concatenation
    print(f"📊 Before deduplication: {df_multiday.shape}")
    df_multiday = df_multiday[~df_multiday.index.duplicated(keep='last')]
    print(f"📊 After deduplication: {df_multiday.shape}")

    # Apply backfill then forward fill to handle remaining NaNs and zeros effectively
    # This ensures continuity and avoids explicit zero-filling for missing data.
    df_multiday = df_multiday.bfill().ffill()
    print(f"📊 After bfill and ffill: {df_multiday.shape}")

def _ensure_bar_timestamp_date_column(df_restructured):
    """
    After reset_index on pivoted MultiIndex OHLC frames, the bar time may appear as
    'timestamp', 'index', or ('timestamp', '') when column levels are MultiIndex.
    """
    if "date" in df_restructured.columns:
        df_restructured["date"] = pd.to_datetime(df_restructured["date"])
        return df_restructured

    if isinstance(df_restructured.columns, pd.MultiIndex):
        for col in df_restructured.columns:
            if isinstance(col, tuple) and col[0] in ("timestamp", "index", "date"):
                out = df_restructured.copy()
                out["date"] = pd.to_datetime(out[col])
                return out

    for name in ("timestamp", "index"):
        if name in df_restructured.columns:
            out = df_restructured.rename(columns={name: "date"})
            out["date"] = pd.to_datetime(out["date"])
            return out

    raise KeyError(
        "Could not locate bar timestamp column after reset_index; "
        f"columns={list(df_restructured.columns)[:8]}"
    )


# ✅ FIXED: Restructure to match cleaned_preprocessed.csv format with robust feature creation
def restructure_to_environment_format(df_multiday):
    """
    Restructure the multi-index data to match the environment's expected format
    where coin is a column and date is a column, similar to cleaned_preprocessed.csv
    """
    print(" Restructuring data to environment format...")

    df_multiday = df_multiday.copy()
    if df_multiday.index.name is None:
        df_multiday.index.name = "timestamp"

    # Reset index to get timestamp as a column
    df_restructured = df_multiday.reset_index()
    df_restructured = _ensure_bar_timestamp_date_column(df_restructured)
    
    # Get all the symbol-specific columns
    symbol_cols = []
    for col in df_restructured.columns:
        if col != 'date' and isinstance(col, tuple) and len(col) == 2:
            symbol_cols.append(col)
    
    # Create the melted dataframe
    melted_data = []
    
    for idx, row in df_restructured.iterrows():
        # ✅ CRITICAL FIX: Extract actual datetime value, not Series
        date_val = row['date']
        if hasattr(date_val, 'iloc'):
            # If it's a Series, extract the first value
            date_val = date_val.iloc[0]
        elif hasattr(date_val, 'item'):
            # If it's a numpy scalar, convert to Python datetime
            date_val = date_val.item()
        
        # Get unique symbols from the column names
        symbols = set()
        for col in symbol_cols:
            if isinstance(col, tuple) and len(col) == 2:
                symbols.add(col[1])  # col[1] is the symbol name
        
        # Ensure we have valid symbols
        symbols = [s for s in symbols if s and s.strip()]
        
        # Create a row for each symbol
        for symbol in symbols:
            row_data = {'date': date_val, 'coin': symbol}  # Use 'coin' as column name from the start
            
            # Add all the features for this symbol
            for feature, sym in symbol_cols:
                if sym == symbol:
                    # ✅ FIX: Handle NaN values properly
                    value = row[(feature, sym)]
                    if hasattr(value, 'iloc'):
                        value = value.iloc[0]
                    elif hasattr(value, 'item'):
                        value = value.item()
                    row_data[feature] = value
            
            melted_data.append(row_data)
    
    # Convert to DataFrame
    df_final = pd.DataFrame(melted_data)
    
    # ✅ FIX: Ensure proper data types before sorting
    df_final['date'] = pd.to_datetime(df_final['date'])
    df_final['coin'] = df_final['coin'].astype(str)
    
    # Sort by date and symbol
    df_final = df_final.sort_values(['date', 'coin']).reset_index(drop=True)
    
    # ✅ CRITICAL: Check what columns actually exist in the data
    print("🔍 Analyzing available columns...")
    available_columns = df_final.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Define expected base columns (case-insensitive check)
    expected_base_cols = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    available_base_cols = []
    
    for expected_col in expected_base_cols:
        # Check for exact match first
        if expected_col in available_columns:
            available_base_cols.append(expected_col)
            print(f"✅ Found: {expected_col}")
        else:
            # Check for case variations
            found = False
            for col in available_columns:
                if col.lower() == expected_col.lower():
                    available_base_cols.append(col)  # Use the actual column name
                    print(f"✅ Found (case variation): {expected_col} -> {col}")
                    found = True
                    break
            if not found:
                print(f"❌ Missing: {expected_col}")
    
    print(f"\n📊 Available base columns: {available_base_cols}")
    
    # ✅ CRITICAL FIX: Create quoteVolume and weightedAverage BEFORE feature calculations
    # Keep these as same-bar proxies so downstream causal shift(1) logic is not double-lagged.
    if 'quoteVolume' not in df_final.columns:
        print("⚠️ Creating 'quoteVolume' column from same-bar notional volume proxy")
        if {'close', 'volume'}.issubset(df_final.columns):
            qv_proxy = df_final['close'].astype(float) * df_final['volume'].astype(float)
        elif {'open', 'volume'}.issubset(df_final.columns):
            qv_proxy = df_final['open'].astype(float) * df_final['volume'].astype(float)
        elif 'volume' in df_final.columns:
            qv_proxy = df_final['volume'].astype(float)
        else:
            qv_proxy = pd.Series(0.0, index=df_final.index)
        df_final['quoteVolume'] = qv_proxy.replace([np.inf, -np.inf], np.nan)
        if 'quoteVolume' not in available_base_cols:
            available_base_cols.append('quoteVolume')
    
    if 'weightedAverage' not in df_final.columns:
        print("⚠️ Creating 'weightedAverage' column from same-bar OHLC proxy")
        if {'open', 'high', 'low', 'close'}.issubset(df_final.columns):
            wa_proxy = (
                df_final['open'].astype(float) +
                df_final['high'].astype(float) +
                df_final['low'].astype(float) +
                df_final['close'].astype(float)
            ) / 4.0
        elif {'high', 'low', 'close'}.issubset(df_final.columns):
            wa_proxy = (
                df_final['high'].astype(float) +
                df_final['low'].astype(float) +
                df_final['close'].astype(float)
            ) / 3.0
        elif 'close' in df_final.columns:
            wa_proxy = df_final['close'].astype(float)
        else:
            wa_proxy = pd.Series(0.0, index=df_final.index)
        df_final['weightedAverage'] = wa_proxy.replace([np.inf, -np.inf], np.nan)
        if 'weightedAverage' not in available_base_cols:
            available_base_cols.append('weightedAverage')

    # Fill any residual NaNs in proxy columns causally (per coin, ffill then bfill)
    for proxy_col in ['quoteVolume', 'weightedAverage']:
        if proxy_col in df_final.columns:
            df_final[proxy_col] = (
                df_final.groupby('coin')[proxy_col]
                .transform(lambda s: s.ffill().bfill())
                .fillna(0)
            )
    
    print(f"✅ Updated available_base_cols: {available_base_cols}")
    
    # ✅ NORMALIZE: Scale down large values to match crypto scale
    print("\n📊 Normalizing data scale to match crypto data...")
    
    # # Normalize volume to crypto scale (divide by 10000 to get reasonable values)
    # if 'volume' in available_base_cols:
    #     df_final['volume'] = df_final['volume'] / 10000.0
    #     print("✅ Normalized volume (divided by 10000)")
    
    # Normalize prices to crypto scale (divide by 100 to get reasonable values)
    # price_cols = ['high', 'low', 'open', 'close']
    # for col in price_cols:
    #     if col in available_base_cols:
    #         df_final[col] = df_final[col] / 100.0
    #         print(f"✅ Normalized {col} (divided by 100)")
    
    # =========================================================================
    # NEAR-HORIZON CAUSAL FEATURE CONSTRUCTION (AUGMENTED)
    #
    # Goal:
    # - Keep strict causality (no same-bar H/L/C leakage)
    # - Keep signals near the decision horizon (no deep 15+ bar effective lag)
    # - Make vh/vl/vc informative at open_t by turning them into causal gap proxies
    #   against the previous completed bar (prev H/L/C vs current open), in sigma units.
    # - Keep the offline file asset-local. Basket-relative features are built
    #   dynamically by enviorment.py after the episode selects its active assets.
    #
    # Design:
    # - vh/vl/vc: causal "gap-z" channels at open_t:
    #     vh_t = zscore( log(high_{t-1} / open_t) )
    #     vl_t = zscore( log(low_{t-1}  / open_t) )
    #     vc_t = zscore( log(close_{t-1} / open_t) )
    #   Mean/std are estimated from past gaps only (shift(1)), so the z-score has a
    #   probability-like interpretation (event rarity) and is more stationary.
    # - weightedAverage_s: causal wa_gap vs open — zscore(log(WA_{t-1}/open_t)), same pattern as vc.
    #   wa_sigma emitted for the critic.
    # - volume_s/quoteVolume_s: causal-surprise zscore(log(vol_{t-1}/baseline)), sigma-normalized.
    #   volume_sigma and quoteVolume_sigma emitted for the critic.
    # - open_s_1m: immediate open-to-open return (preserved).
    # - open_s and its descendants are intentionally not emitted offline.
    # - Light NEAR_HORIZON_SMOOTHING on all distributional features to reduce step noise.
    # - Historical full-universe Beta/RS/open_s code is retained below as
    #   comments for provenance, but must not execute for random 8-of-N baskets.
    # =========================================================================

    ACT_AT_OPEN = bool(getattr(config, "ACT_AT_OPEN", True))
    NEAR_HORIZON_SMOOTHING = 5
    CAUSAL_ROLLING_WINDOW = 15  # Rolling window for vh/vl/vc (matches live_model_integration)
    PRED_WINDOW = 30            # Rolling window for distributional intrabar forecast
    PRED_MIN = 10               # Minimum periods for distributional forecast rolling stats
    SIGMA_CLIP = 1.5            # Clip predictions to +/- this many sigma (price features)
    VOL_SIGMA_CLIP = 1.0        # Tighter sigma clip for volume (more fat-tailed distribution)
    QVOL_SIGMA_CLIP = 1.25      # Slightly wider clip for dollar-volume channel

    df_final = df_final.sort_values(['coin', 'date']).reset_index(drop=True)

    # Gap-z windows:
    # We intentionally normalize the "gap to previous bar" signals in sigma-units to reduce
    # heteroskedasticity and make the signals closer to stationary across regimes.
    GAP_Z_WINDOW = int(getattr(config, "GAP_Z_WINDOW", 60))
    GAP_Z_MIN = int(getattr(config, "GAP_Z_MIN", 20))
    GAP_Z_CLIP = float(getattr(config, "GAP_Z_CLIP", 4.0))
    GAP_SMOOTHING = int(getattr(config, "GAP_SMOOTHING", 1))

    VOL_Z_WINDOW = int(getattr(config, "VOL_Z_WINDOW", 60))
    VOL_Z_MIN = int(getattr(config, "VOL_Z_MIN", 20))
    VOL_Z_CLIP = float(getattr(config, "VOL_Z_CLIP", 4.0))
    QVOL_Z_WINDOW = int(getattr(config, "QVOL_Z_WINDOW", 60))
    QVOL_Z_MIN = int(getattr(config, "QVOL_Z_MIN", 20))
    QVOL_Z_CLIP = float(getattr(config, "QVOL_Z_CLIP", 4.0))
    WA_Z_WINDOW = int(getattr(config, "WA_Z_WINDOW", 60))
    WA_Z_MIN = int(getattr(config, "WA_Z_MIN", 20))
    WA_Z_CLIP = float(getattr(config, "WA_Z_CLIP", 4.0))

    print(f"\n📊 Computing NEAR-HORIZON causal features...")
    print(f"   vh/vl/vc: causal gap-z vs prev bar H/L/C over current open — window={GAP_Z_WINDOW}, clip={GAP_Z_CLIP}σ")
    print(f"   weightedAverage_s: wa_gap vs open — zscore(log(WA_{{t-1}}/open_t)), window={WA_Z_WINDOW}, clip={WA_Z_CLIP}σ, wa_sigma emitted")
    print(f"   volume_s/quoteVolume_s: causal-surprise zscore(log(vol/baseline)) — window={VOL_Z_WINDOW}/{QVOL_Z_WINDOW}, clip={VOL_Z_CLIP}/{QVOL_Z_CLIP}σ")
    print(f"   Smoothing={NEAR_HORIZON_SMOOTHING}, GAP_SMOOTHING={GAP_SMOOTHING}")
    print(f"   Episode-relative Beta/RS/open_s deferred to PortfolioEnv")

    # HISTORICAL FULL-UNIVERSE IMPLEMENTATION — intentionally disabled.
    # The mean below included assets that were absent from a random 8-of-N
    # episode, so visible token features depended on unobserved assets.
    # print("   📊 Calculating Market Index for Beta/RS...")
    # try:
    #     pivoted_open = df_final.pivot(index='date', columns='coin', values='open')
    #     all_returns = np.log(pivoted_open / pivoted_open.shift(1)).fillna(0)
    #     market_returns = all_returns.mean(axis=1)
    # except Exception as e:
    #     print(f"   ⚠️ Error calculating market index: {e}. Beta/RS will be 0.")
    #     all_returns = pd.DataFrame()
    #     market_returns = pd.Series()

    # Must include config.SIGMA_GOVERNED_BASES signal names for sigma-aware critic logic.
    # mu_* channels expose raw rolling distribution means so the agent sees mean and spread separately.
    # vh/vl/vc/volume_s/weightedAverage_s + mu_* roll_7/14/30 variants remain available.
    base_features_for_rolling = [
        'vh', 'vl', 'vc', 'volume_s',
        'quoteVolume_s', 'weightedAverage_s', 'bar_range',
        'mu_h', 'mu_l', 'mu_c', 'mu_wa', 'mu_vol',
    ]
    _roll_windows = getattr(config, 'ROLLING_WINDOWS', ['7', '14', '30'])
    ROLLING_WINDOWS_INT = [int(w) for w in _roll_windows]

    for coin_idx, coin in enumerate(df_final['coin'].unique()):
        coin_mask = df_final['coin'] == coin
        coin_data = df_final[coin_mask].copy()
        coin_data_indexed = coin_data.set_index('date').sort_index()
        n = len(coin_data_indexed)

        if n < 6:
            continue

        open_1m = coin_data_indexed['open'].astype(float).replace(0, np.nan)
        high_1m = coin_data_indexed['high'].astype(float).replace(0, np.nan)
        low_1m = coin_data_indexed['low'].astype(float).replace(0, np.nan)
        close_1m = coin_data_indexed['close'].astype(float).replace(0, np.nan)
        volume_1m = coin_data_indexed['volume'].astype(float).replace(0, np.nan)
        quote_1m = coin_data_indexed['quoteVolume'].astype(float).replace(0, np.nan)
        weighted_1m = coin_data_indexed['weightedAverage'].astype(float).replace(0, np.nan)

        # HISTORICAL FULL-UNIVERSE IMPLEMENTATION — intentionally disabled.
        # if not market_returns.empty and coin in all_returns.columns:
        #     coin_rets_now = all_returns[coin].reindex(coin_data_indexed.index).fillna(0)
        #     mkt_rets_now = market_returns.reindex(coin_data_indexed.index).fillna(0)
        # else:
        #     coin_rets_now = pd.Series(0.0, index=coin_data_indexed.index)
        #     mkt_rets_now = pd.Series(0.0, index=coin_data_indexed.index)

        # Causal minute-to-minute open return (most immediate timing signal)
        open_s_1m = np.log(open_1m / open_1m.shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
        df_final.loc[coin_mask, 'open_s_1m'] = open_s_1m.values

        # Regime tilt — computed ONCE, shared by all three distributional forecasts below
        # (H/L/C, weightedAverage, volume). Still causal: uses only shift(1) open returns.
        open_vol = open_s_1m.shift(1).rolling(30, min_periods=10).std().replace(0, np.nan)
        open_vol_safe = (
            open_vol.where(open_vol > 1e-9)
            .fillna(open_s_1m.abs().rolling(30, min_periods=5).mean())
            .replace(0, np.nan)
            .fillna(1e-6)
        )
        mom_fast = open_s_1m.shift(1).rolling(5, min_periods=1).mean()
        mom_slow  = open_s_1m.shift(1).rolling(15, min_periods=1).mean()
        regime_z  = ((mom_fast - mom_slow) / open_vol_safe).clip(-2.0, 2.0).fillna(0)

        # Causal predictive auxiliaries:
        # - reversal_z: recent open shock mean-reversion pressure
        # - clv_prev/range_z: prior intrabar close-location and range regime
        open_mu = open_s_1m.shift(1).rolling(30, min_periods=10).mean()
        open_z = ((open_s_1m - open_mu) / open_vol_safe).clip(-3.0, 3.0).fillna(0)
        reversal_z = -open_z

        # HISTORICAL FULL-UNIVERSE IMPLEMENTATION — intentionally disabled.
        # rel_open = (coin_rets_now - mkt_rets_now).fillna(0)
        # rel_open_z = (rel_open / open_vol_safe).clip(-3.0, 3.0).fillna(0)

        range_den = (high_1m - low_1m).replace(0, np.nan)
        clv = (((close_1m - low_1m) - (high_1m - close_1m)) / range_den).clip(-1.0, 1.0)
        clv = clv.replace([np.inf, -np.inf], np.nan).fillna(0)
        clv_prev = clv.shift(1).fillna(0)

        intrabar_range = np.log(high_1m / low_1m).replace([np.inf, -np.inf], np.nan)
        range_mu = intrabar_range.shift(1).rolling(30, min_periods=10).mean()
        range_sd = intrabar_range.shift(1).rolling(30, min_periods=10).std().replace(0, np.nan)
        range_z = ((intrabar_range.shift(1) - range_mu) / range_sd).clip(-2.0, 2.0).fillna(0)

        # Causal H/L/C context at decision time — GAP-Z NORMALIZATION
        #
        # We act at open_t. The strongest causal proxy for the unobserved same-bar close/open
        # ratio is the previous-close vs current-open "gap":
        #   gap_c_t = log(close_{t-1} / open_t)
        #
        # Similarly, previous high/low anchored against the current open provide causal
        # support/resistance distances:
        #   gap_h_t = log(high_{t-1} / open_t)
        #   gap_l_t = log(low_{t-1} / open_t)
        #
        # To reduce heteroskedasticity, we convert these gaps into sigma-units using ONLY
        # past gaps to estimate mean/std (shift(1)):
        #   gap_z_t = (gap_t - mean(gap)_{t-1}) / std(gap)_{t-1}
        #
        # This makes |gap_z| approximately interpretable as "event rarity" and tends to be
        # much easier for the critic to learn from than raw bp-level gaps.
        if ACT_AT_OPEN:
            # Gap primitives (strictly causal at open_t; uses previous completed bar)
            open_base = open_1m.replace(0, np.nan)
            gap_h = np.log(high_1m.shift(1) / open_base).replace([np.inf, -np.inf], np.nan)
            gap_l = np.log(low_1m.shift(1) / open_base).replace([np.inf, -np.inf], np.nan)
            gap_c = np.log(close_1m.shift(1) / open_base).replace([np.inf, -np.inf], np.nan)

            # Rolling mean/std estimated from PAST gaps only (shift(1)) to avoid any lookahead.
            mu_h_main = gap_h.shift(1).rolling(GAP_Z_WINDOW, min_periods=GAP_Z_MIN).mean()
            mu_l_main = gap_l.shift(1).rolling(GAP_Z_WINDOW, min_periods=GAP_Z_MIN).mean()
            mu_c_main = gap_c.shift(1).rolling(GAP_Z_WINDOW, min_periods=GAP_Z_MIN).mean()

            sd_h_main = gap_h.shift(1).rolling(GAP_Z_WINDOW, min_periods=GAP_Z_MIN).std()
            sd_l_main = gap_l.shift(1).rolling(GAP_Z_WINDOW, min_periods=GAP_Z_MIN).std()
            sd_c_main = gap_c.shift(1).rolling(GAP_Z_WINDOW, min_periods=GAP_Z_MIN).std()

            # Warmup fallbacks (shorter windows so values exist quickly).
            mu_h_fb = gap_h.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean()
            mu_l_fb = gap_l.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean()
            mu_c_fb = gap_c.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean()

            sd_h_fb = gap_h.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=3).std()
            sd_l_fb = gap_l.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=3).std()
            sd_c_fb = gap_c.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=3).std()

            mu_h_eff = mu_h_main.where(mu_h_main.notna(), mu_h_fb)
            mu_l_eff = mu_l_main.where(mu_l_main.notna(), mu_l_fb)
            mu_c_eff = mu_c_main.where(mu_c_main.notna(), mu_c_fb)

            sd_h_eff = sd_h_main.where(sd_h_main.notna(), sd_h_fb)
            sd_l_eff = sd_l_main.where(sd_l_main.notna(), sd_l_fb)
            sd_c_eff = sd_c_main.where(sd_c_main.notna(), sd_c_fb)

            # Numerical safety: std must be positive; fallback to open_vol_safe for scale.
            sd_h_safe = sd_h_eff.where(sd_h_eff > 1e-9).fillna(open_vol_safe).fillna(1e-6)
            sd_l_safe = sd_l_eff.where(sd_l_eff > 1e-9).fillna(open_vol_safe).fillna(1e-6)
            sd_c_safe = sd_c_eff.where(sd_c_eff > 1e-9).fillna(open_vol_safe).fillna(1e-6)

            # Sigma-unit gaps (probability-like scale). Clip to reduce tail risk for function approximation.
            vh_raw = ((gap_h - mu_h_eff) / sd_h_safe).clip(-GAP_Z_CLIP, GAP_Z_CLIP)
            vl_raw = ((gap_l - mu_l_eff) / sd_l_safe).clip(-GAP_Z_CLIP, GAP_Z_CLIP)
            vc_raw = ((gap_c - mu_c_eff) / sd_c_safe).clip(-GAP_Z_CLIP, GAP_Z_CLIP)

            # If any z-score is still missing (e.g., first row), fall back to a scaled raw gap.
            vh_fb_z = (gap_h / open_vol_safe).clip(-GAP_Z_CLIP, GAP_Z_CLIP)
            vl_fb_z = (gap_l / open_vol_safe).clip(-GAP_Z_CLIP, GAP_Z_CLIP)
            vc_fb_z = (gap_c / open_vol_safe).clip(-GAP_Z_CLIP, GAP_Z_CLIP)
            vh_raw = vh_raw.where(vh_raw.notna(), vh_fb_z)
            vl_raw = vl_raw.where(vl_raw.notna(), vl_fb_z)
            vc_raw = vc_raw.where(vc_raw.notna(), vc_fb_z)

            # Expose rolling mean/scale as independent channels (mean + spread).
            mu_h_obs = mu_h_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
            mu_l_obs = mu_l_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
            mu_c_obs = mu_c_eff.replace([np.inf, -np.inf], np.nan).fillna(0)

            vh_sigma = sd_h_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
            vl_sigma = sd_l_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
            vc_sigma = sd_c_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            vh_raw = np.log(high_1m / open_1m)
            vl_raw = np.log(low_1m / open_1m)
            vc_raw = np.log(close_1m / open_1m)
            mu_h_obs = vh_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            mu_l_obs = vl_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            mu_c_obs = vc_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            vh_sigma = pd.Series(0.0, index=coin_data_indexed.index)
            vl_sigma = pd.Series(0.0, index=coin_data_indexed.index)
            vc_sigma = pd.Series(0.0, index=coin_data_indexed.index)

        vh_raw = vh_raw.replace([np.inf, -np.inf], np.nan)
        vl_raw = vl_raw.replace([np.inf, -np.inf], np.nan)
        vc_raw = vc_raw.replace([np.inf, -np.inf], np.nan)

        # Near-horizon smoothed versions (for gap-z signals we keep them sharp by default)
        vh = vh_raw.rolling(GAP_SMOOTHING, min_periods=1).mean().fillna(0)
        vl = vl_raw.rolling(GAP_SMOOTHING, min_periods=1).mean().fillna(0)
        vc = vc_raw.rolling(GAP_SMOOTHING, min_periods=1).mean().fillna(0)

        # HISTORICAL FULL-UNIVERSE open_s — intentionally disabled.
        # open_s_fast = open_s_1m
        # open_s_slow = open_s_1m.rolling(NEAR_HORIZON_SMOOTHING, min_periods=1).mean()
        # open_s_rel = rel_open.rolling(3, min_periods=1).mean().fillna(0)
        # open_s_reversal = (reversal_z * open_vol_safe).replace([np.inf, -np.inf], np.nan).fillna(0)
        # open_s = (
        #     0.40 * open_s_fast +
        #     0.20 * open_s_slow +
        #     0.25 * open_s_rel +
        #     0.15 * open_s_reversal
        # ).replace([np.inf, -np.inf], 0).fillna(0)

        # -------------------------------------------------------------------------
        # weightedAverage_s — CAUSAL GAP-Z (wa_gap vs open, same pattern as vh/vl/vc)
        # At open_t we have weightedAverage_{t-1}. wa_gap_t = log(WA_{t-1} / open_t).
        # Z-score using past wa_gap values only for stationarity and cross-asset comparability.
        # -------------------------------------------------------------------------
        if ACT_AT_OPEN:
            open_base = open_1m.replace(0, np.nan)
            wa_gap = np.log(weighted_1m.shift(1) / open_base).replace([np.inf, -np.inf], np.nan)

            mu_wa_main = wa_gap.shift(1).rolling(WA_Z_WINDOW, min_periods=WA_Z_MIN).mean()
            sd_wa_main = wa_gap.shift(1).rolling(WA_Z_WINDOW, min_periods=WA_Z_MIN).std()

            mu_wa_fb = wa_gap.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean()
            sd_wa_fb = wa_gap.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=3).std()

            mu_wa_eff = mu_wa_main.where(mu_wa_main.notna(), mu_wa_fb)
            sd_wa_eff = sd_wa_main.where(sd_wa_main.notna(), sd_wa_fb)

            sd_wa_safe = sd_wa_eff.where(sd_wa_eff > 1e-9).fillna(open_vol_safe).fillna(1e-6)

            wa_raw = ((wa_gap - mu_wa_eff) / sd_wa_safe).clip(-WA_Z_CLIP, WA_Z_CLIP)
            wa_fb_z = (wa_gap / open_vol_safe).clip(-WA_Z_CLIP, WA_Z_CLIP)
            wa_raw = wa_raw.where(wa_raw.notna(), wa_fb_z)

            mu_wa_obs = mu_wa_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
            wa_sigma = sd_wa_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            wa_raw = np.log(weighted_1m / open_1m).replace([np.inf, -np.inf], np.nan)
            mu_wa_obs = wa_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            wa_sigma = pd.Series(0.0, index=coin_data_indexed.index)

        wa_raw = wa_raw.replace([np.inf, -np.inf], np.nan)
        weightedAverage_s = wa_raw.rolling(GAP_SMOOTHING, min_periods=1).mean().fillna(0)

        # -------------------------------------------------------------------------
        # volume_s / quoteVolume_s — CAUSAL-SURPRISE Z-SCORE (same pattern as vh/vl/vc)
        # vol_excess_{t-1} = log(volume_{t-1} / baseline_vol_{t-1})
        # baseline_vol_{t-1} = mean(volume_{t-2}, volume_{t-3}, ..., volume_{t-W})
        # volume_s_t = clip((vol_excess_{t-1} - mu_vol) / sigma_vol, -K, +K)
        # Same for quoteVolume_s with dollar volume. Sigma emitted for critic.
        # -------------------------------------------------------------------------
        if ACT_AT_OPEN:
            # baseline_vol_{t-1} = mean(volume_{t-2}, ..., volume_{t-W-1}) — exclude current from baseline
            share_baseline = (volume_1m.shift(2)
                              .rolling(VOL_Z_WINDOW, min_periods=1)
                              .mean()
                              .replace(0, np.nan))
            quote_baseline = (quote_1m.shift(2)
                              .rolling(QVOL_Z_WINDOW, min_periods=1)
                              .mean()
                              .replace(0, np.nan))
            vol_excess = np.log(volume_1m.shift(1) / share_baseline).replace([np.inf, -np.inf], np.nan)
            qvol_excess = np.log(quote_1m.shift(1) / quote_baseline).replace([np.inf, -np.inf], np.nan)

            mu_vol_main = vol_excess.shift(1).rolling(VOL_Z_WINDOW, min_periods=VOL_Z_MIN).mean()
            sd_vol_main = vol_excess.shift(1).rolling(VOL_Z_WINDOW, min_periods=VOL_Z_MIN).std()
            mu_qv_main = qvol_excess.shift(1).rolling(QVOL_Z_WINDOW, min_periods=QVOL_Z_MIN).mean()
            sd_qv_main = qvol_excess.shift(1).rolling(QVOL_Z_WINDOW, min_periods=QVOL_Z_MIN).std()

            mu_vol_fb = vol_excess.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean()
            sd_vol_fb = vol_excess.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=3).std()
            mu_qv_fb = qvol_excess.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean()
            sd_qv_fb = qvol_excess.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=3).std()

            mu_vol_eff = mu_vol_main.where(mu_vol_main.notna(), mu_vol_fb)
            sd_vol_eff = sd_vol_main.where(sd_vol_main.notna(), sd_vol_fb)
            mu_qv_eff = mu_qv_main.where(mu_qv_main.notna(), mu_qv_fb)
            sd_qv_eff = sd_qv_main.where(sd_qv_main.notna(), sd_qv_fb)

            sd_vol_safe = sd_vol_eff.where(sd_vol_eff > 1e-9).fillna(open_vol_safe).fillna(1e-6)
            sd_qv_safe = sd_qv_eff.where(sd_qv_eff > 1e-9).fillna(open_vol_safe).fillna(1e-6)

            vol_raw = ((vol_excess - mu_vol_eff) / sd_vol_safe).clip(-VOL_Z_CLIP, VOL_Z_CLIP)
            qv_raw = ((qvol_excess - mu_qv_eff) / sd_qv_safe).clip(-QVOL_Z_CLIP, QVOL_Z_CLIP)

            vol_fb_z = (vol_excess / open_vol_safe).clip(-VOL_Z_CLIP, VOL_Z_CLIP)
            qv_fb_z = (qvol_excess / open_vol_safe).clip(-QVOL_Z_CLIP, QVOL_Z_CLIP)
            vol_raw = vol_raw.where(vol_raw.notna(), vol_fb_z)
            qv_raw = qv_raw.where(qv_raw.notna(), qv_fb_z)

            mu_vol_obs = mu_vol_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
            volume_sigma = sd_vol_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
            quoteVolume_sigma = sd_qv_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            share_baseline = volume_1m.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean().replace(0, np.nan)
            quote_baseline = quote_1m.shift(1).rolling(CAUSAL_ROLLING_WINDOW, min_periods=1).mean().replace(0, np.nan)
            vol_raw = np.log(volume_1m.shift(1) / share_baseline).replace([np.inf, -np.inf], np.nan)
            qv_raw = np.log(quote_1m.shift(1) / quote_baseline).replace([np.inf, -np.inf], np.nan)
            mu_vol_obs = vol_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            volume_sigma = pd.Series(0.0, index=coin_data_indexed.index)
            quoteVolume_sigma = pd.Series(0.0, index=coin_data_indexed.index)

        vol_raw = vol_raw.replace([np.inf, -np.inf], np.nan)
        qv_raw = qv_raw.replace([np.inf, -np.inf], np.nan)
        volume_s = vol_raw.rolling(GAP_SMOOTHING, min_periods=1).mean().fillna(0)
        quoteVolume_s = qv_raw.rolling(GAP_SMOOTHING, min_periods=1).mean().fillna(0)

        # Causal range proxy available at open_t: use previous completed bar's log range.
        # (At t, high_{t-1} and low_{t-1} are known; using shift(1) avoids same-bar leakage.)
        if ACT_AT_OPEN:
            bar_range_raw = intrabar_range.shift(1)
        else:
            bar_range_raw = intrabar_range
        bar_range = bar_range_raw.replace([np.inf, -np.inf], np.nan).rolling(
            NEAR_HORIZON_SMOOTHING, min_periods=1
        ).mean().fillna(0)

        features_df = pd.DataFrame({
            'vh': vh, 'vl': vl, 'vc': vc,
            'mu_h': mu_h_obs, 'mu_l': mu_l_obs, 'mu_c': mu_c_obs,
            'vh_sigma': vh_sigma, 'vl_sigma': vl_sigma, 'vc_sigma': vc_sigma,
            'volume_s': volume_s,
            'quoteVolume_s': quoteVolume_s,
            'weightedAverage_s': weightedAverage_s,
            'mu_wa': mu_wa_obs, 'mu_vol': mu_vol_obs,
            'volume_sigma': volume_sigma,
            'quoteVolume_sigma': quoteVolume_sigma,
            'wa_sigma': wa_sigma,
            'bar_range': bar_range,
        }, index=coin_data_indexed.index)

        # HISTORICAL FULL-UNIVERSE beta_60 / rs_roll_14 — intentionally disabled.
        # These are now computed from the selected episode basket in enviorment.py.
        # if not market_returns.empty and coin in all_returns.columns:
        #     coin_rets = all_returns[coin].reindex(features_df.index).fillna(0)
        #     mkt_rets = market_returns.reindex(features_df.index).fillna(0)
        #     coin_rets_hist = coin_rets.shift(1).fillna(0)
        #     mkt_rets_hist = mkt_rets.shift(1).fillna(0)
        #     rolling_cov = coin_rets_hist.rolling(60, min_periods=30).cov(mkt_rets_hist)
        #     rolling_var = mkt_rets_hist.rolling(60, min_periods=30).var()
        #     raw_beta = (rolling_cov / rolling_var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        #     beta_smooth = raw_beta.ewm(span=20, min_periods=5, adjust=False).mean()
        #     features_df['beta_60'] = (0.70 * raw_beta + 0.30 * beta_smooth).clip(-3.0, 3.0).fillna(1.0)
        #     excess = (coin_rets - mkt_rets).fillna(0)
        #     rs_fast = excess.ewm(span=6, min_periods=3, adjust=False).mean()
        #     rs_slow = excess.ewm(span=20, min_periods=5, adjust=False).mean()
        #     rs_signal = 0.65 * rs_fast + 0.35 * (rs_fast - rs_slow)
        #     features_df['rs_roll_14'] = rs_signal.rolling(5, min_periods=1).mean().fillna(0) * 100.0
        # else:
        #     features_df['beta_60'] = 1.0
        #     features_df['rs_roll_14'] = 0.0

        # 3. Efficiency Ratio (14 min) - Kaufman
        # Measures trend smoothness: |Net Change| / Sum(|Step Changes|)
        # Using Open prices because we act at open
        net_change = (open_1m - open_1m.shift(14)).abs()
        sum_abs_change = (open_1m - open_1m.shift(1)).abs().rolling(14, min_periods=14).sum()
        efficiency_raw = (net_change / sum_abs_change.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        direction_14 = np.sign(open_1m - open_1m.shift(14)).fillna(0)
        features_df['efficiency_ratio_14'] = (efficiency_raw * direction_14).clip(-1.0, 1.0).fillna(0)

        # 4. Volatility Ratio (14 vs 60)
        # Expansion/Contraction indicator
        vol_14 = open_s_1m.shift(1).rolling(14, min_periods=5).std()
        vol_60 = open_s_1m.shift(1).rolling(60, min_periods=20).std()
        vol_ratio = np.log(vol_14 / vol_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        features_df['volatility_ratio'] = vol_ratio.clip(-3.0, 3.0).fillna(0)

        for col in base_features_for_rolling:
            if col in features_df.columns:
                for w in ROLLING_WINDOWS_INT:
                    features_df[f'{col}_roll_{w}'] = features_df[col].rolling(
                        window=w, min_periods=1
                    ).mean()

        # HISTORICAL open_s descendants — intentionally disabled with open_s.
        # features_df['open_s_std_14'] = features_df['open_s'].rolling(
        #     window=14, min_periods=1).std().fillna(0)
        # features_df['open_s_std_30'] = features_df['open_s'].rolling(
        #     window=30, min_periods=1).std().fillna(0)
        # if 'open_s_roll_7' in features_df.columns and 'open_s_roll_30' in features_df.columns:
        #     features_df['open_s_accel'] = features_df['open_s_roll_7'] - features_df['open_s_roll_30']
        # else:
        #     features_df['open_s_accel'] = 0.0

        features_df = features_df.fillna(0)

        for col in features_df.columns:
            df_final.loc[coin_mask, col] = features_df[col].values

        if coin_idx == 0:
            sample_idx = min(len(features_df) - 1, NEAR_HORIZON_SMOOTHING + 5)
            print(f"   [{coin}] sample features at step {sample_idx}: vh={features_df['vh'].iloc[sample_idx]:.6f}, "
                  f"open_s_1m={open_s_1m.iloc[sample_idx]:.6f}")
            print(f"   [{coin}] Rich asset-local features: ER={features_df['efficiency_ratio_14'].iloc[sample_idx]:.4f}, "
                  f"VolRatio={features_df['volatility_ratio'].iloc[sample_idx]:.4f}")

    print(f"✅ Near-horizon distributional forecast features computed (vh/vl/vc/weightedAverage_s/volume_s + mu_*)")
    print(f"✅ Preserved open_s_1m (1-minute return) for granular timing")
    print(f"✅ Added coin-local Efficiency Ratio (14m) and Vol Ratio")
    print(f"✅ Deferred episode-relative open_s/Beta/RS and relational features to PortfolioEnv")
    print(f"✅ Feature rolling windows {ROLLING_WINDOWS_INT} in units of 1-min steps")
    print(f"   roll_7 = 7 min, roll_14 = 14 min, roll_30 = 30 min")
    print(f"   Sigma governance: vh/vl/vc + mu_h/mu_l/mu_c + volume_s/mu_vol + weightedAverage_s/mu_wa + roll variants use config.SIGMA_GOVERNED_BASES")

    # ✅ FIX: Fill remaining NaNs WITHOUT lookahead (ffill only, then zeros)
    df_final = df_final.fillna(method='ffill').fillna(0)
    
    # ✅ CRITICAL FIX: Ensure we only have 'coin' column, not 'symbol'
    if 'symbol' in df_final.columns:
        # If we have both 'coin' and 'symbol', drop 'symbol' and ensure 'coin' has the right values
        if 'coin' in df_final.columns:
            # Check if 'coin' column is empty and 'symbol' has values
            if df_final['coin'].isna().all() or df_final['coin'].eq('').all():
                df_final['coin'] = df_final['symbol']
            df_final = df_final.drop(columns=['symbol'])
        else:
            # If only 'symbol' exists, rename it to 'coin'
            df_final = df_final.rename(columns={'symbol': 'coin'})
    
    # ✅ CRITICAL FIX: Ensure we have the exact same columns as cleaned_preprocessed.csv
    # Remove any extra columns that shouldn't be there
    expected_columns = ['date', 'coin', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    extra_columns_to_remove = []
    
    ALLOWED_DERIVED_PREFIXES = (
        'vh', 'vl', 'vc', 'open_s', 'volume_s',
        'quoteVolume_s', 'weightedAverage_s',
        'bar_range', 'beta', 'rs', 'efficiency', 'volatility',
        'mu_',
        'volume_sigma', 'quoteVolume_sigma', 'wa_sigma',
    )
    for col in df_final.columns:
        if col not in expected_columns and not col.startswith(ALLOWED_DERIVED_PREFIXES) and not col.endswith(('_roll_7', '_roll_14', '_roll_30')):
            extra_columns_to_remove.append(col)
    
    if extra_columns_to_remove:
        print(f"🗑️ Removing extra columns: {extra_columns_to_remove}")
        df_final = df_final.drop(columns=extra_columns_to_remove)
    
    # ✅ NOTE: quoteVolume and weightedAverage columns are now created EARLIER (before pct_change loop)
    # This ensures the _s versions are calculated correctly
    
    # ✅ CRITICAL FIX: Reorder columns to match cleaned_preprocessed.csv exactly
    # First, get all the base columns in the right order
    base_columns = ['date', 'coin', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    
    # Then get all the derived columns (vh, vl, vc, *_s, *_roll_*)
    derived_columns = [col for col in df_final.columns if col not in base_columns]
    
    # Sort derived columns to match the order in cleaned_preprocessed.csv
    derived_columns.sort()
    
    # Combine in the right order
    final_column_order = base_columns + derived_columns
    
    # Reorder the dataframe
    df_final = df_final[final_column_order]
    
    # ✅ CRITICAL FIX: Remove any duplicate coin/date combinations
    print(f"📊 Before coin/date deduplication: {df_final.shape}")
    df_final = df_final.drop_duplicates(subset=['date', 'coin'], keep='last')
    print(f"📊 After coin/date deduplication: {df_final.shape}")
    
    # Sort back by date and coin for consistency
    df_final = df_final.sort_values(['date', 'coin']).reset_index(drop=True)
    
    print(f"\n✅ Restructured data shape: {df_final.shape}")
    print(f"📅 Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"�� Coins: {df_final['coin'].unique()}")
    
    # Verify we have all required features
    all_created_features = list(base_features_for_rolling)
    for col in base_features_for_rolling:
        for window in ROLLING_WINDOWS_INT:
            all_created_features.append(f"{col}_roll_{window}")
    for standalone in ['open_s_1m', 'efficiency_ratio_14', 'volatility_ratio',
                        'vh_sigma', 'vl_sigma', 'vc_sigma',
                        'volume_sigma', 'quoteVolume_sigma', 'wa_sigma']:
        if standalone in df_final.columns:
            all_created_features.append(standalone)
    
    missing_features = [f for f in all_created_features if f not in df_final.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    else:
        print(f"✅ All {len(all_created_features)} required features created!")
    
    print(f"\n📊 Final feature count: {len(all_created_features)} features per coin")
    print(f"📊 Total features: {len(all_created_features) * len(df_final['coin'].unique())}")
    
    lead_cols = [
        c for c in [
            "date", "coin", "high", "low", "open", "close",
            "volume", "quoteVolume", "weightedAverage",
        ]
        if c in df_final.columns
    ]
    preview_features = [c for c in derived_columns if c in df_final.columns][:8]
    sample_cols = lead_cols + preview_features
    print(f"\n📊 Sample data (date, coin, OHLCV, then first derived features):")
    print(f"   columns: {sample_cols}")
    print(df_final[sample_cols].head(10))
    
    
    return df_final

# Restructure the data
if __name__ == "__main__":
    df_environment_format = restructure_to_environment_format(df_multiday)

    # Save the restructured data
    df_environment_format.to_csv("data/causal_cleaned_preprocessed_restructured_og.csv", index=True)
    print("�� Saved restructured data to 'data/causal_cleaned_preprocessed_restructured_og.csv'")

    # ✅ FIXED: Proper episode length calculation
    print("\n🔍 DEBUGGING DATA STRUCTURE:")
    print(f"Original df_multiday shape: {df_multiday.shape}")
    print(f"Restructured data shape: {df_environment_format.shape}")
    print(f"Unique dates: {df_environment_format['date'].nunique()}")
    print(f"Unique coins: {df_environment_format['coin'].nunique()}")
    print(f"Rows per coin per day: {df_environment_format.shape[0] / (df_environment_format['date'].nunique() * df_environment_format['coin'].nunique()):.1f}")

    # Analyze trading day structure
    print("\n📈 Analyzing trading day structure...")

    # Ensure only 'symbol' is present (drop 'coin' if it slipped in earlier)
    if 'coin' in df_environment_format.columns and 'coin' in df_environment_format.columns:
        df_environment_format = df_environment_format.drop(columns=['coin'])

    # Steps per day should be the count of unique minute timestamps in [13:30, 20:00] per day,
    # not multiplied by number of symbols. Count unique minutes per day.
    df_environment_format['day'] = df_environment_format['date'].dt.date
    df_environment_format['minute'] = df_environment_format['date'].dt.floor('min')

    # Count unique minutes per day (deduplicate across symbols)
    daily_counts = (
        df_environment_format[['day', 'minute']]
        .drop_duplicates()
        .groupby('day')
        .size()
    )

    print(f"📊 Trading days analyzed: {len(daily_counts)}")
    print(f"📊 Average steps per day: {daily_counts.mean():.1f}")
    print(f"📊 Median steps per day: {daily_counts.median():.1f}")
    print(f"�� Min steps per day: {daily_counts.min()}")
    print(f"�� Max steps per day: {daily_counts.max()}")
    print(f"�� Standard deviation: {daily_counts.std():.1f}")

    # Calculate expected steps based on time increment
    # From your code: time_incriment = 1 (1-minute intervals)
    expected_minutes = (datetime.datetime.combine(datetime.datetime.min, datetime.time(20, 0)) - 
                      datetime.datetime.combine(datetime.datetime.min, datetime.time(13, 30))).total_seconds() / 60
    expected_steps = int(expected_minutes)  # Should be ~390 minutes (6.5 hours)

    print(f"\n�� Expected steps per day (13:30-20:00): {expected_steps}")
    print(f"📊 Average completeness: {(daily_counts.mean() / expected_steps):.1%}")

    # Identify incomplete days
    incomplete_days = daily_counts[daily_counts < 0.95 * expected_steps]
    if len(incomplete_days) > 0:
        print(f"\n⚠️  Incomplete trading days (<95% complete): {len(incomplete_days)}")
        print("   These days may have missing data or early market closes")
        for day, steps in incomplete_days.head(5).items():
            completeness = steps / expected_steps
            print(f"   {day}: {steps}/{expected_steps} steps ({completeness:.1%})")

    # Enhanced visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Steps per day distribution
    plt.subplot(2, 2, 1)
    plt.hist(daily_counts.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(daily_counts.mean(), color='red', linestyle='--', label=f'Mean: {daily_counts.mean():.1f}')
    plt.axvline(daily_counts.median(), color='orange', linestyle='--', label=f'Median: {daily_counts.median():.1f}')
    plt.xlabel('Steps per Trading Day')
    plt.ylabel('Frequency')
    plt.title('Distribution of Steps per Trading Day')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Steps over time
    plt.subplot(2, 2, 2)
    plt.plot(daily_counts.index, daily_counts.values, marker='o', alpha=0.7)
    plt.axhline(daily_counts.mean(), color='red', linestyle='--', label=f'Mean: {daily_counts.mean():.1f}')
    plt.xlabel('Date')
    plt.ylabel('Steps')
    plt.title('Steps per Day Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 3: Completeness distribution
    completeness_ratios = daily_counts / expected_steps
    plt.subplot(2, 2, 3)
    plt.hist(completeness_ratios, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='100% Complete')
    plt.xlabel('Completeness Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Day Completeness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Box plot of steps
    plt.subplot(2, 2, 4)
    plt.boxplot(daily_counts.values)
    plt.ylabel('Steps per Day')
    plt.title('Box Plot of Steps per Day')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n🎯 Recommended episode length: {min(391, int(round(daily_counts.median())))} steps")
    print(f"📊 Data ready for RL environment training!")

    # Enhanced RL training analysis
    print("\n" + "="*60)
    print("🧠 RL TRAINING ANALYSIS")
    print("="*60)

    # Calculate what percentage of days meet different thresholds
    for threshold in [300, 350, 380, 390]:
        pct_above = (daily_counts >= threshold).mean() * 100
        print(f"📊 {pct_above:.1f}% of days have ≥{threshold} steps")

    print(f"\n📈 DATA UTILIZATION:")
    total_trading_days = len(daily_counts)
    print(f"   • Total trading days: {total_trading_days}")
    print(f"   • With random starts: ~{total_trading_days * 3} possible episodes")
    print(f"   • With sliding windows: ~{total_trading_days * 5} possible episodes")

    print(f"\n🎯 RECOMMENDED SETTINGS:")
    print(f"   • Episode length: {daily_counts.median():.0f} steps")
    print(f"   • Update config.py: FILE = 'data/causal_cleaned_preprocessed_restructured_og.csv'")
    print(f"   • Update config.py: EPISODE_LENGTH = {daily_counts.median():.0f}")

    # Additional RL training strategy analysis
    print("\n" + "="*60)
    print("🧠 RL TRAINING STRATEGY ANALYSIS")
    print("="*60)

    avg_steps = daily_counts.mean()
    median_steps = daily_counts.median()

    print("🤔 TRAINING APPROACH OPTIONS:")
    print("\n1️⃣  FIXED DAY START APPROACH:")
    print("   ✅ Pros:")
    print("      • Consistent episode structure")
    print("      • Agent learns market open patterns")
    print("      • Easier to debug and reproduce")
    print("      • Matches real trading behavior")
    print("   ❌ Cons:")
    print("      • Less data variety")
    print("      • May overfit to market open patterns")
    print("      • Limited exploration of mid-day scenarios")

    print("\n2️⃣  RANDOM START POINT APPROACH:")
    print("   ✅ Pros:")
    print("      • Maximum data utilization")
    print("      • Better exploration of different market conditions")
    print("      • More diverse training scenarios")
    print("      • Higher sample efficiency")
    print("   ❌ Cons:")
    print("      • Inconsistent episode lengths")
    print("      • May learn unrealistic transitions")
    print("      • Harder to interpret behavior")

    print("\n3️⃣  HYBRID APPROACH (RECOMMENDED):")
    print("   ✅ Best of both worlds:")
    print("      • Use average day length as episode length")
    print("      • Allow random start points within valid ranges")
    print("      • Ensure minimum episode length for learning")
    print("      • Add day boundary awareness")

    print(f"\n�� SPECIFIC RECOMMENDATIONS:")
    print(f"   • Episode length: {median_steps:.0f} steps (median)")
    print(f"   • Minimum episode length: {daily_counts.quantile(0.1):.0f} steps (10th percentile)")
    print(f"   • Maximum episode length: {daily_counts.quantile(0.9):.0f} steps (90th percentile)")

    df_multiday = pd.read_csv('data/causal_cleaned_preprocessed_restructured_og.csv')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_multiday.shape)

    df_multiday.tail(30)

     # %%
    # import pandas as pd
    # df_multiday = pd.read_csv('data/causal_cleaned_preprocessed_restructured_og.csv')
    # df_multiday.tail(30)
    # %%
