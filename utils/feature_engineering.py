import pandas as pd
import numpy as np
import logging

# Assuming setup_logging is in a sibling 'logging.py' or accessible
# from .logging import setup_logging
# For simplicity, using a basic logger here if the import path is complex.
logger = logging.getLogger(__name__)

# --- Standalone Indicator Calculation Functions ---
# These functions are self-contained and can be used by any part of the application.

def calculate_atr(data, period=14):
    """Calculates Average True Range (ATR)."""
    if not isinstance(data, pd.DataFrame) or data.empty or not all(c in data.columns for c in ['high', 'low', 'close']) or len(data) < period:
        return 0.0
    try:
        df = data.copy()
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift()).fillna(0.0)
        low_close = abs(df['low'] - df['close'].shift()).fillna(0.0)
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        if tr.empty:
            return 0.0
        atr = tr.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        return 0.0 if pd.isna(atr) or np.isinf(atr) else float(atr)
    except Exception as e:
        logger.warning(f"ATR calculation error: {e}. Returning 0.0")
        return 0.0

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (RSI)."""
    if not isinstance(data, pd.DataFrame) or data.empty or 'close' not in data.columns or len(data) < window:
        return 50.0
    try:
        df = data.copy()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).fillna(0.0)
        loss = -delta.where(delta < 0, 0.0).fillna(0.0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        if avg_loss.empty or pd.isna(avg_loss.iloc[-1]) or avg_loss.iloc[-1] == 0:
            rs = float('inf') if not avg_gain.empty and pd.notna(avg_gain.iloc[-1]) and avg_gain.iloc[-1] > 0 else 0.0
        else:
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return 50.0 if pd.isna(rsi) or np.isinf(rsi) else float(rsi)
    except Exception as e:
        logger.warning(f"RSI calculation error: {e}. Returning 50.0")
        return 50.0

def bollinger_band_width(data, window=20, std_dev_multiplier=2.0):
    """Calculates Bollinger Band Width."""
    if not isinstance(data, pd.DataFrame) or data.empty or 'close' not in data.columns or len(data) < window:
        return 0.0
    try:
        df = data.copy()
        ma_series = df['close'].rolling(window=window).mean()
        std_series = df['close'].rolling(window=window).std()
        if ma_series.empty or std_series.empty or pd.isna(ma_series.iloc[-1]) or pd.isna(std_series.iloc[-1]) or ma_series.iloc[-1] == 0:
            return 0.0
        ma = ma_series.iloc[-1]
        std = std_series.iloc[-1]
        upper = ma + std_dev_multiplier * std
        lower = ma - std_dev_multiplier * std
        bb_width = (upper - lower) / ma
        return 0.0 if pd.isna(bb_width) or np.isinf(bb_width) else float(bb_width)
    except Exception as e:
        logger.warning(f"Bollinger Band Width calculation error: {e}. Returning 0.0")
        return 0.0

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD, Signal Line, and Histogram."""
    if not isinstance(data, pd.DataFrame) or data.empty or 'close' not in data.columns or len(data) < slow_period:
        return 0.0, 0.0, 0.0
    try:
        df = data.copy()
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line_series = ema_fast - ema_slow
        signal_line_series = macd_line_series.ewm(span=signal_period, adjust=False).mean()
        histogram_series = macd_line_series - signal_line_series

        if macd_line_series.empty or signal_line_series.empty or histogram_series.empty:
            return 0.0, 0.0, 0.0

        macd_val = macd_line_series.iloc[-1]
        signal_val = signal_line_series.iloc[-1]
        hist_val = histogram_series.iloc[-1]
        return (
            float(macd_val) if pd.notna(macd_val) and np.isfinite(macd_val) else 0.0,
            float(signal_val) if pd.notna(signal_val) and np.isfinite(signal_val) else 0.0,
            float(hist_val) if pd.notna(hist_val) and np.isfinite(hist_val) else 0.0
        )
    except Exception as e:
        logger.warning(f"MACD calculation error: {e}. Returning 0,0,0")
        return 0.0, 0.0, 0.0

# --- Main Feature Extraction Function ---

def extract_ml_features(
    symbol, data, signal_direction, bot_instance, dxy_data=None
):
    """
    Extracts a feature vector for the ML model.
    This function is now decoupled from the main TradingBot class.

    Args:
        symbol (str): The trading symbol.
        data (pd.DataFrame): The OHLCV data for the symbol.
        signal_direction (str): 'buy' or 'sell'.
        bot_instance (TradingBot): The main bot instance to access config and strategies.
        dxy_data (pd.DataFrame, optional): DXY data for correlation.

    Returns:
        list: A list of floats representing the feature vector, or None if extraction fails.
    """
    default_feature_vector_length = 34

    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.warning(f"ML Features ({symbol} {signal_direction}): Input data is empty or not a DataFrame.")
        return [0.0] * default_feature_vector_length

    if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
        try:
            data['time'] = pd.to_datetime(data['time'])
        except Exception:
            pass

    data_copy = data.copy()
    current_close_price = data_copy['close'].iloc[-1] if not data_copy.empty and 'close' in data_copy.columns and pd.notna(data_copy['close'].iloc[-1]) else 0.0

    config = bot_instance.config
    feat_eng_cfg = config.get('feature_engineering', {})
    sessions_config = config.get('trading_sessions', {})

    # Initialize all feature variables
    adx_value_feat, plus_di_feat, minus_di_feat, di_diff_feat = 0.0, 0.0, 0.0, 0.0
    price_kc_upper_dist_feat, price_kc_lower_dist_feat, kc_band_width_feat = 0.0, 0.0, 0.0
    snr_dist_a, snr_dist_v, snr_fresh_a, snr_fresh_v, snr_strength = 0.0, 0.0, 0.0, 0.0, 0.0
    volatility_range_norm = 0.0
    roc_rsi_5, roc_rsi_10, roc_adx_5, roc_adx_10 = 0.0, 0.0, 0.0, 0.0
    hour_of_day, day_of_week = 0.0, 0.0
    dxy_correlation_feature = 0.0
    is_asian_session, is_london_session, is_ny_session = 0.0, 0.0, 0.0

    # Base indicator calculations
    atr_val = calculate_atr(data_copy)
    rsi_val = calculate_rsi(data_copy)
    macd_line, signal_line_val, histogram_val = calculate_macd(data_copy)
    macd_adj = (macd_line / atr_val) if atr_val > 0 else 0.0
    hist_adj = (histogram_val / atr_val) if atr_val > 0 else 0.0

    # Indicator features from strategies
    adx_strategy = bot_instance._adx_strategy_for_features
    if adx_strategy:
        adx_indicators = adx_strategy.get_indicator_values(data_copy)
        if adx_indicators and adx_indicators.get('adx') is not None and not adx_indicators['adx'].empty:
            adx_value_feat = adx_indicators['adx'].iloc[-1]
            plus_di_feat = adx_indicators['plus_di'].iloc[-1]
            minus_di_feat = adx_indicators['minus_di'].iloc[-1]
            di_diff_feat = plus_di_feat - minus_di_feat

    keltner_strategy = bot_instance._keltner_strategy_for_features
    if keltner_strategy:
        kc_indicators = keltner_strategy.get_indicator_values(data_copy)
        if kc_indicators and kc_indicators.get('upper_band') is not None and not kc_indicators['upper_band'].empty:
            kc_upper, kc_middle, kc_lower = kc_indicators['upper_band'].iloc[-1], kc_indicators['middle_band'].iloc[-1], kc_indicators['lower_band'].iloc[-1]
            if current_close_price != 0 and pd.notna(kc_upper) and kc_upper != 0: price_kc_upper_dist_feat = (current_close_price - kc_upper) / kc_upper
            if current_close_price != 0 and pd.notna(kc_lower) and kc_lower != 0: price_kc_lower_dist_feat = (current_close_price - kc_lower) / kc_lower
            if pd.notna(kc_middle) and kc_middle != 0: kc_band_width_feat = (kc_upper - kc_lower) / kc_middle

    # MalaysianSNR Features
    snr_strategy = next((s for s in bot_instance.strategies if s.name == "MalaysianSnR"), None)
    if snr_strategy and 'time' in data_copy.columns:
         _, snr_strength_val, snr_features_dict = snr_strategy.get_signal(data_copy, symbol, "M5", return_features=True)
         if snr_features_dict:
             snr_dist_a, snr_dist_v = snr_features_dict.get('dist_a', 0.0), snr_features_dict.get('dist_v', 0.0)
             snr_fresh_a, snr_fresh_v = float(snr_features_dict.get('fresh_a', 0)), float(snr_features_dict.get('fresh_v', 0))
             snr_strength = float(snr_strength_val) if pd.notna(snr_strength_val) else 0.0

    # Rate of Change features
    for period in feat_eng_cfg.get('rate_of_change_periods', []):
        if len(data_copy) > period:
            past_rsi_val = calculate_rsi(data_copy.iloc[:-period])
            if pd.notna(rsi_val) and pd.notna(past_rsi_val):
                if period == 5: roc_rsi_5 = (rsi_val - past_rsi_val) / period
                if period == 10: roc_rsi_10 = (rsi_val - past_rsi_val) / period

            if adx_strategy:
                past_adx_indicators = adx_strategy.get_indicator_values(data_copy.iloc[:-period])
                if past_adx_indicators and past_adx_indicators.get('adx') is not None and not past_adx_indicators['adx'].empty:
                    adx_past_val = past_adx_indicators['adx'].iloc[-1]
                    if pd.notna(adx_value_feat) and pd.notna(adx_past_val):
                        if period == 5: roc_adx_5 = (adx_value_feat - adx_past_val) / period
                        if period == 10: roc_adx_10 = (adx_value_feat - adx_past_val) / period

    # Time-based features
    if feat_eng_cfg.get('time_based_features') and 'time' in data_copy.columns:
        last_time = data_copy['time'].iloc[-1]
        if pd.notna(last_time):
            hour_of_day = float(last_time.hour)
            day_of_week = float(last_time.weekday())
            # Session features
            # This logic can be simplified if bot_instance has a helper function
            # For now, replicate logic here.
            current_time_utc = last_time.time()
            all_sessions_def = sessions_config.get('sessions', {})
            for session_name, session_times in all_sessions_def.items():
                try:
                    start_time = pd.to_datetime(session_times['start']).time()
                    end_time = pd.to_datetime(session_times['end']).time()
                    in_session = (start_time <= current_time_utc <= end_time) if start_time <= end_time else (current_time_utc >= start_time or current_time_utc <= end_time)
                    if in_session:
                        if session_name == 'asian': is_asian_session = 1.0
                        elif session_name == 'london': is_london_session = 1.0
                        elif session_name == 'ny': is_ny_session = 1.0
                except (ValueError, KeyError):
                    continue


    # DXY Correlation feature
    if symbol.upper() == 'XAUUSDM' and feat_eng_cfg.get('dxy_correlation_for_xauusd') and dxy_data is not None and not dxy_data.empty:
        if 'close' in dxy_data.columns and len(dxy_data['close']) > 1 and dxy_data['close'].iloc[0] != 0:
            dxy_correlation_feature = (dxy_data['close'].iloc[-1] - dxy_data['close'].iloc[0]) / dxy_data['close'].iloc[0]

    # Volatility range
    if not data_copy.empty and all(c in data.columns for c in ['high','low','close']) and current_close_price != 0:
         volatility_range_norm = (data_copy['high'].max() - data_copy['low'].min()) / current_close_price

    # Assemble feature vector
    features_list = [
        data_copy['close'].pct_change().mean() if 'close' in data_copy and not data_copy['close'].pct_change().empty else 0.0,
        1.0 if signal_direction == 'buy' else 0.0,
        data_copy['tick_volume'].mean() if 'tick_volume' in data_copy and not data_copy['tick_volume'].empty else 0.0,
        rsi_val,
        atr_val,
        bollinger_band_width(data_copy),
        macd_adj,
        hist_adj,
        snr_dist_a, snr_dist_v, snr_fresh_a, snr_fresh_v,
        snr_strength,
        volatility_range_norm,
        adx_value_feat,
        di_diff_feat,
        price_kc_upper_dist_feat, price_kc_lower_dist_feat, kc_band_width_feat,
        roc_rsi_5, roc_rsi_10,
        roc_adx_5, roc_adx_10,
        hour_of_day, day_of_week,
        dxy_correlation_feature,
        is_asian_session,
        is_london_session,
        is_ny_session
    ]

    # Sanitize and pad
    final_features = [0.0 if pd.isna(f) or np.isinf(f) else float(f) for f in features_list]
    while len(final_features) < default_feature_vector_length:
        final_features.append(0.0)

    return final_features[:default_feature_vector_length]
