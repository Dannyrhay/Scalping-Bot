import MetaTrader5 as mt5
import pandas as pd
from utils.logging import setup_logging
import os
from .data_cache import data_cache_instance
from datetime import datetime

logger = setup_logging()

TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M3': mt5.TIMEFRAME_M3,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}

def connect_mt5(config_credentials=None):
    """
    Initialize connection to MetaTrader 5 terminal.
    """
    login_id_env = os.environ.get('MT5_LOGIN')
    password_env = os.environ.get('MT5_PASSWORD')
    server_env = os.environ.get('MT5_SERVER')
    path_env = os.environ.get('MT5_TERMINAL_PATH')

    login_id, password, server, path = None, None, None, None

    if login_id_env and password_env and server_env:
        logger.info("Using MT5 credentials from environment variables.")
        login_id, password, server, path = login_id_env, password_env, server_env, path_env
    elif config_credentials:
        logger.info("Using MT5 credentials from configuration file.")
        login_id = config_credentials.get('login')
        password = config_credentials.get('password')
        server = config_credentials.get('server')
        path = config_credentials.get('mt5_terminal_path')
    else:
        logger.error("MT5 credentials not found.")
        return False

    if not (login_id and password and server):
        logger.error("Essential MT5 credentials (login, password, server) are missing.")
        return False

    try:
        login_id = int(login_id)
    except (ValueError, TypeError):
        logger.error(f"MT5_LOGIN must be an integer. Got: {login_id}")
        return False

    mt5_path_arg = path if path and path.strip() else ""

    try:
        if not mt5.initialize(login=login_id, password=password, server=server, path=mt5_path_arg):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        logger.info("MT5 connection established successfully.")
        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to MT5: {e}", exc_info=True)
        return False

def get_data(symbol, timeframe_str, bars=None, from_date=None, to_date=None):
    """
    Fetch historical data, using a cache for efficiency. This version is optimized for scalping.
    """
    timeframe = TIMEFRAME_MAP.get(timeframe_str)
    if timeframe is None:
        logger.error(f"Invalid timeframe_str: {timeframe_str}")
        return None

    try:
        if not mt5.terminal_info():
            logger.warning("MT5 not initialized. Cannot fetch data.")
            return None

        # Direct fetch for date ranges (typically for backtesting, not live scalping)
        if from_date is not None and to_date is not None:
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data fetched for {symbol} on {timeframe_str} using date range.")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df

        if bars is None:
            logger.error("get_data for live trading requires the 'bars' argument.")
            return None

        # --- FIX IS HERE ---
        # For scalping, we only need a small buffer beyond the requested bars.
        # This significantly reduces memory usage and initial fetch time.
        max_bars_to_store = bars * 2 if bars else 200 # e.g., if config asks for 100, we store 200.

        cached_df, _ = data_cache_instance.get_cached_ohlc(symbol, timeframe_str)

        # If cache is empty or insufficient, do a full fetch
        if cached_df is None or len(cached_df) < bars:
            logger.info(f"CACHE MISS/INSUFFICIENT for {symbol} {timeframe_str}. Performing full fetch of {max_bars_to_store} bars.")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max_bars_to_store)
            if rates is None or len(rates) == 0:
                logger.error(f"No initial data fetched for {symbol} on {timeframe_str}")
                return None
            df_new = pd.DataFrame(rates)
            df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
            data_cache_instance.update_ohlc_cache(symbol, timeframe_str, df_new, max_bars_to_store)
            cached_df, _ = data_cache_instance.get_cached_ohlc(symbol, timeframe_str)
        else:
            # If cache exists, just fetch the newest bars since the last cached bar
            last_bar_time = cached_df['time'].iloc[-1]
            date_from_update = last_bar_time + pd.Timedelta(seconds=1)
            new_rates = mt5.copy_rates_range(symbol, timeframe, date_from_update, datetime.utcnow())
            if new_rates is not None and len(new_rates) > 0:
                df_new = pd.DataFrame(new_rates)
                df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
                data_cache_instance.update_ohlc_cache(symbol, timeframe_str, df_new, max_bars_to_store)
                cached_df, _ = data_cache_instance.get_cached_ohlc(symbol, timeframe_str)
                logger.debug(f"Fetched {len(df_new)} new bars for {symbol} on {timeframe_str} and updated cache.")

        if cached_df is not None and not cached_df.empty:
            # Return only the number of bars requested by the strategy
            return cached_df.tail(bars).copy()
        else:
            logger.error(f"No data available for {symbol} on {timeframe_str} after fetch attempt.")
            return None

    except Exception as e:
        logger.error(f"Error in get_data for {symbol}: {e}", exc_info=True)
        return None