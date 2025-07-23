import threading
import pandas as pd
from datetime import datetime
import time
from .logging import setup_logging # Assuming this path is correct

logger = setup_logging()

class DataCache:
    def __init__(self):
        # cache format: {(symbol, timeframe_str): {'df': pd.DataFrame, 'last_updated_ts': float, 'max_bars_stored': int}}
        self.ohlc_cache = {}
        # htf_cache format: {(symbol, timeframe_str): {'trend': str, 'levels': list, 'last_updated_ts': float}}
        # Note: Changed htf_context_cache key to include timeframe_str for consistency, though typically it's 'D1'
        self.htf_context_cache = {}
        self.cache_lock = threading.Lock()

    def get_cached_ohlc(self, symbol, timeframe_str):
        """
        Tries to retrieve OHLC data from the cache.
        Returns a tuple: (cached DataFrame, last_bar_timestamp_unix) or (None, None).
        """
        with self.cache_lock:
            cache_key = (symbol, timeframe_str)
            cached_entry = self.ohlc_cache.get(cache_key)

            if not cached_entry:
                logger.debug(f"CACHE MISS (no entry): {symbol} {timeframe_str}")
                return None, None

            df_cached = cached_entry['df']
            if df_cached.empty or 'time' not in df_cached.columns:
                logger.debug(f"CACHE MISS (empty or no time column): {symbol} {timeframe_str}")
                return None, None

            # Ensure 'time' is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_cached['time']):
                try:
                    df_cached['time'] = pd.to_datetime(df_cached['time'])
                except Exception as e:
                    logger.error(f"Error converting cached 'time' to datetime for {symbol} {timeframe_str}: {e}")
                    return None, None # Cannot proceed if time conversion fails

            last_bar_timestamp_unix = df_cached['time'].iloc[-1].timestamp()
            logger.debug(f"CACHE HIT (entry found): {symbol} {timeframe_str}. Cached bars: {len(df_cached)}, Last bar time: {datetime.fromtimestamp(last_bar_timestamp_unix)}")
            return df_cached.copy(), last_bar_timestamp_unix # Return a copy

    def update_ohlc_cache(self, symbol, timeframe_str, df_new, max_bars_to_store):
        """
        Updates the OHLC cache with new data.
        Merges new data with existing if possible, otherwise replaces.
        Keeps only the most recent max_bars_to_store.
        """
        with self.cache_lock:
            if df_new is None or df_new.empty:
                logger.debug(f"CACHE UPDATE SKIPPED (new data empty): {symbol} {timeframe_str}")
                return

            df_new_copy = df_new.copy()
            if 'time' not in df_new_copy.columns:
                logger.error(f"CACHE UPDATE FAIL: DataFrame for {symbol} {timeframe_str} missing 'time' column.")
                return

            # Ensure 'time' column is datetime and sorted
            try:
                df_new_copy['time'] = pd.to_datetime(df_new_copy['time'])
                df_new_copy = df_new_copy.drop_duplicates(subset=['time'], keep='last').sort_values(by='time', ascending=True)
            except Exception as e:
                logger.error(f"Error processing new data 'time' column for {symbol} {timeframe_str}: {e}")
                return

            cache_key = (symbol, timeframe_str)
            cached_entry = self.ohlc_cache.get(cache_key)

            final_df = df_new_copy
            if cached_entry and not cached_entry['df'].empty:
                df_old = cached_entry['df']
                # Ensure 'time' in old_df is also datetime and sorted
                if not pd.api.types.is_datetime64_any_dtype(df_old['time']):
                    try:
                        df_old['time'] = pd.to_datetime(df_old['time'])
                    except Exception as e:
                        logger.error(f"Error converting old cached 'time' to datetime for {symbol} {timeframe_str}: {e}")
                        # Proceed with new data only if old data is problematic

                if pd.api.types.is_datetime64_any_dtype(df_old['time']): # Check again after potential conversion
                    df_old = df_old.sort_values(by='time', ascending=True)
                    combined_df = pd.concat([df_old, df_new_copy], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['time'], keep='last').sort_values(by='time', ascending=True)
                    final_df = combined_df
                else:
                    logger.warning(f"Proceeding with only new data for {symbol} {timeframe_str} due to old data time column issue.")


            # Trim to max_bars_to_store
            if len(final_df) > max_bars_to_store:
                final_df = final_df.tail(max_bars_to_store)

            self.ohlc_cache[cache_key] = {
                'df': final_df,
                'last_updated_ts': datetime.now().timestamp(), # Timestamp of cache update itself
                'max_bars_stored': max_bars_to_store
            }
            logger.debug(f"CACHE UPDATED: {symbol} {timeframe_str}. Now has {len(final_df)} bars. Max stored: {max_bars_to_store}")

    def get_cached_htf_context(self, symbol, higher_tf_str):
        """Gets cached higher timeframe context if fresh."""
        with self.cache_lock:
            cache_key = (symbol, higher_tf_str)
            cached_data = self.htf_context_cache.get(cache_key)
            if cached_data:
                # D1 data changes once a day. Refresh if older than ~20 hours.
                # This is a simple check; a more robust check would involve comparing the date of the last D1 bar.
                if time.time() - cached_data['last_updated_ts'] < 20 * 60 * 60: # 20 hours
                    logger.debug(f"HTF CACHE HIT: {symbol} {higher_tf_str}")
                    return cached_data['trend'], cached_data['levels']
                else:
                    logger.debug(f"HTF CACHE STALE: {symbol} {higher_tf_str}")
            else:
                logger.debug(f"HTF CACHE MISS: {symbol} {higher_tf_str}")
            return None, None

    def update_htf_context_cache(self, symbol, higher_tf_str, trend, levels):
        """Updates higher timeframe context cache."""
        with self.cache_lock:
            cache_key = (symbol, higher_tf_str)
            self.htf_context_cache[cache_key] = {
                'trend': trend,
                'levels': levels,
                'last_updated_ts': time.time()
            }
            logger.debug(f"HTF CACHE UPDATED: {symbol} {higher_tf_str} with trend {trend}")

# Global instance of the cache
# This should be imported by other modules that need access to the cache.
# Ensure utils.data_cache can be resolved if this file is in utils/data_cache.py
# If data_cache.py is in the root, then from data_cache import data_cache_instance
try:
    # Attempt to make it available for import if this file is in a 'utils' directory
    # This allows other files in 'utils' or the main script to import it.
    # For example, in mt5_connection.py (if also in utils): from .data_cache import data_cache_instance
    # Or from main.py: from utils.data_cache import data_cache_instance
    pass
except ImportError:
    # Fallback if the above structure isn't met, assuming it's in the main path
    pass

data_cache_instance = DataCache()
