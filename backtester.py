import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import json
import MetaTrader5 as mt5

# --- Core Backtesting and Project Imports ---
try:
    from backtesting import Backtest, Strategy
except ImportError:
    print("Backtesting.py library not found. Please install it: pip install backtesting")
    # Define dummy classes if backtesting.py is not installed, so the script can be loaded.
    class Strategy: pass
    class Backtest: pass

try:
    # Import all strategies from the strategies directory
    from strategies.sma import SMAStrategy
    from strategies.bollinger_bands import BollingerBandsStrategy
    from strategies.liquidity_sweep import LiquiditySweepStrategy
    from strategies.fibonacci import FibonacciStrategy
    from strategies.malaysian_snr import MalaysianSnRStrategy
    from strategies.smc import SMCStrategy
    from strategies.adx_strategy import ADXStrategy
    from strategies.keltner_channels_strategy import KeltnerChannelsStrategy
    from strategies.scalping_strategy import ScalpingStrategy # Added ScalpingStrategy

    # Import necessary components from the main application and utils
    from strategies.ml_model import MLValidator
    from utils.mt5_connection import connect_mt5, get_data as get_historical_data_from_mt5
    from utils.logging import setup_logging
    from main import TradingBot # Used to access feature extraction and config
except ImportError as e:
    print(f"Error importing project modules in backtester.py: {e}. Ensure paths are correct and all dependencies are installed.")
    # Define dummy classes and functions if imports fail to allow the script to be parsed.
    class BaseStrategyPlaceholder:
        def __init__(self, name, **kwargs): self.name = name
        def get_signal(self, data, **kwargs): return 'hold', 0.0
        def set_config(self, cfg): pass
    SMAStrategy = BollingerBandsStrategy = LiquiditySweepStrategy = FibonacciStrategy = MalaysianSnRStrategy = SMCStrategy = ADXStrategy = KeltnerChannelsStrategy = ScalpingStrategy = BaseStrategyPlaceholder # Added ScalpingStrategy to placeholders
    class MLValidator:
        def __init__(self, config): self.config = config; self.predictors = {}; self.features_to_remove_indices = []
        def fit(self, symbol, X, y, direction): pass
        def predict_proba(self, symbol, X, direction): return np.array([[0.5, 0.5]] * (len(X) if hasattr(X, '__len__') else 1))
        def is_fitted(self, symbol, direction): return False
        def create_predictors_for_all_symbols(self): pass # Dummy method
        def _get_filtered_X(self, X): return X # Dummy filter
    def get_historical_data_from_mt5(*args, **kwargs): return pd.DataFrame()
    def connect_mt5(*args, **kwargs): return False
    class TradingBot:
        def __init__(self): self.config = {}; self.strategies = []; self.ml_validator = None
        def extract_ml_features(self, symbol, data, signal, dxy_data=None): return [0.0]*33 # Updated length for new features
        def initialize_strategies(self): return []
        def load_config_and_reinitialize(self): pass
        def calculate_atr(self, data, period=14): return 0.0
        def calculate_rsi(self, data, window=14): return 50.0 # Added RSI
        def bollinger_band_width(self, data, window=20): return 0.0 # Added BB Width
        def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9): return 0.0, 0.0, 0.0 # Added MACD
        def get_dxy_data_for_correlation(self, primary_symbol_data): return None # Added DXY
        def is_trading_hours(self, symbol): return True # Default to always trading
        def in_cooldown(self, symbol): return False # Default to no cooldown
        def can_open_new_trade(self, symbol, signal_to_open): return True # Default to always can open
        def get_trend(self, symbol, timeframe_str): return 'neutral' # Added get_trend
        def mt5_tf_to_minutes(self, mt5_timeframe_str): return 5 # Added TF to minutes

    import logging as pylogging
    logger = pylogging.getLogger("dummy_backtester_logger")
    setup_logging = lambda: logger

# --- Global Variables and Setup ---
logger = setup_logging()
backtest_ml_validator = None
bot_config_for_backtesting = None
_temp_bot_instance_for_backtesting = None


# --- Helper Functions for Indicator Calculations (Copied from main.py for consistency) ---
# NOTE: These are now typically handled by the _temp_bot_instance_for_backtesting
# but kept here for fallback or if individual functions are preferred.
def calculate_atr_local(data, period=14):
    """Calculates Average True Range. Essential for SL/TP logic."""
    if not isinstance(data, pd.DataFrame) or data.empty or not all(c in data.columns for c in ['high','low','close']) or len(data) < period:
        return 0.0
    try:
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift()).fillna(0.0)
        low_close = abs(data['low'] - data['close'].shift()).fillna(0.0)
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        if tr.empty: return 0.0
        atr = tr.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        return 0.0 if pd.isna(atr) or np.isinf(atr) else float(atr)
    except Exception as e:
        logger.warning(f"Local ATR calculation error: {e}. Returning 0.0")
        return 0.0

def calculate_rsi_local(data, window=14):
    """Calculates Relative Strength Index."""
    if not isinstance(data, pd.DataFrame) or data.empty or 'close' not in data.columns or len(data) < window: return 50.0
    try:
        delta = data['close'].diff()
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
        logger.warning(f"Local RSI calculation error: {e}. Returning 50.0")
        return 50.0

def bollinger_band_width_local(data, window=20, std_dev_multiplier=2.0):
    """Calculates Bollinger Band Width."""
    if not isinstance(data, pd.DataFrame) or data.empty or 'close' not in data.columns or len(data) < window: return 0.0
    try:
        ma_series = data['close'].rolling(window=window).mean()
        std_series = data['close'].rolling(window=window).std()

        if ma_series.empty or std_series.empty or pd.isna(ma_series.iloc[-1]) or pd.isna(std_series.iloc[-1]) or ma_series.iloc[-1] == 0:
            return 0.0

        ma = ma_series.iloc[-1]
        std = std_series.iloc[-1]

        upper = ma + std_dev_multiplier * std
        lower = ma - std_dev_multiplier * std

        bb_width = (upper - lower) / ma
        return 0.0 if pd.isna(bb_width) or np.isinf(bb_width) else float(bb_width)
    except Exception as e:
        logger.warning(f"Local Bollinger Band Width calculation error: {e}. Returning 0.0")
        return 0.0

def calculate_macd_local(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD and Signal Line."""
    if not isinstance(data, pd.DataFrame) or data.empty or 'close' not in data.columns or len(data) < slow_period : return 0.0, 0.0, 0.0
    try:
        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        if ema_fast.empty or ema_slow.empty: return 0.0,0.0,0.0

        macd_line_series = ema_fast - ema_slow
        if macd_line_series.empty: return 0.0,0.0,0.0

        signal_line_series = macd_line_series.ewm(span=signal_period, adjust=False).mean()
        if signal_line_series.empty: return 0.0,0.0,0.0

        histogram_series = macd_line_series - signal_line_series
        if histogram_series.empty: return 0.0,0.0,0.0

        macd_val = macd_line_series.iloc[-1]
        signal_val = signal_line_series.iloc[-1]
        hist_val = histogram_series.iloc[-1]

        return (
            float(macd_val) if pd.notna(macd_val) and np.isfinite(macd_val) else 0.0,
            float(signal_val) if pd.notna(signal_val) and np.isfinite(signal_val) else 0.0,
            float(hist_val) if pd.notna(hist_val) and np.isfinite(hist_val) else 0.0
        )
    except Exception as e:
        logger.warning(f"Local MACD calculation error: {e}. Returning 0,0,0")
        return 0.0, 0.0, 0.0


# --- Core Backtesting Logic ---

def _get_bot_instance_for_backtesting():
    """
    Initializes a temporary instance of the main TradingBot to access its methods
    like feature extraction and to load the master configuration file.
    This ensures that the backtester uses the same logic and configuration
    as the live trading bot.
    """
    global _temp_bot_instance_for_backtesting, bot_config_for_backtesting, backtest_ml_validator
    if _temp_bot_instance_for_backtesting is None:
        try:
            # Initialize TradingBot which will load config and MLValidator
            _temp_bot_instance_for_backtesting = TradingBot()
            bot_config_for_backtesting = _temp_bot_instance_for_backtesting.config
            backtest_ml_validator = _temp_bot_instance_for_backtesting.ml_validator # Get the MLValidator from the bot
            logger.info("Temporary TradingBot instance created for backtesting.")
        except Exception as e:
            logger.critical(f"Failed to initialize TradingBot for backtesting utilities: {e}", exc_info=True)
            # Fallback: try to load config directly and create dummy MLValidator
            try:
                # Assuming config.json is in a 'config' directory one level up from backtester.py
                config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
                with open(config_path, 'r') as f:
                    bot_config_for_backtesting = json.load(f)
                logger.warning(f"Using config loaded directly from {config_path} as a fallback.")
                # Create a dummy bot instance and assign the loaded config
                _temp_bot_instance_for_backtesting = TradingBot()
                _temp_bot_instance_for_backtesting.config = bot_config_for_backtesting
                backtest_ml_validator = MLValidator(bot_config_for_backtesting) # Manually create MLValidator
                _temp_bot_instance_for_backtesting.ml_validator = backtest_ml_validator # Assign to dummy bot
            except Exception as e_fallback:
                 logger.critical(f"CRITICAL: Fallback config loading also failed: {e_fallback}. Backtester cannot function properly.")
                 bot_config_for_backtesting = {} # Empty config to prevent crashes
                 _temp_bot_instance_for_backtesting = TradingBot() # Still create a dummy bot for method access
                 backtest_ml_validator = MLValidator({}) # Dummy MLValidator

    return _temp_bot_instance_for_backtesting

def train_ml_model_for_backtest_period(symbol, timeframe_str, train_df):
    """
    Trains the ML model on data *prior* to the backtest period.
    This function now mirrors the target definition logic from main.py,
    using the TradingBot instance for feature extraction.
    """
    global backtest_ml_validator, bot_config_for_backtesting
    bot_instance = _get_bot_instance_for_backtesting()
    if not bot_config_for_backtesting:
        logger.error("Bot configuration not loaded. Cannot train ML model for backtest.")
        return False
    if not bot_instance.ml_validator:
        logger.error("MLValidator not initialized in TradingBot instance. Cannot train ML model.")
        return False

    logger.info(f"Starting ML model pre-training for backtest: {symbol} on {timeframe_str} with {len(train_df)} bars.")
    backtest_ml_validator = bot_instance.ml_validator # Ensure we're using the bot's configured MLValidator

    # Configuration for ML training from config file
    training_window = bot_config_for_backtesting.get('ml_training_window', 120)
    prediction_horizon = bot_config_for_backtesting.get('ml_prediction_horizon', 16)
    target_def = bot_config_for_backtesting.get('ml_target_definition', {})

    if len(train_df) < training_window + prediction_horizon + 1: # Added +1 for accurate slicing
        logger.warning(f"Insufficient data for ML pre-training ({len(train_df)} bars) for {symbol}. Need at least {training_window + prediction_horizon + 1}.")
        return False

    # DXY data handling for correlation feature, using bot's method
    dxy_correlation_data = None
    feat_eng_cfg = bot_config_for_backtesting.get('feature_engineering', {})
    if symbol.upper() == 'XAUUSDM' and feat_eng_cfg.get('dxy_correlation_for_xauusd', False):
        logger.info(f"Fetching DXY data for {symbol} ML training correlation.")
        dxy_correlation_data = bot_instance.get_dxy_data_for_correlation(train_df.copy())
        if dxy_correlation_data is not None and not dxy_correlation_data.empty:
            logger.info("DXY correlation data fetched and aligned for ML training.")
        else:
            logger.warning("Could not fetch DXY data for ML training, proceeding without it.")
            dxy_correlation_data = None

    # Loop for both buy and sell models
    for direction in ['buy', 'sell']:
        features, labels = [], []
        logger.info(f"Preparing training data for {direction.upper()} model for {symbol}...")

        # Ensure 'time' column is datetime
        if 'time' in train_df.columns and not pd.api.types.is_datetime64_any_dtype(train_df['time']):
            try: train_df['time'] = pd.to_datetime(train_df['time'])
            except Exception as e: logger.error(f"Error converting 'time' column to datetime in train_df: {e}")

        for i in range(training_window, len(train_df) - prediction_horizon):
            window_data = train_df.iloc[i - training_window:i].copy()

            dxy_window_data = None
            if dxy_correlation_data is not None and not dxy_correlation_data.empty:
                 # Ensure dxy_window_data is aligned with window_data's time range
                 window_start_time = window_data['time'].min()
                 window_end_time = window_data['time'].max()
                 dxy_window_data = dxy_correlation_data[
                     (dxy_correlation_data['time'] >= window_start_time) &
                     (dxy_correlation_data['time'] <= window_end_time)
                 ].copy()


            try:
                # Use the exact same feature extraction as the live bot
                feature_vector = bot_instance.extract_ml_features(symbol, window_data, direction, dxy_window_data)

                if feature_vector is None or any(pd.isna(f) or np.isinf(f) for f in feature_vector):
                    # logger.debug(f"Skipping feature vector due to NaN/Inf or None at index {i}.")
                    continue

                # --- REPLICATE TARGET DEFINITION from main.py ---
                entry_price = train_df['close'].iloc[i]
                future_data_slice = train_df.iloc[i+1 : i + 1 + prediction_horizon]

                if entry_price == 0 or future_data_slice.empty:
                    continue

                outcome = 0 # Default to loss/no-win
                if target_def.get("method") == "atr_multiplier":
                    atr_params = target_def.get("params", {})
                    atr_period = atr_params.get("atr_period", 14)
                    atr_multiplier = atr_params.get("atr_multiplier", 1.5)

                    # Calculate ATR on the window leading up to the target definition point
                    atr_for_target = bot_instance.calculate_atr(window_data, period=atr_period)
                    if atr_for_target > 0:
                        profit_target_price_dist = atr_for_target * atr_multiplier
                        if direction == 'buy':
                            peak_price_in_horizon = future_data_slice['high'].max()
                            if (peak_price_in_horizon - entry_price) >= profit_target_price_dist:
                                outcome = 1
                        else: # sell
                            trough_price_in_horizon = future_data_slice['low'].min()
                            if (entry_price - trough_price_in_horizon) >= profit_target_price_dist:
                                outcome = 1
                else: # Fallback to simple profit if not ATR multiplier
                    if not future_data_slice.empty:
                        future_price = future_data_slice['close'].iloc[-1] # Use last price in horizon
                        profit_percent = (future_price - entry_price) / entry_price
                        outcome = 1 if (direction == 'buy' and profit_percent >= 0.001) or \
                                       (direction == 'sell' and profit_percent <= -0.001) else 0 # Simple threshold

                features.append(feature_vector)
                labels.append(outcome)

            except Exception as e:
                logger.warning(f"Error preparing training data for {symbol} ({direction}) at index {i}: {e}", exc_info=False)
                continue

        if not features or len(np.unique(labels)) < 2:
            logger.warning(f"No valid training data or only one class for {symbol} {direction.upper()} model. Skipping fit.")
            continue

        try:
            # Fit the model using the prepared features and labels
            # Ensure X is a list of lists or 2D array for the MLValidator
            backtest_ml_validator.fit(symbol, np.array(features).tolist(), labels, direction)
            if backtest_ml_validator.is_fitted(symbol, direction):
                 logger.info(f"ML model for {symbol} ({direction.upper()}) trained successfully for backtest with {len(features)} samples.")
            else:
                 logger.warning(f"ML model fitting for {symbol} ({direction.upper()}) did not succeed.")
        except Exception as e:
            logger.error(f"Failed backtest ML model training for {symbol} ({direction.upper()}): {e}", exc_info=True)

    return True


class GenericBacktestStrategy(Strategy):
    """
    A generic strategy for Backtesting.py that wraps the bot's live logic.
    This version includes risk-based position sizing and filtering to better match live trading.
    It directly calls methods from a global TradingBot instance.
    """
    # These will be populated by bt.run() using kwargs
    strategy_name_to_run = "SMA"
    strategy_specific_params = {}
    symbol_bt = None
    timeframe_bt = None
    ml_validation_enabled_param = True
    symbol_info_bt = None

    def init(self):
        """
        Initializes the wrapper strategy and the underlying 'real' strategy.
        Also initializes the global TradingBot instance if not already done.
        """
        global _temp_bot_instance_for_backtesting, bot_config_for_backtesting, backtest_ml_validator
        self.bot_instance = _get_bot_instance_for_backtesting()
        bot_config_for_backtesting = self.bot_instance.config # Ensure local config is synced
        backtest_ml_validator = self.bot_instance.ml_validator # Ensure local ML validator is synced

        self.original_strategy_instance = self._create_original_strategy()
        if not self.original_strategy_instance:
            logger.critical(f"Could not instantiate original strategy: {self.strategy_name_to_run}. Backtest will halt.")
            # It's difficult to 'halt' in Backtesting.py init, but subsequent next() calls will return.
            self.stop_run = True # Custom flag to stop processing in next()
            return

        self.stop_run = False
        actual_strategy_name = self.original_strategy_instance.name
        logger.info(f"Initialized backtest for requested strategy: '{self.strategy_name_to_run}', Instantiated as: '{actual_strategy_name}' for {self.symbol_bt} {self.timeframe_bt}. ML validation: {self.ml_validation_enabled_param}")

        if not self.symbol_info_bt:
             logger.warning("Symbol info not provided to backtest strategy. Sizing may be inaccurate.")

        # Initialize internal state for managing trades, mirroring live bot.
        # This will simulate cooldown and max_trades_per_symbol.
        self.bt_last_trade_times = {}
        self.bt_open_positions_count = 0 # Keep track of open positions
        self.bt_consecutive_losses = 0 # Not directly used in backtesting.py for pausing, but good to track


    def _create_original_strategy(self):
        """Dynamically creates an instance of the selected strategy using parameters from the bot's config."""
        cfg = bot_config_for_backtesting or {}
        s_params = self.strategy_specific_params # Params from UI can override config

        strategy_class_map = {
            "SMA": SMAStrategy, "BollingerBands": BollingerBandsStrategy, "LiquiditySweep": LiquiditySweepStrategy,
            "Fibonacci": FibonacciStrategy, "MalaysianSnR": MalaysianSnRStrategy, "SMC": SMCStrategy,
            "ADX": ADXStrategy, "KeltnerChannels": KeltnerChannelsStrategy, "Scalping": ScalpingStrategy # Added Scalping
        }

        StrategyClass = strategy_class_map.get(self.strategy_name_to_run)
        if not StrategyClass:
            logger.error(f"Unknown strategy name for backtesting: {self.strategy_name_to_run}")
            return None

        try:
            # This logic mirrors part of initialize_strategies in main.py
            if self.strategy_name_to_run == "SMA":
                return StrategyClass(name="SMA", short_period=int(s_params.get('short_period', cfg.get('sma_short_period', 10))), long_period=int(s_params.get('long_period', cfg.get('sma_long_period', 100))))
            elif self.strategy_name_to_run == "BollingerBands":
                return StrategyClass(name="BollingerBands", window=int(s_params.get('window', cfg.get('bb_window', 20))), std_dev=float(s_params.get('std_dev', cfg.get('bb_std_dev', 2.0))))
            elif self.strategy_name_to_run == "LiquiditySweep":
                return StrategyClass(name="LiquiditySweep", period=int(s_params.get('period', cfg.get('liquidity_sweep_period', 10))))
            elif self.strategy_name_to_run == "Fibonacci":
                levels_param = s_params.get('levels', cfg.get('fibonacci_levels', [0.382, 0.5, 0.618]))
                if isinstance(levels_param, str):
                    try: levels_param = [float(x.strip()) for x in levels_param.split(',')]
                    except: levels_param = [0.382, 0.5, 0.618] # Fallback
                return StrategyClass(name="Fibonacci", levels=levels_param, window=int(s_params.get('window', cfg.get('fibonacci_window', 20))), proximity_factor=float(s_params.get('proximity_factor', cfg.get('fibonacci_proximity_factor', 0.015))))
            elif self.strategy_name_to_run == "MalaysianSnR":
                instance = StrategyClass(name="MalaysianSnR", window=int(s_params.get('window', cfg.get('snr_window', 10))), freshness_window=int(s_params.get('freshness_window', cfg.get('snr_freshness_window', 3))), threshold=float(s_params.get('threshold', cfg.get('snr_threshold', 0.003))))
                instance.set_config(cfg) # Pass full config to MalaysianSnR
                return instance
            elif self.strategy_name_to_run == "SMC":
                instance = StrategyClass(name="SMC", higher_timeframe=s_params.get('higher_timeframe', cfg.get('smc_higher_timeframe', 'D1')), min_ob_size=float(s_params.get('min_ob_size', cfg.get('smc_min_ob_size', 0.0003))), fvg_threshold=float(s_params.get('fvg_threshold', cfg.get('smc_fvg_threshold', 0.0003))), liquidity_tolerance=float(s_params.get('liquidity_tolerance',cfg.get('smc_liquidity_tolerance',0.005))), trade_cooldown=int(s_params.get('trade_cooldown',cfg.get('smc_trade_cooldown',30))))
                instance.set_config(cfg) # Pass full config to SMC
                return instance
            elif self.strategy_name_to_run == "ADX":
                return StrategyClass(name="ADX", adx_period=int(s_params.get('adx_period', cfg.get('adx_period', 14))), di_period=int(s_params.get('di_period', cfg.get('adx_di_period', 14))), adx_threshold=int(s_params.get('adx_threshold', cfg.get('adx_threshold', 25))), strength_factor=float(s_params.get('strength_factor', cfg.get('adx_strength_factor', 0.02))))
            elif self.strategy_name_to_run == "KeltnerChannels":
                 return StrategyClass(name="KeltnerChannels", ema_period=int(s_params.get('ema_period', cfg.get('keltner_ema_period', 20))), atr_period=int(s_params.get('atr_period', cfg.get('keltner_atr_period', 10))), atr_multiplier=float(s_params.get('atr_multiplier', cfg.get('keltner_atr_multiplier', 2.0))), strength=float(s_params.get('strength', cfg.get('keltner_signal_strength', 0.7))))
            elif self.strategy_name_to_run == "Scalping": # Initialize Scalping Strategy
                 return StrategyClass(name="Scalping")
            else:
                return StrategyClass(name=self.strategy_name_to_run)
        except Exception as e:
            logger.error(f"Error instantiating strategy {self.strategy_name_to_run} with params {s_params}: {e}", exc_info=True)
            return None

    def next(self):
        """
        The main backtesting loop called for each bar of data.
        This method now tightly replicates the live bot's signal analysis,
        ML validation, and trade execution logic.
        """
        if self.stop_run or not self.original_strategy_instance:
            logger.debug(f"Halting next() for {self.symbol_bt}. Strategy not initialized or stop_run flag is set.")
            return

        # Prepare data for strategy and feature extraction
        df_for_strategy = self.data.df.copy()
        df_for_strategy.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'tick_volume'}, inplace=True)
        if 'time' not in df_for_strategy.columns:
            df_for_strategy['time'] = df_for_strategy.index.to_series() # Convert index to a column named 'time'

        # Ensure 'time' column is timezone-aware if the original data might be, or make it so.
        if pd.api.types.is_datetime64_any_dtype(df_for_strategy['time']) and df_for_strategy['time'].dt.tz is None:
            df_for_strategy['time'] = df_for_strategy['time'].dt.tz_localize(timezone.utc)
        elif 'time' in df_for_strategy.columns and not pd.api.types.is_datetime64_any_dtype(df_for_strategy['time']):
            try: df_for_strategy['time'] = pd.to_datetime(df_for_strategy['time'], utc=True)
            except Exception as e: logger.error(f"Failed to convert 'time' column to datetime in backtest strategy: {e}")

        # Get signal from the original strategy
        signal, strength = 'hold', 0.0
        try:
            # Pass a copy to strategy to prevent in-place modification issues
            signal, strength = self.original_strategy_instance.get_signal(df_for_strategy.copy(), symbol=self.symbol_bt, timeframe=self.timeframe_bt)
            if not isinstance(signal, str): # Handle cases where strategy might return non-string signal
                logger.warning(f"Strategy {self.original_strategy_instance.name} returned non-string signal: {signal}. Defaulting to 'hold'.")
                signal = 'hold'
            if pd.isna(strength): strength = 0.0
        except ValueError as ve:
            if 'The truth value of a DataFrame is ambiguous' in str(ve):
                logger.critical(f"CRITICAL PANDAS BUG in {self.original_strategy_instance.name}.get_signal. The strategy code likely contains an 'if some_dataframe:' condition. This is ambiguous. Use 'if not some_dataframe.empty:' instead. Treating signal as 'hold'.")
                signal, strength = 'hold', 0.0
            else:
                logger.error(f"Unhandled ValueError in {self.original_strategy_instance.name}.get_signal: {ve}", exc_info=False)
                signal, strength = 'hold', 0.0
        except Exception as e:
            logger.error(f"Generic error in {self.original_strategy_instance.name}.get_signal: {e}", exc_info=False)
            signal, strength = 'hold', 0.0

        logger.debug(f"Strategy Signal for {self.symbol_bt} {self.timeframe_bt} @ {df_for_strategy['time'].iloc[-1]}: {signal.upper()} (Strength: {strength:.2f})")

        # --- Replicate main.py's `analyze_signals` logic for filtering and consensus ---
        final_consensus_signal = 'hold'
        if signal != 'hold': # Only analyze if strategy generated a signal
            current_signals = {self.original_strategy_instance.name: signal}
            strengths = {self.original_strategy_instance.name: strength}

            # Simulate TradingBot's analyze_signals to apply ADX filter, boosts, and consensus threshold
            # Note: This is a simplified version, true consensus would need all strategies
            # We are currently only backtesting ONE strategy, so consensus logic is effectively bypassed
            # to check this single strategy's signal after potential ADX filtering.

            cfg = bot_config_for_backtesting
            adx_filter_cfg = cfg.get('adx_signal_filter', {})
            apply_adx_filter = adx_filter_cfg.get('enabled', False)

            # For backtester, we need to create a dummy adx_strategy_instance or use bot_instance's internal one
            # The TradingBot instance should have its strategies initialized during _get_bot_instance_for_backtesting
            adx_strategy_instance_bot = next((s for s in self.bot_instance.strategies if isinstance(s, ADXStrategy)), None)

            if apply_adx_filter and adx_strategy_instance_bot:
                try:
                    adx_indicator_values = adx_strategy_instance_bot.get_indicator_values(df_for_strategy.copy())
                    if adx_indicator_values and adx_indicator_values.get('adx') is not None and not adx_indicator_values['adx'].empty :
                        latest_adx_value = adx_indicator_values['adx'].iloc[-1]
                        latest_plus_di = adx_indicator_values['plus_di'].iloc[-1]
                        latest_minus_di = adx_indicator_values['minus_di'].iloc[-1]

                        min_adx_for_entry_symbol_key = f"min_adx_for_entry_{self.symbol_bt}"
                        min_adx_for_entry = adx_filter_cfg.get(min_adx_for_entry_symbol_key, adx_filter_cfg.get('min_adx_for_entry', 20))
                        require_di = adx_filter_cfg.get('require_di_confirmation', True)

                        if latest_adx_value < min_adx_for_entry:
                            logger.info(f"ADX Filter invalidated {signal} from {self.original_strategy_instance.name}. Reason: ADX {latest_adx_value:.2f} < threshold {min_adx_for_entry}.")
                            signal = 'hold' # Override signal
                        elif require_di and not (pd.isna(latest_plus_di) or pd.isna(latest_minus_di)):
                            di_confirms_buy = latest_plus_di > latest_minus_di
                            di_confirms_sell = latest_minus_di > latest_plus_di
                            if (signal == 'buy' and not di_confirms_buy) or (signal == 'sell' and not di_confirms_sell):
                                logger.info(f"ADX Filter invalidated {signal} from {self.original_strategy_instance.name}. Reason: DI confirmation failed.")
                                signal = 'hold' # Override signal
                    else:
                        logger.warning(f"ADX Filter enabled but indicator values for {self.symbol_bt} on {self.timeframe_bt} are invalid. Disabling ADX filter for this cycle.")
                        apply_adx_filter = False
                except Exception as e:
                    logger.error(f"Error applying ADX filter in backtest: {e}", exc_info=False)
                    apply_adx_filter = False

            # After ADX filter, if signal is still active, apply ML validation
            if signal != 'hold' and self.ml_validation_enabled_param and backtest_ml_validator:
                ml_training_window = cfg.get('ml_training_window', 120)
                if len(df_for_strategy) >= ml_training_window:
                    feature_data_df_ml = df_for_strategy.iloc[-ml_training_window:].copy()

                    # Fetch DXY data for ML if enabled, using the bot instance
                    dxy_data_bt = None
                    feat_eng_cfg = cfg.get('feature_engineering', {})
                    if self.symbol_bt.upper() == 'XAUUSDM' and feat_eng_cfg.get('dxy_correlation_for_xauusd', False):
                        dxy_data_bt = self.bot_instance.get_dxy_data_for_correlation(feature_data_df_ml.copy())

                    try:
                        features = self.bot_instance.extract_ml_features(self.symbol_bt, feature_data_df_ml, signal, dxy_data_bt)
                        if features is None or not features or any(pd.isna(f) or np.isinf(f) for f in features):
                            logger.warning(f"ML Features for {self.symbol_bt} ({signal}) are invalid or empty. Skipping ML validation.")
                            signal = 'hold'
                        elif backtest_ml_validator.is_fitted(self.symbol_bt, signal):
                            ml_prob = backtest_ml_validator.predict_proba(self.symbol_bt, [features], signal)[0][1]
                            conf_thresholds = cfg.get('ml_confidence_thresholds', {}).get(signal, {})
                            ml_thresh = conf_thresholds.get(self.symbol_bt, conf_thresholds.get("default", 0.55))

                            if ml_prob < ml_thresh:
                                logger.info(f"ML validation rejected {signal} for {self.symbol_bt}. Confidence {ml_prob:.4f} < threshold {ml_thresh:.4f}.")
                                signal = 'hold' # Override signal
                            else:
                                logger.debug(f"ML validation passed for {self.symbol_bt} ({signal}). Confidence {ml_prob:.4f} >= threshold {ml_thresh:.4f}.")
                        else:
                            logger.warning(f"ML validation enabled but model for {self.symbol_bt} ({signal}) is not fitted. Skipping ML validation.")
                    except Exception as e:
                        logger.error(f"ML prediction error in backtest 'next' loop for {self.symbol_bt} ({signal}): {e}", exc_info=False)
                        signal = 'hold' # Override signal in case of error
                else:
                    logger.warning(f"Insufficient data for ML feature extraction ({len(df_for_strategy)} bars, need {ml_training_window}). Skipping ML validation for {self.symbol_bt}.")
                    # Do not set signal to 'hold' here unless absolutely necessary,
                    # as core strategy might still be valid without ML filter.
                    # For consistency with main.py, if ML is *required* and data is insufficient,
                    # the signal would indeed be held. Let's keep it consistent.
                    if cfg.get('ml_enabled', True): # Assuming ML is globally enabled, if data is too short, hold
                        signal = 'hold'

        final_consensus_signal = signal # The `signal` variable has been filtered by ADX and ML

        # --- Replicate main.py's `execute_trade` logic for trade entry conditions ---
        if final_consensus_signal != 'hold' and not self.position:
            cfg = bot_config_for_backtesting
            current_price = self.data.Close[-1] # Use Backtesting.py's price access

            # Check general trading conditions (cooldown, max trades, trading hours, consecutive losses)
            if self.bt_last_trade_times.get(self.symbol_bt) and \
               (df_for_strategy['time'].iloc[-1] - self.bt_last_trade_times[self.symbol_bt]) < \
               timedelta(minutes=cfg.get('cooldown_period_minutes', 15)):
                logger.debug(f"Trade REJECTED ({self.symbol_bt}): In cooldown period.")
                return # Skip trade

            max_trades_for_symbol = cfg.get('max_trades_per_symbol', 2)
            if self.bt_open_positions_count >= max_trades_for_symbol:
                logger.debug(f"Trade REJECTED ({self.symbol_bt}): Max open trades reached ({self.bt_open_positions_count}/{max_trades_for_symbol}).")
                return # Skip trade

            # In backtesting, we don't have MT5 trading hours directly. Assume always tradable unless specific logic
            # is added to filter historical data by actual trading hours.
            # If the config specifies trading hours, you might want to filter the historical data
            # before passing it to Backtest() or implement this check based on `data['time'].iloc[-1]`.
            # For simplicity, we'll assume historical data passed is within relevant hours for backtesting.
            # is_trading_hours_flag = self.bot_instance.is_trading_hours(self.symbol_bt) # This relies on current real-time
            # if not is_trading_hours_flag:
            #     logger.debug(f"Trade REJECTED ({self.symbol_bt}): Outside trading hours (historical data).")
            #     return

            # H1 Trend Filter (using bot's get_trend method)
            h1_trend_config = cfg.get('h1_trend_filter', {'enabled': True, 'allow_neutral': True})
            if h1_trend_config.get('enabled', True):
                # For backtesting, you'd need to fetch or derive H1 data for the historical period
                # This is a simplification: for true historical H1 trend, you'd need to pre-load H1 data.
                # Assuming get_trend can operate on historical data if available or simply return neutral.
                h1_trend = self.bot_instance.get_trend(self.symbol_bt, timeframe_str='H1') # This would typically fetch live H1 data
                logger.debug(f"H1 Trend for {self.symbol_bt}: {h1_trend.upper()}")
                if h1_trend == 'neutral' and not h1_trend_config.get('allow_neutral', True):
                    logger.info(f"Trade REJECTED ({self.symbol_bt} {final_consensus_signal}): H1 trend NEUTRAL and neutral trades disallowed.")
                    return
                if (final_consensus_signal == 'buy' and h1_trend == 'downtrend') or \
                   (final_consensus_signal == 'sell' and h1_trend == 'uptrend'):
                    logger.info(f"Trade REJECTED ({self.symbol_bt} {final_consensus_signal}): Signal against H1 trend {h1_trend.upper()}.")
                    return

            # ML confidence threshold check already done above.

            # 1. Calculate SL and TP prices (using bot's calculate_atr for consistency)
            sl_tp_atr_period = cfg.get('atr_period_for_sl_tp', 14)
            sl_tp_range_window = cfg.get('recent_range_window_for_sl_tp', 20)

            # Use bot's ATR calculation directly
            atr = self.bot_instance.calculate_atr(df_for_strategy.copy(), period=sl_tp_atr_period)
            if atr <= 0: atr = current_price * cfg.get('atr_fallback_factor', 0.005)

            price_range = 0
            if len(df_for_strategy) >= sl_tp_range_window:
                price_range = df_for_strategy['high'].iloc[-sl_tp_range_window:].max() - df_for_strategy['low'].iloc[-sl_tp_range_window:].min()
            if price_range <= 0: price_range = current_price * cfg.get('range_fallback_factor', 0.01)

            sl_distance = max(cfg.get('sl_atr_multiplier', 2.0) * atr, price_range * cfg.get('sl_range_factor', 0.5))
            tp_distance = sl_distance * cfg.get('risk_reward_ratio', 1.5)

            if final_consensus_signal == 'buy':
                sl_price = current_price - sl_distance
                tp_price = current_price + tp_distance
            else: # sell
                sl_price = current_price + sl_distance
                tp_price = current_price - tp_distance

            # Backtesting.py handles minimum stop levels automatically to some extent.
            # No explicit 'min_stop_price_distance' check needed here for Backtesting.py's order functions.
            # However, for realistic simulation, ensure SL/TP are not too close to current price.
            # Using point from symbol_info_bt if available
            symbol_point = self.symbol_info_bt.point if self.symbol_info_bt else 0.00001 # Default if not available
            min_stop_pips_for_validation = cfg.get('min_stop_multiplier', 1.5) * (self.symbol_info_bt.trade_stops_level if self.symbol_info_bt else 10) * symbol_point

            if final_consensus_signal == 'buy':
                if (tp_price - current_price) < min_stop_pips_for_validation or (current_price - sl_price) < min_stop_pips_for_validation:
                    logger.warning(f"Trade REJECTED ({self.symbol_bt} {final_consensus_signal}): SL/TP too close to entry (below min stop distance).")
                    return
            else: # sell
                if (current_price - tp_price) < min_stop_pips_for_validation or (sl_price - current_price) < min_stop_pips_for_validation:
                    logger.warning(f"Trade REJECTED ({self.symbol_bt} {final_consensus_signal}): SL/TP too close to entry (below min stop distance).")
                    return

            # 2. Calculate position size based on risk
            global_risk_pct = cfg.get('risk_percent_per_trade', 0.01)
            sym_risk_cfg = cfg.get('risk_params', {}).get(self.symbol_bt, {})
            risk_percent = sym_risk_cfg.get('max_risk_per_trade', global_risk_pct)

            # In Backtesting.py, `size` is a fraction of available equity, not a fixed lot.
            # We calculate `risk_amt` and then determine the `size` based on this.
            risk_amt = self.equity * risk_percent # `self.equity` is available in Backtesting.py

            # This calculation requires 'point' and 'trade_tick_value' which are specific to MT5
            # For backtesting.py, we can simplify this or use the provided symbol_info_bt
            if self.symbol_info_bt and self.symbol_info_bt.point > 0 and self.symbol_info_bt.trade_tick_value > 0 and self.symbol_info_bt.trade_tick_size > 0:
                val_per_pip_1_lot = (self.symbol_info_bt.trade_tick_value / self.symbol_info_bt.trade_tick_size) * self.symbol_info_bt.point
            else:
                # Fallback if symbol_info is missing crucial elements
                logger.warning(f"Symbol info for {self.symbol_bt} missing crucial details (point, trade_tick_value, trade_tick_size). Using simplified lot sizing.")
                val_per_pip_1_lot = current_price * 0.0001 # A very rough estimate

            sl_pips_for_lot_calc = abs(current_price - sl_price) / symbol_point
            if sl_pips_for_lot_calc <= 0:
                logger.warning(f"Calculated SL pips is zero or negative for {self.symbol_bt}. Skipping trade.")
                return

            if val_per_pip_1_lot <= 0:
                logger.warning(f"Calculated value per pip for 1 lot is zero or negative for {self.symbol_bt}. Skipping trade.")
                return

            try:
                # This approximates the lot size. Backtesting.py's 'size' is a fraction of equity.
                # Here, we calculate a target lot based on risk, then infer the 'size' for backtesting.py.
                target_lot_based_on_risk = risk_amt / (sl_pips_for_lot_calc * val_per_pip_1_lot)

                # Backtesting.py's 'size' parameter for self.buy/self.sell refers to position size
                # relative to the initial capital or current equity. A common way to approximate
                # risk-based sizing in Backtesting.py is to calculate how much capital 'value'
                # is risked per share/lot, and then scale that to total risk_amt.
                # Let's simplify and use `target_lot_based_on_risk` directly for lot estimation for Backtesting.py
                # This means 'size' in bt.buy() is effectively 'number of units' rather than percentage of equity.
                # This needs careful calibration if actual lot sizing is critical for your backtest.
                # For now, let's treat `size` as the number of "units" based on the MT5 lot logic.
                # Backtesting.py automatically handles margin/cash requirements.
                lot_size = target_lot_based_on_risk

                # Apply volume step/min/max from symbol_info_bt
                if self.symbol_info_bt:
                    volume_step = self.symbol_info_bt.volume_step
                    volume_min = self.symbol_info_bt.volume_min
                    volume_max = self.symbol_info_bt.volume_max

                    if volume_step > 0:
                        lot_size = round(lot_size / volume_step) * volume_step
                    else: # Default step, e.g., 0.01
                         lot_size = round(lot_size, 2)

                    lot_size = max(volume_min, min(lot_size, volume_max))
                else:
                    lot_size = max(0.01, round(lot_size, 2)) # Default min lot

                if lot_size <= 0:
                    logger.warning(f"Calculated trade size is {lot_size:.4f}. Skipping trade (too small/zero).")
                    return

                logger.debug(f"Trade size calculated for {self.symbol_bt} {final_consensus_signal}: {lot_size:.4f} lots. SL: {sl_price:.{self.symbol_info_bt.digits if self.symbol_info_bt else 5}f}, TP: {tp_price:.{self.symbol_info_bt.digits if self.symbol_info_bt else 5}f}")

                # 3. Execute trade with calculated size, SL, and TP
                # Backtesting.py `self.buy` / `self.sell` automatically manage positions.
                # `self.position` gives the current open position.
                if final_consensus_signal == 'buy':
                    self.buy(sl=sl_price, tp=tp_price, size=lot_size)
                elif final_consensus_signal == 'sell':
                    self.sell(sl=sl_price, tp=tp_price, size=lot_size)

                # Update internal state for cooldown simulation
                # FIXED: The timestamp is already timezone-aware, so we remove the call to tz_localize.
                self.bt_last_trade_times[self.symbol_bt] = df_for_strategy['time'].iloc[-1]
                self.bt_open_positions_count = len(self.positions) # Update after placing order

            except Exception as e:
                logger.error(f"Error during trade sizing or execution for {self.symbol_bt} ({final_consensus_signal}): {e}", exc_info=True)
                # This could happen if symbol_info_bt or calculated values are problematic.
                # Do not immediately raise, allow backtest to continue but log errors.
                return

        # --- Handle position management (simulated secure profit, trailing stop) ---
        # Backtesting.py's `self.position.sl` can be modified directly within `next()`
        # However, Backtesting.py's built-in `trailing_stop` is simpler if basic trailing is needed.
        # To truly replicate the full `manage_open_trades` from main.py, you would need to iterate
        # through `self.positions` and apply the logic.

        # For simplicity, let's just make sure the initial SL/TP are set, and
        # any `modify_position_sl` from `main.py` would need to be simulated manually
        # by checking `self.position` and updating `self.position.sl`.

        # Backtesting.py automatically handles SL/TP hits.
        # To replicate profit securing / trailing stop, you'd need to add logic here:
        if self.position:
            cfg = bot_config_for_backtesting
            ps_sl_config_main = cfg.get('profit_securing_stop_loss', {"enabled": False})

            if ps_sl_config_main.get("enabled", False):
                pos = self.position # Current open position
                current_price = self.data.Close[-1]
                entry_price = pos.entry_price
                trade_type = mt5.ORDER_TYPE_BUY if pos.is_long else mt5.ORDER_TYPE_SELL

                symbol_point = self.symbol_info_bt.point if self.symbol_info_bt else 0.00001
                current_profit_pips = 0
                if symbol_point > 0:
                    current_profit_pips = ((current_price - entry_price) / symbol_point) if pos.is_long else ((entry_price - current_price) / symbol_point)

                ps_sl_symbol_specific_config = ps_sl_config_main.get(self.symbol_bt, ps_sl_config_main.get("default_settings", {}))

                # Trailing stop logic for Backtesting.py - this simplifies the complex logic from main.py
                # This requires careful thought as Backtesting.py has its own `trailing_stop` method,
                # but it might not map directly to your complex trailing logic.
                # For precise replication, you would track `highest_profit_pips_overall` yourself.

                # Simplified example: if profit is X pips, move SL to breakeven + Y pips
                # This is just an illustrative simplified profit securing.
                # For full replication, you'd need to manage a state dictionary like `self.trade_management_states`
                # as done in `main.py` and implement the `initial_secure_done` and `trail_reference_price` logic.
                trigger_pips = ps_sl_symbol_specific_config.get("trigger_profit_pips", 50)
                secure_pips_fixed = ps_sl_symbol_specific_config.get("secure_profit_fixed_pips", 10)

                # Check for initial profit securing
                if current_profit_pips >= trigger_pips:
                    new_sl_price = 0
                    if pos.is_long:
                        new_sl_price = max(pos.sl, entry_price + (secure_pips_fixed * symbol_point))
                    else:
                        new_sl_price = min(pos.sl, entry_price - (secure_pips_fixed * symbol_point))

                    if pos.sl != new_sl_price: # Only modify if different
                        # Backtesting.py's position object has a mutable `sl` attribute
                        pos.sl = new_sl_price
                        logger.debug(f"BACKTEST ({self.symbol_bt}): Secured profit. SL moved to {new_sl_price:.{self.symbol_info_bt.digits if self.symbol_info_bt else 5}f}.")

                # Further trailing logic can be added here following main.py's structure
                # This would involve tracking peak profit and adjusting SL based on trailing_method.
                # This is a place where you'd have to choose between Backtesting.py's native trailing
                # or manually implementing the full complex logic from main.py.
                # Manual implementation will require more internal state in GenericBacktestStrategy.
                # For this update, focusing on getting trades to happen first, then fine-tuning profit management.


# Function to fetch data for backtest, largely unchanged
def fetch_data_for_range_backtest(symbol, tf_str, start_date, end_date):
    """Fetches historical data from MT5 for a specific date range."""
    logger.info(f"Fetching data for {symbol} ({tf_str}) from {start_date.date()} to {end_date.date()}")

    df = get_historical_data_from_mt5(symbol, tf_str, from_date=start_date, to_date=end_date)
    if df is None or df.empty:
        logger.warning(f"No data returned from get_historical_data_from_mt5 for {symbol} in range.")
        return pd.DataFrame()

    required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Fetched data is missing required columns. Got: {df.columns.tolist()}")
        return pd.DataFrame()

    # Ensure 'time' column is datetime and timezone-aware (UTC)
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], utc=True)
    elif df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(timezone.utc)

    logger.info(f"Fetched {len(df)} bars for {symbol} ({tf_str}).")
    return df


def run_backtest_main(strategy_name, symbol, timeframe_str, start_date_str, end_date_str,
                      initial_cash, commission_bps, strategy_config_params, ml_enabled=True):
    """
    Main function to orchestrate the backtest process.
    Connects to MT5, fetches data, trains ML model (if enabled), runs backtest, and plots results.
    Note: For symbols like XAUUSD, a higher initial_cash (e.g., 100000) is recommended.
    """
    global bot_config_for_backtesting
    logger.info("--- Starting Backtest Run ---")

    # Ensure the TradingBot instance and its config are loaded
    bot_instance = _get_bot_instance_for_backtesting()
    if not bot_config_for_backtesting:
        return {"error": "Critical: Bot configuration could not be loaded for backtesting."}

    # Ensure MT5 is connected for symbol info and data fetching
    if not connect_mt5(bot_config_for_backtesting.get('mt5_credentials')):
        return {"error": "MT5 connection failed. Cannot run backtest."}

    try:
        # FIXED: Adjust date range to prevent fetching only a single bar.
        # errors='coerce' handles invalid date strings gracefully.
        start_dt_base = pd.to_datetime(start_date_str, errors='coerce', utc=True)
        end_dt_base = pd.to_datetime(end_date_str, errors='coerce', utc=True)

        if pd.NaT in [start_dt_base, end_dt_base]:
            return {"error": f"Invalid date format. Please use YYYY-MM-DD. Got start={start_date_str}, end={end_date_str}"}

        # Add time adjustments to ensure the full date range is included in the data fetch.
        start_date = start_dt_base + timedelta(hours=1)
        end_date = end_dt_base + timedelta(days=1)

    except (ValueError, TypeError) as e:
        return {"error": f"Invalid date format or parsing error: {e}. Please use YYYY-MM-DD."}

    # Fetch symbol info once at the start for accurate calculations
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"Failed to fetch symbol_info for {symbol} from MT5. Error: {mt5.last_error()}")
        return {"error": f"Failed to fetch symbol_info for {symbol} from MT5. Cannot run backtest."}

    # ML Pre-training
    ml_enabled_for_this_run = ml_enabled
    if ml_enabled:
        ml_training_duration_days = bot_config_for_backtesting.get('ml_backtest_training_days', 365)
        # Ensure training period ENDS BEFORE backtest_data_df STARTS to avoid data leakage
        ml_training_end_date = start_date - timedelta(microseconds=1) # Just before backtest start
        ml_training_start_date = ml_training_end_date - timedelta(days=ml_training_duration_days)

        logger.info(f"ML Pre-training data range: {ml_training_start_date} to {ml_training_end_date}")
        ml_train_df = fetch_data_for_range_backtest(symbol, timeframe_str, ml_training_start_date, ml_training_end_date)

        if ml_train_df.empty or len(ml_train_df) < (bot_config_for_backtesting.get('ml_training_window', 120) + bot_config_for_backtesting.get('ml_prediction_horizon', 16) + 1):
            logger.warning(f"ML enabled, but insufficient data for pre-training ({len(ml_train_df)} bars for {symbol} {timeframe_str}). Disabling ML for this run.")
            ml_enabled_for_this_run = False
        elif not train_ml_model_for_backtest_period(symbol, timeframe_str, ml_train_df):
            logger.warning("ML model pre-training failed. Disabling ML for this run.")
            ml_enabled_for_this_run = False
        else:
            logger.info("ML model successfully pre-trained for backtest.")


    # Fetch data for the actual backtest period
    backtest_data_df = fetch_data_for_range_backtest(symbol, timeframe_str, start_date, end_date)
    if backtest_data_df.empty:
        return {"error": f"No historical data found for {symbol} on {timeframe_str} in range {start_date_str} - {end_date_str}."}

    # ADDED: Data validation to ensure enough bars for a meaningful backtest.
    if len(backtest_data_df) < 100:
        return {"error": f"Insufficient data ({len(backtest_data_df)} bars). Need min 100 bars."}

    # Prepare data for Backtesting.py
    # Backtesting.py expects 'Open', 'High', 'Low', 'Close', 'Volume' and a DatetimeIndex
    backtest_data_df.set_index('time', inplace=True)
    backtest_data_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    backtest_data_df = backtest_data_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    if backtest_data_df.empty:
        return {"error": "Historical data became empty after processing (e.g., dropping NaNs). No data to backtest."}

    # Check for non-numeric columns in data, as Backtesting.py is strict
    non_numeric_cols = backtest_data_df.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        logger.error(f"Data contains non-numeric columns: {non_numeric_cols.tolist()}. Please ensure all OHLCV columns are numeric.")
        return {"error": f"Data contains non-numeric columns: {non_numeric_cols.tolist()}. Cannot run backtest."}


    strategy_params_for_bt_run = {
        'strategy_name_to_run': strategy_name,
        'strategy_specific_params': strategy_config_params,
        'symbol_bt': symbol,
        'timeframe_bt': timeframe_str,
        'ml_validation_enabled_param': ml_enabled_for_this_run,
        'symbol_info_bt': symbol_info # Pass symbol info to the strategy
    }

    stats = None
    try:
        bt = Backtest(backtest_data_df, GenericBacktestStrategy,
                      cash=initial_cash, commission=commission_bps / 10000.0,
                      trade_on_close=True, exclusive_orders=True) # trade_on_close=True is often better for backtesting

        logger.info(f"Running backtest for {strategy_name} on {symbol} ({timeframe_str})...")
        stats = bt.run(**strategy_params_for_bt_run)
        logger.info("Backtest.run() completed.")

    except Exception as e:
        logger.error(f"Error during Backtest.run(): {e}", exc_info=True)
        return {"error": f"Backtesting engine error: {str(e)}"}

    plot_dir = os.path.join("static", "backtest_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f"plot_{symbol}_{timeframe_str}_{strategy_name}_{int(datetime.now(timezone.utc).timestamp())}.html"
    full_plot_path = os.path.join(plot_dir, plot_filename)
    plot_url_path = f"/static/backtest_plots/{plot_filename}"

    try:
        # Plotting uses the same internal data as the stats object.
        # If stats object indicates no trades, the plot will reflect that.
        bt.plot(filename=full_plot_path, open_browser=False)
        logger.info(f"Backtest plot saved to: {full_plot_path}")
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        plot_url_path = None

    serializable_stats = {}
    if isinstance(stats, pd.Series):
        for k, v in stats.items():
            if isinstance(v, (pd.Timestamp, datetime)): serializable_stats[k] = v.isoformat()
            elif isinstance(v, timedelta): serializable_stats[k] = str(v)
            elif isinstance(v, (np.integer, np.floating, np.bool_)): serializable_stats[k] = v.item() # .item() for scalar numpy types
            elif pd.isna(v): serializable_stats[k] = None
            elif not isinstance(v, (pd.DataFrame, pd.Series, Strategy)): serializable_stats[k] = v
            else: # Convert other complex types to string representation if they aren't handled
                serializable_stats[k] = str(v)

    trades_list = []
    if stats is not None and '_trades' in stats and isinstance(stats['_trades'], pd.DataFrame):
        trades_df = stats['_trades'].copy()
        # Convert datetime columns to string for JSON serialization
        for col in ['EntryTime', 'ExitTime']:
            if col in trades_df.columns:
                # Ensure datetime conversion is robust to NaT (Not a Time)
                trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

        # Convert any numpy types within the DataFrame to Python native types
        trades_df = trades_df.replace({np.nan: None}) # Replace NaN with None for JSON
        trades_df = trades_df.applymap(lambda x: x.item() if isinstance(x, (np.integer, np.floating, np.bool_)) else x)

        trades_list = trades_df.to_dict(orient='records')

    results = {
        "stats": serializable_stats,
        "plot_file": plot_url_path,
        "trades": trades_list
    }
    logger.info("--- Backtest Run Completed ---")
    return results

if __name__ == '__main__':
    logger.info("backtester.py can be run directly for testing. Ensure MT5 is configured and running.")
   