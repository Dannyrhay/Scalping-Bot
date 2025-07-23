import pandas as pd
import logging
from .base_strategy import BaseStrategy
import numpy as np
import pandas_ta as ta

logger = logging.getLogger(__name__)

class AtrEmaScalper(BaseStrategy):
    """
    Implements an advanced scalping strategy with dynamic parameters,
    trend filtering, and multi-timeframe confirmation.
    """
    def __init__(self, name, **params):
        super().__init__(name)
        self.signal_strength = 1.0
        # Default parameters, will be overwritten by config
        self.ema_fast_period = 14
        self.ema_slow_period = 28
        self.trend_filter_enabled = False
        self.dynamic_params_enabled = False
        self.mtf_enabled = False
        # Initialize with empty config
        self.params = {}
        if params:
            self.set_config(params)

    def set_config(self, config: dict):
        """Loads parameters from the configuration file."""
        self.params = config
        # Main scalper params (used if dynamic params are off)
        scalper_params = self.params.get('atr_ema_scalper_params', {})
        self.ema_fast_period = scalper_params.get('ema_fast_period', 14)
        self.ema_slow_period = scalper_params.get('ema_slow_period', 28)

        # Trend Filter settings
        trend_params = self.params.get('trend_filter', {})
        self.trend_filter_enabled = trend_params.get('enabled', False)
        self.trend_ema_period = trend_params.get('ema_period', 50)
        self.trend_slope_period = trend_params.get('slope_period', 3)

        # Dynamic Parameter settings
        dyn_params = self.params.get('dynamic_parameters', {})
        self.dynamic_params_enabled = dyn_params.get('enabled', False)
        self.adx_period = dyn_params.get('adx_period', 14)
        self.trending_threshold = dyn_params.get('trending_threshold', 25)
        self.ranging_threshold = dyn_params.get('ranging_threshold', 20)
        self.trending_settings = dyn_params.get('trending_params', {})
        self.ranging_settings = dyn_params.get('ranging_params', {})

        # Multi-Timeframe settings
        mtf_params = self.params.get('multi_timeframe_confirmation', {})
        self.mtf_enabled = mtf_params.get('enabled', False)

        logger.info("AtrEmaScalper config loaded/reloaded.")

    def _calculate_indicators(self, data: pd.DataFrame, fast_ema: int, slow_ema: int):
        """Helper to calculate all required indicators."""
        df = data.copy()
        if df.empty:
            return df

        # Calculate fast and slow EMAs for signals
        df['ema_fast'] = df['close'].ewm(span=fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_ema, adjust=False).mean()

        # Calculate Trend Filter EMA and its slope
        if self.trend_filter_enabled:
            df['ema_trend'] = df['close'].ewm(span=self.trend_ema_period, adjust=False).mean()
            # Positive slope = 1, Negative slope = -1
            df['ema_trend_slope'] = np.sign(df['ema_trend'].diff(self.trend_slope_period))

        # Calculate ADX for Dynamic Parameters
        if self.dynamic_params_enabled:
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
            if adx_data is not None and not adx_data.empty:
                 df['adx'] = adx_data[f'ADX_{self.adx_period}']

        return df

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """
        Generates a trade signal based on the configured strategy logic.
        """
        # --- 1. Determine EMA Parameters (Dynamic or Fixed) ---
        fast_ema, slow_ema = self.ema_fast_period, self.ema_slow_period
        market_regime = "default"

        if self.dynamic_params_enabled:
            # Calculate ADX just to determine regime
            adx_data = ta.adx(data['high'], data['low'], data['close'], length=self.adx_period)
            if adx_data is not None and not adx_data.empty and f'ADX_{self.adx_period}' in adx_data.columns:
                latest_adx = adx_data[f'ADX_{self.adx_period}'].iloc[-1]
                if latest_adx > self.trending_threshold:
                    fast_ema = self.trending_settings.get('ema_fast_period', fast_ema)
                    slow_ema = self.trending_settings.get('ema_slow_period', slow_ema)
                    market_regime = "Trending"
                elif latest_adx < self.ranging_threshold:
                    fast_ema = self.ranging_settings.get('ema_fast_period', fast_ema)
                    slow_ema = self.ranging_settings.get('ema_slow_period', slow_ema)
                    market_regime = "Ranging"
                else:
                    market_regime = "Neutral" # In-between, use default/ranging params
                    fast_ema = self.ranging_settings.get('ema_fast_period', fast_ema)
                    slow_ema = self.ranging_settings.get('ema_slow_period', slow_ema)

        # Ensure we have enough data for the longest calculation
        required_len = max(fast_ema, slow_ema, self.trend_ema_period if self.trend_filter_enabled else 0) + self.trend_slope_period + 2
        if len(data) < required_len:
            return 'hold', 0.0, None

        # --- 2. Calculate all indicators with the chosen parameters ---
        df = self._calculate_indicators(data, fast_ema, slow_ema)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(latest.ema_slow) or pd.isna(prev.ema_slow):
            return 'hold', 0.0, None

        # --- 3. Apply Trend Filter ---
        if self.trend_filter_enabled:
            if pd.isna(latest.get('ema_trend_slope')) or latest.ema_trend_slope == 0:
                return 'hold', 0.0, None # Not enough data for slope or flat trend

            # For buys, price must be above trend EMA and trend must be sloping up
            if latest['close'] < latest['ema_trend'] or latest['ema_trend_slope'] < 0:
                long_condition = False # Override any buy signal if trend is down
            else:
                long_condition = prev['ema_fast'] < prev['ema_slow'] and latest['ema_fast'] > latest['ema_slow']

            # For sells, price must be below trend EMA and trend must be sloping down
            if latest['close'] > latest['ema_trend'] or latest['ema_trend_slope'] > 0:
                short_condition = False # Override any sell signal if trend is up
            else:
                short_condition = prev['ema_fast'] > prev['ema_slow'] and latest['ema_fast'] < latest['ema_slow']
        else:
            # Original logic without trend filter
            long_condition = prev['ema_fast'] < prev['ema_slow'] and latest['ema_fast'] > latest['ema_slow']
            short_condition = prev['ema_fast'] > prev['ema_slow'] and latest['ema_fast'] < latest['ema_slow']

        signal_params = {'ema_fast': fast_ema, 'ema_slow': slow_ema}

        # --- 4. Generate Signal ---
        if long_condition:
            logger.info(f"AtrEmaScalper ({symbol} {timeframe}): BUY signal. Regime: {market_regime}. EMAs({fast_ema}/{slow_ema}).")
            return 'buy', self.signal_strength, signal_params

        if short_condition:
            logger.info(f"AtrEmaScalper ({symbol} {timeframe}): SELL signal. Regime: {market_regime}. EMAs({fast_ema}/{slow_ema}).")
            return 'sell', self.signal_strength, signal_params

        return 'hold', 0.0, None

    def check_confirmation(self, confirmation_data: pd.DataFrame, signal: str, signal_params: dict):
        """
        Checks if the higher timeframe data confirms the signal.
        """
        if not signal_params or 'ema_fast' not in signal_params or 'ema_slow' not in signal_params:
            logger.warning("Confirmation check failed: Missing signal parameters.")
            return False

        fast_ema_period = signal_params['ema_fast']
        slow_ema_period = signal_params['ema_slow']

        df = confirmation_data.copy()
        df['ema_fast'] = df['close'].ewm(span=fast_ema_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_ema_period, adjust=False).mean()

        if len(df) < 2:
            return False

        latest = df.iloc[-1]
        if pd.isna(latest.ema_fast) or pd.isna(latest.ema_slow):
            return False

        if signal == 'buy':
            # For a buy signal, confirm that the fast EMA is above the slow EMA on the higher timeframe.
            return latest['ema_fast'] > latest['ema_slow']
        elif signal == 'sell':
            # For a sell signal, confirm that the fast EMA is below the slow EMA on the higher timeframe.
            return latest['ema_fast'] < latest['ema_slow']

        return False
