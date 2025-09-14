import pandas as pd
import logging
from .base_strategy import BaseStrategy
import pandas_ta as ta

logger = logging.getLogger(__name__)

class HybridStrategy(BaseStrategy):
    """
    Implements a comprehensive strategy using:
    - 200 EMA for main trend direction.
    - ADAPTIVE EMA crossover for the entry trigger based on market regime.
    - ADX for trend strength confirmation.
    - Stochastic Oscillator for final buy/sell confirmation.

    MODIFIED: Now includes both 'Crossover' and 'Pullback/Trend Continuation' entry logic
    to increase trade frequency in established trends.
    """
    def __init__(self, name, **params):
        super().__init__(name)
        self.params = {}
        # Set default values, which will be overwritten by the config
        self.trend_ema_period = 200
        self.adx_period = 14
        self.adx_strength_threshold = 25
        self.stoch_k = 10
        self.stoch_d = 3
        self.stoch_smooth_k = 3
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        # Adaptive EMA defaults
        self.ema_fast_period = 14 # Default fast
        self.ema_slow_period = 28 # Default slow
        self.trending_threshold = 25
        self.ranging_threshold = 20
        self.trending_settings = {}
        self.ranging_settings = {}
        if params:
            self.set_config(params)

    def set_config(self, config: dict):
        """Loads parameters from the configuration file."""
        self.params = config
        hybrid_params = self.params.get('hybrid_strategy_params', {})

        # Main filters
        self.trend_ema_period = hybrid_params.get('trend_ema_period', 200)
        self.adx_period = hybrid_params.get('adx_period', 14)
        self.adx_strength_threshold = hybrid_params.get('adx_strength_threshold', 25)

        # Stochastic params
        self.stoch_k = hybrid_params.get('stoch_k', 10)
        self.stoch_d = hybrid_params.get('stoch_d', 3)
        self.stoch_smooth_k = hybrid_params.get('stoch_smooth_k', 3)
        self.stoch_oversold = hybrid_params.get('stoch_oversold', 20)
        self.stoch_overbought = hybrid_params.get('stoch_overbought', 80)

        # Adaptive EMA params
        self.ema_fast_period = hybrid_params.get('ema_fast_period', 14)
        self.ema_slow_period = hybrid_params.get('ema_slow_period', 28)
        self.trending_threshold = hybrid_params.get('trending_threshold', 25)
        self.ranging_threshold = hybrid_params.get('ranging_threshold', 20)
        self.trending_settings = hybrid_params.get('trending_params', {})
        self.ranging_settings = hybrid_params.get('ranging_params', {})

        logger.info("HybridStrategy config reloaded with ADAPTIVE EMAs and Stochastic.")

    def _calculate_indicators(self, data: pd.DataFrame, fast_ema: int, slow_ema: int):
        """Helper to calculate all required indicators."""
        df = data.copy()
        if df.empty: return df

        # 1. EMAs for trend and trigger
        df['ema_trend_filter'] = ta.ema(df['close'], length=self.trend_ema_period)
        df['ema_fast'] = ta.ema(df['close'], length=fast_ema)
        df['ema_slow'] = ta.ema(df['close'], length=slow_ema)

        # 2. ADX for trend strength and regime detection
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        if adx_data is not None and not adx_data.empty:
            df['adx'] = adx_data[f'ADX_{self.adx_period}']

        # 3. Stochastic Oscillator for confirmation
        stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=self.stoch_k, d=self.stoch_d, smooth_k=self.stoch_smooth_k)
        if stoch_data is not None and not stoch_data.empty:
            df['stoch_signal'] = stoch_data[f'STOCHd_{self.stoch_k}_{self.stoch_d}_{self.stoch_smooth_k}']

        return df

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """
        Generates a trade signal based on the full hybrid logic with adaptive EMAs.
        """
        # --- 1. Determine Adaptive EMA Parameters ---
        fast_ema, slow_ema = self.ema_fast_period, self.ema_slow_period
        market_regime = "default"

        # Calculate ADX first to determine market regime
        adx_data = ta.adx(data['high'], data['low'], data['close'], length=self.adx_period)
        if adx_data is not None and not adx_data.empty and f'ADX_{self.adx_period}' in adx_data.columns:
            latest_adx_for_regime = adx_data[f'ADX_{self.adx_period}'].iloc[-1]
            if latest_adx_for_regime > self.trending_threshold:
                fast_ema = self.trending_settings.get('ema_fast_period', fast_ema)
                slow_ema = self.trending_settings.get('ema_slow_period', slow_ema)
                market_regime = "Trending"
            elif latest_adx_for_regime < self.ranging_threshold:
                fast_ema = self.ranging_settings.get('ema_fast_period', fast_ema)
                slow_ema = self.ranging_settings.get('ema_slow_period', slow_ema)
                market_regime = "Ranging"

        required_len = max(self.trend_ema_period, fast_ema, slow_ema) + 5
        if len(data) < required_len:
            return 'hold', 0.0, None

        # --- 2. Calculate all indicators with chosen EMA params ---
        df = self._calculate_indicators(data, fast_ema, slow_ema)
        if len(df) < 2:
            return 'hold', 0.0, None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Check that all indicator values are valid
        required_cols = ['ema_trend_filter', 'ema_fast', 'ema_slow', 'adx', 'stoch_signal']
        if any(pd.isna(latest.get(col)) for col in required_cols) or pd.isna(prev.get('stoch_signal')):
            return 'hold', 0.0, None

        # --- NEW: Dynamic ADX Strength Threshold Based on Regime ---
        # Adapt threshold to market conditions for more frequent trades in varying regimes
        if market_regime == "Trending":
            dynamic_adx_threshold = 25  # Keep strict for strong trends
        elif market_regime == "Ranging":
            dynamic_adx_threshold = 15  # Relax for ranging to capture pullbacks
        else:
            dynamic_adx_threshold = 20  # Balanced for default/moderate

        # --- 3. Define Conditions ---
        # Main Filters
        buy_trend_ok = latest['close'] > latest['ema_trend_filter']
        sell_trend_ok = latest['close'] < latest['ema_trend_filter']
        strength_ok = latest['adx'] >= dynamic_adx_threshold

        # Stochastic Confirmation (used by both entry types)
        buy_confirmation = prev['stoch_signal'] <= self.stoch_oversold and latest['stoch_signal'] > self.stoch_oversold
        sell_confirmation = prev['stoch_signal'] >= self.stoch_overbought and latest['stoch_signal'] < self.stoch_overbought

        # --- NEW: Define two different types of entry triggers ---
        # A. Crossover Entry (Original Logic): Catches the start of a trend.
        buy_crossover_trigger = prev['ema_fast'] <= prev['ema_slow'] and latest['ema_fast'] > latest['ema_slow']
        sell_crossover_trigger = prev['ema_fast'] >= prev['ema_slow'] and latest['ema_fast'] < latest['ema_slow']

        # B. Pullback (Trend Continuation) Entry: Catches a dip in an established trend.
        buy_pullback_trigger = latest['ema_fast'] > latest['ema_slow']
        sell_pullback_trigger = latest['ema_fast'] < latest['ema_slow']

        # --- 4. Generate Final Signal ---
        # A buy signal is generated if the main trend and strength are OK, AND the stochastic confirms it,
        # AND (EITHER a new crossover just happened OR we are entering on a pullback in an established trend).
        if buy_trend_ok and strength_ok and buy_confirmation and (buy_crossover_trigger or buy_pullback_trigger):
            entry_type = "Crossover" if buy_crossover_trigger else "Pullback"
            logger.info(f"HybridStrategy ({symbol} {timeframe}): BUY signal. Type: {entry_type}. Regime: {market_regime}. Conditions met.")
            return 'buy', 1.0, {}

        # A sell signal is generated with the same logic in the opposite direction.
        if sell_trend_ok and strength_ok and sell_confirmation and (sell_crossover_trigger or sell_pullback_trigger):
            entry_type = "Crossover" if sell_crossover_trigger else "Pullback"
            logger.info(f"HybridStrategy ({symbol} {timeframe}): SELL signal. Type: {entry_type}. Regime: {market_regime}. Conditions met.")
            return 'sell', 1.0, {}

        return 'hold', 0.0, None