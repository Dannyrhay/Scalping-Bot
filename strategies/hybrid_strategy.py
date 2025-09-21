import pandas as pd
import logging
from .base_strategy import BaseStrategy
import pandas_ta as ta

logger = logging.getLogger(__name__)

class HybridStrategy(BaseStrategy):
    """
    HybridStrategy (refactored):
      - ADX used as regime classifier (Trending / Moderate / Ranging)
      - Flexible stochastic confirmation (%K/%D cross or %K rising)
      - Pullback entries require a price dip/spike relative to ema_fast
      - Noise filter using small candle body vs ATR ratio
      - Rich debug logging with reasons for HOLD/BUY/SELL

    NOTE: This class does NOT handle SL/TP or position sizing (those remain in main.py).
    """

    def __init__(self, name, **params):
        super().__init__(name)
        self.params = {}
        # defaults (can be overridden by set_config)
        self.trend_ema_period = 200
        self.adx_period = 14
        self.adx_strength_threshold = 25
        self.adx_moderate_threshold = 15
        self.stoch_k = 10
        self.stoch_d = 3
        self.stoch_smooth_k = 3
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        self.ema_fast_period = 14
        self.ema_slow_period = 28
        self.trending_threshold = 25
        self.ranging_threshold = 20
        self.trending_settings = {}
        self.ranging_settings = {}
        # noise filter
        self.min_candle_body_atr_ratio = 0.15  # require candle body >= 15% of ATR by default
        # meaningful move multiplier (how deep a pullback must be relative to ATR)
        self.pullback_min_atr_multiplier = 0.25

        if params:
            self.set_config(params)

    def set_config(self, config: dict):
        """Load parameters from provided config dict (keeps backward compatibility)."""
        self.params = config
        hybrid_params = self.params.get('hybrid_strategy_params', {})
        dyn_params = self.params.get('dynamic_parameters', {})

        # core
        self.trend_ema_period = hybrid_params.get('trend_ema_period', self.trend_ema_period)
        self.adx_period = hybrid_params.get('adx_period', self.adx_period)
        self.adx_strength_threshold = hybrid_params.get('adx_strength_threshold', self.adx_strength_threshold)

        # stochastic
        self.stoch_k = hybrid_params.get('stoch_k', self.stoch_k)
        self.stoch_d = hybrid_params.get('stoch_d', self.stoch_d)
        self.stoch_smooth_k = hybrid_params.get('stoch_smooth_k', self.stoch_smooth_k)
        self.stoch_oversold = hybrid_params.get('stoch_oversold', self.stoch_oversold)
        self.stoch_overbought = hybrid_params.get('stoch_overbought', self.stoch_overbought)

        # EMAs & regime thresholds
        self.ema_fast_period = hybrid_params.get('ema_fast_period', self.ema_fast_period)
        self.ema_slow_period = hybrid_params.get('ema_slow_period', self.ema_slow_period)

        # dynamic/regime params (fallback to hybrid block if not present)
        self.trending_threshold = dyn_params.get('trending_threshold', hybrid_params.get('trending_threshold', self.trending_threshold))
        self.ranging_threshold = dyn_params.get('ranging_threshold', hybrid_params.get('ranging_threshold', self.ranging_threshold))
        self.trending_settings = dyn_params.get('trending_params', hybrid_params.get('trending_params', self.trending_settings))
        self.ranging_settings = dyn_params.get('ranging_params', hybrid_params.get('ranging_params', self.ranging_settings))

        # noise / pullback tuning
        self.min_candle_body_atr_ratio = hybrid_params.get('min_candle_body_atr_ratio', self.min_candle_body_atr_ratio)
        self.pullback_min_atr_multiplier = hybrid_params.get('pullback_min_atr_multiplier', self.pullback_min_atr_multiplier)

        logger.info("HybridStrategy config loaded/updated.")

    def _calculate_indicators(self, data: pd.DataFrame, fast_ema: int, slow_ema: int):
        df = data.copy()
        if df.empty:
            return df

        # EMAs
        df['ema_trend_filter'] = ta.ema(df['close'], length=self.trend_ema_period)
        df['ema_fast'] = ta.ema(df['close'], length=fast_ema)
        df['ema_slow'] = ta.ema(df['close'], length=slow_ema)

        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        if adx is not None and not adx.empty:
            df['adx'] = adx[f'ADX_{self.adx_period}']

        # Stochastic K and D
        stoch = ta.stoch(df['high'], df['low'], df['close'],
                         k=self.stoch_k, d=self.stoch_d, smooth_k=self.stoch_smooth_k)
        if stoch is not None and not stoch.empty:
            k_col = f'STOCHk_{self.stoch_k}_{self.stoch_d}_{self.stoch_smooth_k}'
            d_col = f'STOCHd_{self.stoch_k}_{self.stoch_d}_{self.stoch_smooth_k}'
            if k_col in stoch.columns:
                df['stoch_k'] = stoch[k_col]
            if d_col in stoch.columns:
                df['stoch_d'] = stoch[d_col]

        # ATR (used for noise/pullback checks)
        atr_len = max(14, self.adx_period)
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
        if atr_series is not None and not atr_series.empty:
            df['atr'] = atr_series


        # Candle body size
        df['candle_body'] = (df['close'] - df['open']).abs()

        return df

    def _regime_from_adx(self, adx_val: float):
        """Return regime string and a simple tier flag for logic decisions."""
        if adx_val is None or pd.isna(adx_val):
            return "Unknown"
        if adx_val >= self.trending_threshold:
            return "Trending"
        if adx_val < self.ranging_threshold:
            return "Ranging"
        return "Moderate"

    def _stoch_buy_fast(self, prev, latest):
        """Fast stochastic confirmation: K crosses above D (or K rising from low)."""
        # require non-na
        if pd.isna(prev.get('stoch_k')) or pd.isna(prev.get('stoch_d')) or pd.isna(latest.get('stoch_k')) or pd.isna(latest.get('stoch_d')):
            return False
        cross_up = (prev['stoch_k'] <= prev['stoch_d']) and (latest['stoch_k'] > latest['stoch_d'])
        rising_from_low = (prev['stoch_k'] < latest['stoch_k']) and (prev['stoch_k'] <= self.stoch_oversold)
        return cross_up or rising_from_low

    def _stoch_sell_fast(self, prev, latest):
        if pd.isna(prev.get('stoch_k')) or pd.isna(prev.get('stoch_d')) or pd.isna(latest.get('stoch_k')) or pd.isna(latest.get('stoch_d')):
            return False
        cross_dn = (prev['stoch_k'] >= prev['stoch_d']) and (latest['stoch_k'] < latest['stoch_d'])
        falling_from_high = (prev['stoch_k'] > latest['stoch_k']) and (prev['stoch_k'] >= self.stoch_overbought)
        return cross_dn or falling_from_high

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """
        Returns ('buy'|'sell'|'hold', strength_float, meta_dict)
        meta_dict may include 'reason', 'regime', 'adx'
        """
        # Determine EMA params based on dynamic parameters if present
        fast_ema, slow_ema = self.ema_fast_period, self.ema_slow_period
        market_regime = "Unknown"
        adx_val = None

        # Compute ADX for regime detection (use last available bars)
        try:
            adx_df = ta.adx(data['high'], data['low'], data['close'], length=self.adx_period)
            col = f'ADX_{self.adx_period}'
            if adx_df is not None and col in adx_df.columns and not adx_df.empty:
                adx_val = adx_df[col].iloc[-1]
                market_regime = self._regime_from_adx(adx_val)
                # adapt EMAs if user set regime-specific params
                if market_regime == "Trending" and isinstance(self.trending_settings, dict):
                    fast_ema = self.trending_settings.get('ema_fast_period', fast_ema)
                    slow_ema = self.trending_settings.get('ema_slow_period', slow_ema)
                elif market_regime == "Ranging" and isinstance(self.ranging_settings, dict):
                    fast_ema = self.ranging_settings.get('ema_fast_period', fast_ema)
                    slow_ema = self.ranging_settings.get('ema_slow_period', slow_ema)
        except Exception as e:
            logger.debug(f"HybridStrategy ({symbol} {timeframe}): ADX regime calc failed: {e}")

        # Minimal lookback
        required_len = max(self.trend_ema_period, fast_ema, slow_ema, self.adx_period, 20) + 2
        if len(data) < required_len:
            return 'hold', 0.0, {'reason': f'Not enough data ({len(data)}/{required_len})'}

        df = self._calculate_indicators(data, fast_ema, slow_ema)
        if len(df) < 2:
            return 'hold', 0.0, {'reason': 'Not enough indicator rows'}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Validate required indicator values
        required_cols = ['ema_trend_filter', 'ema_fast', 'ema_slow', 'stoch_k', 'stoch_d', 'atr', 'candle_body']
        missing = [c for c in required_cols if c not in df.columns or pd.isna(latest.get(c))]
        if missing:
            return 'hold', 0.0, {'reason': f'missing_indicators: {missing}'}

        # Basic trend direction (using trend EMA)
        buy_trend_ok = latest['close'] > latest['ema_trend_filter']
        sell_trend_ok = latest['close'] < latest['ema_trend_filter']

        # ADX-tier flags
        adx_val = latest.get('adx') if pd.notna(latest.get('adx')) else adx_val
        regime = self._regime_from_adx(adx_val) if adx_val is not None else market_regime

        # Candle body vs ATR noise filter
        atr = latest.get('atr', 0.0)
        candle_body = latest.get('candle_body', 0.0)
        if atr and not pd.isna(atr) and atr > 0:
            small_candle = (candle_body / atr) < self.min_candle_body_atr_ratio
        else:
            small_candle = False

        # EMA entry triggers
        buy_crossover = (prev['ema_fast'] <= prev['ema_slow']) and (latest['ema_fast'] > latest['ema_slow'])
        sell_crossover = (prev['ema_fast'] >= prev['ema_slow']) and (latest['ema_fast'] < latest['ema_slow'])

        # Pullback: price must dip below ema_fast (buy) or spike above ema_fast (sell) by a minimum relative to ATR
        pullback_depth = self.pullback_min_atr_multiplier * atr if atr and not pd.isna(atr) else 0.0
        buy_pullback = (latest['ema_fast'] > latest['ema_slow']) and ((latest['ema_fast'] - latest['close']) >= pullback_depth)
        sell_pullback = (latest['ema_fast'] < latest['ema_slow']) and ((latest['close'] - latest['ema_fast']) >= pullback_depth)

        # Stochastic confirmations (fast/flexible)
        stoch_buy_fast = self._stoch_buy_fast(prev, latest)
        stoch_sell_fast = self._stoch_sell_fast(prev, latest)

        # Compose decision rules (tiered by regime)
        meta = {'regime': regime, 'adx': float(adx_val) if adx_val is not None else None}

        # Reject signals in chop (very small candles) to reduce false entries
        if small_candle:
            meta['reason'] = f'small_candle (body/ATR < {self.min_candle_body_atr_ratio})'
            logger.debug(f"HybridStrategy ({symbol} {timeframe}): HOLD due to small candle. {meta}")
            return 'hold', 0.0, meta

        # --- BUY logic ---
        # Trending: prefer strict confirmations (crossover+stoch_fast OR pullback+stoch_fast)
        if buy_trend_ok:
            if regime == "Trending":
                if (buy_crossover and stoch_buy_fast):
                    meta['reason'] = 'Trending: crossover + stoch_fast'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): BUY (Trending). {meta}")
                    return 'buy', 1.0, meta
                if (buy_pullback and stoch_buy_fast):
                    meta['reason'] = 'Trending: pullback + stoch_fast'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): BUY (Trending pullback). {meta}")
                    return 'buy', 0.9, meta
            elif regime == "Moderate":
                # allow lighter confirmations: pullback + either stoch_fast OR K rising from low
                if (buy_pullback and (stoch_buy_fast or (prev.get('stoch_k') <= self.stoch_oversold and latest.get('stoch_k') > prev.get('stoch_k')))):
                    meta['reason'] = 'Moderate: pullback + lighter stoch'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): BUY (Moderate). {meta}")
                    return 'buy', 0.85, meta
                # allow crossover with stoch fast but only if price above trend EMA
                if (buy_crossover and stoch_buy_fast):
                    meta['reason'] = 'Moderate: crossover + stoch_fast'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): BUY (Moderate crossover). {meta}")
                    return 'buy', 0.75, meta
            else:  # Ranging or Unknown
                # Be conservative in ranging: only accept strong stoch bounce from oversold + small pullback
                if prev.get('stoch_k') <= self.stoch_oversold and latest.get('stoch_k') > prev.get('stoch_k') and buy_pullback:
                    meta['reason'] = 'Ranging: mean-reversion style pullback + stoch rise'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): BUY (Ranging mean-reversion). {meta}")
                    return 'buy', 0.7, meta

        # --- SELL logic (mirror) ---
        if sell_trend_ok:
            if regime == "Trending":
                if (sell_crossover and stoch_sell_fast):
                    meta['reason'] = 'Trending: crossover + stoch_fast'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): SELL (Trending). {meta}")
                    return 'sell', 1.0, meta
                if (sell_pullback and stoch_sell_fast):
                    meta['reason'] = 'Trending: pullback + stoch_fast'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): SELL (Trending pullback). {meta}")
                    return 'sell', 0.9, meta
            elif regime == "Moderate":
                if (sell_pullback and (stoch_sell_fast or (prev.get('stoch_k') >= self.stoch_overbought and latest.get('stoch_k') < prev.get('stoch_k')))):
                    meta['reason'] = 'Moderate: pullback + lighter stoch'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): SELL (Moderate). {meta}")
                    return 'sell', 0.85, meta
                if (sell_crossover and stoch_sell_fast):
                    meta['reason'] = 'Moderate: crossover + stoch_fast'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): SELL (Moderate crossover). {meta}")
                    return 'sell', 0.75, meta
            else:
                if prev.get('stoch_k') >= self.stoch_overbought and latest.get('stoch_k') < prev.get('stoch_k') and sell_pullback:
                    meta['reason'] = 'Ranging: mean-reversion style pullback + stoch fall'
                    logger.info(f"HybridStrategy ({symbol} {timeframe}): SELL (Ranging mean-reversion). {meta}")
                    return 'sell', 0.7, meta

        # Default: hold, provide diagnostics
        meta['reason'] = ("HOLD default. "
                          f"buy_trend_ok={buy_trend_ok}, sell_trend_ok={sell_trend_ok}, "
                          f"buy_crossover={buy_crossover}, sell_crossover={sell_crossover}, "
                          f"buy_pullback={buy_pullback}, sell_pullback={sell_pullback}, "
                          f"stoch_buy_fast={stoch_buy_fast}, stoch_sell_fast={stoch_sell_fast}, "
                          f"small_candle={small_candle}")
        logger.debug(f"HybridStrategy ({symbol} {timeframe}): HOLD. {meta}")
        return 'hold', 0.0, meta
