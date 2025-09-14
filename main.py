import logging
import MetaTrader5 as mt5
import pandas as pd
import json
import time
import numpy as np
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from utils.mt5_connection import connect_mt5, get_data
from utils.logging import setup_logging
from utils.trade_history import save_trade, update_trade_status
from utils.db_connector import MongoDBConnection
from strategies.atr_ema_scalper import AtrEmaScalper
from strategies.hybrid_strategy import HybridStrategy
from utils.news_manager import NewsManager
from pymongo import DESCENDING, errors

logger = setup_logging()

class TradingBot:
    """
    A trading bot streamlined and optimized for high-frequency scalping.
    It uses a single, fast strategy and aggressive risk management.
    """
    def __init__(self):
        self.bot_running = False
        self.config_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.bot_thread = None
        self.last_error_message = None
        self.config = {}
        self.trade_management_states = {}
        self.last_trade_times = {}
        self.consecutive_failures = 0
        self.load_config_and_reinitialize() # Initial load
        self.news_manager = NewsManager(self.config)
        self.db = MongoDBConnection.connect()
        if not self.db:
            logger.critical("MongoDB connection failed. Trade history will be unavailable.")

    def load_config_and_reinitialize(self):
        """Loads the configuration file and re-initializes bot components."""
        with self.config_lock:
            logger.info("Loading configuration and reinitializing bot...")
            try:
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
                if not os.path.exists(config_path):
                     # Fallback for different execution structure
                    config_path = os.path.join('config', 'config.json')
                    if not os.path.exists(config_path):
                        raise FileNotFoundError(f"Config file not found.")

                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"CRITICAL: Failed to load config.json: {e}", exc_info=True)
                if not self.config: raise RuntimeError(f"Failed to load initial config: {e}")
                logger.warning("Using previously loaded config due to error.")
                return

            self.strategies = self.initialize_strategies()
            # Pass the full config to each strategy
            for strategy in self.strategies:
                if hasattr(strategy, 'set_config'):
                    strategy.set_config(self.config)

            if hasattr(self, 'news_manager') and self.news_manager:
                self.news_manager.config = self.config

            self.cooldown_period = timedelta(minutes=self.config.get('cooldown_period_minutes', 0.25))
            logger.info("Bot configuration and components reloaded successfully.")

    def initialize_strategies(self):
        """Initializes the active strategy from the config."""
        strategy_constructors = {
            "AtrEmaScalper": (AtrEmaScalper, lambda c: c), # Pass the whole config
            "HybridStrategy": (HybridStrategy, lambda c: c)
        }
        initialized_strategies = []
        active_strategy_names = self.config.get('active_strategies', [])
        logger.info(f"Active strategies to be loaded: {active_strategy_names}")

        for name in active_strategy_names:
            if name in strategy_constructors:
                StrategyClass, params_lambda = strategy_constructors[name]
                try:
                    # The strategy now takes the entire config dict
                    params_dict = params_lambda(self.config)
                    initialized_strategies.append(StrategyClass(name=name, **params_dict))
                    logger.info(f"Successfully initialized strategy: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {name}: {e}", exc_info=True)

        if not initialized_strategies:
            logger.critical("No strategies were initialized. The bot cannot function. Check 'active_strategies' in config.json.")

        return initialized_strategies

    def manage_open_trades(self):
        """Implements aggressive risk management tailored for scalping."""
        ps_sl_config = self.config.get('profit_securing_stop_loss', {})
        time_exit_config = self.config.get('time_based_exit', {})

        if not (ps_sl_config.get("enabled", False) or time_exit_config.get("enabled", False)):
            return

        open_positions = mt5.positions_get()
        if open_positions is None:
            return

        for pos in open_positions:
            if pos.magic != 123456: continue

            symbol, ticket, entry_price = pos.symbol, pos.ticket, pos.price_open
            original_sl, trade_type, volume = pos.sl, pos.type, pos.volume

            tick = mt5.symbol_info_tick(symbol)
            if not tick or tick.time == 0: continue

            current_price = tick.bid if trade_type == mt5.ORDER_TYPE_BUY else tick.ask
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info: continue
            point, digits = symbol_info.point, symbol_info.digits

            current_profit_pips = ((current_price - entry_price) / point) if trade_type == mt5.ORDER_TYPE_BUY else ((entry_price - current_price) / point)

            if ps_sl_config.get("enabled", False):
                ps_params = ps_sl_config.get("default_settings", {})
                trade_state = self.trade_management_states.setdefault(ticket, {"initial_secure_done": False})

                if not trade_state["initial_secure_done"] and current_profit_pips >= ps_params.get("trigger_profit_pips", 10):
                    pips_to_secure = ps_params.get("secure_profit_fixed_pips", 2)
                    new_secure_sl = entry_price + (pips_to_secure * point) if trade_type == mt5.ORDER_TYPE_BUY else entry_price - (pips_to_secure * point)

                    if self.should_move_sl(trade_type, new_secure_sl, original_sl):
                        if self.modify_position_sl(ticket, new_secure_sl, symbol):
                            logger.info(f"MANAGE (Secure SL - {ticket}): Moved SL to breakeven+ at {new_secure_sl:.{digits}f}.")
                            trade_state["initial_secure_done"] = True

                elif trade_state["initial_secure_done"] and ps_params.get("trailing_active", False):
                    pips_behind = ps_params.get("trailing_fixed_pips_value", 8)
                    new_trailing_sl = current_price - (pips_behind * point) if trade_type == mt5.ORDER_TYPE_BUY else current_price + (pips_behind * point)

                    if self.should_move_sl(trade_type, new_trailing_sl, pos.sl):
                         self.modify_position_sl(ticket, new_trailing_sl, symbol)

            entry_tf = self.get_entry_timeframe_from_comment(pos.comment)
            if time_exit_config.get("enabled", False) and entry_tf in time_exit_config.get("apply_to_timeframes", []):
                time_params = time_exit_config.get("default", {})
                max_bars = time_params.get("max_bars_open", 5)
                entry_tf_minutes = self.mt5_tf_to_minutes(entry_tf)

                if entry_tf_minutes > 0:
                    seconds_open = (datetime.now(timezone.utc) - datetime.fromtimestamp(pos.time, tz=timezone.utc)).total_seconds()
                    bars_open = seconds_open / (entry_tf_minutes * 60)
                    if bars_open >= max_bars:
                        logger.info(f"MANAGE (Time Exit - {ticket}): Closing trade. Exceeded max bars ({max_bars}) on {entry_tf}.")
                        self.close_position_by_ticket(ticket, symbol, volume, trade_type, f"{entry_tf} Time Exit")
                        break

    def execute_trade(self, symbol, signal, data, timeframe_str):
        if self.news_manager.is_trade_prohibited(symbol):
            logger.info(f"EXECUTE_TRADE REJECTED ({symbol}): Blocked by news filter.")
            return False

        if self.in_cooldown(symbol) or not self.can_open_new_trade(symbol):
            logger.debug(f"EXECUTE_TRADE REJECTED ({symbol}): In cooldown or max trades reached.")
            return False

        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not symbol_info or not tick or tick.time == 0:
            logger.error(f"EXECUTE_TRADE ({symbol}): Invalid symbol info or tick data.")
            self.consecutive_failures += 1
            return False

        price_entry = tick.ask if signal == 'buy' else tick.bid
        min_stop_points = symbol_info.trade_stops_level * symbol_info.point
        safety_buffer_points = symbol_info.point * 2
        guaranteed_min_dist = min_stop_points + safety_buffer_points

        # --- SL/TP Calculation ---
        atr = self.calculate_atr(data.copy(), period=self.config.get('atr_period_for_sl_tp', 14))
        if pd.isna(atr) or atr <= 0:
            atr = data['close'].iloc[-1] * 0.0005 # Fallback ATR
        atr_sl_distance = self.config.get('sl_atr_multiplier', 1.5) * atr
        sl_distance = max(atr_sl_distance, guaranteed_min_dist)
        tp_distance = sl_distance * self.config.get('risk_reward_ratio', 1.5)

        sl = price_entry - sl_distance if signal == 'buy' else price_entry + sl_distance
        tp = price_entry + tp_distance if signal == 'buy' else price_entry - tp_distance
        sl = round(sl, symbol_info.digits)
        tp = round(tp, symbol_info.digits)

        acc_info = mt5.account_info()
        if not acc_info:
            logger.error("Could not fetch account info.")
            return False

        # --- Dynamic Volatility Risk Management ---
        risk_config = self.config.get('dynamic_volatility_risk', {})
        risk_percent = self.config.get('risk_percent_per_trade', 0.01) # Default
        if risk_config.get('enabled', False):
            avg_atr_period = risk_config.get('atr_avg_period', 50)
            df_atr = data.copy()
            df_atr['tr'] = (df_atr['high'] - df_atr['low']).abs()
            avg_atr = df_atr['tr'].rolling(window=avg_atr_period).mean().iloc[-1]

            if pd.notna(avg_atr) and avg_atr > 0:
                high_thresh = risk_config.get('volatility_threshold_high', 1.5)
                low_thresh = risk_config.get('volatility_threshold_low', 0.7)

                if atr > avg_atr * high_thresh:
                    risk_percent = risk_config.get('risk_percent_high_vol', risk_percent)
                    logger.info(f"VOLATILITY: High (Current ATR: {atr:.5f} > Avg ATR: {avg_atr:.5f}). Using risk: {risk_percent*100}%")
                elif atr < avg_atr * low_thresh:
                    risk_percent = risk_config.get('risk_percent_low_vol', risk_percent)
                    logger.info(f"VOLATILITY: Low (Current ATR: {atr:.5f} < Avg ATR: {avg_atr:.5f}). Using risk: {risk_percent*100}%")
                else:
                    risk_percent = risk_config.get('risk_percent_normal', risk_percent)
                    logger.info(f"VOLATILITY: Normal. Using risk: {risk_percent*100}%")

        # --- Lot Size Calculation ---
        risk_amt = acc_info.balance * risk_percent
        sl_pips = abs(price_entry - sl) / symbol_info.point if symbol_info.point > 0 else 0
        if sl_pips == 0:
            logger.error(f"EXECUTE_TRADE ({symbol}): Calculated SL pips is zero. Aborting trade.")
            return False

        val_per_pip_1_lot = (symbol_info.trade_tick_value / symbol_info.trade_tick_size) * symbol_info.point
        lot = risk_amt / (sl_pips * val_per_pip_1_lot) if val_per_pip_1_lot > 0 else symbol_info.volume_min
        lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step if symbol_info.volume_step > 0 else round(lot, 2)
        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        if lot < symbol_info.volume_min: lot = symbol_info.volume_min

        # --- Margin Check & Order Execution ---
        order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
        margin_required = mt5.order_calc_margin(order_type, symbol, lot, price_entry)
        if margin_required is None or margin_required > acc_info.margin_free:
            logger.error(f"EXECUTE_TRADE REJECTED ({symbol}): Insufficient margin. Required: {margin_required}, Available: {acc_info.margin_free}")
            self.consecutive_failures += 1
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot,
            "type": order_type, "price": price_entry, "sl": sl, "tp": tp,
            "deviation": 20, "magic": 123456,
            "comment": f"AtrEmaScalper {timeframe_str} {signal.upper()}",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trade EXECUTED: {signal.upper()} {symbol} @ {result.price:.{symbol_info.digits}f} Lot: {result.volume}. Order: {result.order}")
            self.last_trade_times[symbol] = datetime.now(timezone.utc)
            save_trade({
                "symbol": symbol, "timeframe": timeframe_str, "signal": signal.upper(),
                "entry_price": result.price, "sl_price": sl, "tp_price": tp, "lot_size": result.volume,
                "strategies": ["AtrEmaScalper"], "entry_time": datetime.now(timezone.utc),
                "status": "open", "profit_loss": 0.0, "account_balance": acc_info.balance,
                "order_id": result.order, "deal_id": result.deal
            })
            self.consecutive_failures = 0
            return True
        else:
            error_msg = result.comment if result else mt5.last_error()
            retcode = result.retcode if result else 'N/A'
            logger.error(f"Trade execution FAILED for {symbol}: {error_msg} (RetCode: {retcode})")
            self.consecutive_failures += 1
            return False

    def monitor_market(self):
        mtf_config = self.config.get('multi_timeframe_confirmation', {})
        mtf_enabled = mtf_config.get('enabled', False)
        confirmation_tf = mtf_config.get('confirmation_timeframe', 'M15')

        while self.bot_running:
            try:
                self.check_closed_trades()
                self.manage_open_trades()

                for symbol in self.config.get('symbols', []):
                    if not self.bot_running: break
                    if not self.is_trading_hours(symbol): continue

                    for tf_str in self.config.get('timeframes', ['M1', 'M5']):
                        if not self.bot_running: break
                        if not self.strategies:
                            logger.error("No strategies loaded, stopping monitor loop.")
                            self.stop_bot_logic()
                            return

                        data = get_data(symbol, tf_str, bars=self.config.get('bars', 100))
                        if data is None or data.empty: continue

                        strategy = self.strategies[0]
                        signal, strength, params = strategy.get_signal(data.copy(), symbol=symbol, timeframe=tf_str)

                        # --- Multi-Timeframe Confirmation Logic ---
                        if signal != 'hold' and mtf_enabled:
                            logger.info(f"MONITOR ({symbol} {tf_str}): Base signal '{signal.upper()}' received. Checking {confirmation_tf} for confirmation...")
                            confirm_data = get_data(symbol, confirmation_tf, bars=self.config.get('bars', 100))
                            if confirm_data is None or confirm_data.empty:
                                logger.warning(f"Could not fetch data for confirmation timeframe {confirmation_tf}. Skipping signal.")
                                signal = 'hold' # Invalidate signal
                            else:
                                is_confirmed = strategy.check_confirmation(confirm_data.copy(), signal, params)
                                if not is_confirmed:
                                    logger.info(f"MONITOR ({symbol} {tf_str}): Signal '{signal.upper()}' REJECTED by {confirmation_tf} confirmation.")
                                    signal = 'hold' # Invalidate signal
                                else:
                                    logger.info(f"MONITOR ({symbol} {tf_str}): Signal '{signal.upper()}' CONFIRMED by {confirmation_tf}.")

                        if signal != 'hold':
                            logger.info(f"MONITOR ({symbol} {tf_str}): Final signal is '{signal.upper()}'. Attempting trade.")
                            # Pass the original timeframe data to execute_trade
                            self.execute_trade(symbol, signal, data.copy(), tf_str)

                if self.consecutive_failures >= self.config.get('max_consecutive_trade_failures', 5):
                    logger.critical(f"{self.consecutive_failures} consecutive failures. Stopping bot for safety.")
                    self.stop_bot_logic()

                time.sleep(self.config.get('monitoring_interval_seconds', 1))

            except Exception as e:
                logger.critical(f"Critical error in monitor_market: {e}", exc_info=True)
                self.last_error_message = str(e)
                self.stop_bot_logic()

    # --- Helper and Utility Functions (largely unchanged) ---
    @staticmethod
    def calculate_atr(data, period=14):
        if not isinstance(data, pd.DataFrame) or data.empty or len(data) < period: return 0.0
        df = data.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift())
        df['l-pc'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        atr = df['tr'].ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        return atr if pd.notna(atr) else 0.0

    def modify_position_sl(self, ticket, new_sl, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info: return False
        request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": round(new_sl, symbol_info.digits), "symbol": symbol}
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Successfully modified SL for ticket {ticket} to {request['sl']}.")
            return True
        logger.warning(f"Failed to modify SL for ticket {ticket}. Error: {result.comment if result else 'N/A'}")
        return False

    def should_move_sl(self, trade_type, new_sl, current_sl):
        if current_sl == 0.0: return True
        if trade_type == mt5.ORDER_TYPE_BUY:
            return new_sl > current_sl
        else:
            return new_sl < current_sl

    def close_position_by_ticket(self, ticket, symbol, volume, trade_type, comment):
        close_order_type = mt5.ORDER_TYPE_SELL if trade_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return False
        price = tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                   "type": close_order_type, "position": ticket, "price": price,
                   "deviation": 20, "magic": 123456, "comment": comment[:31]}
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Close order sent for position {ticket}. Comment: {comment}")
            if ticket in self.trade_management_states:
                del self.trade_management_states[ticket]
            return True
        return False

    def get_entry_timeframe_from_comment(self, comment_str):
        if not isinstance(comment_str, str): return None
        match = re.search(r"([MHDW][1-9]\d*)", comment_str)
        return match.group(1) if match else None

    def mt5_tf_to_minutes(self, tf_str):
        return {'M1': 1,'M3': 3, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}.get(tf_str.upper(), 0)

    def in_cooldown(self, symbol):
        last_trade = self.last_trade_times.get(symbol)
        return last_trade and (datetime.now(timezone.utc) - last_trade) < self.cooldown_period

    def can_open_new_trade(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None: return False
        return len([p for p in positions if p.magic == 123456]) < self.config.get('max_trades_per_symbol', 1)

    def is_trading_hours(self, symbol):
        schedule = self.config.get('trading_hours', {}).get(symbol)
        if not schedule: return True
        now_utc = datetime.now(timezone.utc)
        if now_utc.strftime('%A') not in schedule.get('days', []): return False
        start_time = datetime.strptime(schedule.get('start', '00:00'), '%H:%M').time()
        end_time = datetime.strptime(schedule.get('end', '23:59'), '%H:%M').time()
        if start_time <= end_time:
            return start_time <= now_utc.time() <= end_time
        else:
            return now_utc.time() >= start_time or now_utc.time() <= end_time

    def check_closed_trades(self):
        trades_collection = MongoDBConnection.get_trades_collection()
        if not trades_collection: return
        try:
            open_trades = list(trades_collection.find({"status": "open"}))
            if not open_trades: return
            open_tickets_in_mt5 = {p.ticket for p in mt5.positions_get() or []}
            for trade in open_trades:
                ticket = trade.get("order_id")
                if ticket not in open_tickets_in_mt5:
                    logger.info(f"Reconciling closed trade: {ticket}")
                    deals = mt5.history_deals_get(position=ticket)
                    if deals:
                        profit = sum(d.profit for d in deals if d.position_id == ticket)
                        exit_time = datetime.fromtimestamp(deals[-1].time, tz=timezone.utc)
                        exit_price = deals[-1].price
                        update_trade_status(ticket, {"status": "closed", "profit_loss": profit, "exit_time": exit_time, "exit_price": exit_price, "exit_reason": "Reconciled"})
                    else:
                        update_trade_status(ticket, {"status": "closed_unknown", "exit_reason": "Not in MT5, no history"})
        except errors.PyMongoError as e:
            logger.error(f"MongoDB error during trade reconciliation: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during trade reconciliation: {e}", exc_info=True)

    def get_bot_status_for_ui(self):
        with self.status_lock:
            status = "RUNNING" if self.bot_running else "STOPPED"
            if self.last_error_message: status = "ERROR"
        acc_info = mt5.account_info()
        mt5_connected = mt5.terminal_info() is not None
        if not mt5_connected: status = "MT5_DISCONNECTED"
        return {"bot_status": status, "balance": acc_info.balance if acc_info else "N/A", "equity": acc_info.equity if acc_info else "N/A", "last_error": self.last_error_message}

    def get_open_positions_for_ui(self):
        positions = mt5.positions_get()
        if positions is None: return []
        ui_positions = []
        for pos in positions:
            if pos.magic == 123456:
                ui_positions.append({
                    "ticket": pos.ticket, "symbol": pos.symbol, "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume, "entry_price": pos.price_open, "current_price": pos.price_current,
                    "sl": pos.sl, "tp": pos.tp, "pnl_currency": pos.profit,
                    "entry_time": datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S'),
                    "comment": pos.comment
                })
        return ui_positions

    def get_trade_history_for_ui(self, page=1, limit=20):
        trades_collection = MongoDBConnection.get_trades_collection()
        if not trades_collection: return [], 0
        try:
            query = {"status": {"$in": ["closed", "closed_auto", "closed_unknown"]}}
            total_trades = trades_collection.count_documents(query)
            skip_amount = (page - 1) * limit
            trade_cursor = trades_collection.find(query).sort("entry_time", DESCENDING).skip(skip_amount).limit(limit)
            trades_data = []
            for row in trade_cursor:
                row["_id"] = str(row["_id"])
                trades_data.append(row)
            return trades_data, total_trades
        except errors.PyMongoError as e:
            logger.error(f"Error fetching trade history from MongoDB for UI: {e}", exc_info=True)
            return [], 0

    def delete_bot_logs(self):
        global logger
        log_file_path = 'logs/trading_ea.log'

        if logger: logger.info(f"Attempting to delete log file: {log_file_path}")
        else: print(f"Logger not available when attempting to delete log file: {log_file_path}")

        try:
            handler_to_remove = None
            if logger and hasattr(logger, 'handlers'):
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler) and hasattr(handler, 'baseFilename') and handler.baseFilename and os.path.abspath(handler.baseFilename) == os.path.abspath(log_file_path):
                        handler_to_remove = handler
                        break

            if handler_to_remove:
                handler_to_remove.close()
                logger.removeHandler(handler_to_remove)

            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                msg = f"Log file {log_file_path} deleted successfully."
            else:
                msg = f"Log file {log_file_path} not found, no deletion needed."

            logger = setup_logging()
            logger.info(msg + " Logging re-initialized.")
            return True, msg
        except Exception as e:
            try:
                logger = setup_logging()
                logger.error(f"Error during log deletion for '{log_file_path}': {e}", exc_info=True)
            except Exception as e2:
                print(f"CRITICAL: Error deleting log '{log_file_path}': {e}. FAILED to re-init logging: {e2}")
            return False, f"Error deleting log file: {str(e)}"

    def start_bot_logic(self):
        with self.status_lock:
            if self.bot_running:
                logger.info("Bot is already running. Cannot start again.")
                return False
            self.bot_running = True
            self.last_error_message = None
            self.consecutive_failures = 0
            logger.info("Failure counter reset to 0.")

        if self.bot_thread is None or not self.bot_thread.is_alive():
            self.bot_thread = threading.Thread(target=self.monitor_market, daemon=True)
            self.bot_thread.start()
            logger.info("Scalping bot started.")
            return True
        return False

    def stop_bot_logic(self):
        with self.status_lock:
            if not self.bot_running:
                logger.info("Bot is already stopped.")
                return False
            self.bot_running = False
        logger.info("Scalping bot stopping.")
        return True

    def get_equity_curve_data(self, limit_points=100):
        logger.info(f"Generating equity curve data from MongoDB (limit: {limit_points} points).")
        trades_collection = MongoDBConnection.get_trades_collection()
        labels, equity_values = [], []

        if not trades_collection:
            logger.warning("MongoDB not connected or trades collection not found. Trying current MT5 equity.")
            acc_info = mt5.account_info()
            if acc_info:
                return {"labels": [datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')], "equity": [round(acc_info.equity, 2)]}
            return {"labels": [], "equity": [], "error": "MongoDB unavailable and MT5 connection failed."}

        try:
            first_trade_doc = trades_collection.find_one({"account_balance": {"$exists": True, "$type": "number"}}, sort=[("entry_time", 1)])

            if not first_trade_doc:
                logger.info("No trades with numeric account_balance found. Using current MT5 equity.")
                acc_info = mt5.account_info()
                if acc_info:
                    return {"labels": [datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')], "equity": [round(acc_info.equity, 2)]}
                return {"labels": [], "equity": [], "error": "No starting point in MongoDB and MT5 unavailable."}

            initial_equity = first_trade_doc.get('account_balance', 0.0)
            try: initial_equity = float(initial_equity)
            except (ValueError, TypeError): initial_equity = 0.0; logger.warning("Initial equity from DB was not a number.")

            if first_trade_doc.get('status') in ['closed', 'closed_auto'] and pd.notna(first_trade_doc.get('profit_loss')):
                try:
                    initial_equity -= float(first_trade_doc.get('profit_loss', 0.0))
                except (ValueError, TypeError): pass

            base_ts_dt = first_trade_doc.get('entry_time')
            if not isinstance(base_ts_dt, datetime):
                 base_ts_dt = datetime.now(timezone.utc)
                 logger.warning(f"Invalid entry_time for first trade {first_trade_doc.get('order_id')}, using current time for equity curve start.")
            if base_ts_dt.tzinfo is None: base_ts_dt = base_ts_dt.replace(tzinfo=timezone.utc)

            labels.append(base_ts_dt.strftime('%Y-%m-%d %H:%M'))
            equity_values.append(round(initial_equity, 2))
            running_equity = initial_equity

            closed_trades_cursor = trades_collection.find(
                {"status": {"$in": ["closed", "closed_auto"]}, "exit_time": {"$exists": True, "$type": "date"}, "profit_loss": {"$exists": True, "$type": "number"}},
                sort=[("exit_time", 1)]
            )

            for trade in closed_trades_cursor:
                exit_time_dt, profit_loss_val = trade.get('exit_time'), trade.get('profit_loss', 0.0)
                try: profit_loss_val = float(profit_loss_val)
                except (ValueError, TypeError): profit_loss_val = 0.0

                if isinstance(exit_time_dt, datetime):
                    if exit_time_dt.tzinfo is None: exit_time_dt = exit_time_dt.replace(tzinfo=timezone.utc)
                    running_equity += profit_loss_val
                    labels.append(exit_time_dt.strftime('%Y-%m-%d %H:%M'))
                    equity_values.append(round(running_equity, 2))
                else:
                    logger.warning(f"Skipping trade {trade.get('order_id')} for equity curve due to invalid exit_time type: {type(exit_time_dt)}")

            acc_info = mt5.account_info()
            if acc_info:
                current_mt5_equity = round(acc_info.equity, 2)
                current_time_label = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
                if not equity_values or (labels[-1] != current_time_label or equity_values[-1] != current_mt5_equity):
                     if not labels or pd.to_datetime(current_time_label) >= pd.to_datetime(labels[-1]):
                        labels.append(current_time_label)
                        equity_values.append(current_mt5_equity)

            if not equity_values:
                return {"labels": [], "equity": [], "error": "Could not construct equity points."}

            if len(equity_values) > limit_points:
                slice_start = len(equity_values) - limit_points
                labels, equity_values = labels[slice_start:], equity_values[slice_start:]

            return {"labels": labels, "equity": equity_values}

        except errors.PyMongoError as e:
            logger.error(f"MongoDB error generating equity curve: {e}", exc_info=True)
            return {"labels": [], "equity": [], "error": f"MongoDB error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error generating equity curve: {e}", exc_info=True)
            return {"labels": [], "equity": [], "error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    bot_config = {}
    try:
        with open(os.path.join('config', 'config.json'), 'r') as f:
            bot_config = json.load(f)
    except Exception as e:
        logger.error(f"Could not load config for direct run: {e}")

    if not connect_mt5(bot_config.get('mt5_credentials')):
        logger.critical("MT5 connection failed. Bot cannot start.")
        exit(1)

    bot = TradingBot()
    bot.start_bot_logic()

    try:
        while True:
            time.sleep(1)
            if not bot.bot_running:
                logger.critical("Main loop: Bot has stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        logger.info("Bot supervisor stopped by user.")
    finally:
        if bot.bot_running:
            bot.stop_bot_logic()
        MongoDBConnection.close_connection()
        mt5.shutdown()
        logger.info("Bot shutdown complete.")
