from datetime import datetime, timezone
import sys
from flask import Flask, jsonify, logging as flask_logging, request, render_template, send_from_directory # Renamed logging to flask_logging
from flask_cors import CORS
import os
import json
import threading
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Define TRADE_LOG_FILE at the global scope for app.py's own use (e.g., fallbacks)
TRADE_LOG_FILE = "trades_history_fallback.csv"

try:
    from dotenv import load_dotenv
    load_dotenv()

    from main import TradingBot
    from utils.mt5_connection import connect_mt5, TIMEFRAME_MAP as MT5_TIMEFRAME_MAP_APP
    from utils.logging import setup_logging
    from backtester import run_backtest_main, _get_bot_instance_for_backtesting as init_backtester_config

except ImportError as e:
    print(f"CRITICAL Error during initial imports in app.py: {e}. Some features might not work.", file=sys.stderr)
    print("Ensure main.py, utils, and backtester.py are in PYTHONPATH or correct relative paths.", file=sys.stderr)
    # Define dummy classes and functions for fallback
    class TradingBot:
        def __init__(self): self.config = {}; self.last_error_message = "IMPORT_ERROR"; self.bot_running = False
        def get_equity_curve_data(self, limit_points=100): return {"labels": [], "equity": [], "error": "Fallback Bot: Equity data unavailable."}
        def get_bot_status_for_ui(self): return {"bot_status": "ERROR_IMPORT", "balance": "N/A", "equity": "N/A", "free_margin": "N/A", "margin_level": "N/A", "last_ml_retrain": "Never", "last_error": "Import error."}
        def get_open_positions_for_ui(self): return []
        def get_trade_history_for_ui(self,p,l): return [],0
        def get_logs_for_ui(self,l,f): return ["Import error during app startup."]
        def load_config_and_reinitialize(self): pass
    def connect_mt5(cfg=None): return False
    MT5_TIMEFRAME_MAP_APP = {}
    import logging as pylogging_fallback
    _fallback_logger = pylogging_fallback.getLogger("fallback_app_logger")
    if not _fallback_logger.hasHandlers():
        _h = pylogging_fallback.StreamHandler(sys.stderr)
        _f = pylogging_fallback.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _h.setFormatter(_f)
        _fallback_logger.addHandler(_h)
        _fallback_logger.setLevel(pylogging_fallback.INFO)
    setup_logging = lambda: _fallback_logger
    def run_backtest_main(*args, **kwargs): return {"error": "Backtester module import failed."}
    init_backtester_config = lambda: print("Skipping backtester config init due to import error.", file=sys.stderr)

logger = setup_logging()

# --- Flask App Setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

if not os.path.exists(os.path.join(TEMPLATE_DIR, 'dashboard.html')):
    logger.warning(f"dashboard.html not found in {TEMPLATE_DIR}. Attempting to use current directory.")
    TEMPLATE_DIR = BASE_DIR

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

trading_bot_instance = None
bot_initialization_lock = threading.Lock()

def initialize_bot():
    global trading_bot_instance
    with bot_initialization_lock:
        if trading_bot_instance is None:
            logger.info("Flask app: Initializing TradingBot instance...")
            try:
                trading_bot_instance = TradingBot()
                if trading_bot_instance and hasattr(trading_bot_instance, 'config') and trading_bot_instance.config:
                    mt5_creds = trading_bot_instance.config.get('mt5_credentials')
                    if mt5_creds:
                        logger.info("Flask app: Attempting automatic MT5 connection on startup...")
                        if connect_mt5(mt5_creds):
                            logger.info("Flask app: MT5 connected successfully on startup.")
                            if hasattr(trading_bot_instance, 'ensure_symbols_selected'):
                                trading_bot_instance.ensure_symbols_selected()
                        else:
                            logger.warning("Flask app: Automatic MT5 connection failed on startup.")
                    else:
                        logger.warning("Flask app: MT5 credentials not found in bot config.")
                else:
                    logger.warning("Flask app: TradingBot instance or config not available for auto MT5 connection.")

                if 'init_backtester_config' in globals() and callable(init_backtester_config):
                    init_backtester_config()
                    logger.info("Flask app: Backtester configuration access initialized.")
                else:
                    logger.warning("Flask app: init_backtester_config function not found.")

            except Exception as e_init:
                logger.critical(f"Flask app: CRITICAL error during TradingBot initialization: {e_init}", exc_info=True)
                if trading_bot_instance is None:
                    trading_bot_instance = TradingBot()
                    trading_bot_instance.last_error_message = f"Bot Init Failed: {e_init}"
    return trading_bot_instance

def get_bot():
    global trading_bot_instance
    if trading_bot_instance is None:
        initialize_bot()
    return trading_bot_instance

# --- Standard Routes ---
@app.route('/')
def index():
    get_bot()
    return render_template('dashboard.html')

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/api/ml_models', methods=['GET'])
def get_ml_models():
    """Provides the list of available ML models to the UI."""
    models = ["RandomForest"]
    # Check if XGBoost was successfully imported in ml_model.py
    # This requires a way to check that flag, or just check here again.
    try:
        import xgboost
        models.append("XGBoost")
    except ImportError:
        pass
    return jsonify({"models": models})

@app.route('/api/equity_curve', methods=['GET'])
def get_equity_curve_api():
    bot = get_bot()
    if not bot:
        return jsonify({"labels": [], "equity": [], "error": "Bot not initialized."}), 503

    if hasattr(bot, 'get_equity_curve_data'):
        try:
            limit = request.args.get('limit', '100', type=int)
            equity_data = bot.get_equity_curve_data(limit_points=limit)
            if "error" in equity_data:
                 return jsonify(equity_data), 500
            return jsonify(equity_data)
        except Exception as e:
            logger.error(f"Error in /api/equity_curve endpoint: {e}", exc_info=True)
            return jsonify({"labels": [], "equity": [], "error": f"Internal server error: {str(e)}"}), 500
    else:
        return jsonify({"labels": [], "equity": [], "error": "Equity data feature unavailable in bot."}), 501


@app.route('/api/status', methods=['GET'])
def get_status():
    bot = get_bot()
    if bot and hasattr(bot, 'get_bot_status_for_ui'):
        return jsonify(bot.get_bot_status_for_ui())
    # Fallback status
    return jsonify({ "bot_status": "ERROR_API", "balance": "N/A", "equity": "N/A", "free_margin": "N/A", "margin_level": "N/A", "last_ml_retrain": "Never", "last_error": "Bot instance unavailable."}), 503

@app.route('/api/open_positions', methods=['GET'])
def get_open_positions():
    bot = get_bot()
    if bot and hasattr(bot, 'get_open_positions_for_ui'):
        try:
            return jsonify(bot.get_open_positions_for_ui())
        except Exception as e:
            logger.error(f"Error in /api/open_positions: {e}", exc_info=True)
            return jsonify({"error": "Failed to fetch open positions."}), 500
    return jsonify([]), 503

@app.route('/api/trade_history', methods=['GET'])
def get_trade_history():
    bot = get_bot()
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)

    if bot and hasattr(bot, 'get_trade_history_for_ui'):
        try:
            history, total_trades = bot.get_trade_history_for_ui(page, limit)
            return jsonify({"trades": history, "total_trades": total_trades, "page": page, "limit": limit})
        except Exception as e:
            logger.error(f"Error in /api/trade_history from bot method: {e}", exc_info=True)
            return jsonify({"trades": [], "total_trades": 0, "error": str(e)}), 500

    # Fallback to CSV read
    try:
        if os.path.exists(TRADE_LOG_FILE):
            df = pd.read_csv(TRADE_LOG_FILE).fillna("N/A")
            df = df.sort_values(by='entry_time', ascending=False)
            total = len(df)
            paginated_df = df.iloc[(page - 1) * limit:page * limit]
            return jsonify({"trades": paginated_df.to_dict('records'), "total_trades": total})
        else:
            return jsonify({"trades": [], "total_trades": 0})
    except Exception as e:
        logger.error(f"Error reading trade history from CSV fallback: {e}", exc_info=True)
        return jsonify({"trades": [], "total_trades": 0, "error": "Failed to read history file."}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    bot = get_bot()
    limit = request.args.get('limit', 50, type=int)
    level = request.args.get('level', 'ALL', type=str).upper()
    if bot and hasattr(bot, 'get_logs_for_ui'):
        try:
            return jsonify(bot.get_logs_for_ui(limit=limit, level_filter=level if level != "ALL" else None))
        except Exception as e:
            logger.error(f"Error getting logs from bot: {e}", exc_info=True)
            return jsonify([f"Error fetching logs from bot: {str(e)}"]), 500
    return jsonify(["Log service unavailable."]), 503

@app.route('/api/download_log', methods=['GET'])
def download_log():
    log_dir = os.path.join(BASE_DIR, 'logs')
    log_file = 'trading_ea.log'
    try:
        return send_from_directory(log_dir, log_file, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Log file not found."}), 404

@app.route('/api/config', methods=['GET'])
def get_config_values():
    bot = get_bot()
    if bot and hasattr(bot, 'config') and bot.config:
        return jsonify(bot.config)
    try:
        config_file_path = os.path.join(BASE_DIR, 'config', 'config.json')
        with open(config_file_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": "Could not load configuration."}), 500

@app.route('/api/config/update', methods=['POST'])
def update_config_values():
    bot = get_bot()
    if not bot:
        return jsonify({"success": False, "message": "Bot not initialized."}), 503
    try:
        new_data = request.get_json()
        config_path = os.path.join(BASE_DIR, 'config', 'config.json')
        with open(config_path, 'r') as f:
            live_config = json.load(f)

        def merge_configs(original, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    merge_configs(original[key], value)
                else:
                    original[key] = value
            return original

        updated_config = merge_configs(live_config, new_data)
        with open(config_path, 'w') as f:
            json.dump(updated_config, f, indent=2)

        if hasattr(bot, 'load_config_and_reinitialize'):
            bot.load_config_and_reinitialize()
            if 'mt5_credentials' in new_data:
                 if not mt5.terminal_info():
                     if connect_mt5(updated_config.get('mt5_credentials')):
                         logger.info("MT5 reconnected successfully after config update.")
                         if hasattr(bot, 'ensure_symbols_selected'): bot.ensure_symbols_selected()
                     else:
                         return jsonify({"success": True, "message": "Config updated, but MT5 connection failed."}), 200
            return jsonify({"success": True, "message": "Configuration updated and bot reinitialized."})
        else:
            return jsonify({"success": True, "message": "Config updated. Bot may need manual restart."})

    except Exception as e:
        logger.error(f"Error updating configuration: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

# --- Control Routes ---
@app.route('/api/control/start_bot', methods=['POST'])
def start_bot_api():
    bot = get_bot()
    if not bot: return jsonify({"success": False, "message": "Bot not initialized."}), 503
    if hasattr(bot, 'start_bot_logic'):
        if not mt5.terminal_info():
            mt5_creds = bot.config.get('mt5_credentials', {})
            if not connect_mt5(mt5_creds):
                return jsonify({"success": False, "message": "Failed to connect to MT5."}), 500
            if hasattr(bot, 'ensure_symbols_selected'): bot.ensure_symbols_selected()

        if bot.start_bot_logic():
            return jsonify({"success": True, "message": "Bot started."})
        else:
            return jsonify({"success": False, "message": "Bot already running."})
    return jsonify({"success": False, "message": "Control function not available."}), 501

@app.route('/api/control/stop_bot', methods=['POST'])
def stop_bot_api():
    bot = get_bot()
    if not bot: return jsonify({"success": False, "message": "Bot not initialized."}), 503
    if hasattr(bot, 'stop_bot_logic'):
        if bot.stop_bot_logic():
            return jsonify({"success": True, "message": "Bot stopping."})
        else:
            return jsonify({"success": False, "message": "Bot already stopped."})
    return jsonify({"success": False, "message": "Control function not available."}), 501

@app.route('/api/control/trigger_retrain', methods=['POST'])
def trigger_retrain_api():
    bot = get_bot()
    if not bot: return jsonify({"success": False, "message": "Bot not initialized."}), 503
    if hasattr(bot, 'trigger_manual_retrain'):
        success, message = bot.trigger_manual_retrain()
        return jsonify({"success": success, "message": message})
    return jsonify({"success": False, "message": "Control function not available."}), 501

@app.route('/api/control/close_trade/<int:ticket_id>', methods=['POST'])
def close_trade_api(ticket_id):
    bot = get_bot()
    if not bot: return jsonify({"success": False, "message": "Bot not initialized."}), 503
    if hasattr(bot, 'manual_close_trade_by_ticket'):
        success, message = bot.manual_close_trade_by_ticket(ticket_id)
        return jsonify({"success": success, "message": message})
    return jsonify({"success": False, "message": "Control function not available."}), 501

@app.route('/api/control/delete_logs', methods=['POST'])
def delete_logs_api():
    bot = get_bot()
    if not bot: return jsonify({"success": False, "message": "Bot not initialized."}), 503
    if hasattr(bot, 'delete_bot_logs'):
        success, message = bot.delete_bot_logs()
        return jsonify({"success": success, "message": message})
    return jsonify({"success": False, "message": "Log deletion function not available."}), 501

@app.route('/api/control/force_full_ml_training', methods=['POST'])
def force_full_ml_training_api():
    bot = get_bot()
    if not bot: return jsonify({"success": False, "message": "Bot not initialized."}), 503
    if hasattr(bot, 'force_full_ml_retraining'):
        success, message = bot.force_full_ml_retraining()
        return jsonify({"success": success, "message": message})
    return jsonify({"success": False, "message": "Forced ML training function not available."}), 501

# --- Backtesting API Endpoints ---
@app.route('/api/backtest/strategies', methods=['GET'])
def get_backtest_strategies_api():
    try:
        # --- MODIFIED LINE ---
        supported_strategies = [ "Scalping", "SMA", "BollingerBands", "LiquiditySweep","ADX", "KeltnerChannels","Fibonacci", "MalaysianSnR", "SMC" ]
        return jsonify({"strategies": supported_strategies})
    except Exception as e:
        return jsonify({"error": f"Could not load strategies: {str(e)}"}), 500

@app.route('/api/backtest/run', methods=['POST'])
def handle_run_backtest_api():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No data provided."}), 400

        strategy_name = data.get('strategy_name')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_cash = float(data.get('initial_cash', 10000))
        commission_bps = float(data.get('commission_bps', 0))
        ml_enabled = data.get('ml_enabled', True)
        strategy_specific_params = data.get('strategy_params', {})

        if not all([strategy_name, symbol, timeframe, start_date, end_date]):
            return jsonify({"error": "Missing required backtest parameters."}), 400

        logger.info(f"Received backtest request: Strat={strategy_name}, Sym={symbol}, TF={timeframe}, Period={start_date}-{end_date}, ML={ml_enabled}")

        results = run_backtest_main(
            strategy_name, symbol, timeframe,
            start_date, end_date, initial_cash, commission_bps,
            strategy_specific_params, ml_enabled
        )

        if "error" in results:
            logger.error(f"Backtest run failed: {results['error']}")
            return jsonify(results), 400

        return jsonify(results)
    except ValueError as ve:
        return jsonify({"error": f"Invalid parameter value: {str(ve)}"}), 400
    except Exception as e:
        logger.error(f"Critical error in /api/backtest/run endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    logger.info("Attempting to initialize bot and start Flask development server...")
    initialize_bot()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)