import pandas as pd
# from .logging import setup_logging # Assuming this is in utils.logging
import logging # Using standard logging for this module if setup_logging is complex to import standalone
import os
import ast
import numpy as np
from datetime import datetime, timedelta, timezone # For handling datetime objects

# Import MongoDB connection utility and PyMongo errors
from .db_connector import MongoDBConnection # Assuming db_connector.py is in the same directory (utils)
from pymongo import errors

logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def save_trade(trade_data, update=False): 
    """
    Save or update a trade in MongoDB.
    Args:
        trade_data (dict): The trade data to save. Must include 'order_id'.
                           'entry_time' and 'exit_time' should be datetime objects.
        update (bool): If True, tries to update an existing document.
                       (This logic might be better handled by specific update functions like update_trade_status)
    """
    trades_collection = MongoDBConnection.get_trades_collection()
    if not trades_collection:
        logger.error(f"Cannot save trade, MongoDB not connected or collection not found. Trade for order_id {trade_data.get('order_id')} not saved.")
        # Optionally, implement a fallback to CSV here if critical
        return

    try:
        if 'order_id' not in trade_data or trade_data['order_id'] is None:
            logger.error(f"Trade data missing order_id: {trade_data}")
            return

        # Ensure order_id is an int for consistency, though MongoDB is flexible
        try:
            order_id_val = int(trade_data['order_id'])
            trade_data['order_id'] = order_id_val # Standardize to int
        except ValueError:
            logger.error(f"Invalid order_id format for trade: {trade_data.get('order_id')}. Cannot save.")
            return

        # Prepare a copy of the trade data for MongoDB
        # MongoDB prefers datetime objects for date fields
        mongo_trade_data = trade_data.copy()

        # Convert relevant string dates to datetime objects if they aren't already
        # 'entry_time' and 'exit_time' are expected to be datetime objects by the time they reach here
        # from main.py's execute_trade or close_position_by_ticket.
        # If they are strings, they should be parsed.
        for time_field in ['entry_time', 'exit_time']:
            if time_field in mongo_trade_data and isinstance(mongo_trade_data[time_field], str):
                try:
                    # Attempt to parse ISO format strings
                    dt_obj = datetime.fromisoformat(mongo_trade_data[time_field])
                    # Ensure it's timezone-aware (UTC)
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    mongo_trade_data[time_field] = dt_obj
                except ValueError:
                    logger.warning(f"Could not parse string '{mongo_trade_data[time_field]}' for field '{time_field}' into datetime. Storing as string.")
            elif time_field in mongo_trade_data and isinstance(mongo_trade_data[time_field], datetime):
                # Ensure it's timezone-aware (UTC)
                if mongo_trade_data[time_field].tzinfo is None:
                     mongo_trade_data[time_field] = mongo_trade_data[time_field].replace(tzinfo=timezone.utc)


        # Ensure numeric fields are numbers (MongoDB can store various number types)
        numeric_fields = ['entry_price', 'sl_price', 'tp_price', 'lot_size',
                          'profit_loss', 'account_balance', 'ml_confidence', 'exit_price']
        for field in numeric_fields:
            if field in mongo_trade_data and mongo_trade_data[field] is not None:
                try:
                    mongo_trade_data[field] = float(mongo_trade_data[field])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {field} value '{mongo_trade_data[field]}' to float for order {order_id_val}. It might be stored as is or skipped.")
                    # Decide handling: store as is, or remove, or set to a default (e.g., 0.0 or None)
                    # mongo_trade_data[field] = None # Example: set to None if conversion fails

        # Use update_one with upsert=True to insert if not exists, or update if exists.
        # This is generally preferred over checking existence first.
        # The filter is based on 'order_id'.
        query_filter = {"order_id": order_id_val}
        update_operation = {"$set": mongo_trade_data}

        result = trades_collection.update_one(query_filter, update_operation, upsert=True)

        if result.upserted_id:
            logger.info(f"Trade {order_id_val} inserted into MongoDB with new _id: {result.upserted_id}")
        elif result.matched_count > 0 and result.modified_count > 0:
            logger.info(f"Trade {order_id_val} updated in MongoDB.")
        elif result.matched_count > 0 and result.modified_count == 0:
            logger.info(f"Trade {order_id_val} found in MongoDB, but no fields were modified by the update.")
        else: # Should not happen with upsert=True if no error
            logger.warning(f"Trade {order_id_val} neither inserted nor updated. Result: {result.raw_result}")

    except errors.PyMongoError as e: # Catch specific MongoDB errors
        logger.error(f"MongoDB error saving/updating trade for order_id {trade_data.get('order_id', 'N/A')}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in save_trade (Order ID: {trade_data.get('order_id', 'N/A')}): {e}", exc_info=True)


def update_trade_status(order_id_to_update, update_data_dict):
    """
    Updates specific fields of an existing trade in MongoDB.
    Args:
        order_id_to_update (int or str): The order_id of the trade to update.
        update_data_dict (dict): A dictionary containing fields to update.
                                 'exit_time' should be a datetime object.
    """
    trades_collection = MongoDBConnection.get_trades_collection()
    if not trades_collection:
        logger.error(f"Cannot update trade status for order_id {order_id_to_update}, MongoDB not connected or collection not found.")
        return False # Indicate failure

    try:
        order_id_val = int(order_id_to_update)
    except ValueError:
        logger.error(f"Invalid order_id format for update: {order_id_to_update}. Cannot update.")
        return False

    update_payload = update_data_dict.copy()

    # Ensure datetime fields are datetime objects
    if 'exit_time' in update_payload and isinstance(update_payload['exit_time'], str):
        try:
            dt_obj = datetime.fromisoformat(update_payload['exit_time'])
            if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            update_payload['exit_time'] = dt_obj
        except ValueError:
            logger.warning(f"Could not parse 'exit_time' string '{update_payload['exit_time']}' for order {order_id_val}. Storing as string or original type.")
    elif 'exit_time' in update_payload and isinstance(update_payload['exit_time'], datetime):
         if update_payload['exit_time'].tzinfo is None:
             update_payload['exit_time'] = update_payload['exit_time'].replace(tzinfo=timezone.utc)


    # Ensure numeric fields are numbers
    numeric_fields_update = ['profit_loss', 'exit_price', 'sl_price', 'account_balance'] # Add any other numerics
    for field in numeric_fields_update:
        if field in update_payload and update_payload[field] is not None:
            try:
                update_payload[field] = float(update_payload[field])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {field} '{update_payload[field]}' to float for order {order_id_val} update. Skipping update for this field or storing as is.")
                # Decide: del update_payload[field], or store as is if MongoDB handles mixed types well for your queries.
                # For safety, you might want to remove it if type is critical:
                # del update_payload[field]

    query_filter = {"order_id": order_id_val}
    update_operation = {"$set": update_payload}

    try:
        result = trades_collection.update_one(query_filter, update_operation)

        if result.matched_count > 0:
            if result.modified_count > 0:
                logger.info(f"Trade {order_id_val} status updated in MongoDB. Fields: {list(update_payload.keys())}")
            else:
                logger.info(f"Trade {order_id_val} found, but no fields were modified by the update. Data might be the same.")
            return True
        else:
            logger.warning(f"Trade {order_id_val} not found in MongoDB for status update.")
            # If not found, you might want to save it as a new trade if that's the desired behavior,
            # or log it as a potential issue (e.g., a trade that should exist but doesn't).
            # For now, just logs a warning.
            return False # Indicate not found for update

    except errors.PyMongoError as e:
        logger.error(f"MongoDB error updating trade status for order {order_id_val}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in update_trade_status for order {order_id_val}: {e}", exc_info=True)
        return False


def safe_parse_strategies(strat_data):
    """Safely parses a string representation of a list of strategies."""
    if isinstance(strat_data, list):
        return strat_data
    if pd.isna(strat_data) or not isinstance(strat_data, str) or \
       strat_data.strip().lower() in ['', 'nan', '[]', 'n/a', 'none']:
        return []
    try:
        parsed = ast.literal_eval(strat_data)
        if isinstance(parsed, list):
            return parsed
        else:
            logger.warning(f"Parsed strategies string '{strat_data}' was not a list: {type(parsed)}. Returning empty list.")
            return []
    except (ValueError, SyntaxError, TypeError) as e: # Added TypeError
        logger.warning(f"Could not parse strategies string: '{strat_data}' (Error: {e}). Returning empty list.")
        return []


def get_strategy_weights(lookback_days=30):
    """
    Calculates dynamic strategy weights based on recent historical performance from MongoDB.
    Performance score is a combination of Profit Factor and Win Rate.
    Args:
        lookback_days (int): How many days of trade history to consider for the calculation.
    """
    # Default weights, returned if no data is available
    default_initial_weights = {
        "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0, "Fibonacci": 1.0,
        "MalaysianSnR": 1.0, "BollingerBands": 1.0, "ADX": 1.0, "KeltnerChannels": 1.0,
        "Scalping": 1.0, "MLPrediction": 1.0
    }

    trades_collection = MongoDBConnection.get_trades_collection()
    if not trades_collection:
        logger.warning("Cannot calculate strategy weights, MongoDB not connected. Returning default weights.")
        return default_initial_weights

    try:
        # Add a time-based filter for recent performance
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Query for recent closed trades with necessary fields
        closed_trades_cursor = trades_collection.find({
            "status": {"$in": ["closed", "closed_auto"]},
            "profit_loss": {"$exists": True, "$type": "number"}, # Ensure profit_loss is a number
            "strategies": {"$exists": True},
            "exit_time": {"$gte": cutoff_date} # Filter for recent trades
        })

        df_closed = pd.DataFrame(list(closed_trades_cursor))

        if df_closed.empty:
            logger.info(f"No closed trades found in the last {lookback_days} days for weight calculation. Using default weights.")
            return default_initial_weights

        # Data Cleaning and Preparation
        df_closed['profit_loss'] = pd.to_numeric(df_closed['profit_loss'], errors='coerce').fillna(0.0)

        # Handle 'strategies' field which could be a string or list
        if 'strategies' in df_closed.columns and not df_closed['strategies'].dropna().empty:
            # Check the type of the first non-null entry to decide on parsing
            first_strat_entry = df_closed['strategies'].dropna().iloc[0] if not df_closed['strategies'].dropna().empty else None
            if isinstance(first_strat_entry, str):
                df_closed['parsed_strategies'] = df_closed['strategies'].apply(safe_parse_strategies)
            else: # Assume it's already a list (or compatible type)
                df_closed['parsed_strategies'] = df_closed['strategies']
        else: # Handle case where column is missing or all entries are NaN
            df_closed['parsed_strategies'] = pd.Series([[] for _ in range(len(df_closed))], index=df_closed.index)

        weights = {}
        all_strategy_names = list(default_initial_weights.keys())
        min_trades_for_calc = 5 # Minimum trades required to calculate a dynamic weight

        for strategy_name in all_strategy_names:
            # Filter DataFrame for trades where this strategy was involved
            strategy_trades = df_closed[df_closed['parsed_strategies'].apply(
                lambda strat_list: isinstance(strat_list, list) and strategy_name in strat_list
            )]

            # If not enough trades, assign default weight
            if len(strategy_trades) < min_trades_for_calc:
                weights[strategy_name] = 1.0
                continue

            # --- Performance Metrics Calculation ---
            total_trades = len(strategy_trades)
            winning_trades = strategy_trades[strategy_trades['profit_loss'] > 0]
            losing_trades = strategy_trades[strategy_trades['profit_loss'] < 0]

            win_rate = len(winning_trades) / total_trades

            gross_profit = winning_trades['profit_loss'].sum()
            gross_loss = abs(losing_trades['profit_loss'].sum())

            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                # If there are no losses, profit factor is technically infinite.
                # Assign a high but capped value to reward perfect performance without skewing results.
                profit_factor = 10.0 if gross_profit > 0 else 1.0 # 1.0 if no profit and no loss

            # --- Combine Metrics into a Single Weight ---
            # Normalization and combination logic.
            # A win rate of 50% should be neutral (1.0).
            # A profit factor of 1.5 should be neutral (1.0).

            # Win Rate Score: Maps [0, 1] to roughly [0.75, 1.25]
            win_rate_score = 1.0 + (win_rate - 0.5) * 0.5

            # Profit Factor Score: Uses tanh to squash the value, centered around a PF of 1.5
            # Maps PF to roughly [0.5, 1.5]
            pf_score = 1.0 + np.tanh(profit_factor - 1.5) * 0.5

            # Combine the scores. Averaging them gives equal importance.
            calculated_weight = (win_rate_score + pf_score) / 2.0

            # Bound the final weight to prevent extreme values, e.g., [0.5, 2.0]
            final_weight = max(0.5, min(2.0, calculated_weight))
            weights[strategy_name] = final_weight

            logger.debug(f"Weight Calc for '{strategy_name}': Trades={total_trades}, WR={win_rate:.2f}, PF={profit_factor:.2f} -> Final Weight={final_weight:.3f}")

        # Ensure all strategies have a weight
        for strat_name in all_strategy_names:
            if strat_name not in weights:
                weights[strat_name] = 1.0 # Default for strategies with no recent trades

        logger.info(f"Calculated dynamic strategy weights (last {lookback_days} days): { {k: round(v, 2) for k, v in weights.items()} }")
        return weights

    except errors.PyMongoError as e:
        logger.error(f"MongoDB error calculating strategy weights: {e}", exc_info=True)
        return default_initial_weights
    except Exception as e:
        logger.error(f"Unexpected error calculating strategy weights: {e}", exc_info=True)
        return default_initial_weights


if __name__ == '__main__':
    # This block is for direct testing of this module, if needed.
    # Ensure MongoDB is running and accessible.
    # Set MONGODB_URI and MONGODB_DATABASE_NAME environment variables for testing.

    logger.info("Testing trade_history.py with MongoDB...")
    db_conn_test = MongoDBConnection.connect() # Establish connection

    if db_conn_test:
        logger.info(f"Connected to MongoDB: {db_conn_test.name} for testing.")

        # Example: Test save_trade
        test_trade_new = {
            "order_id": 999901, "symbol": "XAUUSDm", "timeframe": "M5", "signal": "BUY",
            "entry_price": 2300.50, "sl_price": 2295.00, "tp_price": 2310.50, "lot_size": 0.01,
            "strategies": ["SMA", "ADX"], "ml_confidence": 0.75,
            "entry_time": datetime.now(timezone.utc),
            "status": "open", "profit_loss": 0.0,
            "account_balance": 10000.00,
            "failure_reason": "", "exit_reason": "", "deal_id": "D001"
        }
        save_trade(test_trade_new)

        # Example: Test update_trade_status
        update_data_test = {
            "status": "closed_auto",
            "profit_loss": 10.50,
            "exit_price": 2301.55,
            "exit_time": datetime.now(timezone.utc) + timedelta(minutes=30),
            "exit_reason": "Auto TP Reversal"
        }
        update_trade_status(999901, update_data_test)

        # Example: Test get_strategy_weights
        calculated_weights = get_strategy_weights()
        logger.info(f"Test: Calculated strategy weights: {calculated_weights}")

        MongoDBConnection.close_connection() # Close connection when done
    else:
        logger.error("Failed to connect to MongoDB for testing trade_history.py.")
