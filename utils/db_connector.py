# utils/db_connector.py
import os
from pymongo import MongoClient, errors
from pymongo.server_api import ServerApi
import logging
from dotenv import load_dotenv
import certifi

load_dotenv()
# Assuming you use the existing logger from setup_logging
# from utils.logging import setup_logging # If your logger is in a different path
# logger = setup_logging()

# For simplicity here, using a basic logger:
logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MongoDBConnection:
    _client = None
    _db = None

    # Defaults from your provided connection string
    DEFAULT_MONGO_URI = "mongodb+srv://EA_TradingBot:pykOSLpNP9k1jMre@cluster0.0mk8h.mongodb.net/"
    DEFAULT_DB_NAME = "EA_TradingBot"

    @classmethod
    def connect(cls):
        """
        Establishes a connection to MongoDB.
        Prioritizes environment variables MONGODB_URI and MONGODB_DATABASE_NAME.
        Falls back to hardcoded defaults if environment variables are not set.
        """
        if cls._client is None:
            uri = os.environ.get("MONGODB_URI", cls.DEFAULT_MONGO_URI)
            db_name = os.environ.get("MONGODB_DATABASE_NAME", cls.DEFAULT_DB_NAME)

            # Append db_name to URI if it's not already part of it and using srv string
            # For srv strings, the database name is often specified in the connection string directly
            # or can be accessed after connecting to the client.
            # PyMongo handles this well; typically, you connect to the cluster
            # and then select the database.

            # For srv URIs, it's common to specify the default database in the connection string
            # like mongodb+srv://user:pass@host/default_db_name?options
            # If your URI already contains the db_name, PyMongo will use it.
            # If not, cls._client[db_name] will select/create it.

            logger.info(f"Attempting to connect to MongoDB URI: {uri} (DB Name: {db_name})")

            try:
                # --- FIXED: Added tlsCAFile parameter for SSL validation ---
                ca = certifi.where()
                # For modern PyMongo versions with ServerApi
                # Create a new client and connect to the server
                # The ServerApi part is for ensuring compatibility with MongoDB versions.
                # You might not strictly need it if your MongoDB version is very recent and doesn't require it.
                cls._client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=10000, tlsCAFile=ca) # Increased timeout and added certifi

                # Send a ping to confirm a successful connection
                cls._client.admin.command('ping')
                logger.info("Pinged your deployment. You successfully connected to MongoDB!")

                cls._db = cls._client[db_name] # Select the database
                logger.info(f"Successfully connected to MongoDB. Database: {cls._db.name}")

            except errors.ServerSelectionTimeoutError as err:
                logger.error(f"MongoDB connection failed (Timeout): {err}. URI: {uri}, DB: {db_name}")
                cls._client = None
                cls._db = None
            except errors.ConnectionFailure as err:
                logger.error(f"MongoDB connection failed (ConnectionFailure): {err}. URI: {uri}, DB: {db_name}")
                cls._client = None
                cls._db = None
            except errors.ConfigurationError as err: # Catch configuration errors (e.g. invalid URI)
                logger.error(f"MongoDB configuration error: {err}. URI: {uri}, DB: {db_name}")
                cls._client = None
                cls._db = None
            except Exception as e:
                logger.error(f"An unexpected error occurred during MongoDB connection: {e}", exc_info=True)
                cls._client = None
                cls._db = None
        return cls._db

    @classmethod
    def get_db(cls):
        if cls._db is None:
            logger.warning("Database not connected. Call MongoDBConnection.connect() first or check connection errors.")
            # Optionally, attempt to reconnect or raise an error
            # For now, returning None, and callers should handle it.
        return cls._db

    @classmethod
    def close_connection(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("MongoDB connection closed.")

    @classmethod
    def get_trades_collection(cls):
        """Returns the 'trades' collection. Ensures DB is connected."""
        db = cls.get_db()
        if db:
            return db.trades # 'trades' is the collection name
        return None

# Example of how to use it (optional, for testing)
if __name__ == '__main__':
    # To test this, set your MONGODB_URI and MONGODB_DATABASE_NAME environment variables
    # or rely on the defaults.
    # Example:
    # export MONGODB_URI="mongodb+srv://your_user:your_password@your_cluster.mongodb.net/"
    # export MONGODB_DATABASE_NAME="your_db_name"

    db_instance = MongoDBConnection.connect()
    if db_instance:
        logger.info(f"Connected to DB: {db_instance.name}")
        trades_collection = MongoDBConnection.get_trades_collection()
        if trades_collection is not None:
            logger.info(f"Trades collection object: {trades_collection}")
            try:
                # Example: Insert a test document
                # test_doc_id = trades_collection.insert_one({"test_trade": "example", "timestamp": datetime.now(timezone.utc)}).inserted_id
                # logger.info(f"Inserted test document with ID: {test_doc_id}")
                # count = trades_collection.count_documents({})
                # logger.info(f"Number of documents in trades collection: {count}")
                pass
            except Exception as e:
                logger.error(f"Error interacting with trades collection: {e}")
        else:
            logger.error("Could not get trades collection.")
        MongoDBConnection.close_connection()
    else:
        logger.error("Failed to connect to MongoDB for the example.")
