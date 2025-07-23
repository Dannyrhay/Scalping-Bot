import logging
from datetime import datetime
import pytz

class UTCTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        utc_dt = datetime.fromtimestamp(record.created, tz=pytz.UTC)
        if datefmt:
            return utc_dt.strftime(datefmt)
        return utc_dt.strftime('%Y-%m-%d %H:%M:%S %Z')

def setup_logging():
    """Set up logging to file and console with UTC timestamps."""
    logger = logging.getLogger('trading_ea')
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = UTCTimeFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    file_handler = logging.FileHandler('logs/trading_ea.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger