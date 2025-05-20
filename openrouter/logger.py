import os
import logging
import datetime

LOG_ROOT = "./logs"

def setup_log_folder():
    """Create the main log folder and a subfolder for today's date."""
    if not os.path.exists(LOG_ROOT):
        os.makedirs(LOG_ROOT)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    daily_log_folder = os.path.join(LOG_ROOT, today)
    if not os.path.exists(daily_log_folder):
        os.makedirs(daily_log_folder)
    
    return daily_log_folder

def setup_logger(log_file_path):
    """Set up a logger to write chat responses to the specified log file."""
    logger = logging.getLogger("ChatLogger")
    logger.setLevel(logging.INFO)
    
    # Create file handler to write to log file
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    
    # Formatter to include a timestamp in each log entry
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Avoid duplicate log entries by checking if handlers already exist
    if not logger.handlers:
        logger.addHandler(fh)
    
    return logger

def log_chat_response(logger, chat_response):
    """Log a chat response using the provided logger."""
    logger.info(chat_response)

def init_logger():
    """Initialize the logger and return the logger instance."""
    daily_log_folder = setup_log_folder()
    log_file_path = os.path.join(daily_log_folder, "chat_responses.log")
    logger = setup_logger(log_file_path)
    return logger
