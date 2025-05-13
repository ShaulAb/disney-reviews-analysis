"""
Logging utility for the Disney Reviews Analysis project.
"""
import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_level=logging.INFO):
    """
    Set up and configure a logger.
    
    Args:
        name (str): Logger name, typically the module name
        log_level (int): Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create file handler
    current_date = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(f"logs/{current_date}_{name}.log")
    file_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 