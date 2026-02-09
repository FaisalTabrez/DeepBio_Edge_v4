import logging
import os
from src.edge.config_init import LOGS_PATH

# Define a global logger name
LOGGER_NAME = "DeepBio-Logger"

def setup_localized_logging():
    """
    Configures the centralized logging system.
    - Clears 'session.log' on startup.
    - Writes to standard terminal (Developer View).
    - Writes to 'session.log' (UI View).
    """
    log_file = LOGS_PATH / 'session.log'
    
    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear handlers if they exist (to avoid duplication on reload)
    if logger.handlers:
        logger.handlers.clear()
        
    # 1. File Handler (The UI Bridge)
    # mode='w' clears the file implicitly on startup
    file_handler = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 2. Console Handler (The Developer Terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', 
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("--- DEEPBIO-SCAN SESSION INITIALIZED ---")
    logging.info(f"Logging Bridge established at: {log_file}")

def get_logger():
    return logging.getLogger(LOGGER_NAME)
