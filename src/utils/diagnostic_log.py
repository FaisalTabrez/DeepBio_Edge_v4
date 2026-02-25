import sys
import os
import logging

# Fix path
sys.path.append(os.getcwd())

from src.edge.config_init import LOGS_PATH, BASE_PATH
from src.edge.logger import setup_localized_logging

print(f"BASE_PATH detected as: {BASE_PATH}")
print(f"LOGS_PATH detected as: {LOGS_PATH}")

try:
    setup_localized_logging()
    print("Logger setup called.")
    
    # Write a test log
    logging.info("Test Log Entry from Diagnostic Script")
    
    log_file = LOGS_PATH / 'session.log'
    if log_file.exists():
        print(f"SUCCESS: Log file created at {log_file}")
        with open(log_file, 'r') as f:
            print("--- Log Content ---")
            print(f.read())
            print("-------------------")
    else:
        print(f"FAILURE: Log file not found at {log_file}")

except Exception as e:
    print(f"ERROR: {e}")
