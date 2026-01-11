import logging
import os
from datetime import datetime

Log_File = f"log_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Create a unique log file name with timestamp.
logs_path = os.path.join(os.getcwd(), "logs", Log_File)  # Construct the path for logs directory including the file name.
os.makedirs(logs_path, exist_ok=True)  # Create the logs directory if it doesn't exist.

LOG_FILE_PATH = os.path.join(logs_path, Log_File)  # Full path to the log file.

logging.basicConfig(  # Set up basic logging configuration.
    filename=LOG_FILE_PATH,  # Log messages will be written to this file.
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Format: timestamp, line number, logger name, level, message.
    level=logging.INFO,  # Log INFO level and above.
)

