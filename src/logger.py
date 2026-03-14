import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path=os.path.join(os.getcwd(), 'logs', LOG_FILE)
#this means that the log file will be created in a 'logs' directory within the current working directory, and the log file will be named with the current date and time.
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

