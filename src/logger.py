import logging
import os
from datetime import datetime

# folder_format
folder_format = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# folder_path -> current directory, folder name, folder format
# below code means get cwd create logs folder and inside create log_folder
folder_path = os.path.join(os.getcwd(),"logs",folder_format)  # getcwd meand get current working director

# make directory of log, so if it already exist there will be no error
os.makedirs(folder_path,exist_ok=True)

# log_file_path -> create log file inside log folder
log_file_path = os.path.join(folder_path,folder_format)

# basicConfig
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)


