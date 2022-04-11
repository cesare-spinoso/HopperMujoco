import logging
import os
from datetime import datetime



class Logger:
    def __init__(self, path, logger_name="log"):
        logging.basicConfig(
            filename=os.path.join(path, f"{logger_name}.txt"),
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
        self.logger = logging.getLogger()
        self.location = path

    def log(self, msg):
        print(msg)
        self.logger.info(f"{msg}")

def start_logging(logger_name="log"):
    """
    Starts an instance of the Logger class to log the training results.
    :return logger: the Logger class instance for this training session
    :return results_path: the location of the log files
    """
    # making sure there is a results folder
    results_folder = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # making a folder for this current training
    results_path = os.path.join(
        os.getcwd(), "results/" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    )
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    # start an instance of the Logger class :)
    logger = Logger(results_path, logger_name)

    return logger