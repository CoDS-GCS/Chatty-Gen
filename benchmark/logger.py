import logging
import os
import pathlib
from appconfig import config

exp_log_filename = f"debug-log.log"
log_filepath = os.path.join(config.outputdir, f"{exp_log_filename}")

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.logger.setLevel(logging.DEBUG)
            cls._instance.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            cls._instance.file_handler = None
            cls._instance.stream_handler = None
        return cls._instance

    def configure_file_handler(self, file_path):
        self.file_handler = logging.FileHandler(file_path)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def configure_stream_handler(self):
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.INFO)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def configure_logger(self, file_path=None):
        if file_path:
            self.configure_file_handler(file_path)
        self.configure_stream_handler()

    def get_logger(self):
        return self.logger

logger_obj = Logger()
directory = pathlib.Path(log_filepath).parent
directory.mkdir(parents=True, exist_ok=True)
logger_obj.configure_logger(file_path=log_filepath)

# Getting the configured logger instance
logger = logger_obj.get_logger()


# Example usage:
if __name__ == "__main__":
    pass
    # # Creating and configuring the logger
    # logger = Logger()
    # logger.configure_logger(file_path="app.log")

    # # Getting the configured logger instance
    # configured_logger = logger.get_logger()

    # # Usage in code base
    # configured_logger.debug("Debug message")
    # configured_logger.info("Info message")
    # configured_logger.warning("Warning message")
    # configured_logger.error("Error message")
    # configured_logger.critical("Critical message")
