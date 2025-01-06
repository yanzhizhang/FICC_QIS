# common/log.py

import datetime as dt
import logging
import os

import pytz
import requests
from requests.auth import HTTPBasicAuth

from common.config import Config, ModelPlatform


class Log:
    _instance = None  # Class-level instance variable for Singleton pattern

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Log, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        """
        Initialize the logger based on configuration.

        If 'esurl' is in the config, a remote logger will be used.
        Otherwise, a local file logger will be used.
        """
        if not os.path.exists(Config.LOCAL_LOG_DIRECTORY):
            os.makedirs(Config.LOCAL_LOG_DIRECTORY)

        if Config.ModelPlatform == ModelPlatform.LOCAL:
            filename = Config.LOCAL_LOG_FILE
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                filename=filename,
                filemode="a",
            )
        else:
            self.url = f"{Config.LOG_SERVER['url']}/{Config.LOG_SERVER['index_name']}/{Config.LOG_SERVER['type']}"
            self.username = Config.LOG_SERVER["username"]
            self.password = Config.LOG_SERVER["password"]
            self.model = Config.LOG_SERVER["model"]

    def _get_current_time(self):
        """Returns the current time in 'Asia/Shanghai' timezone."""
        return dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

    def _send_remote_log(self, log_type, content):
        """Helper method to send log data to the remote server."""
        document_data = {
            "date": self._get_current_time(),
            "content": "".join(str(arg) for arg in content),
            "model": self.model,
            "type": log_type,
        }

        response = requests.post(
            self.url,
            json=document_data,
            auth=HTTPBasicAuth(self.username, self.password),
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 201:
            print(f"Failed to add document. Status code: {response.status_code}, Response: {response.text}")

    def _log_local_message(self, level, *args):
        """Helper method to log messages locally."""
        content = "".join(str(arg) for arg in args)
        getattr(logging, level)(content)

    def debug(self, *args):
        """Logs a debug message."""
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            self._log_local_message("debug", *args)
        else:
            self._log_local_message("debug", *args)
            # self._send_remote_log("DEBUG", args)

    def info(self, *args):
        """Logs an info message."""
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            self._log_local_message("info", *args)
        else:
            self._log_local_message("info", *args)
            # self._send_remote_log("INFO", args)

    def warning(self, *args):
        """Logs a warning message."""
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            self._log_local_message("warning", *args)
        else:
            self._log_local_message("warning", *args)
            # self._send_remote_log("WARNING", args)

    def error(self, *args):
        """Logs an error message."""
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            self._log_local_message("error", *args)
        else:
            self._log_local_message("error", *args)
            # self._send_remote_log("ERROR", args)

    def critical(self, *args):
        """Logs a critical message."""
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            self._log_local_message("critical", *args)
        else:
            self._log_local_message("critical", *args)
            # self._send_remote_log("CRITICAL", args)
