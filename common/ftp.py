# -*- encoding: utf-8 -*-
import os
import re
import ftplib
from common.config import Config
from common.log import Log

Log = Log()


class FTPClient:
    """FTP client for uploading and downloading files."""

    def __init__(self):
        self.host = Config.FTP_SERVER["host"]
        self.port = Config.FTP_SERVER["port"]
        self.username = Config.FTP_SERVER["username"]
        self.password = Config.FTP_SERVER["password"]
        self.conn = ftplib.FTP()
        self.conn.encoding = "utf-8"  # Use utf-8 encoding for file names
        self.connect()

    def connect(self):
        """Establishes an FTP connection and logs in."""
        try:
            self.conn.connect(self.host, self.port)
            self.conn.login(self.username, self.password)
            Log.info(f"{self.conn.welcome} - Login successful")
        except Exception as e:
            Log.error(f"Failed to connect or log in: {e}")

    def close(self):
        """Closes the FTP connection."""
        try:
            self.conn.quit()
            Log.info("Connection closed successfully.")
        except Exception as e:
            Log.error(f"Error closing connection: {e}")

    def __enter__(self):
        """Enable usage of FTPClient as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures connection is closed when context exits."""
        self.close()

    def _is_directory(self, path):
        """Checks if a given script_path is a directory on the FTP server."""
        try:
            self.conn.cwd(path)
            self.conn.cwd("..")
            return True
        except ftplib.error_perm:
            return False

    def download_file(self, local_path, remote_path):
        """Downloads a file from the FTP server."""
        try:
            with open(local_path, "wb") as file_handler:
                self.conn.retrbinary(f"RETR {remote_path}", file_handler.write)
            Log.info(f"Downloaded file: {remote_path} to {local_path}")
            return True
        except Exception as e:
            Log.error(f"Error downloading file {remote_path}: {e}")
            return False

    def download_directory(self, local_dir, remote_dir, file_filter=""):
        """Downloads a directory from the FTP server, recursively."""
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            self.conn.cwd(remote_dir)
            remote_files = self.conn.nlst()
            for file_name in remote_files:
                local_path = os.path.join(local_dir, file_name)
                if "." not in file_name:
                    # Recursively download subdirectories
                    self.download_directory(local_path, file_name, file_filter)
                elif not file_filter or re.search(file_filter, file_name):
                    self.download_file(local_path, file_name)
            self.conn.cwd("..")
            Log.info(f"Downloaded directory: {remote_dir} to {local_dir}")
            return True
        except Exception as e:
            Log.error(f"Error downloading directory {remote_dir}: {e}")
            return False

    def list_directory(self, remote_dir, file_filter=None, include_path=False):
        """Lists files in a remote directory with an optional filter."""
        try:
            self.conn.cwd(remote_dir)
            file_names = self.conn.nlst()
            if file_filter:
                file_names = [f for f in file_names if re.search(file_filter, f)]
            if include_path:
                file_names = [os.path.join(remote_dir, f) for f in file_names]
            return file_names
        except Exception as e:
            Log.error(f"Error listing directory {remote_dir}: {e}")
            return []

    def upload_file(self, local_path, remote_path="."):
        """Uploads a file to the FTP server."""
        try:
            remote_directory = os.path.dirname(remote_path)
            if self._is_directory(remote_directory):
                with open(local_path, "rb") as file_handler:
                    self.conn.storbinary(f"STOR {remote_path}", file_handler)
                Log.info(f"Uploaded file: {local_path} to {remote_path}")
                return True
            else:
                Log.error(f"Remote script_path {remote_path} is invalid.")
                return False
        except Exception as e:
            Log.error(f"Error uploading file {local_path}: {e}")
            return False

    def upload_directory(self, local_dir, remote_dir="."):
        """Uploads a directory to the FTP server, recursively."""
        if not os.path.isdir(local_dir):
            Log.error(f"Local directory {local_dir} does not exist.")
            return False

        if not self._is_directory(remote_dir):
            try:
                self.conn.mkd(remote_dir)
            except Exception as e:
                Log.error(f"Error creating remote directory {remote_dir}: {e}")
                return False

        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = os.path.join(remote_dir, item)
            if os.path.isdir(local_path):
                self.upload_directory(local_path, remote_path)
            else:
                self.upload_file(local_path, remote_path)
        Log.info(f"Finished uploading directory: {local_dir}")
        return True


if __name__ == "__main__":
    local_file = os.path.join(os.getcwd(), "data/ftp_download.xlsx")
    REMOTE_FILE = "/指数指标_20231218.xlsx"

    with FTPClient() as ftp:
        RES = ftp.download_file(local_file, REMOTE_FILE)
        Log.info("Download successful:", RES)
