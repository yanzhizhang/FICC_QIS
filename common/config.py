import os
from enum import Enum


class ModelPlatform(Enum):
    """
    Enumeration representing different model platforms.
    Attributes:
        LOCAL (int): Represents the local model platform.
        PYTHON_PLATFORM (int): Represents the Python-based model platform.
    """

    LOCAL = 0
    PYTHON_PLATFORM = 1


class Config:
    """
    Configuration class for managing connection settings to various services.

    This class holds configuration details for connecting to multiple databases, an FTP server, and a log server. It centralizes these settings, making it easier to manage and access connection parameters throughout the application.

    Attributes:
        MODEL_PLATFORM (ModelPlatform): The platform on which the model runs.
        DATABASES (dict): A dictionary containing connection settings for various databases.
        FTP_SERVER (dict): Connection settings for the FTP server.
        LOG_SERVER (dict): Configuration settings for the log server.
        GS_MINIO_PATH (str): Path configuration for the MinIO server.
    """

    ModelPlatform = ModelPlatform.PYTHON_PLATFORM

    # Database connection settings
    DATABASES = {
        "wind": {
            "host": "10.181.113.239",
            "port": 1433,
            "database": "WindDB",
            "username": "gdsyxypj",
            "password": "g1dsyxypj@Gtja",
        },
        "eficc": {  # EFICC research database
            "host": "10.181.123.18",
            "port": 9006,
            "database": "stock",
            "username": "stock",
            "password": "Gtjagdsy_123456",
        },
        "gs": {
            "host": "10.180.100.41",
            "port": 1521,
            "database": "bmaster",
            "username": "gs",
            "password": "gtja990818",
        },
        "tg": {
            "host": "10.180.100.41",
            "port": 1521,
            "database": "bmaster",
            "username": "tg",
            "password": "gtja990818",
        },
        "test": {
            "host": "10.169.4.168",
            "port": 1521,
            "database": "orcl",
            "username": "gs",
            "password": "gtja990818",
        },
        "bloomberg.ficc": {
            "host": "10.187.129.188",
            "port": 33061,
            "database": "ficc",
            "username": "root",
            "password": "gTja@2020##",
        },
    }

    # FTP server connection settings
    FTP_SERVER = {
        "host": "10.181.139.58",
        "port": 36000,
        "username": "zhongzhaiData",
        "password": "zhongzhai#_1234",
    }

    # Log server settings
    LOG_SERVER = {
        "url": "http://10.116.22.12:9100",
        "index_name": "gslog",
        "type": "_doc",
        "username": "elastic",
        "password": "Gtjagdsy_123456",
        "model": "GS2025Q1",  # Modify as needed
    }

    # Local logging settings
    LOCAL_LOG_FILE = os.path.join(os.getcwd(), "..", "log", "application.log")
    LOCAL_LOG_DIRECTORY = os.path.join(os.getcwd(), "..", "log")

    # MinIO server script_path configuration
    GS_MINIO_PATH = "gsmyc"  # Modify as needed
