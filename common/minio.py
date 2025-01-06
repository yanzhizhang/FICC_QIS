import requests

from common.log import Log

Log = Log()


class Minio:
    """Minio class for handling file uploads and downloads to and from a remote server."""

    BASE_URL = "http://gateway.fxyjjlyq-fxyj.svc.cluster.local/api/common/file"

    def __init__(self):
        # Any initialization required can go here, e.g., API client, etc.
        pass

    def save_to_remote(self, local_file_name: str, remote_file_path: str) -> bool:
        """
        Uploads a file to a remote server.

        Args:
            file_name (str): The local file name to be uploaded.
            remote_file_path (str): The remote directory script_path where the file should be uploaded.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            requests.exceptions.RequestException: If there is an issue with the HTTP request.
        """
        # Prepare URL and parameters
        params = {
            "filePath": f"{remote_file_path}/",
            "originalFilename": local_file_name,
        }

        try:
            # Fetch the upload URL
            response = requests.put(f"{Minio.BASE_URL}/upload", params=params, timeout=10)
            response.raise_for_status()  # Raises exception for bad responses
            upload_url = response.json()["data"]["uploadFileUrl"]

            # Upload the file
            with open(local_file_name, "rb") as f:
                upload_response = requests.put(upload_url, data=f, timeout=10)
                upload_response.raise_for_status()  # Ensure successful upload
                Log.info(f"File {local_file_name} uploaded successfully to {upload_url}")

            return True

        except FileNotFoundError:
            Log.error(f"File {local_file_name} not found.")
            raise
        except requests.exceptions.RequestException as e:
            Log.error(f"Upload failed: {e}")
            raise
        except Exception as e:
            Log.error(f"Unexpected error during upload: {e}")
            raise

    def download_from_remote(self, original_file_name: str, remote_file_path: str) -> bool:
        """
        Downloads a file from a remote server and saves it locally.

        Args:
            file_name (str): The local file name to save the downloaded content.
            remote_file_path (str): The remote directory script_path from where the file should be downloaded.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.
        """
        # Prepare URL and parameters
        params = {
            "filePath": f"{remote_file_path}/",
            "originalFilename": original_file_name,
        }

        try:
            # Get the download URL
            download_url_resp = requests.get(f"{Minio.BASE_URL}/download", params=params, timeout=10)
            download_url_resp.raise_for_status()  # Raises exception for bad responses
            if download_url := download_url_resp.json().get("data"):
                # Download the file
                file_req = requests.get(download_url, timeout=10)
                file_req.raise_for_status()  # Ensure successful file retrieval

                # Save the file locally
                with open(original_file_name, "wb") as wf:
                    wf.write(file_req.content)
                Log.info(f"File {original_file_name} downloaded successfully from {download_url}")
                return True
            else:
                Log.error(f"Download URL for {original_file_name} not found.")
                return False

        except requests.exceptions.RequestException as e:
            Log.error(f"Download failed: {e}")
            raise
        except Exception as e:
            Log.error(f"Unexpected error during download: {e}")
            raise


def main():
    minio_instance = Minio()  # Create an instance of the Minio class
    file_name = "example.txt"
    remote_file_path = "/uploads"

    if minio_instance.save_to_remote(file_name, remote_file_path):
        print(f"File {file_name} uploaded successfully.")
    else:
        print(f"Failed to upload {file_name}.")

    minio_instance = Minio()  # Create an instance of the Minio class
    file_name = "example.txt"
    remote_file_path = "/downloads"

    if minio_instance.download_from_remote(file_name, remote_file_path):
        print(f"File {file_name} downloaded successfully.")
    else:
        print(f"Failed to download {file_name}.")
