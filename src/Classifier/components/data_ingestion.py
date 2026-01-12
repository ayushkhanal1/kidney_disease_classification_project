import os
import zipfile
import gdown
from src.Classifier.utils.common import get_size
from src.logger import logging
from src.Classifier.entity.config_entity import (DataIngestionConfig)


class dataingestion:
    """
    Component for ingesting data.
    Handles downloading the dataset from a URL and unzipping it.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the data ingestion component with configuration.
        """
        self.config = config

    def download_file(self) -> str:
        """
        Downloads the zip file from the source URL if it doesn't already exist.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            logging.info(f"Downloading file from :[{dataset_url}] into :[{zip_download_dir}]")
            
            # Simple gdown call using the URL directly to download the dataset
            gdown.download(dataset_url, str(zip_download_dir), quiet=False, fuzzy=True)
            
            logging.info(f"downloaded data from {dataset_url} into {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
        Extracts the downloaded zip file into the unzipped directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            logging.info(f"Extracting zip file: [{self.config.local_data_file}] into dir: [{unzip_path}]")
            zip_ref.extractall(unzip_path)
            logging.info(f"extracted zip file: [{self.config.local_data_file}] into dir: [{unzip_path}]")
