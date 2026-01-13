import os
import zipfile
import gdown
from src.Classifier.utils.common import get_size
from src.logger import logging
from src.Classifier.entity.config_entity import (DataIngestionConfig)


class dataingestion:
    """
    Component for ingesting data.
    
    This is the first step of the project: bringing raw data into our environment.
    It automates the manual process of downloading and extracting files.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the component with paths and URLs specified in configuration.
        """
        self.config = config

    def download_file(self) -> str:
        """
        Downloads the dataset zip file.
        
        Using 'gdown' allows us to download directly from Google Drive.
        We check if the file already exists to avoid redundant downloads.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            
            # Ensure the directory exists before downloading
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            
            logging.info(f"Downloading file from :[{dataset_url}]")
            
            # 'fuzzy=True' helps gdown handle different types of Drive links
            gdown.download(dataset_url, str(zip_download_dir), quiet=False, fuzzy=True)
            
            logging.info(f"Downloaded record: {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
        Extracts the zip file contents.
        
        Unzipping organizes the data into the 'artifacts/data_ingestion' folder 
        so the next stages can easily access the images.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            logging.info(f"Extracting to: {unzip_path}")
            zip_ref.extractall(unzip_path)
