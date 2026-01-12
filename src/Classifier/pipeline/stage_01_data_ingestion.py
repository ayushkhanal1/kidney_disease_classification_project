from src.logger import logging
from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.data_ingestion import dataingestion
STAGE_NAME = "Data Ingestion" 

class DataIngestionTrainingPipeline:
    """
    Orchestrates the data ingestion process.
    This class brings together configuration management and the data ingestion component.
    """
    def __init__(self):
        pass
    
    def main(self):
        """
        Executes the data ingestion steps:
        1. Access the configuration manager.
        2. Get the specific data ingestion configuration.
        3. Trigger the download and extraction processes.
        """
        # Step 1: Load the configuration
        config = ConfigurationManager()
        
        # Step 2: Get data ingestion specific configuration
        data_ingestion_config = config.get_data_ingestion_config()

        # Step 3: Use the configuration to download and extract the dataset
        data_ingestion = dataingestion(config=data_ingestion_config)
        data_ingestion.download_file()      # Downloads the dataset zip
        data_ingestion.extract_zip_file()   # Extracts the zip file into the artifact directory


if __name__ == '__main__':
    try:
        logging.info(f"\n\n ----------{STAGE_NAME} started ------------------- \n\n")
        
        # Instantiate and run the data ingestion training pipeline
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        logging.info("Data ingestion pipeline execution initiated")
        
        data_ingestion_pipeline.main()
        
        logging.info("Data ingestion pipeline execution completed successfully")
        logging.info(f"\n\n ----------{STAGE_NAME} completed ------------------- \n\n")

    except Exception as e:
        # Log any error to the log file or console
        logging.exception(e)
        raise e
