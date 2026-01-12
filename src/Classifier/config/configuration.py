from src.Classifier.constants import *
from src.Classifier.utils.common import read_yaml, create_directories
from src.Classifier.entity.config_entity import (DataIngestionConfig)
from src.Classifier.entity.config_entity import (PrepareBaseModelConfig)
class ConfigurationManager:
    """
    Manages the configuration for the entire project.
    Reads YAML files and provides configuration objects for different components.
    """
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):
        """
        Initializes ConfigurationManager with config and params file paths.
        Creates the root directory for artifacts.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Extracts data ingestion configuration from the main config and returns a DataIngestionConfig object.
        """
        config = self.config.data_ingestion
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
