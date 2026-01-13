import os
from src.Classifier.constants import *
from src.Classifier.utils.common import read_yaml, create_directories
from src.Classifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig)
class ConfigurationManager:
    """
    CONFIGURATION MANAGER
    ---------------------
    This class is the 'brain' of the project's settings. Instead of hardcoding 
    paths and values inside the components, we use this manager to:
    1. Read the raw settings from YAML files (config.yaml, params.yaml).
    2. Convert those settings into structured 'Entity' objects (Dataclasses).
    
    This separation makes the code highly modular and easy to update.
    """
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):
        """
        Initializes the manager by loading the YAML files and creating 
        the base root directory for all artifacts.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Create the 'artifacts' folder if it doesn't exist
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
        """
        Extracts prepare base model configuration and return PrepareBaseModelConfig object.
        Initializes the root directory for model artifacts and maps parameters from params.yaml.
        """
        config = self.config.prepare_base_model
        
        # Ensure the directory for storing model artifacts exists
        create_directories([config.root_dir])

        # Create and return the configuration object with values from config.yaml and params.yaml
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

    def get_training_config(self) -> TrainingConfig:
        """
        Extracts training configuration and return TrainingConfig object.
        Initializes the root directory for training artifacts and maps parameters.
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        # We construct the path to the extracted dataset.
        # It's located in the unzip_dir (artifacts/data_ingestion) under the 'kidney-ct-scan-image' folder.
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        
        # Ensure the directory for storing training artifacts (models, plots) exists
        create_directories([Path(training.root_dir)])

        # Map all values from config.yaml and params.yaml into our TrainingConfig entity
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_config


    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Extracts evaluation configuration and return EvaluationConfig object.
        Maps the model path, data path, and MLflow URI for experiment tracking.
        """
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5", # Pointing to the latest trained model
            training_data="artifacts/data_ingestion/kidney-ct-scan-image", # Dataset for validation
            # URI for MLflow experiment tracking (connected to DagsHub in this case)
            mlflow_uri="https://dagshub.com/ayukhanalsh100/kidney_disease_classification_project.mlflow",
            all_params=self.params, # Passing hyperparameters to log them in MLflow
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config

