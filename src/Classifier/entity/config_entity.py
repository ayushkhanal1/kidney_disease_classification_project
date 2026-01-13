from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:  
    """
    Configuration for the data ingestion component.
    Defines the paths and URL required for downloading and extracting data.
    """
    root_dir: Path        # Directory where data ingestion artifacts will be stored
    source_URL: str       # URL from which the data will be downloaded
    local_data_file: Path # Path where the downloaded zip file will be saved
    unzip_dir: Path       # Directory where the zip file will be extracted


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    Configuration for the base model preparation component.
    Contains paths and parameters required to load and customize a pre-trained model.
    """
    root_dir: Path                 # Base directory for the component's output
    base_model_path: Path          # Path where the original pre-trained model will be saved
    updated_base_model_path: Path  # Path where the modified (customized) model will be saved
    params_image_size: list        # Resolution of input images (e.g., [224, 224, 3])
    params_learning_rate: float    # Learning rate for the optimizer during compilation
    params_include_top: bool       # Whether to include the original fully-connected top layer of the pre-trained model
    params_weights: str            # Specifies the pre-trained weights to use (e.g., "imagenet")
    params_classes: int            # Number of output classes for our classification task


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the model training component.
    Defines paths and parameters required for the training process.
    
    NOTE: Using 'frozen=True' makes this class immutable, meaning you 
    can't accidentally change these values once they're set during training.
    """
    root_dir: Path                # Directory where training logs and results are saved
    trained_model_path: Path      # Full path (including filename) to save the final trained .h5 model
    updated_base_model_path: Path # Path to the custom base model (the one with your added top layers)
    training_data: Path           # Folder containing the dataset (e.g., 'Normal' and 'Tumor' subfolders)
    params_epochs: int            # How many times the model sees the entire dataset
    params_batch_size: int        # Number of images processed at once before updating model weights
    params_is_augmentation: bool  # Toggle for 'Data Augmentation' to help prevent overfitting
    params_image_size: list       # Image resolution as defined in params.yaml (e.g., [224, 224, 3])
    params_learning_rate: float   # The step size for the optimizer during weight updates


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for the model evaluation component.
    Defines the parameters for validating the model and logging to MLflow.
    """
    path_of_model: Path     # Path to the trained .h5 model file
    training_data: Path     # Folder containing the data to be used for evaluation
    all_params: dict        # All hyperparameters from params.yaml for logging purposes
    mlflow_uri: str         # Remote URI for MLflow (e.g., DagsHub tracking URL)
    params_image_size: list # Expected image resolution
    params_batch_size: int  # Number of images to process in each evaluation batch