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
    """
    root_dir: Path                # Directory for training artifacts
    trained_model_path: Path      # Path where the final trained model will be saved
    updated_base_model_path: Path # Path to the base model that will be trained
    training_data: Path           # Path to the directory containing training images
    params_epochs: int            # Number of training epochs
    params_batch_size: int        # Size of each training batch
    params_is_augmentation: bool  # Whether to apply data augmentation
    params_image_size: list       # Input size for the model
    params_learning_rate: float   # Learning rate for the optimizer
