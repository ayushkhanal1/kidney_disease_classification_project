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
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int