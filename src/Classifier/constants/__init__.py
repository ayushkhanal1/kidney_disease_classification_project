from pathlib import Path

"""
CENTRAL CONSTANTS
-----------------
These paths define where our project-wide settings live.
By keeping them here, we avoid hardcoding file paths in multiple places.
"""

# Config: Defines artifact paths and URLs
CONFIG_FILE_PATH = Path("config/config.yaml") 

# Params: Defines hyperparameters (Epochs, Learning Rate, Batch Size)
PARAMS_FILE_PATH = Path("params.yaml")
