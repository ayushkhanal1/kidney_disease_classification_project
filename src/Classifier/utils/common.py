import os
from box.exceptions import BoxValueError
import yaml
from src.logger import logging 
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content wrapped in a ConfigBox.
    
    ConfigBox allows accessing dictionary keys as attributes (e.g., config.key instead of config['key']).

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other errors during file reading.

    Returns:
        ConfigBox: The content of the YAML file.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list): A list containing paths of directories to be created.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves data into a JSON file.

    Args:
        path (Path): The destination path for the JSON file.
        data (dict): The dictionary data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads data from a JSON file and returns it as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: The JSON data accessible via attributes.
    """
    with open(path) as f:
        content = json.load(f)

    logging.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data into a binary file using joblib.

    Args:
        data (Any): The data (e.g., model, object) to be saved.
        path (Path): The destination path for the binary file.
    """
    joblib.dump(value=data, filename=path)
    logging.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: The object loaded from the file.
    """
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Calculates and returns the size of a file in Kilobytes (KB).

    Args:
        path (Path): Path to the file.

    Returns:
        str: A string representing the size (e.g., "~ 10 KB").
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    """
    Decodes a base64 encoded image string and saves it to a file.

    Args:
        imgstring (str): The base64 encoded image string.
        fileName (str): The name/path of the file to save the decoded image.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    """
    Encodes an image file into a base64 string.

    Args:
        croppedImagePath (str): Path to the image file.

    Returns:
        bytes: The base64 encoded version of the image.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
