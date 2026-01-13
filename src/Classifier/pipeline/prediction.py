import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf


class PredictionPipeline:
    """
    PREDICTION PIPELINE
    -------------------
    This class handles the end-to-end process of taking a raw image and 
    predicting whether it shows a Kidney Tumor or is Normal.
    
    It encapsulates:
    1. Loading the trained model (.h5 file).
    2. Preprocessing the image to match the model's expected input.
    3. Running the prediction and interpreting the result.
    """
    def __init__(self, filename):
        """
        Initializes the pipeline with the path to the image to be classified.
        Loads the pre-trained model once during startup for better performance.
        
        Args:
            filename (str): The path to the image file (e.g., 'inputImage.jpg').
        """
        self.filename = filename
        # Load model once during initialization to improve prediction speed
        # We load the model from the artifacts directory created during the Training stage.
        # This is the 'final' model after weights have been optimized.
        self.model = load_model(os.path.join("model","model.h5"))

    
    def predict(self):
        """
        Runs the prediction loop: Load -> Preprocess -> Predict -> Interpret.
        
        Returns:
            list: A list containing a dictionary with the prediction result (e.g., 'Tumor' or 'Normal').
        """
        # Reference the pre-loaded model
        model = self.model

        # 1. Load the image from the specified path
        # target_size must match the resolution the model was trained on (224x224 for VGG16)
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        
        # 2. Convert the image (RGB) into a numerical array of pixels
        test_image = image.img_to_array(test_image)
        
        # 3. Expand Dimensions:
        # Keras expects a "batch" of images. Even for one image, we need to make it 
        # look like a list of images: (224, 224, 3) -> (1, 224, 224, 3)
        test_image = np.expand_dims(test_image, axis=0)
        
        # 4. Model Prediction:
        # model.predict returns probabilities for each class. 
        # argmax picks the index with the highest probability.
        result = np.argmax(model.predict(test_image), axis=1)
        print(f"Prediction index: {result}")

        # 5. Result Interpretation:
        # In our dataset, we typically map indices to human-readable labels.
        if result[0] == 1:
            prediction = 'Tumor'
            return [{"image": prediction}]
        else:
            prediction = 'Normal'
            return [{"image": prediction}]