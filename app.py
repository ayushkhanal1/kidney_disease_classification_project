from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Classifier.utils.common import decodeImage
from Classifier.pipeline.prediction import PredictionPipeline



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """
    Renders the main dashboard of the web application.
    """
    return render_template('index.html')




@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    """
    Triggers the Machine Learning pipeline.
    In a real-world scenario, we use 'dvc repro' to intelligently run 
    only the modified parts of the pipeline.
    """
    # os.system("python main.py") # Alternative direct execution
    os.system("dvc repro")        # Recommended: DVC handles pipeline tracking
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    Receives an image string (Base64) from the frontend, decodes it, 
    and returns the classification result (Normal/Tumor).
    """
    image = request.json['image']
    
    # 1. Decode the Base64 image and save it as 'inputImage.jpg'
    decodeImage(image, clApp.filename)
    
    # 2. Use the PredictionPipeline to classify the saved image
    result = clApp.classifier.predict()
    
    # 3. Send the result back to the frontend as JSON
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS