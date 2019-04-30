import base64
import numpy as np
import io
from PIL import Image
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import tensorflow as tf

def get_model():
        global model
        model = load_model('basic_cnn.h5')
        global graph
        graph = tf.get_default_graph()
        print('Model Loaded From memory')
        
print("Loading Keras Model......")
get_model()

from flask import Flask, request, jsonify, render_template
app = Flask(__name__, template_folder='html')

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods = ['GET','POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((125,125))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    with graph.as_default():
        prediction = model.predict(image).tolist()
        
        print(prediction)

    response = prediction
    return jsonify(response)
    
if __name__ == "__main__":
    app.run(port=8080, debug=False)