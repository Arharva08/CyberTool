from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)
model = MobileNet(weights='imagenet')


@app.route('/deepfake', methods=['POST'])
def detect_deepfake():
    img = request.files['file']
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    result = decode_predictions(prediction, top=3)[0]

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
