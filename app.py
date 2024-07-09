from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import base64

app = Flask(__name__)
def deserialize_sparse_categorical_crossentropy(config, custom_objects=None):
    """Manually deserializes SparseCategoricalCrossentropy for older models."""
    return tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=config['config'].get('reduction'),
        name=config['config'].get('name'),
        from_logits=config['config'].get('from_logits'),
        ignore_index=config['config'].get('ignore_class')
    )

# Load your model, passing the custom deserialization function
custom_objects = {'SparseCategoricalCrossentropy': deserialize_sparse_categorical_crossentropy}
try:
    MODEL = tf.keras.models.load_model('CNNmodel.h5', custom_objects=custom_objects)
except OSError as e:
    print(f"Error loading model: {e}")

CLASS_NAMES = ["benign", "malignant"]

class ImageData:
    def __init__(self, content):
        self.content = content

def read_file_as_image(data):
    image_bytes = base64.b64decode(data)
    image = np.array(Image.open(BytesIO(image_bytes)))
    return image



@app.route('/')
def hello_world():
    return 'Backend is successfully running on cloud'


@app.route('/predict/', methods=['POST'])
def predict():
    try:
        image_data = request.json['content']
        image = read_file_as_image(image_data)
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        if confidence >= 0.60:
            return jsonify({"class": predicted_class, "confidence": confidence}), 200
        else:
            return jsonify({"class": "unknown", "confidence": confidence}), 200
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

if __name__ == "__main__":
    app.run()
