from flask import Flask, request, jsonify, render_template
from keras import models
from keras import utils
import numpy as np

app = Flask(__name__)
model = models.load_model('models/mdfp_1.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file found"

    file = request.files['file']
    file.save('pictures/' + file.filename)

    img = utils.load_img('pictures/' + file.filename,
                         target_size=(48,48),
                         color_mode="grayscale")
    img_array = utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction_result = model.predict(img_array)
    predicted_class = np.argmax(prediction_result[0])
    class_labels = ['angry', 'disgust',
                    'fear', 'happy',
                    'neutral', 'sad',
                    'surprise']
    predicted_label = class_labels[predicted_class]
    return render_template('result.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
