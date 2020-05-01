from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
# from albumentations import (
#     Compose, LongestMaxSize, PadIfNeeded
# )

from albumentations_min import PadIfNeeded, LongestMaxSide

import numpy as np
from gevent.pywsgi import WSGIServer
import onnxruntime
app = Flask(__name__)

ort_session = onnxruntime.InferenceSession(os.path.abspath('ml/resnet18.onnx'))
CLASSES = ["a Floorplan", "not a Floorplan"]

def process_image(img_path):
    img_color = cv2.imread(img_path)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return PadIfNeeded(LongestMaxSide(img_color, 224))
    # augmentation = Compose([
    #     LongestMaxSide(max_size=224),
    #     PadIfNeeded(
    #         min_height=224,
    #         min_width=224,
    #         border_mode=0)
    # ])
    # augmented = augmentation(**{
    #     'image': img_color
    # })
    # return augmented["image"]

def get_prediction(img_path):
    list_img = [process_image(img_path)]
    data = np.array(list_img)[:, :, :, :].transpose(0, 3, 1, 2).astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: data}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs).item()
    return CLASSES[prediction]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img_file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(img_file.filename))
        img_file.save(file_path)

        return get_prediction(file_path)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    print("\n########################################")
    print('--- Running on port {} ---'.format(port))
    print("########################################\n")
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()
