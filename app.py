from flask import Flask, render_template, request
from ml.model import ResNet18Model
import gdown
import torch
import os
from werkzeug.utils import secure_filename
import cv2
from albumentations import (
    Compose, LongestMaxSize, PadIfNeeded
)
import numpy as np
from gevent.pywsgi import WSGIServer
app = Flask(__name__)

model = ResNet18Model()
model_file_name = 'floorplan_resnet18.pth'
google_drive_id = '1NZ-r8lcEAvo8ThCbjSnVfqUVNJ6tmuy0'
gdown.download(f'https://drive.google.com/uc?id={google_drive_id}', model_file_name)
# if not os.path.exists('floorplan_resnet18.pth'):
#     subprocess.Popen(['scripts/dl-gdrive', google_drive_id , model_file_name])

if torch.cuda.is_available():
    trained_weights = torch.load(model_file_name)
else:
    trained_weights = torch.load(model_file_name, map_location='cpu')

model.load_state_dict(trained_weights['state_dict'])
model.eval()

CLASSES = ["a Floorplan", "not a Floorplan"]

def process_image(img_path):
    img_color = cv2.imread(img_path)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    augmentation = Compose([
        LongestMaxSize(max_size=224),
        PadIfNeeded(
            min_height=224,
            min_width=224,
            border_mode=0)
    ])
    augmented = augmentation(**{
        'image': img_color
    })
    return augmented["image"]

def get_prediction(img_path):
    list_img = [process_image(img_path)]
    data = torch.from_numpy(np.array(list_img)[:, :, :, :].transpose(0, 3, 1, 2).astype(np.float32))
    output = model(data)
    prediction = torch.argmax(output).item()
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
