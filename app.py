from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2
import numpy as np

import gdown

# Path where model will be saved
model_path = "best.pt"

# Check if model already exists
if not os.path.exists(model_path):
    # Replace with your actual file ID
    file_id = "1TCeW4rYvHMHkg5w_nghibrHkXLz3qXNL"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
#-------------------------------------------------------------------------
app = Flask(__name__)
model = YOLO('best.pt')  # path to your trained model

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(path)

            results = model(path)
            res_plotted = results[0].plot()
            cv2.imwrite(path, res_plotted)  # overwrite with boxes

            return render_template('index.html', user_image=path)
    return render_template('index.html', user_image=None)

if __name__ == '__main__':
    app.run(debug=True)