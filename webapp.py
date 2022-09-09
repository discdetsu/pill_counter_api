from PIL import Image
import io
import torch
import cv2
import numpy as np
from PIL import Image

from flask import Flask,request

model = torch.hub.load("yolov7-main", 'custom', "PillDetectorYOLOv7.pt", source='local', force_reload=True)

model.eval()

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files:
        img = Image.open(request.files['file'].stream)
        width, height = img.size
        results = model(img, size=640)
        results = results.pandas().xyxy[0].values.tolist()

        res = {'width': width, 'height': height, 'pill_counts': len(results),'annotations': []}
        for pill in results:
            ann = {'class': 'pill', 'bbox': [pill[0], pill[1], pill[2], pill[3]]}
            res['annotations'].append(ann)
        return res
    else:
        'Error with uploading image!'

if __name__ == "__main__":
    app.run(debug=True) 
