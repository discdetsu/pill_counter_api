from fastapi import FastAPI,File
from fastapi.responses import ORJSONResponse,FileResponse

from PIL import Image
import io

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


model = "PillDetectorYOLOv7.onnx"
providers = ['CPUExecutionProvider'] # CPU only
session = ort.InferenceSession(model, providers=providers)

# From yolov7 repo
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)



app = FastAPI()

@app.get("/")
def root():
    return "It's work!" 

favicon_path = 'favicon.ico'

@app.get('/favicon.ico')
async def favicon():
    return FileResponse(favicon_path)

@app.post("/image/", response_class=ORJSONResponse)
async def read_items(file: bytes = File()):
    img = Image.open(io.BytesIO(file))
    width, height = img.size
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]:im}

    outputs = session.run(outname, inp)[0]
    bboxes = []
    for i,(_,x0,y0,x1,y1,_,_) in enumerate(outputs):

        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        bboxes.append(box)
    res = {'width': width, 'height': height, 'pill_counts': len(outputs),'annotations': bboxes}

    return ORJSONResponse(res)