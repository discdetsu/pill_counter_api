from fastapi import FastAPI,File
import torch
from PIL import Image
import io
from fastapi.responses import ORJSONResponse,FileResponse

import warnings
warnings.filterwarnings("ignore")

model = torch.hub.load("yolov7_local", 'custom', "PillDetectorYOLOv7.pt", source='local', force_reload=True)
# results = model('1.jpg', size=640)
# results = results.pandas().xyxy[0].values.tolist()

# img = mpimg.imread('1.jpg')
# annotations = []
# for pill in results:
#     ann = {'class': 'pill', 'bbox': [int(pill[0]), int(pill[1]), int(pill[2]), int(pill[3])]}
#     annotations.append(ann)
# print(annotations)


app = FastAPI()

@app.get("/")
def root():
    return "It's work!" 

favicon_path = 'favicon.ico'

@app.get('/favicon.ico')
async def favicon():
    return FileResponse(favicon_path)
# @app.post("/plot/", responses = {
#         200: {
#             "content": {"image/jpg": {}}
#         }
#     },
#     response_class=Response
# )
# async def create_file(file: bytes = File()):
#     img = Image.open(io.BytesIO(file))
#     results = model(img, size=640)
#     results.render() 
#     res = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB)
#     img_encode = cv2.imencode('.jpg', res)[1].tobytes()

#     return Response(content=img_encode, media_type="image/jpg")

@app.post("/image/", response_class=ORJSONResponse)
async def read_items(file: bytes = File()):
    img = Image.open(io.BytesIO(file))
    width, height = img.size
    results = model(img, size=640)
    results = results.pandas().xyxy[0].values.tolist()

    res = {'width': width, 'height': height, 'pill_counts': len(results),'annotations': []}
    for pill in results:
        ann = {'class': 'pill', 'bbox': [pill[0], pill[1], pill[2], pill[3]]}
        res['annotations'].append(ann)
    return ORJSONResponse(res)

# pip install orjson