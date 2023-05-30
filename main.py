

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.responses import JSONResponse

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.lite.Interpreter(model_path="./models/model.tflite")
classes = ["diseased",  "healthy",]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("got request")
        image = await file.read()

        # Learn about its input and output details
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
        model.allocate_tensors()

        img = Image.open(BytesIO(image)).convert('RGB')
        img = img.resize((224, 224))
        img_np = np.array(img)[None].astype('float32')

        model.set_tensor(input_details[0]['index'], img_np)
        model.invoke()

        class_scores = model.get_tensor(output_details[0]['index'])
        scores = class_scores.tolist()
        predictedClass = classes[class_scores.argmax()]
        print("")
        print("class_scores", class_scores)
        print("Class : ", classes[class_scores.argmax()])
        return JSONResponse(content={"scores": scores, "predictedClass": predictedClass}, status_code=200)

    except Exception as e:
        print(e)
        return JSONResponse(content={"status": "fail", "message": "Something went wrong"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
