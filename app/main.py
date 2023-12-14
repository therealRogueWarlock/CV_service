#!/bin/python3
from io import BytesIO
from PIL import Image


from app.model import cnn_model
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response

app = FastAPI()
@app.post("/detect_objects")
async def detect_objects(image: UploadFile = File(...)):
    # convert uploadfile to Pil image 
    image_file = Image.open(BytesIO(await image.read()))

    # get classes
    predicted_classes, _ = cnn_model.get_objects_detected(image_file)

    return predicted_classes


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

@app.get("/")
def home():
    return "Hallo test: im runnin!"


if __name__ == "__main__":
    import uvicorn
    print("should not run in container")
    uvicorn.run("main:app", host="0.0.0.0", port=80, log_level="info")
