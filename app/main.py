#!/bin/python3
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from model import yolo_model_pretrained

app = FastAPI()

# A dictionary to store image URLs and their corresponding image data
image_urls = {}


@app.get("/")
def home():
    return "Hallo im runnin!"


@app.post("/detect_objects")
async def detect_objects(image: UploadFile = File(...)):
    # convert uploadfile to Pil image 
    image_file = Image.open(BytesIO(await image.read()))

    # get predidted image
    predicted_classes, predicted_img = yolo_model_pretrained.get_objects_detected(image_file)

    # convert pil image to bytes
    predicted_img_bytes = image_to_byte_array(predicted_img)

    response = Response(content=predicted_img_bytes, media_type="image/jpg")
    return response


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, log_level="info")
