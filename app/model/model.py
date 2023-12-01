from PIL import Image
import pickle
from pathlib import Path

print("import model")

BASE_DIR = Path(__file__).resolve(strict=True).parent

print("open pickle file")
with open(f"{BASE_DIR}/object-detection-0.1.0.pkl", 'rb') as file:
    print("load pickle file")
    yolo_model = pickle.load(file)

print("Ml rdy!")

def get_objects_detected(image) -> Image:
    print(type(image))
    print(image)
    
    results = yolo_model(image)

    im:Image = None

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.format = "JPEG"

    return im 
