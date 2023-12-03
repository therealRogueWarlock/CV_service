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

    results = yolo_model.predict(image)

    predicted_im = get_img_with_predicted(results)

    predicted_classes = get_predicted_classes(results)

    return predicted_classes, predicted_im


def get_img_with_predicted(results):
    im: Image = None

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.format = "JPEG"

    return im


def get_predicted_classes(results):
    predicted_classes = {}
    # a dict af all classes
    class_dict = results[0].names

    # extract predicted classes from results
    for tensor in results[0].boxes.cls:
        predicted_class = class_dict[int(tensor.item())]
        if predicted_class not in predicted_classes:
            predicted_classes[predicted_class] = 0
        predicted_classes[predicted_class] += 1

    return predicted_classes
