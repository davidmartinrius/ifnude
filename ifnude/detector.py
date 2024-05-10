import os
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from tqdm import tqdm
import urllib.request

from .detector_utils import preprocess_image


def dummy(*args, **kwargs):
    pass

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

def download_model(model_folder):
    
    model_url = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/detector.onnx"
    model_name = os.path.basename(model_url)
    model_path = os.path.join(model_folder, model_name)
    
    if not os.path.exists(model_path):
        print("Downloading the detection model to", model_path)
        download(model_url, model_path)

    return model_path
        
def download_classes(model_folder):

    classes_url = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/classes"
    classes_path = os.path.join(model_folder, "classes")

    if not os.path.exists(classes_path):
        print("Downloading the classes list to", classes_path)
        download(classes_url, classes_path)

    return classes_path

def detect(img, model_path=None, classes_path=None, mode="default", min_prob=None):

    if not model_path or not classes_path:
        home = Path.home()
        model_folder = os.path.join(home, f".ifnude/")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)
            
    if not model_path:
        model_path = download_model(model_folder)
        
    #if not classes_path:        
    #    classes_path = download_classes(model_folder)
    #classes = [c.strip() for c in open(classes_path).readlines() if c.strip()]

    classes = [
        "EXPOSED_BELLY",
        "EXPOSED_BUTTOCKS"
        "EXPOSED_BREAST_F",
        "EXPOSED_GENITALIA_F",
        "EXPOSED_GENITALIA_M",
        "EXPOSED_BREAST_M"
    ]

    # we are loading the model on every detect() because it crashes otherwise for some reason xD
    detection_model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    if mode == "fast":
        image, scale = preprocess_image(img, min_side=480, max_side=800)
        if not min_prob:
            min_prob = 0.5
    else:
        image, scale = preprocess_image(img)
        if not min_prob:
            min_prob = 0.6

    outputs = detection_model.run(
        [s_i.name for s_i in detection_model.get_outputs()],
        {detection_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
    )

    labels = [op for op in outputs if op.dtype == "int32"][0]
    scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
    boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

    boxes /= scale
    processed_boxes = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < min_prob:
            continue
        box = box.astype(int).tolist()
        label = classes[label]
        if label == "EXPOSED_BELLY":
            continue
        processed_boxes.append(
            {"box": [int(c) for c in box], "score": float(score), "label": label}
        )

    return processed_boxes

def censor(img_path, out_path=None, visualize=False, parts_to_blur=[]):
    if not out_path and not visualize:
        print(
            "No out_path passed and visualize is set to false. There is no point in running this function then."
        )
        return

    image = cv2.imread(img_path)
    boxes = detect(img_path)

    if parts_to_blur:
        boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
    else:
        boxes = [i["box"] for i in boxes]

    for box in boxes:
        image = cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
        )

    return image
