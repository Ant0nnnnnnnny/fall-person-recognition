from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import time
import matplotlib.pyplot as plt
import cv2
def detect_person(imgurl):
    image = Image.open(imgurl)
    img = cv2.imread(imgurl)
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-base')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-base")

    inputs = image_processor(images=image, return_tensors="pt")
    before = time.time()
    outputs = model(**inputs)

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    after = time.time()
    print(after - before)
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        if model.config.id2label[label.item()] =='person':
            boxes.append(box)
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    return boxes
