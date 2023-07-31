import  cv2
import os
from bytetracker import BYTETracker

from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import time
import cv2
import numpy as np
from detector.Detector import FastDetector
def detect_person(frame):
    
    image = Image.fromarray(frame)
    print(image.size)
    
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to('mps')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

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
            boxes.append([*box,score.detach().cpu().numpy(),label.item()])
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    return boxes

tracker = BYTETracker()

capture = cv2.VideoCapture('examples/video3.mp4')
counter =  0
dets = None
class TA:
    def __init__(self) -> None:
          
        self.device = 'cpu'

        self.detector_weight_path = os.path.join('checkpoints','detector.pth')

        self.threshold = 0.25

detector = FastDetector(TA())
while True:
        
        ret, frame = capture.read()
        if not ret:
            break
        before =time.time()
        if counter %1 == 0:

            dets = detector.detect(frame)
            if len(dets) == 0:
                continue
            
            counter = 0
        counter+=1
        online_targets = tracker.update(torch.tensor(np.array(dets)),2)
        end = time.time()
        print(online_targets)
        if len(online_targets) == 0:
            continue
        dets = online_targets
        cv2.putText(frame, "fps: " + str(round(1/(end - before),2)), (5,34),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0,0,255))
        for i in online_targets:
            cv2.rectangle(frame,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])), (0,0,255),4)
        cv2.imshow('video', frame)
        if cv2.waitKey(50) == 27:
            break

