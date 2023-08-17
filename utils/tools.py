import os
import torch

from bytetracker import BYTETracker
from detector.PicoDet import PicoDetector
from utils.multi_estimator import MultiEstimator
from ActionRecognition.Classifier import ActionRecognition


from models.MFNet.MFNet import MFNet
from models.MobileNet.MSNet import MSNet
from models.MSKNet.MSKNet import MSKNet
from models.STGCN.STGCN import STGCN
from models.SMLP.backbone import SMLP
from models.SGN.backbone import SGN
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging

def save_checkpoint(states, is_best, output_dir,model_name,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir,model_name,filename))
    if not os.path.exists( os.path.join(output_dir,model_name)):
        os.mkdir( os.path.join( output_dir,model_name))
   
def load_model(args):

    if args.model_name == 'msknet':
        model = MSKNet(args)
    elif args.model_name == 'msnet':
        model = MSNet(args)
    elif args.model_name == 'mfnet':
        model = MFNet(args)
    elif args.model_name == 'st-gcn':
        model = STGCN(args)
    elif args.model_name == 'smlp':
        model = SMLP(args)
    elif args.model_name == 'sgn':
        model = SGN(args)
    else:
        raise 'Unknown model name.'
    return model.to(args.device)
def inference_with_detector(args, img_path):

    estimator = MultiEstimator(args)
    detector = PicoDetector(args)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    before = time.time()
    boxes = detector.detect(img)
    mid = time.time()
    person_count = len(boxes)
    humans = estimator.inference(img,boxes)
    frame = estimator.vis(img,humans,boxes)

    after = time.time()
    plt.figure(figsize=(10,8))
    plt.imshow(img)
    plt.imshow(frame)
    plt.title('Total {:.2f}ms,detect:{:.2f}ms. \n Pose estimation:{:.2f}ms, avg: {:.2f}ms/person.'.format((after - before)*1000,(mid - before)*1000,(after - mid)*1000,(after - mid)*1000/person_count))
    plt.savefig(img_path.split('.')[0]+'-estimation.png')
    plt.show()

def inference_with_tracker(args,video_path):
    
    estimator = MultiEstimator(args)
    tracker = BYTETracker()
    classifier = ActionRecognition(args)
    capture = cv2.VideoCapture(video_path)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    
    frames_buffer = {}

    print(frame_width,frame_height)
    out = cv2.VideoWriter(os.path.join('video04.mov'), cv2.VideoWriter_fourcc(*"avc1"), 30, (frame_width, frame_height),True) 

    counter =  0
    dets = None
    detector = PicoDetector(args)
    while True:
            
        ret, frame = capture.read()
        if not ret:
            break
        before =time.time()
       
        dets = detector.detect(frame)

        det_time = time.time()

        if len(dets) == 0:
            continue

        online_targets = tracker.update(torch.tensor(np.array(dets)),counter)
       
        print(online_targets)
        counter +=1 
        track_time = time.time() 
        if len(online_targets) == 0:
            continue
        dets = online_targets

        humans, humans_scaled = estimator.inference(frame,dets)
        labels = []
        for idx in range(len(online_targets)):
            if online_targets[idx][4] in frames_buffer.keys():

                frames_buffer[online_targets[idx][4]].append(humans_scaled[idx])
                if len(frames_buffer[online_targets[idx][4]]) >= args.seg:
                    frames_buffer[online_targets[idx][4]] = frames_buffer[online_targets[idx][4]][-args.seg:]
                    labels.append(classifier.infer(np.array(frames_buffer[online_targets[idx][4]]),None))
                else:
                    labels.append('Tracking')
            else :
                frames_buffer.update({online_targets[idx][4]:humans_scaled[idx]})
                labels.append('Tracking')

        print(humans.shape)
        end = time.time()
        logging.info('Total {total:.2f}ms/frame\t Detect cost: {det_c:.2f}ms/frame \t Tracker cost: {t_c:.2f}ms/frame \t Pose estimation: {p_c:.2f}ms'.
                     format(total = (end -before)*1000, 
                            det_c =( det_time - before)*1000, 
                            t_c = (track_time - det_time)*1000, 
                            p_c = (end - track_time)*1000))
        frame = estimator.vis(frame,humans,dets, labels)
        cv2.imshow('video', frame)
        out.write(frame)
        if cv2.waitKey(50) == 27:
            out.release()
            break
        
    out.release()
