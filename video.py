import os
import torch

from tracker.ByteTracker import BYTETracker
from detector.PicoDet import PicoDetector
from utils.multi_estimator import MultiEstimator
from ActionRecognition.Classifier import ActionRecognition
from utils.filter import OneEuroFilter

import argparse

import time
import cv2

import numpy as np
import logging

class Pipeline():

    def __init__(self) -> None:
        
        self.initialize_argparser()

        self.init_model()
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
        logger = logging.getLogger(__name__)

    def init_model(self):
        
        self.estimator = MultiEstimator(self.args)
        self.tracker = BYTETracker(track_thresh=0.1,track_buffer=30, match_thresh=0.95)
        self.classifier = ActionRecognition(self.args)
        self.detector = PicoDetector(self.args)


        self.filter = OneEuroFilter(1)
        self.scaled_filter = OneEuroFilter(1)
        self.frames_buffer = {}

    def video_inference(self):

        counter =  0
        dets = None

        capture = cv2.VideoCapture(self.args.video_path)
        while True:
                
            ret, frame = capture.read()
            if not ret:
                break
            before =time.time()
        
            dets = self.detector.detect(frame)

            det_time = time.time()

            if len(dets) == 0:
                continue

            online_targets = self.tracker.update(torch.tensor(np.array(dets)),counter)
        
            counter += 1 
            track_time = time.time() 
            if len(online_targets) == 0:
                continue
            dets = online_targets

            humans, humans_scaled = self.estimator.inference(frame,dets)
            if self.args.enable_filter:
                for i in range(len(humans)):
                    humans[i] = self.filter.predict(humans[i],1)
                    humans_scaled[i] = self.scaled_filter.predict(humans_scaled[i],1)
                    
            estimation_time = time.time()
            
            labels = []
            probs = []
            for idx in range(len(online_targets)):
                if len(humans_scaled) == 0:
                    logging.info('No skeleton detected')
                    break
                if online_targets[idx][4] in self.frames_buffer.keys():
                    
                    self.frames_buffer[online_targets[idx][4]].append(humans_scaled[idx])
                    if len(self.frames_buffer[online_targets[idx][4]]) >= self.args.seg:
                        self.frames_buffer[online_targets[idx][4]] = self.frames_buffer[online_targets[idx][4]][-self.args.seg:]
                        label, prob = self.classifier.infer(torch.tensor(self.frames_buffer[online_targets[idx][4]]))
                        labels.append(label)
                        probs.append(prob)
                    else:
                        labels.append('Tracking')
                        probs.append(0)
                else :
                    self.frames_buffer.update({online_targets[idx][4]:[humans_scaled[idx]]})
                    labels.append('Tracking')
                    probs.append(0)

            end = time.time()
            # logging.info('Total {total:.2f}ms/frame\t Detect cost: {det_c:.2f}ms/frame \t Tracker cost: {t_c:.2f}ms/frame \t Pose estimation: {p_c:.2f}ms \t Action recognition: {r_c:.2f}ms'.
            #             format(total = (end -before)*1000, 
            #                     det_c =( det_time - before)*1000, 
            #                     t_c = (track_time - det_time)*1000, 
            #                     p_c = (estimation_time - track_time)*1000,
            #                     r_c = (end - estimation_time)*1000),
            #                   )
            
            frame = self.estimator.vis(frame,humans,dets, labels, probs)
            cv2.imshow('video', frame)
        
            if cv2.waitKey(50) == 27:
             
                break

    def initialize_argparser(self) ->None:

        parser = argparse.ArgumentParser(

        description='Arguments for skeleton-based action recognition.')

        parser.add_argument('--img_shape', type=list, default=[256,192], help='the input images shape.')
        parser.add_argument('--device', type=str, default='cpu')
         

        parser.add_argument('--estimator_onnx_path',type = str, default=os.path.join('checkpoints','mfnet','mfnet-large.onnx'))
        parser.add_argument('--detector_weight_path', type=str, default= os.path.join('checkpoints', 'pico.onnx'))
        parser.add_argument('--classifier_weight_path', type=str, default=os.path.join('checkpoints','sgn','c0833seg30.pth')) #os.path.join('labels.txt')
        parser.add_argument('--labels_path', type=str, default=os.path.join('dataset','ActionData','labels.txt'))

        parser.add_argument('--seg', type=int, default=30)
        parser.add_argument('--num_joint', type=int, default=17)
        parser.add_argument('--num_classes', type=int, default=120)
        parser.add_argument('--channels', type=int, default=2)

        parser.add_argument('--enable_filter', type=bool, default=True)
        parser.add_argument('--detector_prob_threshold', type=float, default=0.4)
        parser.add_argument('--detector_iou_threshold', type=float, default=0.3)
        parser.add_argument('--threshold',type=float, default=0.25)

        parser.add_argument('--video_path',type=str, required=True)

        self.args = parser.parse_args()

def inference_with_tracker(args,video_path):
    
    estimator = MultiEstimator(args)
    tracker = BYTETracker()
    classifier = ActionRecognition(args)
    capture = cv2.VideoCapture(video_path)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    
    filter = OneEuroFilter(1)
    scaled_filter = OneEuroFilter(1)
    frames_buffer = {}

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
    
        counter += 1 
        track_time = time.time() 
        if len(online_targets) == 0:
            continue
        dets = online_targets

        humans, humans_scaled = estimator.inference(frame,dets)
        if args.enable_filter:
            for i in range(len(humans)):
                humans[i] = filter.predict(humans[i],1)
                humans_scaled[i] = scaled_filter.predict(humans_scaled[i],1)
        labels = []
        probs = []
        for idx in range(len(online_targets)):
            if len(humans_scaled) == 0:
                logging.info('No skeleton detected')
                continue
            if online_targets[idx][4] in frames_buffer.keys():
                
                frames_buffer[online_targets[idx][4]].append(humans_scaled[idx])
                if len(frames_buffer[online_targets[idx][4]]) >= args.seg:
                    frames_buffer[online_targets[idx][4]] = frames_buffer[online_targets[idx][4]][-args.seg:]
                    label, prob = classifier.infer(torch.tensor(frames_buffer[online_targets[idx][4]]))
                    labels.append(label)
                    probs.append(prob)
                else:
                    labels.append('Tracking')
                    probs.append(0)
            else :
                frames_buffer.update({online_targets[idx][4]:[humans_scaled[idx]]})
                labels.append('Tracking')
                probs.append(0)

        end = time.time()
        logging.info('Total {total:.2f}ms/frame\t Detect cost: {det_c:.2f}ms/frame \t Tracker cost: {t_c:.2f}ms/frame \t Pose estimation: {p_c:.2f}ms'.
                     format(total = (end -before)*1000, 
                            det_c =( det_time - before)*1000, 
                            t_c = (track_time - det_time)*1000, 
                            p_c = (end - track_time)*1000))
        frame = estimator.vis(frame,humans,dets, labels, probs)
        cv2.imshow('video', frame)
        out.write(frame)
        if cv2.waitKey(50) == 27:
            out.release()
            break
        
    out.release()

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.video_inference()