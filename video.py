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
        text_duration = 0
        capture = cv2.VideoCapture(self.args.video_path)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        if self.args.save_result:
            assert self.args.save_path !=None,'Need to specify save path.'
            out = cv2.VideoWriter(os.path.join(self.args.save_path), cv2.VideoWriter_fourcc(*"avc1"), 30, (frame_width, frame_height),True) 
     
        while True:
                
            ret, frame = capture.read()
            if not ret:
                break
            before =time.time()
        
            dets = self.detector.detect(frame)

            det_time = time.time()

            if len(dets) != 0:
                

                online_targets = self.tracker.update(torch.tensor(np.array(dets)),counter)
            
                counter += 1 
                track_time = time.time() 
                dets = online_targets

                humans, humans_scaled = self.estimator.inference(frame,dets)
                if not self.args.disable_filter:
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
                            if round(prob)==0:
                                text_duration = 30
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
                if self.args.verbose:
                    logging.info('Total {total:.2f}ms/frame\t Detect cost: {det_c:.2f}ms/frame \t Tracker cost: {t_c:.2f}ms/frame \t Pose estimation: {p_c:.2f}ms \t Action recognition: {r_c:.2f}ms'.
                                format(total = (end -before)*1000, 
                                        det_c =( det_time - before)*1000, 
                                        t_c = (track_time - det_time)*1000, 
                                        p_c = (estimation_time - track_time)*1000,
                                        r_c = (end - estimation_time)*1000),
                                    )
                
                frame = self.estimator.vis(frame,humans,dets, labels, probs, vis_skeleton = self.args.skeleton_visible)

            if text_duration !=0:

                cv2.putText(frame, "Fall detected !!!!!!!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                text_duration-=1
            if self.args.save_result:
                out.write(frame)

            cv2.imshow('video', frame)

        
            if cv2.waitKey(50) == 27:
                if self.args.save_result:
                    out.release() 
                break
        if self.args.save_result:
            out.release()

    def initialize_argparser(self) ->None:

        parser = argparse.ArgumentParser(

        description='Arguments for skeleton-based action recognition.')

        parser.add_argument('--img_shape', type=list, default=[256,192], help='the input images shape.')
        parser.add_argument('--device', type=str, default='cpu')
         

        parser.add_argument('--estimator_onnx_path',type = str, default=os.path.join('checkpoints','mfnet','mfnet-small.onnx'))
        parser.add_argument('--detector_weight_path', type=str, default= os.path.join('checkpoints', 'pico.onnx'))
        parser.add_argument('--classifier_weight_path', type=str, default=os.path.join('checkpoints','sgn','fall_detect928-DA-SEG30.pth')) #os.path.join('labels.txt')
        parser.add_argument('--labels_path', type=str, default=os.path.join('dataset','FallData','label.txt'))

        parser.add_argument('--seg', type=int, default=30)
        parser.add_argument('--num_joint', type=int, default=17)
        parser.add_argument('--num_classes', type=int, default=1)
        parser.add_argument('--channels', type=int, default=2)
        parser.add_argument('--detector_prob_threshold', type=float, default=0.7)



        parser.add_argument('--disable_filter', action='store_true')
        parser.add_argument('--skeleton_visible',action='store_true')
        parser.add_argument('--verbose',action='store_true')


        parser.add_argument('--video_path',type=str, required=True)
        parser.add_argument('--save_result',action='store_true')
        parser.add_argument('--save_path', type = str, default = None)

        self.args = parser.parse_args()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.video_inference()