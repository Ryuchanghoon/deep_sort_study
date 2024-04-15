import cv2
import numpy as np
import sys
import glob
import os
import time
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


class YoloDetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names

        self.device = self.model.names

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device', self.device)


    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_name, force_reload = True)

        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)

        return model
    

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2

        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))


        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord
    
    
    def class_to_label(self, x):
        return self.classes[int(x)]


    def plot_boxes(self, results, frame, height, width, confidence = 0.3):

        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height


        for i in range(n):
            row = cord[i]


            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)

                if self.class_to_label(labels[i]) == 'person':

                    x_center = x1 + (x2 - x1)
                    y_center = y1 + ((y2 - y1) / 2)

                    tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype = np.float32)
                    confidence = float(row[4].item())
                    feature = 'person'

                    detections.append(([x1, y1, int(x2 -x1), int(y2 - y1)], row[4].item(), 'person'))

        return frame, detections


cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = YoloDetector(model_name = None)

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

 
object_tracker = DeepSort(max_age = 5,
                           n_init = 2,
                           nms_max_overlap = 1.0,
                           nn_budget = None,
                           override_track_class= None,
                           embedder = 'mobilenet',
                           half = True,
                           bgr = True,
                           embedder_gpu = True,
                           embedder_model_name = None,
                           embedder_wts = None,
                           polygon = False,
                           today = None)