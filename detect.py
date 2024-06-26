import os
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from argparse import ArgumentParser

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLODetector:
    def __init__(self,
        weights = None,  # model.pt path(x)
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        imgsz=[640, 640],  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,
        save_txt=False,
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,
        dnn=False,  # use OpenCV DNN for ONNX inference
        save_conf=False,
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        nosave=False,  # do not save images/videos
        save_crop = False,
        update=False,
        ):
        self.weights = weights
        self.imgsz = imgsz
        self.max_det = max_det
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.dnn = dnn
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.update = update
        
        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
    
    def Prediction(self, image):
        dataset = LoadImages(image, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        highest_conf = 0
        highest_conf_box = None
        all_boxes = []

        for path, im, im0s, _, _ in dataset:
            im = torch.from_numpy(im).to(self.device).float() / 255.0  # Normalize and add batch dimension
            if len(im.shape) == 3:
                im = im.unsqueeze(0)

            pred = self.model(im, augment=self.augment, visualize=self.visualize)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            for det in pred:  # Process detections
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()  # Scale boxes
                    for *xyxy, conf, cls in reversed(det):
                        if conf > highest_conf:
                            highest_conf = conf
                            x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                            highest_conf_box = [x1, y1, x2, y2]  # Update highest confidence box
            if highest_conf_box:
                all_boxes.append(highest_conf_box) 

        return all_boxes   # Return the last processed frame and highest confidence box

