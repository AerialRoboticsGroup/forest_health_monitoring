import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

W, H = 512, 512


class DetectionModel:
    def __init__(self, weightsPath, cfgPath, classes, colors, thres=0.0, sep=0.01):
        self.classes = classes
        self.sep = sep
        self.thres = thres
        # self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.colors = colors
        self.model = cv2.dnn.readNet(weightsPath, cfgPath)
        self.initialise()

    def initialise(self):
        net = self.model
        self.layer_names = net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def predict(self, img):
        net = self.model
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []

        width, height = img.shape[1], img.shape[0]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.thres:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.thres, self.sep)

        return boxes, confidences, class_ids, indexes

    def cropDetectedObject(self, boxes, confidences, class_ids, indexes, im):
        pilImg = Image.fromarray(np.uint8(im))
        width, height = im.shape[1], im.shape[0]

        croppedImgs = []
        labelIds = []
        coordBoxes = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # x, y, x+w, y+h
                coordBox = (max(x, 0), max(y, 0), min(int(x + w), width), min(int(y + h), height))
                # coordBox = (max(x, 0), max(y, 0), x+w, y+h)
                # print(boxes[i], coordBox)
                croppedImgs.append(cv2.resize(np.asarray(pilImg.crop(coordBox)), (width, height)))
                labelIds.append(class_ids[i])
                # coordBoxes.append(boxes[i])
                coordBoxes.append(boxes[i])
        return croppedImgs, labelIds, coordBoxes

    def drawDetectionBB(self, boxes, confidences, class_ids, indexes, im, crown=True):
        img = im.copy()
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # readIdx = 0 if not crown else i
                if not crown and i > len(class_ids):
                    i = len(class_ids)-1
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + 20), color, -1)
                if crown:
                    cv2.putText(img, "CL: " + label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                else:
                    cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        return img
