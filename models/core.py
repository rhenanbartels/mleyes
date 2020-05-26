# Inspired by https://github.com/arunponnusamy/cvlib

import cv2
import numpy as np


prototxt = "models/deploy.protoxt"
caffemodel = "models/res10_300x300_ssd_iter_140000.caffemodel"
NET = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)


def detect_face(image, threshold=0.5):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    NET.setInput(blob)

    detections = NET.forward()

    faces = []
    confidences = []

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf < threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        faces.append([startX, startY, endX, endY])
        confidences.append(conf)

    return faces, confidences
