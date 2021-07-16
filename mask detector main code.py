from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
from random import randint
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Initializing the tracker

trackers=cv2.MultiTracker_create()
model = load_model("mask_recog1.h5")
vs = cv2.VideoCapture(r'D:/dataset/m6.mp4')
ret,frame=vs.read()
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('file1.avi',
                         fourcc,
                         10, size)
frame = imutils.resize(frame, width=1200)

colors=[]

#selecting k objects to be tracked using  selectROI

k=2
for i in range(k):
    cv2.imshow('frame',frame)
    bbi=cv2.selectROI('frame',frame)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    tracker = cv2.TrackerKCF_create()
    trackers.add(tracker,frame,bbi)



while True:
    ret,frame=vs.read()
    out.write(frame)
    if not ret:
        break
    frame=imutils.resize(frame,width=1200)
    faces_list = []
    preds = []
    (success,boxes)=trackers.update(frame)
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        x=int(newbox[0])
        y=int(newbox[1])
        w=int(newbox[2])
        h=int(newbox[3])

        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)

        if len(faces_list) > 0:
            for face in faces_list:
                #predicting the label using trained model

                preds = model.predict(face)

                for pred in preds:
                    (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    cv2.imshow('frame',frame)
    key=cv2.waitKey(5) & 0xFF
    if key==ord('q'):
     break
vs.release()
out.release()
cv2.destroyAllWindows()
