import numpy as np
import json
import cv2
import os.path
import os
import sys

#get rid of annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.tracebacklimit = 0

print('Loading Tensorflow...')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


DIR = "./nomaskpics"

print('Loading face detector...')
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print('Loading mask detector...')
maskNet = load_model("mask_detector.model")

print('Starting video...')
cam = VideoStream(src=0).start()

def takepic():
    fileindir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    filename = "./nomaskpics/pics_" + str(fileindir) + ".png"
    cv2.imwrite(filename, frame)

def predictmask(frame, faceNet, maskNet):
    smallframe = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    #get frame dimension and make a blob
    (h, w) = smallframe.shape[:2]
    blob = cv2.dnn.blobFromImage(smallframe, 1.0, (300, 300), (104.0, 177.0, 123.0))
    #pass blob through face detector and get detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    #init faces, locations, and predictions
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        #get confidence of detection
        confidence = detections[0, 0, i, 2]
        #filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #check to make sure that bounding box is within frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            #extract region of interest, convert to RGB, resize to 224x224, and preprocess
            face = smallframe[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                #add face and bounding box to their respective lists
                faces.append(face)
                startx = int(startX * 5)
                starty = int(startY * 5)
                endx = int(endX * 5)
                endy = int(endY * 5)
                locs.append((startx, starty, endx, endy))

    #only make predictions if at least one face was detected
    if len(faces) > 0:
        #make batch predictions on all faces at once in the for loop above
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

while True:
    frame = cam.read()
    #detect face and mask
    (locs, preds) = predictmask(frame, faceNet, maskNet)
    #loop over detected faces and their locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
    #determine mask status from confidence
        if (mask > 0.6):
            label = "Mask : YES"
            color = (0, 255, 0) #green label
        elif (mask < 0.4):
            label = "Mask : NO"
            color = (0, 0, 255) #red label
        else:
            label = "Mask : UNKNOWN"
            color = (255, 255, 0)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        #display label and bounding box rectangle on output frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 3)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    #show output frame
    cv2.imshow("Mask Detector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    #if 'q' key is pressed, break from loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cam.stop()
cam.release()
