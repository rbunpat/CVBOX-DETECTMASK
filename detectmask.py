import numpy as np
import cv2
import os
import sys

from savePicture import saveFace, saveFull
from maskdetect_config import *

# get rid of annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tensorflowloglevel)
sys.tracebacklimit = pythontraceback

print('Loading Tensorflow...')

from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

from imutils.video import VideoStream

print('Loading Face detector...')
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(
    ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print('Loading Mask detector...')
maskNet = load_model(modelpath)

print('Starting video...')
cam = VideoStream(src=videosource).start()


def predictmask(frame, faceNet, maskNet):
    #get frame dimension and make a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

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

            #extract region of interest, convert to RGB, resize to 224x224, and preprocess
            face = frame[startY:endY, startX:endX]

            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                #add face and bounding box to their respective lists
                faces.append(face)

                #idk why do this lmao, might remove later
                startx = int(startX)
                starty = int(startY)
                endx = int(endX)
                endy = int(endY)

                #append bounding box to locs or something idk
                locs.append((startx, starty, endx, endy))

    #only make predictions if at least one face was detected
    if len(faces) > 0:
        #make batch predictions on all faces at once in the for loop above
        faces = np.array(faces, dtype="float")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)


while True:
    #read frame from camera duh
    frame = cam.read()

    #detect face and mask
    (locs, preds) = predictmask(frame, faceNet, maskNet)
    #loop over detected faces and their locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        faceframe = frame[startY:endY, startX:endX]

        #determine mask status from confidence and set label and color
        if (mask > maskconfidence):
            label = masktext
            color = (0, 255, 0)
            saveFull(frame, maskdir)
            saveFace(faceframe, maskfacedir)

        elif (mask < nomaskconfidence):
            label = nomasktext
            color = (0, 0, 255)
            saveFull(frame, nomaskdir)
            saveFace(faceframe, nomaskfacedir)

        else:
            label = unknowntext
            color = (255, 255, 0)
            saveFull(frame, nomaskdir)
            saveFace(faceframe, unknownfacedir)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #display label and bounding box rectangle on output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 3)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # cv2.imshow('face', faceframe)

    #show output frame
    cv2.imshow(windowtitle, frame)

    key = cv2.waitKey(1) & 0xFF
    #if 'q' key is pressed, break from loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cam.stop()
cam.release()
