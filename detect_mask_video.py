import numpy as np
import cv2
import os
import sys

#import config file
import videoconfig as config

#get rid of annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(config.tensorflowloglevel)
sys.tracebacklimit = config.pythontraceback

print('Loading Tensorflow...')

from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

from imutils.video import VideoStream

print('Loading face detector...')
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(
    ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print('Loading mask detector...')
maskNet = load_model(config.modelpath)

print('Starting video...')
cam = VideoStream(src=config.videosource).start()


def takepic(savedir):
    #count number of files in directory and add 1 to get next file name and save image as png file
    fileindir = len([
        name for name in os.listdir(savedir)
        if os.path.isfile(os.path.join(savedir, name))
    ])
    filename = savedir + config.fileprefix + str(fileindir) + config.filetype
    cv2.imwrite(filename, frame)


def predictmask(frame, faceNet, maskNet):
    #make smol frame
    smallframe = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    #get frame dimension and make a blob
    (h, w) = smallframe.shape[:2]
    blob = cv2.dnn.blobFromImage(smallframe, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

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

                #set bounding box to original frame size
                startx = int(startX * 5)
                starty = int(startY * 5)
                endx = int(endX * 5)
                endy = int(endY * 5)

                #append bounding box to locs or something idk
                locs.append((startx, starty, endx, endy))

    #only make predictions if at least one face was detected
    if len(faces) > 0:
        #make batch predictions on all faces at once in the for loop above
        faces = np.array(faces, dtype="float32")
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

        #determine mask status from confidence
        if (mask > config.maskconfidence):
            label = config.masktext
            color = (0, 255, 0)  #green label
            # takepic(maskdir)
            if config.takepic_enable:
                takepic(config.maskdir)

        elif (mask < config.nomaskconfidence):
            label = config.nomasktext
            color = (0, 0, 255)  #red label
            if config.takepic_enable:
                takepic(config.nomaskdir)

        else:
            label = config.unknowntext
            color = (255, 255, 0)
            if config.takepic_enable:
                takepic(config.unknowndir)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #display label and bounding box rectangle on output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 3)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    #show output frame
    cv2.imshow(config.windowtitle, frame)

    key = cv2.waitKey(1) & 0xFF
    #if 'q' key is pressed, break from loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cam.stop()
cam.release()
