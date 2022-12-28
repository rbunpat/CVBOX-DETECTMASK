import sys
import os
import cv2
import numpy as np
import serial

from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QMainWindow, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore

from imutils.video import VideoStream

from imgTools import *
from config import *

import threading

ser = serial.Serial(serialPort, serialBaud)

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

                #append bounding box to locs or something idk
                locs.append((startX, startY, endX, endY))

    #only make predictions if at least one face was detected
    if len(faces) > 0:
        #make batch predictions on all faces at once in the for loop above
        faces = np.array(faces, dtype="float")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

class MaskDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mask Detector")
        self.setStyleSheet("background-color: #121212;")

        # Create a label to display the video stream
        self.label = QLabel(self)
        self.ardudata = QLabel()
        self.ardudata.setStyleSheet("font-family: Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif; font: 80px; color: #00bd7e;")
        self.ardudata.setAlignment(QtCore.Qt.AlignCenter)

        # Create a button to start and stop the detection
        self.button = QPushButton('Start', self)
        self.button.setStyleSheet("background-color: #42cc8c;border-style: outset;border-width: 2px;border-radius: 10px; font: bold 26px;min-width: 10em;padding: 6px;")
        self.button.clicked.connect(self.toggle_detection)

        # Create a vertical layout to hold the label and button
        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0)
        self.layout.addWidget(self.ardudata, 0, 1)
        self.layout.addWidget(self.button, 1, 0)

        # Create a widget to hold the layout, and set it as the central widget of the main window
        self.widget = QWidget(self)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        # Initialize the video stream and mask detection variables
        self.cam = None
        self.faceNet = None
        self.maskNet = None
        self.running = False
        self.detection_thread = None
        self.serial_thread = None

        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.maskNet = load_model(modelpath)
        self.cam = VideoStream('src=videosource').start()

    def toggle_detection(self):
        if self.running:
            # Stop the detection and update the button text
            self.running = False
            self.button.setText('Start')
            self.serial_thread.join()
            self.serial_thread = None
            self.button.setStyleSheet("background-color: #42cc8c;border-style: outset;border-width: 2px;border-radius: 10px; font: bold 26px;min-width: 10em;padding: 6px;")
        else:
            # Start the detection and update the button text
            self.running = True
            self.button.setText('Stop')
            self.button.setStyleSheet("background-color: #FF4742;border-style: outset;border-width: 2px;border-radius: 10px; font: bold 26px;min-width: 10em;padding: 6px;")

            # Run the detection loop in a separate thread
            self.serial_thread = threading.Thread(target=self.waitforserial)
            self.serial_thread.start()

    def detection_loop(self):
        # Read a frame from the video stream
        frame = self.cam.read()

        # Detect faces and masks
        (locs, preds) = predictmask(frame, self.faceNet, self.maskNet)

        # Draw rectangles and labels on the frame
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            self.masklabel = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)
            self.masklabel = "{}: {:.2f}%".format(self.masklabel, max(mask, withoutMask) * 100)

            cv2.putText(frame, self.masklabel, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Convert the frame to a QImage and set it as the image of the label
        image = img_to_qimage(frame)
        self.label.setPixmap(QPixmap.fromImage(image))

    def waitforserial(self):
        while self.running:
            data = ser.readline()
            if data:
                self.detection_loop()
                tempdata = str(data).lstrip("b").strip("'").rstrip("\\r\\n")
                tempdata = "Temperature \n" + tempdata + "°C"
                self.ardudata.setText(tempdata)

def img_to_qimage(img):
    # Convert the image to a NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, color = img.shape
    bytesPerLine = 3 * width

    # Create a QImage and copy the image data into it
    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

    return qImg

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MaskDetectorGUI()
    gui.show()
    sys.exit(app.exec_())
