import os
import cv2
import os
import base64
import requests
from config import *

def saveFace(faceframe, savedir):
    fileindir = len([
        name for name in os.listdir(savedir)
        if os.path.isfile(os.path.join(savedir, name))
    ])
    filename = savedir + fileprefix + str(fileindir) + filetype
    cv2.imwrite(filename, faceframe)

def saveFull(frame, savedir):
    fileindir = len([
        name for name in os.listdir(savedir)
        if os.path.isfile(os.path.join(savedir, name))
    ])
    filename = savedir + fileprefix + str(fileindir) + filetype
    cv2.imwrite(filename, frame)

def sendImage(frame):
    _, image_jpg = cv2.imencode('.jpg', frame)
    image_bytes = image_jpg.tostring()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    response = requests.post(serverURL, json={'image': image_base64})
    res = response.json()
    filename = res['filename']
    return filename
