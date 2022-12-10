import os
import cv2
import os
from maskdetect_config import *

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
