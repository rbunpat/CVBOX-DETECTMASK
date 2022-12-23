#Tensorflow log level
tensorflowloglevel = 3

#Python Traceback
pythontraceback = 1

#Select Camera
videosource = 1

#Image Server Address, download one from https://github.com/rbunpat/CVBOX-FILESERVER
serverURL = 'YOURSERVERURL'

#Serial Config
serialPort = 'YOURCOMPORT'
serialBaud = YOURBAUDRATE

#Mask Detector Model Path
# modelpath = 'mask_detector.model'
modelpath = 'datasetv2.model'

#Choose whether to save images
takepic_enable = False

#Image save directory
maskdir = './maskpics/'
nomaskdir = './nomaskpics/'
unknowndir = './unknownpics/'
maskfacedir = './maskfacecroppedpics/'
nomaskfacedir = './nomaskfacecroppedpics/'
unknownfacedir = './unknownfacecroppedpics/'

#Image file prefix and file type
fileprefix = 'pic_'
filetype = '.png'

#Set minimum confidence for mask detection
maskconfidence = 0.5
nomaskconfidence = 0.5

#Text displayed in mask detection window
masktext = 'Mask : YES'
nomasktext = 'Mask : NO'
unknowntext = 'Mask : UNKNOWN'

#Mask Detection Window Title
windowtitle = 'Mask Detector'
