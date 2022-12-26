#Tensorflow log level
tensorflowloglevel = 3

#Python Traceback
pythontraceback = 1

#Select Camera
videosource = 0

#Image Server Address (see https://github.com/rbunpat/CVBOX-FILESERVER)
serverURL = 'http://localhost/api/upload-image'

#Serial Config
serialPort = 'YOURSERIALPORT'
serialBaud = YOURBAUDRATE

#Mask Detector Model Path
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
