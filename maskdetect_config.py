#Tensorflow log level
tensorflowloglevel = 3

#Python Traceback
pythontraceback = 0

#Select Camera
videosource = 2

#Mask Detector Model Path
modelpath = 'mask_detector.model'

#Choose whether to save images
takepic_enable = True

#Image save directory
maskdir = './maskpics/'
nomaskdir = './nomaskpics/'
unknowndir = './unknownpics/'

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
