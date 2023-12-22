import os
from MultispectralProcessing.MultispectralProcessing import Multispectral

"""This section is sample code for multispectral processing
"""

multispectralProcessor = Multispectral("C:\\Users\\saif_\\output")

nir_img_path = "C:\\Users\\saif_\\Main\\Source\\PrecisionAgriculture\\Data\\Input\\Near_Infrared_Channel\\Train_Images\\Image_001.jpg"
red_img_path = "C:\\Users\\saif_\\Main\\Source\\PrecisionAgriculture\\Data\\Input\\Red_Channel\\Train_Images\\Image_001.jpg"

image_list = os.listdir("path")

for image in image_list:
    ndvi = multispectralProcessor.ndvi(nir_img=nir_img_path, red_img=red_img_path)
    

""" Image Processing using OpenCv
"""
from ImageProcessingApproach.OpenCvMultispectralClassifier import OpenCvClassifier

output_dir = "C:\\Users\\saif_\\output"

# soil_max=0.1, unhealthy_max=0.45 are the values used in the report
multispectralOpenCvClassifier = OpenCvClassifier(0.1, 0.6, "ndvi", output_dir)

ndvi_file_path = output_dir + "Image_001.txt"

result = multispectralOpenCvClassifier.apply_filter(ndvi_file_path)
percent_healthy = result.healthy_percent
unhealthy_percent = result.unhealthy_percent
soil_percent = result.soil_percent

""" Yolov8 training
"""

from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt') # -> models can be changed based on hardware, ex: yolov8s.pt, yolov8m.pt
 
# to use cuda
# model.to('cuda')

# Training.
results = model.train(
   data='pothole_v8.yaml',
   imgsz=1280,
   epochs=50,
   batch=8,
   name='yolov8n_v8_50e'
)

# https://docs.ultralytics.com/modes/predict/#arguments
# model.predict(source, save=True, imgsz=320, conf=0.5,device='xyz')