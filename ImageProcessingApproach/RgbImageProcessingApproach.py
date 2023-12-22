from imp import load_dynamic
import numpy as np
import cv2 
import os


class RgbImageProcessingClassifier():
    def __init__(self):
        self.unhealthy_upper = np.array([24, 252, 241], dtype="uint8")  
        self.unhealthy_lower = np.array([0, 76, 0], dtype="uint8")
        self.healthy_min = np.array([27,64,0], dtype="uint8")
        self.healthy_max = np.array([179, 255, 255], dtype="uint8")
        self.root_dir = "C:\\Users\\saif_\\Main\\Source\\PrecisionAgriculture\\Data\\Input\\RGB_Images\\Train_Images"

    def load_image(self, file_name):
        file_path = os.path.join(self.root_dir, file_name)
        matrix = cv2.imread(file_path)
        return matrix
    
    def extract_unhealthy_from_subimage(self, file_path, x_min, x_max, y_min, y_max):
        img = self.load_image(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[x_min: x_max, y_min: y_max]
        mask = cv2.inRange(img, self.unhealthy_lower, self.unhealthy_upper)
        detected = cv2.bitwise_and(img,img, mask=mask)
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        area = 0
        for c in cnts:
            area += cv2.contourArea(c)

        return area
    
    def extract_healthy_from_subimage(self, file_path, x_min, x_max, y_min, y_max):
        img = self.load_image(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[x_min: x_max, y_min: y_max]
        mask = cv2.inRange(img, self.healthy_min, self.healthy_max)
        detected = cv2.bitwise_and(img,img, mask=mask)
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        area = 0
        for c in cnts:
            area += cv2.contourArea(c)

        return area
    
    