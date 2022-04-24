import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import cv2 as cv2

from planeshift.planeShift import PlaneShift

def start_planeshift():
    plane_shift = PlaneShift(debug=True, device="/dev/video2")
    plane_shift.load_camera_calibration("calibration/calibration.json")
    plane_shift.run()

if __name__ == "__main__":
    start_planeshift()