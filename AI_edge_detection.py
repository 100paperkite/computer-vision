import cv2
import numpy as np
import AI_function as func
import math


# 2-1 Apply the Gaussian filtering to the input image


# 2-2 Implement a function that returns the image gradient
mag,dir = func.compute_image_gradient("lenna.png")
func.non_maximum_suppression_dir(mag,dir)
# func.compute_image_gradient("shapes.png")
# cv2.waitKey()

# 2-3 Implement a function that performs Non-maximum Suppression (NMS)

