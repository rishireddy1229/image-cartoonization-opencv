import cv2

def meanshift_quantization(img):
    return cv2.pyrMeanShiftFiltering(img, 21, 51)