import cv2

def apply_bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)