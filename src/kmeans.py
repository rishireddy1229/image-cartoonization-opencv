import cv2
import numpy as np

def kmeans_quantization(img, k=8):
    data = np.float32(img).reshape((-1, 3))

    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    return result.reshape(img.shape)