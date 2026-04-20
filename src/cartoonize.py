import cv2
from src.preprocess import apply_bilateral_filter
from src.edges import detect_edges
from src.kmeans import kmeans_quantization
from src.meanshift import meanshift_quantization

def cartoonize_image(img, method="kmeans"):
    smooth = apply_bilateral_filter(img)
    edges = detect_edges(img)

    if method == "kmeans":
        color = kmeans_quantization(smooth)
    else:
        color = meanshift_quantization(smooth)

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)

    return cartoon