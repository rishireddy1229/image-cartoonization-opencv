from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import cv2

def evaluate(original, cartoon):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cartoon_gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(original_gray, cartoon_gray)
    mse_score = mean_squared_error(
        original_gray.flatten(),
        cartoon_gray.flatten()
    )

    return ssim_score, mse_score