import cv2
import argparse
import os
from src.cartoonize import cartoonize_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--method", default="kmeans", choices=["kmeans", "meanshift"])
    args = parser.parse_args()

    img = cv2.imread(args.input)

    if img is None:
        print("Error: Image not found")
        return

    output = cartoonize_image(img, method=args.method)

    os.makedirs("output_images", exist_ok=True)

    output_path = f"output_images/cartoon_{args.method}.jpg"
    cv2.imwrite(output_path, output)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()