import os
import cv2

from data_utils.utils import HR_DIR, OUTPUT_BASE_DIR, DOWNSAMPLE_SIZES


def downsample_images(input_dir, output_base_dir, sizes):
    for size in sizes:
        os.makedirs(f"{output_base_dir}_{size}", exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Ensure it's an image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img.shape[:2] == (512, 512):
                for size in sizes:
                    resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                    output_path = os.path.join(f"{output_base_dir}_{size}", filename)
                    cv2.imwrite(output_path, resized_img)

    print("Downsampling complete!")


if __name__ == "__main__":
    downsample_images(HR_DIR, OUTPUT_BASE_DIR, DOWNSAMPLE_SIZES)