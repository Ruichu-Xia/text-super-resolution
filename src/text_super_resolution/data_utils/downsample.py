import os
import cv2

from data_utils.utils import HR_DIR, OUTPUT_BASE_DIR, DOWNSAMPLE_SIZES, FULL_PAGE_DOWNSAMPLE_OUTPUT_DIR, FULL_PAGE_DIR, FULL_PAGE_UNSEEN_DIR, FULL_PAGE_UNSEEN_DOWNSAMPLE_OUTPUT_DIR


def downsample_images(input_dir, output_base_dir, sizes):
    for size in sizes:
        os.makedirs(f"{output_base_dir}_{size}", exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img.shape[:2] == (512, 512):
                for size in sizes:
                    resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                    output_path = os.path.join(f"{output_base_dir}_{size}", filename)
                    cv2.imwrite(output_path, resized_img)

    print("Downsampling complete!")


def downsample_by_factors(input_dir, output_base_dir):
    factors = [2, 4, 8]

    for factor in factors:
        os.makedirs(f"{output_base_dir}_{factor}x", exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            original_height, original_width = img.shape[:2]

            for factor in factors:
                new_width = original_width // factor
                new_height = original_height // factor

                if new_width > 0 and new_height > 0:
                    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    output_path = os.path.join(f"{output_base_dir}_{factor}x", filename)
                    cv2.imwrite(output_path, resized_img)

    print("Downsampling by factors complete!")


if __name__ == "__main__":
    # downsample_images(HR_DIR, OUTPUT_BASE_DIR, DOWNSAMPLE_SIZES)
    # downsample_by_factors(FULL_PAGE_DIR, FULL_PAGE_DOWNSAMPLE_OUTPUT_DIR)
    downsample_by_factors(FULL_PAGE_UNSEEN_DIR, FULL_PAGE_UNSEEN_DOWNSAMPLE_OUTPUT_DIR)