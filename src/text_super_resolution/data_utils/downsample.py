import os
import cv2
import fitz
import numpy as np

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


def downsample_by_factors(input_dir, output_base_dir, factors = [2, 4, 8]):
    

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

def pdf_to_image(pdf_directory, page_num, output_directory):    
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        pdf = fitz.open(pdf_directory)
        
        if page_num < 0 or page_num >= len(pdf):
            raise ValueError(f"Invalid page number. PDF has {len(pdf)} pages (0-{len(pdf)-1})")
        
        page = pdf.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        pdf_name = os.path.splitext(os.path.basename(pdf_directory))[0]
        output_path = os.path.join(output_directory, f"page_{page_num}.png")
        
        cv2.imwrite(output_path, img_data)
        pdf.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error converting PDF page to image: {e}")
        return None    


if __name__ == "__main__":
    # downsample_images(HR_DIR, OUTPUT_BASE_DIR, DOWNSAMPLE_SIZES)
    # downsample_by_factors(FULL_PAGE_DIR, FULL_PAGE_DOWNSAMPLE_OUTPUT_DIR)
    downsample_by_factors(FULL_PAGE_UNSEEN_DIR, FULL_PAGE_UNSEEN_DOWNSAMPLE_OUTPUT_DIR)