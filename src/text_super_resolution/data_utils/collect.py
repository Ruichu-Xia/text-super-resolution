import fitz 
from PIL import Image
import io
import random
import os
import easyocr
import numpy as np

from data_utils.utils import HR_DIR, TEXTBOOK_PATH, CROP_SIZE, NUM_CROPS_PER_PAGE, FULL_PAGE_DIR, NUM_FULL_PAGE, FULL_PAGE_UNSEEN_DIR, FULL_PAGE_UNSEEN_DOWNSAMPLE_OUTPUT_DIR, UNSEEN_TEXTBOOK_PATH

def collect_partial_pages(hr_dir, textbook_path, crop_size, num_crops_per_page): 
    os.makedirs(hr_dir, exist_ok=True)
    ocr_reader = easyocr.Reader(['en'])
    pdf_reader = fitz.open(textbook_path)

    img_idx = 0
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if img.size[0] >= crop_size and img.size[1] >= crop_size:
            for _ in range(num_crops_per_page):
                x_max = img.size[0] - crop_size
                y_max = img.size[1] - crop_size
                x = random.randint(0, x_max)
                y = random.randint(0, y_max)

                cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
                cropped_img = cropped_img.convert('L')

                cropped_img_np = np.array(cropped_img)
                detected_text = ocr_reader.readtext(cropped_img_np)
                if len(detected_text) > 10:  
                    cropped_img.save(f'{hr_dir}/Image_{img_idx}.png')
                    img_idx += 1

    pdf_reader.close()


def extract_random_pages(output_dir, textbook_path, num_pages):
    os.makedirs(output_dir, exist_ok=True)
    pdf_reader = fitz.open(textbook_path)
    
    total_pages = len(pdf_reader)
    selected_pages = random.sample(range(total_pages), min(num_pages, total_pages))
    
    for idx, page_num in enumerate(selected_pages):
        page = pdf_reader.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # High-resolution scaling
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = img.convert('L')
        img.save(f'{output_dir}/Page_{page_num}.png', format='PNG')
    
    pdf_reader.close()
    print(f"Extracted {len(selected_pages)} pages to {output_dir}")


if __name__ == "__main__":
    # collect_partial_pages(HR_DIR, TEXTBOOK_PATH, CROP_SIZE, NUM_CROPS_PER_PAGE)
    # extract_random_pages(FULL_PAGE_DIR, TEXTBOOK_PATH, NUM_FULL_PAGE)
    extract_random_pages(FULL_PAGE_UNSEEN_DIR, UNSEEN_TEXTBOOK_PATH, NUM_FULL_PAGE)