import fitz  # PyMuPDF
import time
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

NOT_READED = 0

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # Image extension (ex: 'png', 'jpeg', etc.)
            
            # Verify if is a supported image format
            if image_ext.lower() in ["png", "jpeg", "jpg"]:
                images.append((page_number + 1, image_bytes))
    return images

def detect_faces_in_image(image_bytes):
    global NOT_READED
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        NOT_READED = NOT_READED+1
        return False
    
    open_cv_image = np.array(image)
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return len(faces) > 0

def find_pages_with_faces(pdf_path):
    images = extract_images_from_pdf(pdf_path)
    pages_with_faces = []
    for page_number, image_bytes in images:
        if detect_faces_in_image(image_bytes):
            pages_with_faces.append(page_number)
    return pages_with_faces

if __name__ == "__main__":
    pdf_path = "data/some_pdf.pdf"
    
    start_time = time.time()
    pages_with_faces = find_pages_with_faces(pdf_path)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if pages_with_faces:
        print(f"Faces were detected on the following pages: {pages_with_faces}")
    else:
        print("No faces detected in the PDF.")

    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Error reading {NOT_READED} images")
