import cv2
import time
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def detect_faces_in_image(image):
    open_cv_image = np.array(image)
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return len(faces) > 0

def find_pages_with_faces(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    pages_with_faces = []
    for i, image in enumerate(images):
        if detect_faces_in_image(image):
            pages_with_faces.append(i + 1)
    return pages_with_faces

if __name__ == "__main__":
    pdf_path = "data/0423656-93.1999.8.26.0053.pdf"
    
    start_time = time.time()
    pages_with_faces = find_pages_with_faces(pdf_path)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if pages_with_faces:
        print(f"Faces were detected on the following pages: {pages_with_faces}")
    else:
        print("No faces detected in the PDF.")

    print(f"Execution time: {execution_time:.2f} seconds")