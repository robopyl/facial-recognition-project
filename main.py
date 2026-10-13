import cv2
import numpy as np

def load_and_preprocess_image(image_path):
    """
    Loads an image from the specified path, converts it to grayscale,
    and normalizes pixel values for analytical processing.
    """
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image could not be loaded. Please check the file path.")
        return None
    
    # Convert to grayscale (reducing dimensionality for facial analysis)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values (scaling from 0-255 to 0.0-1.0)
    normalized_image = gray_image / 255.0
    
    print("Image successfully converted to grayscale and normalized.")
    return gray_image, normalized_image

def detect_faces(gray_image):
    """
    Applies OpenCV's Haar Cascade classifier to detect facial regions 
    within the preprocessed grayscale image.
    """
    # Load the pre-trained Haar Cascade classifier for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Total faces detected: {len(faces)}")
    return faces

if __name__ == "__main__":
    # Example workflow simulation
    # Replace 'test_face.jpg' with a real image path in your directory if available
    sample_image_path = 'test_face.jpg'
    
    print("--- Facial Recognition & Image Preprocessing Pipeline ---")
    
    # Note: If 'test_face.jpg' is missing, this serves as a structural template demonstrating 
    # the data cleaning and feature extraction process for your portfolio.
    processed_data = load_and_preprocess_image(sample_image_path)
    
    if processed_data is not None:
        gray, normalized = processed_data
        detected_faces = detect_faces(gray)
        print("Preprocessing and analysis pipeline executed successfully.")