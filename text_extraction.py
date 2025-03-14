# text_extraction.py
import cv2
import pytesseract
import numpy as np

# Set the path to Tesseract executable (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """Preprocess the image for text extraction."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def extract_text(image_path):
    """Extract text from an image using Tesseract."""
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Check the file path and integrity.")
        return

    # Preprocess the image
    binary = preprocess_image(image)

    # Display the preprocessed image
    cv2.imshow("Preprocessed Image", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract text using Tesseract
    custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) and Page Segmentation Mode (PSM)
    text = pytesseract.image_to_string(binary, config=custom_config)

    return text

def main():
    # Path to the image
    image_path = "Bill.jpg"  # Replace with your image path

    # Extract text from the image
    extracted_text = extract_text(image_path)

    if extracted_text:
        # Print the extracted text
        print("Extracted Text:\n", extracted_text)

        # Save the extracted text to a file
        with open("extracted_text.txt", "w") as file:
            file.write(extracted_text)
        print("Text saved to extracted_text.txt")

if __name__ == "__main__":
    main()