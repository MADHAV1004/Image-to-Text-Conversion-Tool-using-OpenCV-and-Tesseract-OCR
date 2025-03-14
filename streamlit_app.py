# streamlit_app.py
import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# Set the path to Tesseract executable (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the title of the app
st.title("📄 Text Extraction from Images")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert PIL image to NumPy array

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.subheader("Preprocessing")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the preprocessed image
    st.image(binary, caption="Preprocessed Image", use_column_width=True, clamp=True)

    # Extract text using Tesseract
    st.subheader("Extracted Text")
    custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) and Page Segmentation Mode (PSM)

    # Add a progress bar for text extraction
    with st.spinner("Extracting text..."):
        text = pytesseract.image_to_string(binary, config=custom_config)

    # Display extracted text
    st.text_area("Extracted Text", text, height=200)

    # Save the extracted text to a file
    if st.button("Save Extracted Text"):
        with open("extracted_text.txt", "w") as file:
            file.write(text)
        st.success("Text saved to extracted_text.txt")

    # Download extracted text
    st.download_button(
        label="Download Extracted Text",
        data=text,
        file_name="extracted_text.txt",
        mime="text/plain"
    )