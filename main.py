import os
import pytesseract
import cv2
from scanner import scan_document

def get_input_images():
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for file in os.listdir('input'):
        file_path = os.path.join('input', file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_path)
            if ext.lower() in image_extensions:
                image_files.append(file_path)
    return image_files

def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return ""

def process_images():
    image_files = get_input_images()
    if not image_files:
        print("No images found in the input directory.")
        return

    for image_path in image_files:
        try:
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            print(f"Processing {filename}...")

            scanned_image = scan_document(image_path)

            scanned_output_path = os.path.join('output', f"{name}_scanned.png")
            cv2.imwrite(scanned_output_path, scanned_image)
            print(f"Saved scanned image to {scanned_output_path}")

            text = perform_ocr(scanned_image)

            text_output_path = os.path.join('output', f"{name}_text.txt")
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved OCR text to {text_output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("All images processed.")

def main():
    try:        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in PATH.")
        return
    process_images()

if __name__ == "__main__":
    main()
