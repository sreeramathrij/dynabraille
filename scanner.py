import os
import cv2
from paddleocr import PaddleOCR
import numpy as np

# 1. Initialize PaddleOCR globally so it only loads into RAM once.
# use_gpu=False ensures compatibility when you port this to the Raspberry Pi 5.
print("Loading OCR Engine...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
print("Engine Loaded. Starting camera...")

def extract_text(image_frame):
    """
    Passes the captured frame to PaddleOCR and formats the output into a clean string.
    Identifies the page number if it appears isolated at the top or bottom of the page.
    """
    # PaddleOCR accepts numpy arrays (OpenCV frames) directly!
    result = ocr_engine.ocr(image_frame, cls=True)
    
    extracted_text = ""
    page_number = None
    
    if result and result[0]: 
        lines = result[0]
        
        # Sort the bounding boxes vertically (top-to-bottom) based on the Y-coordinate
        lines.sort(key=lambda x: x[0][0][1])
        
        for i, line in enumerate(lines):
            text_snippet = line[1][0].strip()
            
            # --- Page Number Heuristic ---
            # 1. It consists entirely of digits (or "Page X")
            # 2. It is geometrically at the very top or very bottom of the page
            is_mostly_digits = text_snippet.isdigit() or text_snippet.lower().replace("page", "").strip().isdigit()
            is_edge_position = (i <= 1) or (i >= len(lines) - 2)
            
            if is_mostly_digits and is_edge_position and page_number is None:
                # Extract just the numerical part
                page_number = ''.join(filter(str.isdigit, text_snippet))
                # Skip adding the standalone page number to the main body text
                continue
                
            extracted_text += text_snippet + " "
            
    return extracted_text.strip(), page_number

def main():
    samples_dir = "samples"
    
    # 2. Check if samples directory exists
    if not os.path.exists(samples_dir):
        print(f"Error: The directory '{samples_dir}' does not exist.")
        print(f"Please create it and place some test images inside.")
        return

    # Get all valid image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Error: No image files found in '{samples_dir}'.")
        return

    print(f"\n--- FOUND {len(image_files)} IMAGES IN '{samples_dir}' ---")
    print("Press any key to process the next image, or 'q' to QUIT.\n")

    for img_file in image_files:
        img_path = os.path.join(samples_dir, img_file)
        print(f"\n[!] Loading {img_file}...")
        
        # Read the image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Display the image (scaled down for easier viewing on laptop)
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        if w > 1080:
            scale = 1080 / w
            display_frame = cv2.resize(display_frame, (1080, int(h * scale)))
        
        cv2.imshow("Book Scanner Preview", display_frame)
        cv2.waitKey(1) # Brief pause to allow window to render

        print("[!] Running OCR...")
        
        # --- OPTIONAL OPTIMIZATION FOR PI 5 ---
        # If the image is very large, resize it down to speed up the Pi's processing time
        # height, width = frame.shape[:2]
        # new_width = 960
        # new_height = int((new_width / width) * height)
        # frame_resized = cv2.resize(frame, (new_width, new_height))
        # text = extract_text(frame_resized)
        
        # Run extraction on the current frame
        text, page_number = extract_text(frame)
        
        print("-" * 30)
        if page_number:
            print(f">>> PAGE DETECTED: {page_number}")
            print("-" * 30)
            
        if text:
            print("EXTRACTED TEXT:\n", text)
            # ---> HERE is where you will send 'text' to your Gemma formatter
            # ---> or directly to your Arduino UART serial connection.
        else:
            print("No main body text detected on this page.")
        print("-" * 30)

        # Wait for a key press to continue or quit
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Exiting scanner...")
            break

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
