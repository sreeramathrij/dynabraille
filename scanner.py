import os
import cv2
from paddleocr import PaddleOCR
import numpy as np
import pytesseract
import platform
import ollama
import sys

# Force Windows console to support Unicode printing (so Malayalam characters don't crash the script)
sys.stdout.reconfigure(encoding='utf-8')

# Point Tesseract to the local folder where we downloaded the language models
# Windows requires absolute paths for tessdata-dir to prevent TesseractError crashes!
tess_dir = os.path.abspath("tessdata")
TESSDATA_CONFIG = f'--tessdata-dir {tess_dir}'

print("Loading OCR Engines...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
print("Engines Loaded. Starting application loop...")

def preprocess_image(frame):
    """
    Applies standard OpenCV deskew and thresholding to clean noise from the input frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur helps smooth edges before applying Otsu's thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Finding text skew angle using bounding contours
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return gray
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Fix OpenCV angle output conventions
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate image to straighten textual lines
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def extract_text(image_frame):
    """
    English PaddleOCR extraction that returns parsed text, page number, and the true confidence average.
    """
    result = ocr_engine.ocr(image_frame, cls=True)
    
    extracted_text = ""
    page_number = None
    confidences = []
    
    if result and result[0]: 
        lines = result[0]
        # Sort vertically (top-to-bottom) based on Y-coordinate bounding boxes
        lines.sort(key=lambda x: x[0][0][1])
        
        for i, line in enumerate(lines):
            text_snippet = line[1][0].strip()
            confidence_score = float(line[1][1])
            
            is_mostly_digits = text_snippet.isdigit() or text_snippet.lower().replace("page", "").strip().isdigit()
            is_edge_position = (i <= 1) or (i >= len(lines) - 2)
            
            if is_mostly_digits and is_edge_position and page_number is None:
                page_number = ''.join(filter(str.isdigit, text_snippet))
                continue
                
            extracted_text += text_snippet + " "
            confidences.append(confidence_score)
            
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return extracted_text.strip(), page_number, avg_conf

def extract_malayalam_tesseract(image_frame):
    """
    Fallback Malayalam Tesseract engine. Extracts text and computes internal dictionary confidences.
    """
    # Ask pytesseract directly for tabular Dict data so we can calculate the internal confidence
    try:
        data = pytesseract.image_to_data(image_frame, lang='mal', config=TESSDATA_CONFIG, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"    [!] Tesseract is missing or missing language pack ({type(e).__name__}). Cannot fallback to Malayalam.")
        return "", None, 0.0
        
    
    text_parts = []
    conf_scores = []
    
    for i in range(len(data['text'])):
        conf = float(data['conf'][i])
        text = data['text'][i].strip()
        
        # Tesseract outputs -1 confidence for blank prediction nodes or spaces
        if conf > 0 and len(text) > 0: 
            text_parts.append(text)
            conf_scores.append(conf / 100.0) # Map 0-100 to 0.0-1.0
            
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
    extracted_text = " ".join(text_parts)
    return extracted_text, None, avg_conf

def clean_ocr(raw_text):
    """
    Pings the local Ollama daemon to correct OCR noise via an LLM.
    """
    if not raw_text.strip():
        return ""
        
    print(f"    [>] Sending {len(raw_text.split())} words to Gemma for repair...")
    
    prompt = f"You are a strict proofreader. Fix typos, OCR hallucinations, and spacing errors in the text below. DO NOT summarize it. DO NOT omit any paragraphs. IMPORTANT: IF THE TEXT IS IN MALAYALAM, KEEP IT IN MALAYALAM. DO NOT TRANSLATE IT TO ENGLISH. Return the exact same length of text with only the spelling fixed:\n\n{raw_text}"
    try:
        response = ollama.generate(model='gemma:2b', prompt=prompt)
        cleaned = response.get('response', raw_text).strip()
        print(f"    [>] Gemma returned {len(cleaned.split())} words.")
        return cleaned
    except Exception as e:
        print(f"    [!] Gemma pipeline hallucinated or crashed ({e}). Returning pure OCR.")
        return raw_text

def main():
    samples_dir = "samples"
    
    if not os.path.exists(samples_dir):
        print(f"Error: The directory '{samples_dir}' does not exist.")
        return

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
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        if w > 1080:
            scale = 1080 / w
            display_frame = cv2.resize(display_frame, (1080, int(h * scale)))
        
        cv2.imshow("Book Scanner Preview", display_frame)
        cv2.waitKey(1)

        print("[!] Pre-processing image (Deskew + Threshold)...")
        processed_frame = preprocess_image(frame)

        print("[!] Running Primary OCR (PaddleOCR English)...")
        text_paddle, page_number_paddle, conf_paddle = extract_text(processed_frame)
        print(f"    Confidence: {conf_paddle:.2f}")

        final_text = ""
        page_number = page_number_paddle

        # --- DYNAMIC OCR SWITCHING PIPELINE ---
        if conf_paddle > 0.80:
            print("    [+] High confidence! Accepting PaddleOCR English result.")
            final_text = text_paddle
        elif conf_paddle < 0.60:
            print("    [-] Low confidence. Hard reroute to Tesseract (Malayalam)...")
            text_tess, _, conf_tess = extract_malayalam_tesseract(processed_frame)
            print(f"    Tesseract Confidence: {conf_tess:.2f}")
            final_text = text_tess
        else:
            print("    [?] Medium confidence. Running Tesseract to compare engines...")
            text_tess, _, conf_tess = extract_malayalam_tesseract(processed_frame)
            print(f"    PaddleOCR Conf: {conf_paddle:.2f} | Tesseract Conf: {conf_tess:.2f}")
            
            if conf_tess > conf_paddle:
                print("    [+] Tesseract Malayalam scored higher. Accepting override.")
                final_text = text_tess
            else:
                print("    [+] PaddleOCR English scored higher. Sticking with primary extraction.")
                final_text = text_paddle

        print("[!] Post-Processing (LLM Cleanup)...")
        clean_text = clean_ocr(final_text)

        print("-" * 50)
        if page_number:
            print(f">>> PAGE DETECTED: {page_number}")
            print("-" * 50)
            
        if clean_text:
            print("CLEAN TEXT:\n", clean_text)
            # ---> Send to Arduino UART connection
        else:
            print("No main body text detected on this page.")
        print("-" * 50)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Exiting scanner...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
