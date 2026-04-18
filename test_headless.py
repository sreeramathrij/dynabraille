import cv2
import scanner

print('[TEST] Loading image headless...')
img_path = r'c:\Dynabraille\samples\WhatsApp Image 2026-04-18 at 6.11.20 PM.jpeg'
frame = cv2.imread(img_path)

if frame is None:
    print(f'[TEST] Failed to load {img_path}')
    exit(1)

print('[TEST] Testing Preprocess...')
processed = scanner.preprocess_image(frame)

print('[TEST] Testing PaddleOCR English...')
text, page, conf = scanner.extract_text(processed)
print(f'   -> Conf: {conf:.2f}')
print(f'   -> Page: {page}')

print('[TEST] Testing Tesseract Malayalam (Fallback)...')
text_mal, _, conf_mal = scanner.extract_malayalam_tesseract(processed)
print(f'   -> Conf: {conf_mal:.2f}')

print('[TEST] Testing Ollama Gemma connection...')
clean = scanner.clean_ocr(text)
print(f'   -> Clean Output: {clean[:100]}...')
print('[TEST] Successfully validated all endpoints.')
