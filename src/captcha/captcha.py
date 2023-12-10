import cv2
import numpy as np
import pytesseract
from PIL import Image

from .neural_net.neural_net import predict, get_model

def ocr(image):
    text = pytesseract.image_to_string(image)
    
    cleaned_text = ""
    for char in text:
        if char.isalnum():
            cleaned_text += char
    
    print(f"OCR-Text: \"{cleaned_text}\"")
    return cleaned_text

def cleanup_image(image):
    
    # Konvertierung in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Anwenden eines Schwellenwerts
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    masked_image = thresh

    for _ in range(5):
        kernel = np.ones((5, 5), np.float32) / 25
        img_filtered = cv2.filter2D(masked_image, -1, kernel)
        # Überprüfen, ob die Summe der weißen Pixel > 40% in jedem 5x5 Bereich ist
        masked_image[img_filtered < (255 * 0.4)] = 0
        
    kernel = np.ones((2, 2), np.float32) / 4
    img_filtered = cv2.filter2D(masked_image, -1, kernel)

    # Überprüfen, ob weniger als 3 weiße Pixel in jedem 2x2 Bereich sind, wenn ja, dann schwarz färben
    masked_image[img_filtered < (255 * 0.76)] = 0
    
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    # Erstellen einer Maske, die nur die 6 größten Konturen enthält
    contour_mask = np.zeros_like(masked_image)
    cv2.drawContours(contour_mask, contours[:6], -1, 255, -1)
    
    # Anwenden der Maske auf das Bild
    masked_image = cv2.bitwise_and(masked_image, contour_mask)
    
    return masked_image

def extract_characters(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    bounding_boxes.sort(key=lambda x: x[0])

    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the character using the bounding box
        char_image = image[y:y+h, x:x+w]

        # Determine padding for each side
        pad_x = max(0, (35 - w) // 2) + 1
        pad_y = max(0, (35 - h) // 2) + 1

        # Create a new 35x35 image and place the character in the center
        padded_image = cv2.copyMakeBorder(char_image, top=pad_y, bottom=pad_y, left=pad_x, right=pad_x, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Resize if the character image is larger than 35x35
        if padded_image.shape[0] > 35 or padded_image.shape[1] > 35:
            padded_image = cv2.resize(padded_image, (35, 35), interpolation=cv2.INTER_NEAREST)

        character_images.append(padded_image)

    return character_images

image = cv2.imread(R'C:\Projects\Hochschulsport-Bot\src\captcha\captcha.png')

cleaned_image = cleanup_image(image)
character_images = extract_characters(cleaned_image)

for i, character_image in enumerate(character_images):
    cv2.imwrite(f"character_{i}.png", character_image)

for i, character_image in enumerate(character_images):
    print(f"Character {i}:")
    print(f"Shape: {character_image.shape}")

model = get_model()
chars, _ = predict(model, np.array(character_images) / 255.0)

print("Correct characters: Y2AF88")

print(f"Predicted characters: {chars}")

# Speichern des bearbeiteten Bildes
cv2.imwrite('identified.png', cleaned_image)
ocr(Image.fromarray(cleaned_image))
