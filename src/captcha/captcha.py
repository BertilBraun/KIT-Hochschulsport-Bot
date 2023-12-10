import cv2
import numpy as np

from .neural_net.neural_net import pad_image_to_size, predict, get_model, plot_predictions

def cleanup_image(image: np.ndarray) -> np.ndarray:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def extract_characters(image: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    bounding_boxes.sort(key=lambda x: x[0])

    character_images = []
    for x, y, w, h in bounding_boxes:
        # Extract the character using the bounding box
        char_image = image[y:y+h, x:x+w]
        padded_image = pad_image_to_size(char_image)
        character_images.append(padded_image)

    return character_images


def solve_captcha(image: np.ndarray) -> str:
    cleaned_image = cleanup_image(image)
    character_images = extract_characters(cleaned_image)

    chars, confidences = predict(model, np.array(character_images) / 255.0)

    captcha_text = "".join(chars)
    return captcha_text

image = cv2.imread(R'C:\Projects\Hochschulsport-Bot\src\captcha\captcha.png')

cleaned_image = cleanup_image(image)
character_images = extract_characters(cleaned_image)

# Save the extracted characters by placing them next to each other and saving the image
character_image = np.hstack(character_images)
cv2.imwrite("characters.png", character_image)

X_test = np.array(character_images) / 255.0
Y_test = np.array(['Y', '2', 'A', 'F', '8', '8'])

model = get_model()
chars, confidences = predict(model, X_test)

print("Correct characters  : Y2AF88")

plot_predictions(X_test, Y_test, chars, confidences)

print(f"Predicted characters: {''.join(chars)}")
print(f"Confidences: {' '.join([str(round(confidence, 2)) for confidence in confidences])}")

# Speichern des bearbeiteten Bildes
cv2.imwrite('identified.png', cleaned_image)
