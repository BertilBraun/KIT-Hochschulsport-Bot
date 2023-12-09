import cv2
import numpy as np
import pytesseract
from PIL import Image

# Laden des Bildes (ersetzen Sie 'path_to_image' mit dem Pfad zu Ihrem Bild)
image = cv2.imread('captcha.png')

# Konvertierung in Graustufen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Anwenden eines Schwellenwerts
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

masked_image = thresh

for i in range(5):
    # Erstellen eines 5x5 Kernels
    kernel = np.ones((5, 5), np.float32) / 25

    # Anwenden des Kernels auf das Bild
    img_filtered = cv2.filter2D(masked_image, -1, kernel)

    # Überprüfen, ob die Summe der weißen Pixel > 40% in jedem 5x5 Bereich ist
    masked_image[img_filtered < (255 * 0.4)] = 0


# Umwandlung des bearbeiteten Bildes zurück in ein PIL-Bild
final_image = Image.fromarray(masked_image)
captcha_image = final_image

# Verwenden von Tesseract, um Text aus dem Bild zu extrahieren
captcha_text = pytesseract.image_to_string(captcha_image)

print(f"CAPTCHA-Text: {captcha_text}")

# Speichern des bearbeiteten Bildes
cv2.imwrite('identified.png', masked_image)
