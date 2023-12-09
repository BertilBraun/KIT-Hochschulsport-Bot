import sys
import time
import cv2
import numpy as np
import pytesseract
from PIL import Image


def find_and_fill_largest_area(thresh, x, y, w, h, label_mask):
    # Isolieren des Bereichs der Bounding Box
    roi = thresh[y:y+h, x:x+w]
    num_labels, labels = cv2.connectedComponents(roi)

    max_label, max_size = 0, 0
    for label in range(1, num_labels):
        size = np.sum(labels == label)
        if size > max_size:
            max_label, max_size = label, size

    # Maskierung der größten Fläche
    largest_area_mask = (labels == max_label).astype(np.uint8) * 255

    return largest_area_mask

    # Ermitteln der Koordinaten für den Startpunkt der Flut-Füllung
    points = np.column_stack(np.where(largest_area_mask == 255))
    if points.size > 0:
        flood_start_point = tuple(points[0])

        # Durchführen der Flut-Füllung auf der separaten Maske
        cv2.floodFill(label_mask, np.zeros((h, w), np.uint8), flood_start_point, 255, 0, 0, 8)


# Laden des Bildes (ersetzen Sie 'path_to_image' mit dem Pfad zu Ihrem Bild)
image = cv2.imread('captcha.png')

# Konvertierung in Graustufen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Anwenden eines Schwellenwerts
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Finden von Konturen
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Erstellen einer schwarzen Maske
mask = np.zeros_like(thresh)

# Durchlaufen der Konturen und Identifizieren von Buchstabengruppen
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 15 and h > 15:  # Beispielwerte, anpassen nach Bedarf
        # Füllen der Bounding Box in der Maske mit Weiß
        mask[y:y+h, x:x+w] = 255


label_mask = np.zeros_like(thresh)

# Verarbeitung jeder Kontur
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 15 and h > 15:  # Beispielwerte, anpassen nach Bedarf
        largest_area_mask = find_and_fill_largest_area(thresh, x, y, w, h, label_mask)
        
        # set label_mask to largest_area_mask only where largest_area_mask is white
        label_mask[y:y+h, x:x+w] = cv2.bitwise_or(label_mask[y:y+h, x:x+w], largest_area_mask)
        
                
        # TODO finde größte Fläche in der Bounding Box die zusammenhängend und weis ist
        
        
        # TODO flood fill largest area in label_mask with white
        pass

# Durchlaufen der Konturen und Identifizieren von Buchstabengruppen
#for contour in contours:
#    # Berechnen der Bounding Box für jede Kontur
#    x, y, w, h = cv2.boundingRect(contour)
#
#    # Filtern basierend auf der Größe der Bounding Box (angepasst an Ihre spezifischen Anforderungen)
#    if w > 15 and h > 15:  # Beispielwerte, anpassen nach Bedarf
#        # Zeichnen einer Bounding Box um jede identifizierte Buchstabengruppe
#        cv2.rectangle(thresh, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Anzeigen des bearbeiteten Bildes
# Anwenden der Maske auf das Originalbild
masked_image = cv2.bitwise_and(thresh, mask)
masked_image = cv2.bitwise_and(masked_image, label_mask)


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
#cv2.imshow('Identified Letter Groups', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sys.exit(0)


# Dekodieren des Bildes
captcha_image = Image.open("captcha.png")


# Dekodieren und Konvertieren des Bildes in ein Numpy-Array
captcha_image = np.array(captcha_image)

# Anwenden eines Schwellenwerts
_, thresh = cv2.threshold(captcha_image, 1, 255, cv2.THRESH_BINARY_INV)

# Entfernen kleiner Artefakte
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# processed_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Umwandlung des bearbeiteten Bildes zurück in ein PIL-Bild
final_image = Image.fromarray(thresh)
captcha_image = final_image

captcha_image.save("captcha_processed.png")

# Verwenden von Tesseract, um Text aus dem Bild zu extrahieren
captcha_text = pytesseract.image_to_string(captcha_image)

print(f"CAPTCHA-Text: {captcha_text}")

# Eingabe des CAPTCHA-Textes in das Eingabefeld
captcha_input_field = driver.find_element(By.ID, "BS_F_captcha")
captcha_input_field.send_keys(captcha_text.strip())


submit_button = driver.find_element(By.XPATH, "//input[@type='submit'][@value='kostenpflichtig buchen']")
# TODO activate submit_button.click()

time.sleep(1000)
# Schließen des Browsers
driver.quit()
