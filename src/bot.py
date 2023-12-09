import time
import base64
from tkinter.tix import AUTO
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from config import *


def solve_captcha(captcha_image):
    # TODO solve
    captcha_image.save("captcha.png")
    return ""
    
    
# Initialisieren des WebDrivers
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Öffnen einer Webseite
driver.get(LINK)

driver.implicitly_wait(10)

# Verwenden von XPath, um den Kurs mit der Nummer 3700 zu finden
kursnummer_xpath = "//td[@class='bs_sknr' and text()='3700']/ancestor::tr//input[@type='submit']"
submit_button = driver.find_element(By.XPATH, kursnummer_xpath)

print("Kurs gefunden")

submit_button.click()
driver.implicitly_wait(5)

# Wechseln zur neuen Registerkarte
new_tab = driver.window_handles[1]  # Index 1, da 0 die erste (ursprüngliche) Registerkarte ist
driver.switch_to.window(new_tab)

# Eintragen der Daten
geschlecht_button = driver.find_element(By.XPATH, f"//input[@name='sex' and @value='{GESCHLECHT}']")
geschlecht_button.click()

vorname_input = driver.find_element(By.NAME, "vorname")
vorname_input.clear() 
vorname_input.send_keys(VORNAME)

name = driver.find_element(By.NAME, "name")
name.clear()
name.send_keys(NACHNAME)

street = driver.find_element(By.NAME, "strasse")
street.clear()
street.send_keys(STRASSE + " " + HAUSNUMMER)

ort = driver.find_element(By.NAME, "ort")
ort.clear()
ort.send_keys(PLZ + " " + ORT)

status_element = driver.find_element(By.NAME, "statusorig")
status = Select(status_element)
status.select_by_value("S-KIT")

matnr = driver.find_element(By.NAME, "matnr")
matnr.clear()
matnr.send_keys(MATNUMMER)

mail = driver.find_element(By.NAME, "email")
mail.clear()
mail.send_keys(EMAIL)

iban = driver.find_element(By.NAME, "iban")
iban.clear()
iban.send_keys(IBAN)

accept = driver.find_element(By.NAME, "tnbed")
accept.click()

print("Daten eingetragen")
print("Warte auf Absenden der Daten...")

# Absenden der Daten
weiter_button = WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable((By.ID, "bs_submit"))
)
weiter_button.click()

driver.implicitly_wait(5)

print("Daten abgesendet")

# Extrahieren des CAPTCHA-Bildes als Kind des Elements mit der ID 'bs_captcha'
captcha_img_src = driver.find_element(By.CSS_SELECTOR, "#bs_captcha img").get_attribute("src")
captcha_base64 = captcha_img_src.split(",")[1]

# Dekodieren des Bildes
captcha_image = Image.open(BytesIO(base64.b64decode(captcha_base64)))

captcha_text = solve_captcha(captcha_image)

print(f"CAPTCHA-Text: {captcha_text}")

# Eingabe des CAPTCHA-Textes in das Eingabefeld
captcha_input_field = driver.find_element(By.ID, "BS_F_captcha")
captcha_input_field.send_keys(captcha_text.strip())

# repeatedly check if captcha has 6 characters entered, then submit
while len(captcha_input_field.get_attribute("value")) != 6:
    time.sleep(0.1)

submit_button = driver.find_element(By.XPATH, "//input[@type='submit'][@value='kostenpflichtig buchen']")
if AUTOSUBMIT:
    submit_button.click()

time.sleep(1000)
# Schließen des Browsers
driver.quit()
