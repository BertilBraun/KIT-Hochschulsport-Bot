import re
import sys
import time
import base64
import argparse
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


def solve_captcha(captcha_image) -> str:
    # TODO solve
    captcha_image.save("captcha.png")
    return ""
    
    
def pre_validate_user_data():
    has_error = False
    
    for format, value in zip(
        [r"[A-Za-z]+", r"[A-Za-z]+", r"[A-Za-z\w]+", r"[0-9]+", r"[0-9]{5}", r"[A-Za-z\w]+", r"[0-9]{6}", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", r"[A-Z]{2}[0-9]{20}"],
        [VORNAME, NACHNAME, STRASSE, HAUSNUMMER, PLZ, ORT, MATNUMMER, EMAIL, IBAN]
    ):
        if not value:
            print(f"Fehler: Bitte fülle alle Felder aus!")
            has_error = True
        if not re.match(f"^{format}$", value):
            print(f"Fehler: Ungültiges Format für \"{value}\"! Erwartet: {format}")
            has_error = True
            
    if has_error:
        sys.exit(1)
                            
def signup_to_course():
    while True:
        try:
            try_signup_to_course()
            print("Erfolgreich angemeldet!")
            break
        except Exception as e:
            print(e)
            print("Fehler aufgetreten, versuche es erneut...")
            time.sleep(1)
    
def try_signup_to_course():
        
    # Initialisieren des WebDrivers
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    go_to_signup_page(driver)
    handle_signup_data(driver)
    captcha_text = handle_captcha(driver)
    handle_signup(driver, captcha_text)
    
    driver.quit()

def go_to_signup_page(driver):
    
    # Öffnen einer Webseite
    driver.get(LINK)

    # Verwenden von XPath, um den Kurs mit der Nummer 3700 zu finden
    kursnummer_xpath = "//td[@class='bs_sknr' and text()='3700']/ancestor::tr//input[@type='submit']"
    # Warten auf den "Submit"-Button
    submit_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, kursnummer_xpath))
    )

    print("Kurs gefunden")

    submit_button.click()

    # Warten, bis die neue Registerkarte geöffnet wird
    WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) > 1)

    # Wechseln zur neuen Registerkarte
    new_tab = driver.window_handles[1]  # Index 1, da 0 die erste (ursprüngliche) Registerkarte ist
    driver.switch_to.window(new_tab)
    
    
def handle_signup_data(driver):
    # Eintragen der Daten
    geschlecht_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, f"//input[@name='sex' and @value='{GESCHLECHT}']"))
    )
    geschlecht_button.click()

    status_element = driver.find_element(By.NAME, "statusorig")
    status = Select(status_element)
    status.select_by_value("S-KIT")

    for name, value in zip(
        ['vorname', 'name', 'strasse', 'ort', 'matnr', 'email', 'iban'],
        [VORNAME, NACHNAME, STRASSE + " " + HAUSNUMMER, PLZ + " " + ORT, MATNUMMER, EMAIL, IBAN]
    ):
        element = driver.find_element(By.NAME, name)
        element.clear()
        element.send_keys(value)

    accept = driver.find_element(By.NAME, "tnbed")
    accept.click()

    print("Daten eingetragen")
    print("Warte auf Absenden der Daten...")

    # Absenden der Daten
    weiter_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.ID, "bs_submit"))
    )
    weiter_button.click()

    print("Daten abgesendet")


def handle_captcha(driver):
    # Extrahieren des CAPTCHA-Bildes als Kind des Elements mit der ID 'bs_captcha'
    # Warten auf das CAPTCHA-Bild
    captcha_img_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#bs_captcha img"))
    )
    captcha_img_src = captcha_img_element.get_attribute("src")
    captcha_base64 = captcha_img_src.split(",")[1]

    # Dekodieren des Bildes
    captcha_image = Image.open(BytesIO(base64.b64decode(captcha_base64)))

    captcha_text = solve_captcha(captcha_image)

    print(f"CAPTCHA-Text: {captcha_text}")
    return captcha_text

def handle_signup(driver, captcha_text):
    # Eingabe des CAPTCHA-Textes in das Eingabefeld
    captcha_input_field = driver.find_element(By.ID, "BS_F_captcha")
    captcha_input_field.send_keys(captcha_text.strip())

    # repeatedly check if captcha has 6 characters entered, then submit
    while len(captcha_input_field.get_attribute("value")) != 6:
        time.sleep(0.1)

    submit_button = driver.find_element(By.XPATH, "//input[@type='submit'][@value='kostenpflichtig buchen']")
    if AUTOSUBMIT:
        submit_button.click()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skript mit -r oder -o starten.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", "--retry", action="store_true", help="Versuche Anmeldung bis es klappt")
    group.add_argument("-o", "--once", action="store_true", help="Versuche Anmeldung nur einmal")

    args = parser.parse_args()
    
    pre_validate_user_data()

    if args.retry:
        signup_to_course()
    elif args.once:
        try_signup_to_course()