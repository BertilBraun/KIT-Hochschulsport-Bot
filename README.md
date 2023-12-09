# Hochschulsport Bot

Das Problem ist, dass man beim Hochschulsport des KIT immer sehr schnell sein muss, um einen Platz zu bekommen. Dieses Skript soll einem dabei helfen, indem es automatisch versucht, einen Platz zu reservieren.

## Nutzung

Zur Nutzung soll der Link und die Kursnummer angegeben werden, zu welchem man angemeldet werden möchte.

Link: [Judo](https://buchsys-hsp.ifss.kit.edu/angebote/aktueller_zeitraum/_Judo.html)
Kursnummer: 3700

Zudem wird benötigt:

- Geschlecht
- Vorname
- Familienname
- Strasse Nr
- PLZ Ort
- Status: wird als KIT Student angenommen
- E-Mail
- IBAN

## Wie funktioniert es?

Wir rufen den angegebenen Link auf und suchen den Kurs mit der gegebenen Kursnummer. Wenn der Kurs gefunden wurde, wird auf die Anmeldeseite des Kurses navigiert. Dort werden die angegebenen Daten eingetragen und die Anmeldung wird abgeschickt.

## Future Work

- [x] Automatisches Ausfüllen der Daten
- [x] Anmeldung automatisch abschicken
- [ ] Captcha lösen
- [ ] Make it faaaaaassst as fuck boi
  - [ ] Faster submitting than timeout allows
- [x] Automatisches Auswählen des Kurses
- [ ] Bei Kursvergabe Start, automatisch, schnell, wiederholt versuchen, einen Platz zu bekommen
- [ ] Mehrere Kurse gleichzeitig buchen
- [ ] Prevalidate user information
- [ ] Improve README.md
  - [ ] Setup instructions
  - [ ] Committing to the project
  - [ ] etc.
