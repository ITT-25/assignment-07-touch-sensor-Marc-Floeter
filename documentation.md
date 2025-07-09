# Camera-Based Touch Sensor & Text Input
Marc Flöter, Universität Regensburg, Interaktionstechniken und -technologien (SoSe2025)


## Allgemein

Im Rahmen dieser Abgabe wurde ein kamerabasiertes Touch-Interface entwickelt, das Finger- bzw. Stylusbewegungen und Taps über eine von unten gefilmte Plexiglas-Oberfläche optisch erkennt, die zur Diffusion mit einem Blatt Papier belegt ist.
Bewegung und Taps werden auf die Dimensionen und das Koordinatensystem der mitgelieferten Fitt's Law Anwendung gemapped und als DIPPID-Events an diese versendet.
Zusätzlich wurde das System um die Funktionalität zur handschriftlichen Texteingabe erweitert. Zeichenweise Eingaben werden dabei durch ein auf den mitgelieferten NIST-Datensatz selbsttrainiertes CNN-Modell erkannt, und mit `pynput` auf Tasteneingaben des erkannten Zeichens gemapped.


## Benutzung

### Anforderungen

#### Hardware

- Eine Webcam wird innerhalb einer Kartonbox installiert, die nach oben zu einer aufliegenden Plexiglasscheibe gerichtet ist.
- Auf die Scheibe kommt ein Diffusor (Papier), um gleichmäßiges Licht und klare Kontraste für die Bilderkennung zu erzeugen.

#### Software
Zu finden in und installieren aus `requirements.txt`.
Funktionsbeschreibungen unter "Externe Bibliotheken / Codes / Datensätze".


### Start

Das Hauptprogramm `touch_input.py` wird einfach über Python ausgeführt:

```bash
python touch_input.py
```
**ACHTUNG**
Bei mir gibt es hier immer erstmal eine sehr lange Pause, bis die Webcam initialisiert wurde! Falls das bei euch auch so ist, einfach abwarten.

Beim Start, noch bevor etwas angezeigt wird, erfolgt eine automatische Kalibrierung der Lichtverhältnisse (Achte auf Konsolenausgabe für mehr Infos!). Wurde dann die richtige Kamera erkannt (Bei z.B. Laptops gibt es ja oft mehrere, die ID der zu verwendenden kann per Konstante "VIDEO_ID" eingestellt werden, default = 0), beginnt das System, Touch und ggf. Schriftzeichen zu erkennen.


### Bedienung

- Eine Berührung auf der Touch-Oberfläche innerhalb des von der Kamera erfassten Bereiches wird als Bewegung erkannt.
- Eine kurze Berührung (Schwellwert einstellbar als "MAX_TAP_DURATION", default auf 0,2s) wird zusätzlich als "Tap" registriert.
- Bewegungen und ggf. Infos über Taps werden per DIPPID verschickt und können durch eine parallel laufende Fitt's Law Anwendung zur Steuerung verarbeitet werden.
- Handschriftliche Zeichen (z. B. Buchstaben) werden durch längeres Schreiben mit Finger oder Stylus eingegeben.
- Nach einer kurzen Pause (Schwellwert einstellbar als "INPUT_TIMEOUT") wird das geschriebene Zeichen (sofern im Datensatz vorhanden gewesen) automatisch erkannt und als Tastatureingabe simuliert.
- `q` beendet das Programm.

### Einstellungen

Durch die Anpassung einiger Konstanten am Anfang des Codes können Einstellungen für die Benutzung der Anwendung gemacht werden.

- 



## Designentscheidungen und Funktionsweise

### 1. Fingererkennung, Palm-Rejection und Helligkeits-Kalibrierung

- Die Fingererkennung erfolgt über Konturdetektion in einem binären Threshold-Bild.
- Um Handflächen zu ignorieren, wird die Bounding-Box-Fläche auf einen einstellbaren Bereich typischer Fingergrößen beschränkt.
- Die Schwelle für das binäre Bild wird automatisch beim Start über eine Kalibrierung ermittelt. Helligkeits-Schwellwerte zur Erkennung des Fingers/Stylus in Abgrenzung zum "Hintergrund" werden anhand der Durchschnittshelligkeit des Kamerabildes berechnet. 


### 2. Bewegung & Tap

- Taps werden anhand der Dauer eines Touchs unterschieden (Schwellwert 0.2 Sekunden).
- Bewegungen werden kontinuierlich von DIPPID als `"movement"`-Events gesendet.

**DIPPID-Events für Fitt's Law Kommunikation:**
```json
{"movement": {"x": 130, "y": 200}}, 
{"tap": 1}
```

### 3. Rendering & Handschriftenerkennung

- Alle Bewegungen während eines Touchs werden gesammelt.
- Sobald eine Pause einstellbarer Länge erkannt wird, wird das gesamte "Stroke-Image" gerendert.
- Dieses Bild wird auf 28×28 Pixel skaliert (nach meinen Recherchen inzwischen gängig bei z.B. MNIST), normalisiert und zur Einordnung an das CNN-Modell übergeben.
- Die Vorhersage wird nur akzeptiert, wenn die Konfidenz über einem einstellbaren Schwellwert liegt, um falsche Eingaben durch nicht erkennbare Zeichen zu vermeiden.
- PS: Die Erkennungsstrategie "Zeichen für Zeichen" hat ausschließlich zeitliche Gründe und wird eher als Limitation angesehen.


### 4. Modellwahl & Training

- Für die Handschrifterkennung wurde ein eigenes Convolutional Neural Network (CNN) trainiert.
- Als Trainingsdaten wurden ausschließlich die zur Kurseinheit mitgelieferten Daten aus dem Subset des NIST-Datensatzes verwendet.
- Die Architektur und der Trainingsprozess (Ergänzungen in hwr_datasets.ipynb) orientieren sich an früheren Übungen und der Erfahrung daraus (Assignment 5).
- Das Modell wurde im `.keras`-Format gespeichert und wird in `touch_input.py` geladen.
- PS: Theoretisch kam eine Zeichenerkennung á la "$1 Recognizer" in Frage, wurde aber aufgrund der riesigen Menge an Vergleichsbildern aus Effizienzgründen ausgeschlossen.


## Externe Bibliotheken / Codes / Datensätze

- `cv2` (OpenCV): Kamerazugriff, Bildverarbeitung
- `numpy`: Array-Verarbeitung
- `Pillow`: Zeichnen der Zeichen-Pfade
- `keras`: Laden des CNN-Modells
- `pynput`: Simuliert Tastatureingaben
- `DIPPID`: Kommunikation mit dem Fitts’ Law Tool

Das CNN-Modell "text_recognition.keras" und dessen Trainingscode (Ergänzungen in hwr_datasets.ipynb) basieren auf meiner Abgabe zu Assignment 5, die wiederum auf dem im Kurs mitgelieferten Testcode basierte.

Als Trainingsdatensatz wurde das zu dieser Kurseinheit mitgelieferte Subset des NIST-Datensatzes verwendet.


## Limitationen

- Bei sehr schwachem Licht kann die Kalibrierung fehlschlagen, da nicht mehr garantiert werden kann, dass Hintergrund und Finger/Stylus einen ausreichend großen Helligkeitsunterschied aufweisen.
- Die Erkennung basiert dem Verhältnis von Helligkeit/Dunkelheit eines Schattens und der eingestellten Fingergröße. Das variiert stark und ist daher fehleranfällig bzw. muss genau eingestellt werden.
- Das CNN-Modell erkennt aktuell nur einzelne Zeichen, keine Wörter oder Sätze.
