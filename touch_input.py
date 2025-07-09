import cv2, time, socket, json, keras
import numpy as np
from PIL import Image, ImageDraw
from pynput.keyboard import Controller, Key


# EINSTELLUNGEN UND KONSTANTEN #########################################################################

# GUI
## Touch
SHOW_CAM = True # Kamera anzeigen
SHOW_PREPROCESSED = False # Preprocessed Frame statt Originalbild der Kamera anzeigen
SHOW_BOUNDING_BOX = True # Bounding Box um Finger/Stylus anzeigen
BOUNDING_BOX_COLOR = (0, 255, 0) # Farbe der Bounding Box Außenlinie
BOUNDING_BOX_THICKNESS = 2 # Dicke der Bounding Box Außenlinie

## Texteingabe
TEXT_INPUT = True # Text Input aktivieren/deaktivieren
SHOW_PATH = True # Anzeige des Touch-Pfads eines Zeichens auf dem Kamerabild
GUI_LINE_THICKNESS = 4 # Dicke des Pfads auf dem Kamerabild
GUI_LINE_COLOR = (255, 0, 0) # Farbe des Pfads auf dem Kamerabild
SHOW_PATH_IMG = True # Anzeige des preprocessed Auswertungsbilds

# Kamera
VIDEO_ID = 0 # Geräte-ID der Kamera
CALIBRATION_FRAMES = 60 # Frames, bis Helligkeits-Kalibrierung startet (erste Frames meist noch zu dunkel und unscharf)

# Touch-Erkennung
MAX_TAP_DURATION = 0.2 # Maximale Dauer eines Touchs (in s), um noch als Tap erkannt zu werden (alles darüber = nur Movement)
MIN_TOUCH_SIZE = 50 # Minimale Größe eines zu erkennenden Fingers
MAX_TOUCH_SIZE = 5000 # Maximale Größe eines zu erkennenden Fingers
V_DIFF_FINGER_TO_MEAN = 75 # Helligkeitsunterschied von Finger auf Oberfläche zu durchschnittlicher Framebeleuchtung

# Texterkennung
## Texteingabe
INPUT_TIMEOUT = 1.0 # Minimale Dauer der Pause nach einem Touch-Input, bis die Auswertung startet

## Preprocessing
PATH_LINE_THICKNESS = 4 # Dicke des Pfads im Auswertungsbild
PATH_LINE_COLOR = 0 # Farbe des Pfads im Auswertungsbild (schwarz)
PATH_IMG_SIZE = 28 # Größe des Bilds mit dem zu erkennenden Zeichen für das CNN-Modell
PATH_IMG_MARGIN = 2 # Rand um das Zeichen im Auswertungsbild

## Prediction
MIN_PATH_BB_SIZE = 1 # Minimale Größe eines Eingabezeichens, um zur Auswertung zugelassen zu werden
MIN_PREDICTION_CONFIDENCE = 0.5 # Minimale Prediction Confidence des Modells, um als valide Eingabe betrachtet zu werden
LABEL_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] # Liste der Namen der eingebbaren Zeichen

# Fitt's Law Kompatibiltät (DIPPID)
FITTS_LAW_WINDOW_SIZE = 800
IP = '127.0.0.1'
PORT = 5700
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
frame_height = None
frame_width = None

# CNN-Model zur Texterkennung
model = keras.models.load_model("./cnn_model/text_recognition.keras")

# Pynput-Controller für Texteingabe
keyboard = Controller()


########################################################################################################
def main(video_id):
    global frame_height, frame_width

    # Kamera starten
    cap = open_cam(video_id)
    if cap == None:
        return
    
    # Helligkeit kalibrieren
    threshold = calibrate_threshold(cap)
    if threshold is None:
        cap.release()
        return
    
    # Bilddimensionen holen (für Mapping auf Fitt's Law)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Toucherkennung starten
    detect_touch(cap, threshold)


# KAMERA UND KALIBRIERUNG #########################################################################################

# Kamera öffnen
def open_cam(video_id):
    cap = cv2.VideoCapture(video_id)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden!")
        return None
    
    return cap


# Helligkeitsschwellwert anhand durchschnittlicher Beleuchtung der Touchfläche berechnen
def calibrate_threshold(cap):
    print("Starte Kalibrierung... bitte Plexiglasfläche freihalten!")

    for _ in range(CALIBRATION_FRAMES):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Kalibrierung fehlgeschlagen: Fehler beim Lesen des Frames")
        return None

    preprocessed_frame = preprocess_frame(frame) # Vorverarbeiteten Frame verwenden
    v_mean = np.mean(preprocessed_frame)
    print(f"Durchschnittliche Helligkeit: {v_mean}")

    # Zu dunkle Umgebung, falls Durchschnittshelligkeit kleiner als zur Finger/Styluserkennung nötiger Helligkeitsunterschied
    if v_mean < V_DIFF_FINGER_TO_MEAN:
        print("Umgebung zu dunkel!")
        return None
    
    threshold = max(0, v_mean - V_DIFF_FINGER_TO_MEAN)
    print(f"Kalibrierter Helligkeits-Schwellwert: {threshold}")

    return threshold


# Frame zur Helligkeits-Schwellwertberechnung und Toucherkennung preprocessen
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    return blurred_frame


# TOUCH EINGABE ERKENNEN UND VERARBEITEN #####################################################################
def detect_touch(cap, threshold):

    touching = False
    touch_start_time = None
    last_touch_time = None
    last_touch_pos = None

    strokes = [] # Liste aller touch_paths innerhalb eines timeouts
    touch_path = [] # Liste an berührten Punkten seit dem letzten Absetzen

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Frames")
            break
        
        # Vertikal spiegeln
        flipped_frame = cv2.flip(frame, 1)
        
        # Preprocessing und Helligkeitsschwellwert als Maske anwenden
        preprocessed_frame = preprocess_frame(flipped_frame)
        masked_frame = apply_threshold(preprocessed_frame, threshold)
        
        # Touch-Position ermitteln
        current_time = time.time()
        touch_bb = detect_touch_position(masked_frame)

        # Falls eine valide Touch-Position nach einer Pause neu erkannt wurde, zeichne Pfad auf und sende DIPPID Events
        if touch_bb:
            bb_x, bb_y, bb_width, bb_height, center_bb = touch_bb
            last_touch_time = current_time
            last_touch_pos = center_bb

            if not touching:
                print("Touch/Bewegung startet!")
                touch_start_time = current_time
                touch_path.clear()

            touching = True
            touch_path.append(center_bb)

            send_movement_event(center_bb, False) # DIPPID Event senden
            print(f"Touch center pos: {center_bb}")

        # Falls gerade kein Touch erkannt, aber gerade noch einer erkannt wurde, entscheide, ob es ein Tap war, füge den Pfad ggf. zur Liste an gemalten Strichen und starte Auswertung, falls genug Zeit vergangen
        elif touching:
            touch_duration = last_touch_time - touch_start_time 

            # kürzlich abgehoben = Tap
            if touch_duration < MAX_TAP_DURATION:  
                send_movement_event(last_touch_pos, True)
                print("Tap erkannt!")
            else: 
                print("Bewegung beendet!")

            # Füge Pfad zu Liste an Strichen des Zeichens hinzu
            if touch_path:
                strokes.append(touch_path.copy())
                touch_path.clear()

            touching = False

        # Falls kein Touch erkannt und Input Timeout überschritten -> Starte Auswertung der Touch-Texteingabe
        if not touching and strokes and (time.time() - last_touch_time > INPUT_TIMEOUT) and TEXT_INPUT:
            print("Buchstabe abgeschlossen, rendere alle Strokes...")
            strokes_img = render_strokes_image(strokes, PATH_IMG_SIZE, PATH_IMG_SIZE) # Generiere Bild aus Strichen

            if strokes_img is not None:
                # Vorverarbietung für CNN-Modell
                preprocessed_strokes_img = preprocess_strokes_image_for_prediction(strokes_img)

                # Prediction des CNN-Modells, um welches Zeichen es sich handelt
                prediction = model.predict(preprocessed_strokes_img)
                predicted_index = np.argmax(prediction)
                confidence = np.max(prediction)

                if 0 <= predicted_index < len(LABEL_NAMES):
                    predicted_label = LABEL_NAMES[predicted_index]
                
                    print(f"Vorhergesagtes Label: {predicted_label}")
                    print(f"Label-Index: {predicted_index}")
                    print(f"Confidence: {confidence:.2f}")

                    # Falls sich Modell sicher genug ist, dass es dieses Zeichen ist -> Drücke entsprechende Taste
                    if confidence >= MIN_PREDICTION_CONFIDENCE: 
                        trigger_keypress(predicted_label)
                    else:
                        print(f"Confidence der Prediction unter Schwellwert von {MIN_PREDICTION_CONFIDENCE}")
                else:
                    print("Ungültiger Index der Prediction")
                
                # Vorverarbeitetes Bild zur Zeichenerkennung in eigenem Fenster anzeigen (Debug)
                if SHOW_PATH_IMG and strokes_img is not None:
                    cv2.imshow("Strokes Image", strokes_img)
                    
            strokes.clear()

        # Optionales Live-Display
        if SHOW_CAM:
            if SHOW_PREPROCESSED:
                display_frame = masked_frame.copy()
            else:
                display_frame = flipped_frame.copy()

            if touch_bb and SHOW_BOUNDING_BOX:
                cv2.rectangle(display_frame, (bb_x, bb_y), (bb_x + bb_width, bb_y + bb_height), BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS)

            if TEXT_INPUT and SHOW_PATH:
                # Bisher gezeichnete Striche zeichnen
                for stroke in strokes:
                    if len(stroke) > 1:
                        for i in range(1, len(stroke)): # Beginnend bei 1 statt 0, weil sonst kein vorheriger Punkt da ist, zu dem man die Linie ziehen könnte
                            cv2.line(display_frame, stroke[i-1], stroke[i], GUI_LINE_COLOR, GUI_LINE_THICKNESS)
                # Aktuellen Strich zeichnen
                if len(touch_path) > 1:
                    for i in range(1, len(touch_path)): 
                        cv2.line(display_frame, touch_path[i-1], touch_path[i], GUI_LINE_COLOR, GUI_LINE_THICKNESS)
            
            # Kamerabild ggf. mit Zusätzen anzeigen
            cv2.imshow('Visual Touch Sensor', display_frame) 

        # Wait for a key press and check if it's the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Helligkeitsschwellwert zur Erkennung von Touch auf Frame anwenden
def apply_threshold(frame, threshold):
    _, masked_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)

    return masked_frame


# Touch-Positionen Anhand von Konturen im vorverarbeiteten Frame erkennen, die größte Valide als Eingabe benutzen (single touch)
def detect_touch_position(frame):

    # Alle Konturen finden
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    if not contours:
        return None
    
    # Checken, ob Kontur im fingergroßen Bereich ist
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_TOUCH_SIZE < area < MAX_TOUCH_SIZE:
            valid_contours.append(c)

    if not valid_contours:
        return None

    # Größte valide Kontur = Eingabe
    largest_valid_contour = max(valid_contours, key=cv2.contourArea)

    # Bounding-Box um Toucheingabe erstellen
    x, y, width, height = cv2.boundingRect(largest_valid_contour)
    center = (x + width // 2, y + height // 2) # Mitte der Berührung = Eingabeposition

    return x, y, width, height, center


# FITTS' LAW STEUERUNG #################################################################################################

# DIPPID Event zur Steuerung von Fitts' Law erstellen und senden
def send_movement_event(touch_pos, tapping):
    if tapping:
        tap_value = 1
    else:
        tap_value = 0

    mapped_touch_pos = map_coords_to_fitts_law(touch_pos)
    movement_event = {
        "movement": {"x": mapped_touch_pos[0], "y": mapped_touch_pos[1]},
        "tap": tap_value
    }

    sock.sendto(json.dumps(movement_event).encode(), (IP, PORT))


# Touch-Event auf Koordinatensystem von Fitt's Law Anwendung anpassen
def map_coords_to_fitts_law(touch_pos):
    y_inverted = frame_height - touch_pos[1]
    scale_factor_by_height = FITTS_LAW_WINDOW_SIZE / frame_height
    scaled_touch_pos = (int(touch_pos[0] * scale_factor_by_height), int(y_inverted * scale_factor_by_height))

    return scaled_touch_pos


# TEXTEINGABE ###########################################################################################################

# Bild des gezeichneten Zeichens aus aufgezeichneten Touch-Pfaden erstellen
def render_strokes_image(strokes, width, height):
    if not strokes or not any(len(stroke) > 0 for stroke in strokes):
        print("Keine gültigen Striche zum Rendern, Eingabe verworfen")
        return None

    # Leeres Bild erstellen
    strokes_img = Image.new("L", (width, height), color=255)

    # Punkte aller Striche des Zeichens zusammenführen für Skalierungs- und Ausrichtungsberechnungen
    all_points = np.concatenate([np.array(stroke) for stroke in strokes])

    # Bounding Box um alle Striche des Zeichens berechnen
    min_x = np.min(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_x = np.max(all_points[:, 0])
    max_y = np.max(all_points[:, 1])
    bb_width = max_x - min_x
    bb_height = max_y - min_y

    # Zu kleine oder zu große Eingaben ignorieren
    if bb_width < MIN_PATH_BB_SIZE or bb_height < MIN_PATH_BB_SIZE:
        print("Eingabe zu klein oder groß, verworfen")
        return None

    # Ganzes Zeichen auf 28x28px skalieren
    scale_x = (width - PATH_IMG_MARGIN * 2) / bb_width
    scale_y = (height - PATH_IMG_MARGIN * 2) / bb_height
    scale = min(scale_x, scale_y)

    all_points_scaled = []
    for (x, y) in all_points:
        x_scaled = int((x - min_x) * scale)
        y_scaled = int((y - min_y) * scale)
        all_points_scaled.append((x_scaled, y_scaled))
    all_points_scaled = np.array(all_points_scaled)

    # Bounding-Box der skalierten Punkte bestimmen
    min_scaled_x = np.min(all_points_scaled[:, 0])
    min_scaled_y = np.min(all_points_scaled[:, 1])
    max_scaled_x = np.max(all_points_scaled[:, 0])
    max_scaled_y = np.max(all_points_scaled[:, 1])

    bb_center_x = (min_scaled_x + max_scaled_x) / 2
    bb_center_y = (min_scaled_y + max_scaled_y) / 2

    # Mittelpunkt des Zielbildes berechnen
    strokes_img_center_x = width / 2
    strokes_img_center_y = height / 2

    # Offset, um die Mitte der Bounding Box auf die Bildmitte zu legen
    offset_x = strokes_img_center_x - bb_center_x
    offset_y = strokes_img_center_y - bb_center_y

    # Linien durch Punkte pro Stroke ziehen (damit getrennt bleibt, was getrennt gehört)
    for stroke in strokes:
        stroke_np = np.array(stroke)

        # Strich skalieren
        stroke_scaled = []
        for (x, y) in stroke_np:
            x_scaled = int((x - min_x) * scale)
            y_scaled = int((y - min_y) * scale)
            stroke_scaled.append((x_scaled, y_scaled))
        stroke_scaled = np.array(stroke_scaled)

        # Strich ausrichten (anhand der Finalposition des ganzen Zeichens)
        stroke_centered = []
        for (x, y) in stroke_scaled:
            x_centered = int(x + offset_x)
            y_centered = int(y + offset_y)
            stroke_centered.append((x_centered, y_centered))

        # Strich zeichnen
        draw = ImageDraw.Draw(strokes_img)
        for i in range(1, len(stroke_centered)):
            draw.line([stroke_centered[i-1], stroke_centered[i]], fill=PATH_LINE_COLOR, width=PATH_LINE_THICKNESS)
        strokes_img_np = np.array(strokes_img)

    return strokes_img_np


# Bild des gezeichneten Zeichens zur Einordnung durch CNN-Modell preprocessen
def preprocess_strokes_image_for_prediction(img):
    img_normalized = img.astype('float32') / 255.0  # Normalisieren
    img_reshaped = img_normalized.reshape(1, PATH_IMG_SIZE, PATH_IMG_SIZE, 1)

    return img_reshaped


# Tastendruck durch erkanntes gezeichnetes Zeichen auslösen
def trigger_keypress(predicted_char):
    if not predicted_char or len(predicted_char) != 1:
        print(f"Ungültige Prediction: {predicted_char}")
        return
    
    keyboard.press(predicted_char)
    keyboard.release(predicted_char)

    print(f"Erkanntes Zeichen: {predicted_char}")


if __name__ == "__main__":
    main(VIDEO_ID)
