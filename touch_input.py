import cv2, time, socket, json
import numpy as np


# EINSTELLUNGEN UND KONSTANTEN #########################################################################

# ID der Kamera
VIDEO_ID = 0

# Maximale Dauer eines Touchs, um noch als Tap erkannt zu werden (alles darüber = nur Movement)
MAX_TAP_DURATION = 0.2

# Größe eines zu erkennenden Fingers
MIN_TOUCH_SIZE = 50
MAX_TOUCH_SIZE = 1000

# Helligkeitsunterschied von Finger auf Oberfläche zu durchschnittlicher Framebeleuchtung
V_DIFF_FINGER_TO_MEAN = 80

# Bounding Box um Finger/Stylus anzeigen
SHOW_BOUNDING_BOX = True
BOUNDING_BOX_COLOR = (0, 255, 0)
BOUNDING_BOX_THICKNESS = 2

# Kamera anzeigen
SHOW_CAM = True

# Preprocessed Frame statt Originalbild der Kamera anzeigen
SHOW_PREPROCESSED = False

FITTS_LAW_WINDOW_SIZE = 800

CALIBRATION_FRAMES = 60

# DIPPID Kommunikation
IP = '127.0.0.1'
PORT = 5700
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

frame_height = None
frame_width = None


# KAMERABILD AUSLESEN ##################################################################################

def check_cams():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera mit ID {i} verfügbar")
            cap.release()
        else:
            print(f"Kamera mit ID {i} NICHT verfügbar")


def open_cam(video_id):
    cap = cv2.VideoCapture(video_id) # Create a video capture object for the webcam

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden!")
        return None
    
    return cap


def detect_touch(cap, threshold):

    touching = False
    touch_start_time = None
    last_touch_time = None
    last_touch_pos = None

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Frames")
            break
        
        # Vertikal spiegeln
        flipped_frame = cv2.flip(frame, 1)
        
        preprocessed_frame = preprocess_frame(flipped_frame)
        masked_frame = apply_threshold(preprocessed_frame, threshold)
        
        # Touch-Position ermitteln
        current_time = time.time()
        touch_bb = detect_touch_position(masked_frame)

        if touch_bb:
            bb_x, bb_y, bb_width, bb_height, center_bb = touch_bb
            last_touch_time = current_time
            last_touch_pos = center_bb
            if not touching:
                print("Touch/Bewegung startet!")
                touch_start_time = current_time
            touching = True

            send_movement_event(center_bb, False)
            print(f"Touch center pos: {center_bb}")

        elif touching:
            touch_duration = last_touch_time - touch_start_time 
            if touch_duration < MAX_TAP_DURATION:  # kürzlich abgehoben = Tap
                send_movement_event(last_touch_pos, True)
                print("Tap erkannt!")
            else:
                print("Bewegung beendet!")
            touching = False

        # Optionales Live-Display
        if SHOW_CAM:
            if SHOW_PREPROCESSED:
                display_frame = masked_frame.copy()
            else:
                display_frame = flipped_frame.copy()

            if touch_bb and SHOW_BOUNDING_BOX:
                cv2.rectangle(display_frame, (bb_x, bb_y), (bb_x + bb_width, bb_y + bb_height), BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS)


            cv2.imshow('Visual Touch Sensor', display_frame)
        

        # Wait for a key press and check if it's the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


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


def map_coords_to_fitts_law(touch_pos):
    y_inverted = frame_height - touch_pos[1]
    scale_factor_by_height = FITTS_LAW_WINDOW_SIZE / frame_height
    scaled_touch_pos = (int(touch_pos[0] * scale_factor_by_height), int(y_inverted * scale_factor_by_height))

    return scaled_touch_pos


def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    return blurred_frame


def calibrate_threshold(cap):
    print("Starte Kalibrierung... bitte Plexiglasfläche freihalten!")

    for _ in range(CALIBRATION_FRAMES):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Kalibrierung fehlgeschlagen: Fehler beim Lesen des Frames")
        return None

    preprocessed_frame = preprocess_frame(frame)
    v_mean = np.mean(preprocessed_frame)
    print(f"Durchschnittliche Helligkeit: {v_mean}")

    if v_mean < V_DIFF_FINGER_TO_MEAN:
        print("Umgebung zu dunkel!")
        return None
    
    threshold = max(0, v_mean - V_DIFF_FINGER_TO_MEAN)
    print(f"Kalibrierter Threshold: {threshold}")

    return threshold


def apply_threshold(frame, threshold):
    _, masked_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)

    return masked_frame


def detect_touch_position(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    if not contours:
        return None
    
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_TOUCH_SIZE < area < MAX_TOUCH_SIZE:
            valid_contours.append(c)

    if not valid_contours:
        return None

    largest_valid_contour = max(valid_contours, key=cv2.contourArea)

    x, y, width, height = cv2.boundingRect(largest_valid_contour)
    center = (x + width // 2, y + height // 2)

    return x, y, width, height, center


def main(video_id):
    global frame_height, frame_width

    cap = open_cam(video_id)
    if cap == None:
        return
    
    threshold = calibrate_threshold(cap)
    if threshold is None:
        cap.release()
        return
    
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    detect_touch(cap, threshold)


if __name__ == "__main__":
    main(VIDEO_ID)
