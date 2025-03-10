import cv2
import mediapipe as mp
import numpy as np
import time  # Voor tijdsbeheer

# Initialiseer mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Kleurinstellingen
colors = {
    1: (255, 0, 0),    # Blue
    2: (128, 0, 128),  # Purple
    3: (0, 0, 255),    # Red
    4: (0, 255, 0),    # Green
    5: (255, 255, 0)   # Yellow
}

# Start de webcam
cap = cv2.VideoCapture(0)
drawing = False
last_point = None
previous_point = None

# Maak een canvas om de lijnen op te tekenen
canvas = None  # Dit wordt later geïnitialiseerd

# Voeg deze variabelen toe
current_color = (0, 0, 0)  # Begin met zwart
finger_count_stable_time = 0  # Tijd dat het aantal vingers constant is
stable_time_threshold = 5  # Tijd in frames om de kleur te veranderen
previous_finger_count = 0  # Houd het vorige aantal vingers bij

# Lijst om lijnen op te slaan
lines = []  # Elke lijn is een dict met 'start', 'end', 'color', 'timestamp'

# Maak vensters met de mogelijkheid om ze te schalen
cv2.namedWindow('Lijnen', cv2.WINDOW_NORMAL)
cv2.namedWindow('Hand Drawing', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Spiegel het beeld
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialiseer het canvas met dezelfde afmetingen als het frame
    if canvas is None:
        canvas = np.zeros(frame.shape, dtype=np.uint8)

    # Initialiseer finger_count voor deze iteratie
    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Teken de hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Tel het aantal vingers
            thumb_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
            index_up = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
            middle_up = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
            ring_up = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
            pinky_up = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y

            # Tel de vingers
            if thumb_up:
                finger_count += 1
            if index_up:
                finger_count += 1
            if middle_up:
                finger_count += 1
            if ring_up:
                finger_count += 1
            if pinky_up:
                finger_count += 1

            # Debug: Print de status van elke vinger
            print(f"Thumb: {thumb_up}, Index: {index_up}, Middle: {middle_up}, Ring: {ring_up}, Pinky: {pinky_up}")
            print(f"Aantal vingers: {finger_count}")

            # Controleer of het aantal vingers is veranderd
            if finger_count == previous_finger_count:
                finger_count_stable_time += 1  # Verhoog de timer
            else:
                finger_count_stable_time = 0  # Reset de timer

            # Update de kleur alleen als het aantal vingers constant is
            if finger_count_stable_time > stable_time_threshold:
                current_color = colors.get(finger_count, (0, 0, 0))  # Update de kleur

            # Bepaal de huidige positie van de vinger
            current_point = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                             int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))

            # Teken een lijn van de vorige positie naar de huidige positie als er vingers zijn
            if previous_point is not None and finger_count > 0:
                # Voeg de lijn toe aan de lijst met de huidige tijd
                lines.append({
                    'start': previous_point,
                    'end': current_point,
                    'color': current_color,
                    'timestamp': time.time()  # Huidige tijd in seconden
                })

            # Update de vorige positie
            previous_point = current_point

    # Stop met tekenen als er geen vingers zijn
    if finger_count == 0:
        previous_point = None  # Reset de vorige positie

    # Bewaar de huidige finger_count voor de volgende iteratie
    previous_finger_count = finger_count

    # Verwijder lijnen die langer dan 10 seconden op het scherm zijn
    current_time = time.time()
    lines = [line for line in lines if current_time - line['timestamp'] < 10]

    # Teken de lijnen op het canvas met een fade-effect
    for line in lines:
        elapsed_time = current_time - line['timestamp']
        if elapsed_time < 8:
            alpha = 1  # Volledig zichtbaar
        elif elapsed_time < 10:
            alpha = 1 - (elapsed_time - 8) / 2  # Fade-out effect tussen 8 en 10 seconden
        else:
            alpha = 0  # Volledig vervaagd

        color_with_alpha = tuple(int(c * alpha) for c in line['color'])  # Pas de kleur aan op basis van alpha
        cv2.line(canvas, line['start'], line['end'], color_with_alpha, 5)

    # Combineer de canvas met het frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Maak een nieuw venster voor de lijnen op een zwarte achtergrond
    line_window = np.zeros(frame.shape, dtype=np.uint8)  # Maak een zwart venster
    for line in lines:
        cv2.line(line_window, line['start'], line['end'], line['color'], 5)  # Teken de lijnen

    cv2.imshow('Lijnen', line_window)  # Toon het lijnvenster

    cv2.imshow('Hand Drawing', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
