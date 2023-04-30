import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore

cap = cv2.VideoCapture(0)

ptime = 0
ctime = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Not success")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if results.multi_hand_landmarks:
            for handlandmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)

                mp_drawing.draw_landmarks(image,
                                          handlandmarks,
                                          mp_hands.HAND_CONNECTIONS)

        ctime = time.time()
        fps = str(int(1 / (ctime - ptime)))
        ptime = ctime
        cv2.putText(image, fps, (5, 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.imshow("Hand Tracking Frame", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
