import cv2
import mediapipe as mp
import time
import hand_module as hm

cap = cv2.VideoCapture(0)

ptime = 0
ctime = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Not success")
        continue
    detector = hm.handDetector()
    image = detector.drawHand(image)
    ctime = time.time()
    fps = str(int(1 / (ctime - ptime)))
    ptime = ctime
    cv2.putText(image, fps, (5, 40),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Hand Tracking Frame", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
