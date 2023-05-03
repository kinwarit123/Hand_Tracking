import cv2
import mediapipe as mp
import time

def num_count(x=1, y=2):
    num = 0
    TopLms = [4, 8, 12, 16, 20]

    if handCheck() == "right":
        if lmlist[TopLms[0]][x] > lmlist[TopLms[0] - 2][x]:
            num += 1
    else:
        if lmlist[TopLms[0]][x] < lmlist[TopLms[0] - 2][x]:
            num += 1
            
    for fingerIndex in range(1,5):
        if lmlist[TopLms[fingerIndex]][y] < lmlist[TopLms[fingerIndex] - 2][y]:
            num += 1
    return num

def indexFingerDirection(x=1, y=2):
    direction = ["UP", "RIGHT", "DOWN", "LEFT"]
    if max(lmlist[0:][y]) == lmlist[8][y]:
        return direction[0]
    if min(lmlist[0:][x]) == lmlist[8][x]:
        return direction[1]
    if min(lmlist[0:][y]) == lmlist[8][y]:
        return direction[2]
    if max(lmlist[0:][x]) == lmlist[8][x]:
        return direction[3]

def handCheck(x=1, y=2):
    if lmlist[3][x] > lmlist[17][x] and lmlist[5][x] > lmlist[13][x]:
        return "right"
    else:
        return "left"

def putText(text, textposition, addText=""):
    cv2.putText(image, str(text) + str(addText), (textposition),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

ptime = 0
ctime = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
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
        lmlist = []
        
        if results.multi_hand_landmarks:
            for handlandmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image,
                                          handlandmarks,
                                          mp_hands.HAND_CONNECTIONS)
                
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if len(lmlist) != 0:
                # putText(indexFingerDirection(), (5, 70))
                putText(num_count(), (5, 110))
                putText(handCheck(), (5, 150), addText=" Hand")

        ctime = time.time()
        fps = str(int(1 / (ctime - ptime)))
        ptime = ctime
        cv2.putText(image, fps, (5, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow("Hand Tracking Frame", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()