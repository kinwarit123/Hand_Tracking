import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=0,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        # self.static_image_mode = static_image_mode
        # self.max_num_hands = max_num_hands
        # self.model_complexity = model_complexity
        # self.min_detection_confidence = min_detection_confidence
        # self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands  # type: ignore
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore
        self.mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore

        self.hands = self.mp_hands.Hands(
            static_image_mode,
            max_num_hands,
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence
        )

    def drawHand(self, image, draw=True):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.results.multi_hand_landmarks:
            for handlandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(image,
                                                   handlandmarks,
                                                   self.mp_hands.HAND_CONNECTIONS)
        return image

    def posistion(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
        return lmlist

def main():
    cap = cv2.VideoCapture(0)

    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Not success")
            continue
        detector = handDetector()
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


main()