import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.6):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
            model_complexity=1
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Slight contrast boost for better detection in varied lighting
        imgRGB = cv2.convertScaleAbs(imgRGB, alpha=1.2, beta=10)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True, color=(255, 0, 255)):
        """Returns list of [id, cx, cy] for a specific hand."""
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo >= len(self.results.multi_hand_landmarks):
                return lmList
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
        return lmList

    def getHandsInfo(self, img=None):
        """Returns list of (index, label) tuples.
        Bypasses MediaPipe's known index-mapping bug by pairing
        labels with landmarks by their order in the results."""
        hands_info = []
        if self.results and self.results.multi_hand_landmarks:
            handedness_list = self.results.multi_handedness
            for i, _ in enumerate(self.results.multi_hand_landmarks):
                if i < len(handedness_list):
                    label = handedness_list[i].classification[0].label
                else:
                    label = "Unknown"
                hands_info.append((i, label))
        return hands_info

    def fingersUp(self, lmList, handedness="Right"):
        """Returns [thumb, index, middle, ring, pinky] where 1 = up, 0 = down.
        Handedness matters for the thumb: left and right thumbs point
        in opposite x-directions when extended."""
        if len(lmList) == 0:
            return []

        fingers = []

        # Thumb: compare x of tip (4) and IP joint (3)
        if handedness == "Right":
            # On a mirrored feed, a right-hand extended thumb
            # points leftward (lower x) relative to the IP joint
            fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)
        else:
            # Left-hand extended thumb points rightward (higher x)
            fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)

        # Index, Middle, Ring, Pinky: tip above PIP = up
        for fid in [8, 12, 16, 20]:
            fingers.append(1 if lmList[fid][2] < lmList[fid - 2][2] else 0)

        return fingers


# ─── Standalone test / debug runner ────────────────────────
def main():
    import time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("Trying default backend...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    for _ in range(10):
        cap.read()

    detector = handDetector(maxHands=2)
    pTime = 0

    while True:
        success, img = cap.read()
        if not success or img is None:
            continue

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        hands_info = detector.getHandsInfo(img)

        for idx, label in hands_info:
            lm = detector.findPosition(img, handNo=idx, draw=False)
            if lm:
                fingers = detector.fingersUp(lm, label)
                cv2.putText(
                    img, f"{label} (idx {idx}): {fingers}",
                    (10, 100 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()