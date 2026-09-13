import cv2
import time
import numpy as np
import HandTrackingModule as htm
import pyautogui

# ═══════════════════════════════════════════════════════════
#  CAMERA SETUP
# ═══════════════════════════════════════════════════════════
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("Warming up camera...")
for _ in range(10):
    cap.read()
print("✅ Camera ready! Press Q to quit.")

# ═══════════════════════════════════════════════════════════
#  DETECTOR + PYAUTOGUI
# ═══════════════════════════════════════════════════════════
pTime = 0
detector = htm.handDetector(maxHands=2, detectionCon=0.7, trackCon=0.6)

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# ═══════════════════════════════════════════════════════════
#  TUNING
# ═══════════════════════════════════════════════════════════
SMOOTHING       = 0.45
DEAD_ZONE       = 4
CLICK_COOLDOWN  = 0.40
GESTURE_STABLE  = 3
SCROLL_PER_FRAME = 40          # scroll units applied per frame while held
THUMB_UP_PIXELS  = 40          # min vertical delta for thumb direction detection

cam_x_min, cam_x_max = 110, 530
cam_y_min, cam_y_max = 70, 400

# ═══════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════
smooth_x, smooth_y = 0, 0

right_mode = "IDLE"
right_prev_raw = None
right_stable = 0

left_mode = "IDLE"
left_prev_raw = None
left_stable = 0

grabbed = False
last_click_time = 0
left_pulse_until = 0
right_pulse_until = 0

# Edge-tracking for one-shot clicks
last_left_state = False
last_right_state = False


def dist(p1, p2):
    return np.hypot(p1[1] - p2[1], p1[2] - p2[2])


def stabilize(new_g, prev_g, count, needed=GESTURE_STABLE):
    if new_g == prev_g:
        count += 1
    else:
        count = 0
    return new_g, count, (count >= needed)


def move_cursor(px, py):
    global smooth_x, smooth_y
    tx = int(np.interp(px, [cam_x_min, cam_x_max], [0, screen_w - 1]))
    ty = int(np.interp(py, [cam_y_min, cam_y_max], [0, screen_h - 1]))
    tx = max(0, min(screen_w - 1, tx))
    ty = max(0, min(screen_h - 1, ty))

    smooth_x = SMOOTHING * tx + (1 - SMOOTHING) * smooth_x
    smooth_y = SMOOTHING * ty + (1 - SMOOTHING) * smooth_y

    cur = pyautogui.position()
    if abs(smooth_x - cur.x) > DEAD_ZONE or abs(smooth_y - cur.y) > DEAD_ZONE:
        pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)


def thumb_points_up(lmList, handedness):
    """Returns 1 if thumb points up, -1 if down, 0 if not clearly extended."""
    # Landmarks: 4 = thumb tip, 2 = thumb MCP, 3 = thumb IP
    tip_y = lmList[4][2]
    mcp_y = lmList[2][2]

    # Thumb tip-to-index-MCP distance to check extension
    ext_len = dist(lmList[4], lmList[5])
    palm_w = max(dist(lmList[5], lmList[17]), 1)

    # Thumb is extended if tip is far from index MCP relative to palm width
    if ext_len < palm_w * 0.55:
        return 0  # curled — not a scroll gesture

    dy = mcp_y - tip_y    # positive means tip is ABOVE mcp (screen y grows downward)
    if dy > THUMB_UP_PIXELS:
        return 1
    elif dy < -THUMB_UP_PIXELS:
        return -1
    return 0


# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════
while True:
    try:
        success, img = cap.read()
        if not success or img is None:
            continue

        img = cv2.flip(img, 1)
        img = detector.findHands(img, draw=False)
        hands_info = detector.getHandsInfo(img)

        right_lm = None
        left_lm = None
        for idx, label in hands_info:
            lm = detector.findPosition(img, handNo=idx, draw=False)
            if not lm:
                continue
            if label == "Right":
                right_lm = lm
            elif label == "Left":
                left_lm = lm

        # ═══════════════════════════════════════════════════
        #  RIGHT HAND
        # ═══════════════════════════════════════════════════
        right_wants_grab = False

        if right_lm:
            fingers_r = detector.fingersUp(right_lm, "Right")
            index  = right_lm[8]
            middle = right_lm[12]
            lm9    = right_lm[9]

            ext = fingers_r[1] + fingers_r[2] + fingers_r[3] + fingers_r[4]
            thumb_dir = thumb_points_up(right_lm, "Right")

            # ── Classify gesture ───────────────────────────
            if thumb_dir == 1:
                raw_r = "SCROLL_UP"
            elif thumb_dir == -1:
                raw_r = "SCROLL_DOWN"
            elif fingers_r[1] == 1 and fingers_r[2] == 1 and ext == 2:
                raw_r = "MOVE"
            elif ext == 0 and thumb_dir == 0:
                raw_r = "DRAG"
            else:
                raw_r = "IDLE"

            right_prev_raw, right_stable, confirmed = stabilize(
                raw_r, right_prev_raw, right_stable
            )
            if confirmed:
                right_mode = raw_r

            # ── Execute ────────────────────────────────────
            if right_mode == "MOVE":
                mx = (index[1] + middle[1]) // 2
                my = (index[2] + middle[2]) // 2
                move_cursor(mx, my)

                cv2.circle(img, (mx, my), 14, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (mx, my), 14, (255, 255, 255), 2)

            elif right_mode == "SCROLL_UP":
                pyautogui.scroll(SCROLL_PER_FRAME)
                cv2.arrowedLine(img, (320, 240), (320, 120),
                                (0, 255, 0), 8, tipLength=0.4)
                cv2.putText(img, "SCROLL UP", (200, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            elif right_mode == "SCROLL_DOWN":
                pyautogui.scroll(-SCROLL_PER_FRAME)
                cv2.arrowedLine(img, (320, 240), (320, 360),
                                (0, 100, 255), 8, tipLength=0.4)
                cv2.putText(img, "SCROLL DOWN", (180, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 3)

            elif right_mode == "DRAG":
                right_wants_grab = True
                move_cursor(lm9[1], lm9[2])
                cv2.circle(img, (lm9[1], lm9[2]), 16, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "DRAG", (250, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # ═══════════════════════════════════════════════════
        #  LEFT HAND
        # ═══════════════════════════════════════════════════
        left_wants_grab = False

        if left_lm:
            fingers_l = detector.fingersUp(left_lm, "Left")
            index_l  = left_lm[8]
            middle_l = left_lm[12]

            ext_l = fingers_l[1] + fingers_l[2] + fingers_l[3] + fingers_l[4]

            # ── Classify gesture (finger count) ────────────
            if fingers_l[1] == 1 and fingers_l[2] == 1 and ext_l == 2:
                raw_l = "RIGHT_CLICK"
            elif fingers_l[1] == 1 and ext_l == 1:
                raw_l = "LEFT_CLICK"
            elif ext_l == 0:
                raw_l = "SELECT"
            else:
                raw_l = "IDLE"

            left_prev_raw, left_stable, l_confirmed = stabilize(
                raw_l, left_prev_raw, left_stable
            )
            if l_confirmed:
                left_mode = raw_l

            now = time.time()
            can_click = (now - last_click_time) > CLICK_COOLDOWN

            # Edge-triggered clicks: only fire when entering the gesture
            if left_mode == "LEFT_CLICK":
                if not last_left_state and can_click:
                    pyautogui.click(button='left')
                    last_click_time = now
                    left_pulse_until = now + 0.2
                    print("Left click")
                last_left_state = True
            else:
                last_left_state = False

            if left_mode == "RIGHT_CLICK":
                if not last_right_state and can_click:
                    pyautogui.click(button='right')
                    last_click_time = now
                    right_pulse_until = now + 0.2
                    print("Right click")
                last_right_state = True
            else:
                last_right_state = False

            if left_mode == "SELECT":
                left_wants_grab = True

            # Visual feedback
            now_v = time.time()
            if now_v < left_pulse_until:
                cv2.circle(img, (index_l[1], index_l[2]), 22,
                           (0, 255, 0), cv2.FILLED)
            if now_v < right_pulse_until:
                cv2.circle(img, (middle_l[1], middle_l[2]), 22,
                           (255, 255, 0), cv2.FILLED)

            if left_mode == "SELECT":
                cv2.circle(img, (index_l[1], index_l[2]), 16,
                           (255, 100, 0), cv2.FILLED)
                cv2.putText(img, "SELECT", (250, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)

            # Dot on the "active" finger
            cv2.circle(img, (index_l[1], index_l[2]), 8,
                       (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (middle_l[1], middle_l[2]), 8,
                       (255, 200, 0), cv2.FILLED)

        # ═══════════════════════════════════════════════════
        #  GRAB MANAGER
        # ═══════════════════════════════════════════════════
        want_grab = right_wants_grab or left_wants_grab

        if want_grab and not grabbed:
            pyautogui.mouseDown()
            grabbed = True
            print("Grab ON")
        elif not want_grab and grabbed:
            pyautogui.mouseUp()
            grabbed = False
            print("Grab OFF")

        # ═══════════════════════════════════════════════════
        #  HUD
        # ═══════════════════════════════════════════════════
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.rectangle(img, (cam_x_min, cam_y_min),
                      (cam_x_max, cam_y_max), (80, 80, 80), 1)

        cv2.rectangle(img, (0, 0), (640, 60), (30, 30, 30), cv2.FILLED)
        cv2.putText(img, f"R: {right_mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"L: {left_mode}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (500, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if right_lm:
            cv2.putText(img, "R", (610, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        if left_lm:
            cv2.putText(img, "L", (610, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 200, 0), 2)

        cv2.putText(img, "R: 2-Fingers=Move | Thumb-Up/Down=Scroll | Fist=Drag",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)
        cv2.putText(img, "L: 1-Finger=LeftClick | 2-Finger=RightClick | Fist=Select",
                    (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)

        cv2.imshow("Gesture Control — Natural Two-Hand", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"⚠️ Loop error: {e}")
        continue

cap.release()
cv2.destroyAllWindows()