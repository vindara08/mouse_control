import cv2
import time
import numpy as np
from collections import deque, Counter
import HandTrackingModule as htm
import pyautogui
import subprocess

# ═══════════════════════════════════════════════════════════
#  CAMERA SETUP
# ═══════════════════════════════════════════════════════════
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Warming up camera...")
for _ in range(10):
    cap.read()
print("Camera ready! Press Q to quit.")

# ═══════════════════════════════════════════════════════════
#  DETECTOR + PYAUTOGUI
# ═══════════════════════════════════════════════════════════
pTime = 0
detector = htm.handDetector(maxHands=2, detectionCon=0.75, trackCon=0.7)
pyautogui.FAILSAFE = False

# ═══════════════════════════════════════════════════════════
#  TUNING
# ═══════════════════════════════════════════════════════════
GESTURE_STABLE   = 5
COMMAND_COOLDOWN = 0.9
THUMB_UP_PIXELS  = 40
PINCH_DIST       = 50
SEEK_REPEATS     = 3
LEFT_HISTORY_LEN = 7
RESET_FRAMES     = 4

# Two-hand close gesture
CLOSE_STABLE_FRAMES = 6      # both fists must persist this long
CLOSE_COOLDOWN      = 2.0    # seconds before close can fire again

APP_MAP = {
    1: ("Browser",    "Brave.exe"),
    2: ("Notepad",    "notepad.exe"),
    3: ("vscode", "vscode.exe"),
    4: ("Explorer",   "explorer.exe"),
    5: ("whatsapp",    "whatsapps.exe"),
}

# ═══════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════
right_mode = "IDLE"
right_prev_raw = None
right_stable = 0

last_oneshot = None
release_counter = 0

left_history = deque(maxlen=LEFT_HISTORY_LEN)
left_count = -1
left_prev_raw = -1
left_stable = 0
left_fired = False

last_command_time = 0

# Two-hand close state
close_stable = 0
close_last_fire = 0
close_armed = True          # re-arms when hands separate

DEBUG = True


def dist(p1, p2):
    return np.hypot(p1[1] - p2[1], p1[2] - p2[2])


def stabilize(new_g, prev_g, count, needed=GESTURE_STABLE):
    if new_g == prev_g:
        count += 1
    else:
        count = 0
    return new_g, count, (count >= needed)


def can_fire():
    global last_command_time
    now = time.time()
    if (now - last_command_time) > COMMAND_COOLDOWN:
        last_command_time = now
        return True
    return False


def thumb_direction(lmList):
    tip_y = lmList[4][2]
    mcp_y = lmList[2][2]
    ext_len = dist(lmList[4], lmList[5])
    palm_w = max(dist(lmList[5], lmList[17]), 1)

    if ext_len < palm_w * 0.55:
        return 0

    dy = mcp_y - tip_y
    if dy > THUMB_UP_PIXELS:
        return 1
    elif dy < -THUMB_UP_PIXELS:
        return -1
    return 0


def launch_app(app_key):
    if app_key in APP_MAP:
        name, exe = APP_MAP[app_key]
        try:
            subprocess.Popen(f"start {exe}", shell=True)
            print(f" Launched: {name}")
            return True
        except Exception as e:
            print(f" Failed to launch {name}: {e}")
    return False


def close_active_app():
    pyautogui.hotkey('alt', 'f4')
    print(" Closed active window")


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

        # ── Compute finger states once for both hands ──────
        fingers_r = None
        fingers_l = None
        ext_r = None
        ext_l = None

        if right_lm:
            fingers_r = detector.fingersUp(right_lm, "Right")
            ext_r = fingers_r[1] + fingers_r[2] + fingers_r[3] + fingers_r[4]

        if left_lm:
            fingers_l = detector.fingersUp(left_lm, "Left")
            ext_l = fingers_l[1] + fingers_l[2] + fingers_l[3] + fingers_l[4]

        # ═══════════════════════════════════════════════════
        #  TWO-HAND CLOSE GESTURE (check FIRST, before others)
        # ═══════════════════════════════════════════════════
        both_fists = (right_lm is not None and left_lm is not None
                and ext_r == 0 and ext_l == 0)

        now = time.time()
        can_close = (now - close_last_fire) > CLOSE_COOLDOWN

        if both_fists:
            close_stable += 1

            # Fire once when stable
            if (close_stable >= CLOSE_STABLE_FRAMES
                    and close_armed and can_close):
                close_active_app()
                close_last_fire = now
                close_armed = False     # wait for hands to separate
                cv2.putText(img, "CLOSE APP", (170, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Progress indicator while forming the gesture
            cv2.putText(img, f"CLOSE: {close_stable}/{CLOSE_STABLE_FRAMES}",
                        (170, 340), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 100, 255), 2)
        else:
            close_stable = 0
            close_armed = True         # re-arm when hands separate

        # Skip the individual-hand actions if both fists are active
        suppress_individual = both_fists

        # ═══════════════════════════════════════════════════
        #  RIGHT HAND — MEDIA
        # ═══════════════════════════════════════════════════
        if right_lm and not suppress_individual:
            thumb = right_lm[4]
            index = right_lm[8]

            thumb_dir = thumb_direction(right_lm)
            d_pinch = dist(thumb, index)

            if ext_r == 4:
                raw_r = "PLAY_PAUSE"
            elif thumb_dir == 1 and ext_r <= 1:
                raw_r = "VOL_UP"
            elif thumb_dir == -1 and ext_r <= 1:
                raw_r = "VOL_DOWN"
            elif fingers_r[1] == 1 and fingers_r[2] == 1 and ext_r == 2:
                raw_r = "SKIP_FWD"
            elif fingers_r[1] == 1 and ext_r == 1:
                raw_r = "SKIP_BACK"
            elif ext_r == 0 and thumb_dir == 0:
                raw_r = "MUTE"
            elif d_pinch < PINCH_DIST and ext_r <= 2:
                raw_r = "SEEK_FWD"
            else:
                raw_r = "IDLE"

            right_prev_raw, right_stable, confirmed = stabilize(
                raw_r, right_prev_raw, right_stable
            )
            if confirmed:
                right_mode = raw_r

            if right_mode != last_oneshot:
                release_counter += 1
                if release_counter >= RESET_FRAMES:
                    last_oneshot = None
                    release_counter = 0
            else:
                release_counter = 0

            if DEBUG:
                lock_state = last_oneshot if last_oneshot else "—"
                print(f"R: {right_mode:12s} | ext={ext_r} | thumb={thumb_dir:2d} | lock={lock_state}")

            if right_mode == "PLAY_PAUSE":
                if last_oneshot != "PLAY_PAUSE":
                    pyautogui.press('playpause')
                    print("Play/Pause FIRED")
                    last_oneshot = "PLAY_PAUSE"
                    release_counter = 0
                cv2.putText(img, "PLAY / PAUSE", (170, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

            elif right_mode == "MUTE":
                if last_oneshot != "MUTE":
                    pyautogui.press('volumemute')
                    print("Mute FIRED")
                    last_oneshot = "MUTE"
                    release_counter = 0
                cv2.putText(img, "MUTE", (260, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 3)

            elif right_mode == "SKIP_FWD" and can_fire():
                pyautogui.press('right')
                print("Skip Forward")
                cv2.arrowedLine(img, (300, 200), (440, 200),
                                (0, 255, 0), 8, tipLength=0.4)

            elif right_mode == "SKIP_BACK" and can_fire():
                pyautogui.press('left')
                print("Skip Backward")
                cv2.arrowedLine(img, (300, 200), (160, 200),
                                (0, 255, 0), 8, tipLength=0.4)

            elif right_mode == "VOL_UP" and can_fire():
                for _ in range(3):
                    pyautogui.press('volumeup')
                print(" Volume Up")
                cv2.putText(img, "VOLUME UP", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 3)

            elif right_mode == "VOL_DOWN" and can_fire():
                for _ in range(3):
                    pyautogui.press('volumedown')
                print("Volume Down")
                cv2.putText(img, "VOLUME DOWN", (170, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 3)

            elif right_mode == "SEEK_FWD" and can_fire():
                for _ in range(SEEK_REPEATS):
                    pyautogui.press('right')
                    time.sleep(0.03)
                print("Seek forward")

            cv2.circle(img, (index[1], index[2]), 10, (0, 255, 255), cv2.FILLED)

        elif not right_lm:
            last_oneshot = None
            release_counter = 0

        # ═══════════════════════════════════════════════════
        #  LEFT HAND — APP LAUNCHING (1–5 fingers, no fist)
        # ═══════════════════════════════════════════════════
        if left_lm and not suppress_individual:
            index_l = left_lm[8]

            # Total: 5 only if all four fingers + thumb extended
            if ext_l == 4 and fingers_l[0] == 1:
                total = 5
            else:
                total = ext_l

            left_history.append(total)
            counts = Counter(left_history)
            majority = counts.most_common(1)[0][0]

            if majority == left_prev_raw:
                left_stable += 1
            else:
                left_stable = 0
                left_prev_raw = majority
                left_fired = False

            if left_stable >= GESTURE_STABLE and not left_fired:
                left_count = majority
                left_fired = True
                if 1 <= left_count <= 5:
                    launch_app(left_count)
                    cv2.putText(img, f"LAUNCH {left_count}",
                                (200, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 3)

            cv2.circle(img, (index_l[1], index_l[2]), 10, (255, 0, 255),
                    cv2.FILLED)
            cv2.putText(img, f"L: raw={total} maj={majority} count={left_count} stable={left_stable}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 200, 0), 2)

            if DEBUG:
                print(f"L: raw={total}  majority={majority}  stable={left_stable}  "
                    f"count={left_count}  fired={left_fired}")

        elif not left_lm:
            left_history.clear()
            left_stable = 0
            left_prev_raw = -1
            left_fired = False

        # ═══════════════════════════════════════════════════
        #  HUD
        # ═══════════════════════════════════════════════════
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.rectangle(img, (0, 0), (640, 60), (30, 30, 30), cv2.FILLED)
        cv2.putText(img, f"R: {right_mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (500, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if right_lm:
            cv2.putText(img, "R", (610, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        if left_lm:
            cv2.putText(img, "L", (610, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 200, 0), 2)

        cv2.putText(img, "R: Palm=Play | 2F=Fwd | 1F=Back | Thumb=Vol | Fist=Mute | Pinch=Seek",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1)
        cv2.putText(img, "L: 1=Browser | 2=Notepad | 3=Calc | 4=Files | 5=Spotify",
                    (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1)
        cv2.putText(img, "CLOSE APP: both hands as fists",
                    (10, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (100, 100, 255), 1)

        cv2.imshow("Gesture Media Controller", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f" Loop error: {e}")
        continue

cap.release()
cv2.destroyAllWindows()