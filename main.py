
import cv2
import mediapipe as mp
import pyautogui
import math
import time
from pathlib import Path

# ==================================================
# GESTURE AI - HAND GESTURE CONTROLLER
# ==================================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

MODEL_PATH = str(
    Path(__file__).with_name("hand_landmarker.task")
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Settings
PINCH_THRESHOLD = 0.35
PINCH_RELEASE_THRESHOLD = 0.48
SMOOTHING = 0.35
ACTION_COOLDOWN = 0.7
SCROLL_SENSITIVITY = 300
SWIPE_DISTANCE = 0.15

mode = "MOUSE"
paused = False
dragging = False
pinch_active = False
previous_x = None
previous_y = None
scroll_anchor = None
volume_anchor = None
swipe_anchor = None
last_action = 0
previous_gesture = ""
timestamp = 0

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Webcam not available")


def finger_states(hand):
    return [
        hand[tip].y < hand[pip].y
        for tip, pip in [
            (8, 6), (12, 10),
            (16, 14), (20, 18)
        ]
    ]


def recognize_gesture(hand, was_pinching=False):
    fingers = finger_states(hand)
    count = sum(fingers)

    pinch_distance = math.hypot(
        hand[4].x - hand[8].x,
        hand[4].y - hand[8].y
    )

    hand_size = math.hypot(
        hand[0].x - hand[9].x,
        hand[0].y - hand[9].y
    )

    threshold = (
        PINCH_RELEASE_THRESHOLD
        if was_pinching
        else PINCH_THRESHOLD
    )

    is_pinch = (
        hand_size > 0 and
        pinch_distance < hand_size * threshold
    )

    if is_pinch:
        return "PINCH"
    if count == 0:
        return "FIST"
    if count == 4:
        return "OPEN PALM"
    if fingers == [True, False, False, False]:
        return "ONE FINGER"
    if fingers == [True, True, False, False]:
        return "TWO FINGERS"
    if fingers == [True, True, True, False]:
        return "THREE FINGERS"

    return "OTHER"


def move_cursor(hand):
    global previous_x, previous_y

    margin = 0.12

    x = (hand[8].x - margin) / (1 - 2 * margin)
    y = (hand[8].y - margin) / (1 - 2 * margin)

    x = max(0.02, min(0.98, x))
    y = max(0.02, min(0.98, y))

    target_x = int(x * (SCREEN_W - 1))
    target_y = int(y * (SCREEN_H - 1))

    if previous_x is None:
        previous_x = target_x
        previous_y = target_y

    previous_x = int(
        previous_x * (1 - SMOOTHING)
        + target_x * SMOOTHING
    )
    previous_y = int(
        previous_y * (1 - SMOOTHING)
        + target_y * SMOOTHING
    )

    pyautogui.moveTo(previous_x, previous_y)


def release_drag():
    global dragging

    if dragging:
        pyautogui.mouseUp()
        dragging = False


try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp += 33
            result = detector.detect_for_video(
                mp_image, timestamp
            )

            gesture = "NO HAND"
            hand = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                gesture = recognize_gesture(
                    hand, pinch_active
                )

            is_pinch = gesture == "PINCH"
            now = time.monotonic()

            if hand is None:
                release_drag()
                previous_x = None
                previous_y = None
                scroll_anchor = None
                volume_anchor = None
                swipe_anchor = None

            elif not paused:

                if mode == "MOUSE":

                    if gesture == "ONE FINGER":
                        move_cursor(hand)
                    else:
                        previous_x = None
                        previous_y = None

                    if (
                        gesture == "PINCH"
                        and not pinch_active
                        and now - last_action > ACTION_COOLDOWN
                    ):
                        pyautogui.click()
                        last_action = now

                    if (
                        gesture == "THREE FINGERS"
                        and previous_gesture != "THREE FINGERS"
                        and now - last_action > ACTION_COOLDOWN
                    ):
                        pyautogui.click(button="right")
                        last_action = now

                    if gesture == "TWO FINGERS":
                        if scroll_anchor is None:
                            scroll_anchor = hand[8].y

                        delta = scroll_anchor - hand[8].y

                        if abs(delta) > 0.025:
                            amount = int(
                                delta * SCROLL_SENSITIVITY
                            )

                            if amount != 0:
                                pyautogui.scroll(amount)
                                scroll_anchor = hand[8].y
                    else:
                        scroll_anchor = None

                elif mode == "DRAG":

                    if gesture in ("ONE FINGER", "PINCH"):
                        move_cursor(hand)
                    else:
                        previous_x = None
                        previous_y = None

                    if is_pinch and not dragging:
                        pyautogui.mouseDown()
                        dragging = True

                    elif not is_pinch:
                        release_drag()

                elif mode == "VOLUME":

                    if gesture == "OPEN PALM":
                        if volume_anchor is None:
                            volume_anchor = hand[9].y

                        delta = volume_anchor - hand[9].y

                        if abs(delta) > 0.055:
                            if delta > 0:
                                pyautogui.press("volumeup")
                            else:
                                pyautogui.press("volumedown")

                            volume_anchor = hand[9].y
                    else:
                        volume_anchor = None

                elif mode == "PRESENTATION":

                    if gesture == "OPEN PALM":
                        if swipe_anchor is None:
                            swipe_anchor = hand[9].x

                        delta = hand[9].x - swipe_anchor

                        if (
                            abs(delta) > SWIPE_DISTANCE
                            and now - last_action > ACTION_COOLDOWN
                        ):
                            if delta > 0:
                                pyautogui.press("right")
                            else:
                                pyautogui.press("left")

                            last_action = now
                            swipe_anchor = None
                    else:
                        swipe_anchor = None

            else:
                release_drag()

            pinch_active = is_pinch
            previous_gesture = gesture

            # Draw landmarks
            if hand is not None:
                points = []

                for landmark in hand:
                    px = int(landmark.x * width)
                    py = int(landmark.y * height)

                    points.append((px, py))

                    cv2.circle(
                        frame, (px, py), 4,
                        (0, 255, 0), -1
                    )

                for start, end in CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 0), 2
                    )

            # Dashboard
            cv2.rectangle(
                frame, (0, 0), (width, 125),
                (25, 25, 25), -1
            )

            status = "PAUSED" if paused else "ACTIVE"

            cv2.putText(
                frame, f"GestureAI | {mode} | {status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            cv2.putText(
                frame, f"Gesture: {gesture}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.putText(
                frame, "M:Mouse V:Volume P:Slides D:Drag",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.putText(
                frame, "SPACE:Pause Q:Quit",
                (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            
            cv2.namedWindow("GestureAI Controller", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                 "GestureAI Controller",
               cv2.WND_PROP_FULLSCREEN,
               cv2.WINDOW_NORMAL
            )
            cv2.imshow("GestureAI Controller", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                paused = not paused
                release_drag()

            elif key in [
                ord("m"), ord("v"),
                ord("p"), ord("d")
            ]:
                release_drag()

                mode = {
                    ord("m"): "MOUSE",
                    ord("v"): "VOLUME",
                    ord("p"): "PRESENTATION",
                    ord("d"): "DRAG"
                }[key]

                previous_x = None
                previous_y = None
                scroll_anchor = None
                volume_anchor = None
                swipe_anchor = None

finally:
    release_drag()
    camera.release()
    cv2.destroyAllWindows()
