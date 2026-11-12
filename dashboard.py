import math
import time
from pathlib import Path
import tkinter as tk

import cv2
import mediapipe as mp
import pyautogui
import customtkinter as ctk
from PIL import Image

MODEL = Path(__file__).with_name('hand_landmarker.task')
if not MODEL.exists():
    raise FileNotFoundError(f'Missing model: {MODEL}')

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('blue')

class GestureDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('GestureAI | Hand Gesture Controller')
        self.geometry('1120x740')
        self.minsize(850, 590)
        self.configure(fg_color='#0B1220')
        self.protocol('WM_DELETE_WINDOW', self.close)

        self.mode = 'MOUSE'
        self.paused = True  # start safely
        self.dragging = False
        self.pinch_active = False
        self.previous_gesture = ''
        self.previous_x = None
        self.previous_y = None
        self.scroll_anchor = None
        self.volume_anchor = None
        self.swipe_anchor = None
        self.last_action = 0.0
        self.timestamp = 0
        self.last_frame_time = time.monotonic()
        self.camera = None
        self.detector = None
        self.running = True
        self.preview_image = None
        self.last_hand_time = 0.0

        self.build_ui()
        self.after(150, self.initialize_camera)

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color='transparent')
        header.pack(fill='x', padx=24, pady=(20, 14))
        ctk.CTkLabel(header, text='GestureAI', font=ctk.CTkFont(size=30, weight='bold'), text_color='#F8FAFC').pack(side='left')
        ctk.CTkLabel(header, text='  AI Hand Gesture Controller', font=ctk.CTkFont(size=14), text_color='#94A3B8').pack(side='left', pady=8)
        self.status_label = ctk.CTkLabel(header, text='● PAUSED', text_color='#FBBF24', font=ctk.CTkFont(size=15, weight='bold'))
        self.status_label.pack(side='right')

        main = ctk.CTkFrame(self, fg_color='transparent')
        main.pack(fill='both', expand=True, padx=24, pady=(0, 22))
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, fg_color='#172236', corner_radius=18)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        ctk.CTkLabel(left, text='LIVE CAMERA', text_color='#C7D2E4', font=ctk.CTkFont(size=14, weight='bold')).pack(anchor='w', padx=20, pady=(18, 8))
        self.video = ctk.CTkLabel(left, text='Starting camera…', text_color='#94A3B8', fg_color='#0B1220', corner_radius=12)
        self.video.pack(fill='both', expand=True, padx=18, pady=(0, 14))
        self.camera_details = ctk.CTkLabel(left, text='Camera initializing', text_color='#94A3B8', font=ctk.CTkFont(size=12))
        self.camera_details.pack(anchor='w', padx=20, pady=(0, 16))

        right = ctk.CTkFrame(main, fg_color='#172236', corner_radius=18)
        right.grid(row=0, column=1, sticky='nsew')
        ctk.CTkLabel(right, text='CONTROL MODES', text_color='#C7D2E4', font=ctk.CTkFont(size=14, weight='bold')).pack(anchor='w', padx=20, pady=(18, 12))
        buttons = ctk.CTkFrame(right, fg_color='transparent')
        buttons.pack(fill='x', padx=18)
        buttons.grid_columnconfigure((0, 1), weight=1)
        self.mode_buttons = {}
        for i, (mode, title) in enumerate([('MOUSE','Mouse'), ('VOLUME','Volume'), ('PRESENTATION','Slides'), ('DRAG','Drag')]):
            btn = ctk.CTkButton(buttons, text=title, height=62, corner_radius=12,
                                command=lambda m=mode: self.set_mode(m), font=ctk.CTkFont(size=16, weight='bold'))
            btn.grid(row=i//2, column=i%2, padx=5, pady=5, sticky='ew')
            self.mode_buttons[mode] = btn
        self.update_mode_buttons()

        self.gesture_label = ctk.CTkLabel(right, text='Gesture: NO HAND', text_color='#E2E8F0', font=ctk.CTkFont(size=19, weight='bold'))
        self.gesture_label.pack(anchor='w', padx=22, pady=(22, 5))
        self.mode_hint = ctk.CTkLabel(right, text='', text_color='#94A3B8', wraplength=300, justify='left', font=ctk.CTkFont(size=13))
        self.mode_hint.pack(anchor='w', padx=22, pady=(0, 16))
        self.update_hint()

        controls = ctk.CTkFrame(right, fg_color='transparent')
        controls.pack(fill='x', padx=20, pady=(0, 14))
        self.toggle_button = ctk.CTkButton(controls, text='Start controls', height=44, fg_color='#16A34A', hover_color='#15803D', command=self.toggle_pause)
        self.toggle_button.pack(side='left', expand=True, fill='x', padx=(0, 6))
        ctk.CTkButton(controls, text='STOP', height=44, fg_color='#B91C1C', hover_color='#991B1B', command=self.emergency_stop).pack(side='left', expand=True, fill='x', padx=(6, 0))

        ctk.CTkLabel(right, text='Start paused • Esc = emergency stop\nClose window = exit • Corner failsafe enabled',
                     text_color='#94A3B8', justify='left', font=ctk.CTkFont(size=12)).pack(anchor='w', padx=22, pady=(0, 16))
        self.bind_all('<Escape>', lambda event: self.emergency_stop())

    def update_mode_buttons(self):
        for mode, btn in self.mode_buttons.items():
            btn.configure(fg_color='#2563EB' if mode == self.mode else '#26354C',
                          hover_color='#1D4ED8' if mode == self.mode else '#334155')

    def update_hint(self):
        hints = {
            'MOUSE': 'Index: move cursor  •  Pinch: left click\nTwo fingers: scroll  •  Three fingers: right click',
            'VOLUME': 'Show an open palm, then move your hand up or down.',
            'PRESENTATION': 'Show an open palm and swipe horizontally.\nYour slideshow must have keyboard focus.',
            'DRAG': 'Move with index finger. Pinch to hold; move while pinching; release to drop.'
        }
        self.mode_hint.configure(text=hints[self.mode])

    def set_mode(self, mode):
        self.release_drag()
        self.mode = mode
        self.reset_motion()
        self.pinch_active = False
        self.previous_gesture = ''
        self.update_mode_buttons()
        self.update_hint()

    def reset_motion(self):
        self.previous_x = self.previous_y = None
        self.scroll_anchor = self.volume_anchor = self.swipe_anchor = None

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.release_drag()
            self.reset_motion()
        self.refresh_status()

    def emergency_stop(self):
        self.paused = True
        self.release_drag()
        self.reset_motion()
        self.refresh_status()

    def refresh_status(self):
        self.status_label.configure(text='● PAUSED' if self.paused else '● ACTIVE',
                                    text_color='#FBBF24' if self.paused else '#4ADE80')
        self.toggle_button.configure(text='Start controls' if self.paused else 'Pause controls',
                                     fg_color='#16A34A' if self.paused else '#D97706',
                                     hover_color='#15803D' if self.paused else '#B45309')

    def release_drag(self):
        if self.dragging:
            try:
                pyautogui.mouseUp()
            finally:
                self.dragging = False

    def initialize_camera(self):
        try:
            opts = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1
            )
            self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError('Could not open webcam. Close other camera applications.')
            self.camera_details.configure(text='Camera connected • Hand tracking ready')
            self.update_frame()
        except Exception as exc:
            self.camera_details.configure(text=f'Startup error: {exc}', text_color='#FCA5A5')
            self.video.configure(text='Camera could not start')

    @staticmethod
    def gesture_for(hand, was_pinching):
        fingers = [hand[tip].y < hand[pip].y for tip, pip in [(8,6),(12,10),(16,14),(20,18)]]
        dist = math.hypot(hand[4].x-hand[8].x, hand[4].y-hand[8].y)
        size = math.hypot(hand[0].x-hand[9].x, hand[0].y-hand[9].y)
        threshold = 0.48 if was_pinching else 0.35
        if size > 0 and dist < size*threshold:
            return 'PINCH'
        if sum(fingers) == 0:
            return 'FIST'
        if sum(fingers) == 4:
            return 'OPEN PALM'
        if fingers == [True,False,False,False]:
            return 'ONE FINGER'
        if fingers == [True,True,False,False]:
            return 'TWO FINGERS'
        if fingers == [True,True,True,False]:
            return 'THREE FINGERS'
        return 'OTHER'

    def move_cursor(self, hand):
        margin = 0.12
        x = max(0.02, min(0.98, (hand[8].x-margin)/(1-2*margin)))
        y = max(0.02, min(0.98, (hand[8].y-margin)/(1-2*margin)))
        target_x, target_y = int(x*(SCREEN_W-1)), int(y*(SCREEN_H-1))
        if self.previous_x is None:
            self.previous_x, self.previous_y = target_x, target_y
        self.previous_x = int(self.previous_x*0.65 + target_x*0.35)
        self.previous_y = int(self.previous_y*0.65 + target_y*0.35)
        pyautogui.moveTo(self.previous_x, self.previous_y)

    def apply_controls(self, hand, gesture):
        now = time.monotonic()
        is_pinch = gesture == 'PINCH'
        if self.paused:
            self.pinch_active = is_pinch
            self.previous_gesture = gesture
            return

        if self.mode == 'MOUSE':
            if gesture == 'ONE FINGER':
                self.move_cursor(hand)
            else:
                self.previous_x = self.previous_y = None
            if is_pinch and not self.pinch_active and now-self.last_action > 0.7:
                pyautogui.click()
                self.last_action = now
            if gesture == 'THREE FINGERS' and self.previous_gesture != gesture and now-self.last_action > 0.7:
                pyautogui.click(button='right')
                self.last_action = now
            if gesture == 'TWO FINGERS':
                if self.scroll_anchor is None:
                    self.scroll_anchor = hand[8].y
                delta = self.scroll_anchor-hand[8].y
                if abs(delta) > 0.025:
                    amount = int(delta*180)
                    if amount:
                        pyautogui.scroll(amount)
                        self.scroll_anchor = hand[8].y
            else:
                self.scroll_anchor = None

        elif self.mode == 'DRAG':
            if gesture in ('ONE FINGER','PINCH'):
                self.move_cursor(hand)
            else:
                self.previous_x = self.previous_y = None
            if is_pinch and not self.dragging:
                pyautogui.mouseDown()
                self.dragging = True
            elif not is_pinch:
                self.release_drag()

        elif self.mode == 'VOLUME':
            if gesture == 'OPEN PALM':
                if self.volume_anchor is None:
                    self.volume_anchor = hand[9].y
                delta = self.volume_anchor-hand[9].y
                if abs(delta) > 0.055:
                    pyautogui.press('volumeup' if delta > 0 else 'volumedown')
                    self.volume_anchor = hand[9].y
            else:
                self.volume_anchor = None

        elif self.mode == 'PRESENTATION':
            if gesture == 'OPEN PALM':
                if self.swipe_anchor is None:
                    self.swipe_anchor = hand[9].x
                delta = hand[9].x-self.swipe_anchor
                if abs(delta) > 0.15 and now-self.last_action > 0.7:
                    pyautogui.press('right' if delta > 0 else 'left')
                    self.last_action = now
                    self.swipe_anchor = None
            else:
                self.swipe_anchor = None

        self.pinch_active = is_pinch
        self.previous_gesture = gesture

    def update_frame(self):
        if not self.running:
            return
        try:
            ok, frame = self.camera.read()
            if not ok:
                self.camera_details.configure(text='Camera frame unavailable', text_color='#FCA5A5')
                self.after(80, self.update_frame)
                return
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.timestamp += 33
            result = self.detector.detect_for_video(mp_image, self.timestamp)
            gesture = 'NO HAND'
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                gesture = self.gesture_for(hand, self.pinch_active)
                self.apply_controls(hand, gesture)
                pts = [(int(p.x*w), int(p.y*h)) for p in hand]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (100, 185, 255), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (80, 255, 150), -1)
            else:
                self.release_drag()
                self.reset_motion()
                self.pinch_active = False
                self.previous_gesture = ''

            self.gesture_label.configure(text=f'Gesture: {gesture}')
            cv2.rectangle(frame, (0, 0), (w, 52), (11, 18, 32), -1)
            cv2.putText(frame, f'{self.mode}  |  {"PAUSED" if self.paused else "ACTIVE"}  |  {gesture}',
                        (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            preview_w = max(self.video.winfo_width()-12, 320)
            preview_h = max(self.video.winfo_height()-12, 220)
            scale = min(preview_w/w, preview_h/h)
            new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
            frame = cv2.resize(frame, (new_w, new_h))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.preview_image = ctk.CTkImage(light_image=image, dark_image=image, size=(new_w, new_h))
            self.video.configure(image=self.preview_image, text='')
        except pyautogui.FailSafeException:
            self.emergency_stop()
            self.camera_details.configure(text='Failsafe triggered — controls paused', text_color='#FBBF24')
        except Exception as exc:
            self.emergency_stop()
            self.camera_details.configure(text=f'Runtime error: {exc}', text_color='#FCA5A5')
        if self.running:
            self.after(20, self.update_frame)

    def close(self):
        self.running = False
        self.emergency_stop()
        if self.camera is not None:
            self.camera.release()
        if self.detector is not None:
            self.detector.close()
        self.destroy()

if __name__ == '__main__':
    GestureDashboard().mainloop()
