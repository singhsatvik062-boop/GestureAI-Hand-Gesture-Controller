# GestureAI — AI Hand Gesture Controller

GestureAI is a Python-based desktop application that uses computer vision and AI-powered hand tracking to control a computer through hand gestures without requiring a physical mouse.

## Features

- Real-time hand tracking using a webcam
- Mouse cursor movement using the index finger
- Left-click using a thumb–index pinch
- Right-click using three fingers
- Scrolling using two fingers
- System volume control using hand movements
- Presentation slide navigation using hand swipes
- Drag-and-drop functionality
- Dark-themed desktop dashboard with clickable controls
- Start, Pause, and Stop controls

## Technologies Used

- **Python 3.12**
- **MediaPipe** — Pretrained hand-landmark detection
- **OpenCV** — Webcam capture and image processing
- **PyAutoGUI** — Mouse and keyboard automation
- **CustomTkinter** — Desktop graphical interface
- **Pillow** — Image handling
- **NumPy** — Numerical operations

## Installation

Clone the repository:

```bash
git clone https://github.com/singhsatvik062-boop/GestureAI-Hand-Gesture-Controller.git
cd GestureAI-Hand-Gesture-Controller
```

Create and activate a virtual environment on Windows:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install mediapipe==0.10.33 opencv-python pyautogui customtkinter pillow
```

## Run the Application

Launch the desktop dashboard:

```bash
python dashboard.py
```

Alternatively, run the original controller:

```bash
python main.py
```

## Gesture Controls

| Gesture | Function |
|---|---|
| One finger | Move mouse cursor |
| Thumb + index pinch | Left-click |
| Two fingers | Scroll |
| Three fingers | Right-click |
| Open palm | Volume and presentation controls |
| Pinch and move | Drag and drop |

## How It Works

1. OpenCV captures live video from the webcam.
2. MediaPipe identifies 21 landmarks on the detected hand.
3. GestureAI analyzes fingertip positions and distances.
4. Rule-based logic identifies the user's gesture.
5. PyAutoGUI executes the corresponding computer action.
6. The dashboard displays the live video and control status.

## Project Information

**Project:** GestureAI — AI Hand Gesture Controller

**Category:** Artificial Intelligence / Computer Vision / Human–Computer Interaction

**Language:** Python

**Platform:** Windows

**Model:** Pretrained MediaPipe Hand Landmarker

## Future Scope

- Improved gesture recognition accuracy
- Customizable gesture shortcuts
- Support for additional operating systems
- Multi-hand gesture recognition
- More advanced accessibility features

## License

The project source code is provided for educational purposes. Third-party libraries and pretrained model assets remain subject to their respective licenses.
