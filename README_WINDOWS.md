# AI Driver Safety — Windows Version

## Requirements
- Windows 10 or Windows 11
- Python 3.11 recommended
- Built-in or USB webcam
- Internet connection for the first dependency installation

## 1. Install Python
Install Python 3.11 from:
https://www.python.org/downloads/

During installation, enable **Add Python to PATH**.

## 2. Setup
Double-click:

`setup_windows.bat`

Wait for it to finish.

## 3. Test the installation
Double-click:

`test_windows.bat`

It should report versions for OpenCV, MediaPipe, Torch and Streamlit, plus whether the camera is available.

## 4. Run the driver-safety application
Double-click:

`run_windows.bat`

A webcam window should open.

Press **Q** in the camera window to stop.

## 5. Run the dashboard
Double-click:

`dashboard_windows.bat`

The Streamlit dashboard will open in your browser.

## Important
This is an academic/research prototype. The included GRU checkpoint is a bootstrap model trained on generated feature sequences. It is not a validated real-world driver-drowsiness model. For final research claims, train/evaluate with real labeled driver data and subject-independent test splits.

## Troubleshooting
- If Windows blocks camera access: Settings → Privacy & security → Camera → enable camera access and allow desktop apps.
- If the camera is unavailable, close Teams/Zoom/Camera/OBS and run `test_windows.bat` again.
- If installation fails, use Python 3.11 rather than a newer Python release.
