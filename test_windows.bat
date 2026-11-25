\
@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -c "import cv2, mediapipe as mp, torch, streamlit; print('OpenCV:', cv2.__version__); print('MediaPipe:', mp.__version__); print('Torch:', torch.__version__); print('Streamlit:', streamlit.__version__); print('FaceMesh API:', hasattr(mp, 'solutions')); c=cv2.VideoCapture(0); print('Camera available:', c.isOpened()); c.release()"
pause
