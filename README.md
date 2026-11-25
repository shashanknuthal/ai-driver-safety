# AI Driver Safety

A research prototype for webcam based driver drowsiness and distraction cues. The application extracts facial landmarks, applies configurable rules to estimate risk, displays visual/audio alerts, and records events locally. An optional temporal GRU produces an auxiliary prediction after it has collected a feature sequence.

> **Safety notice:** This is an academic prototype, not a certified driver monitoring or safety system. Do not use it to make driving decisions or claim real-world accuracy. Webcam lighting, camera placement, glasses, face orientation, and individual differences can affect the output.

## Current capabilities

- Webcam capture with face landmark processing
- Eye aspect ratio (EAR), mouth aspect ratio (MAR), and head pose features
- Heuristic drowsiness and distraction scores with configurable thresholds
- Optional temporal GRU prediction (reported separately from the heuristic score)
- Visual and audio alerts
- Local SQLite event logging
- Streamlit event dashboard
- Feature collection and a training script for temporal models
- Windows setup, run, dashboard, and environment-check batch files

## How it works

The live application runs `main.py`. Face features from `features/face_processor.py` feed into `features/risk_engine.py`, which computes the displayed risk status with configurable heuristics. When PyTorch and the bundled checkpoint are available, the risk engine also requests a prediction from `models/runtime_model.py`. That prediction is auxiliary and does not currently calibrate or replace the heuristic score.

## Quick start

Python 3.10 or 3.11 is recommended. A working webcam is needed for live capture.

### Windows

1. Install Python 3.11 and enable **Add Python to PATH**.
2. Run `setup_windows.bat`.
3. Run `test_windows.bat` to check installed packages and camera access.
4. Run `run_windows.bat`. Press **Q** in the camera window to quit.
5. Run `dashboard_windows.bat` to open the event dashboard.

See [README_WINDOWS.md](README_WINDOWS.md) for Windows setup notes.

### macOS or Linux

```bash
python3.11 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Optional: install PyTorch to enable the bundled GRU inference path
python -m pip install torch
python main.py
```

In another activated terminal, start the dashboard:

```bash
streamlit run dashboard/app.py
```

Press `q` in the webcam window to stop the live application. Camera permissions may need to be enabled in your operating system settings.

## Configuration

Edit [`config.yaml`](config.yaml) to adjust the camera index and resolution, EAR/MAR and head-pose thresholds, risk levels, alert options, and SQLite database path. Changes to thresholds should be validated on representative, consented data before drawing conclusions.

## Model and evaluation status

The repository includes a small BiGRU checkpoint and feature scaler so the optional inference path can run. The checkpoint was trained on generated feature sequences. The reported `1.0000` bootstrap test accuracy measures performance on held-out generated sequences only; it is **not** a benchmark of real drivers, and must not be presented as real-world accuracy.

For meaningful evaluation, collect or use appropriately licensed real labeled data, split by participant (not by adjacent frames), and report per-class precision, recall, F1, confusion matrix, latency, and frame rate. Record the dataset source, consent/privacy conditions, preprocessing, and split procedure. Training notes are in [`training/README.md`](training/README.md).

## Privacy and data

Camera frames are processed locally by this prototype and are not sent to a remote service. Event records are written to the configured local SQLite database. If you use `collect_data.py`, follow applicable consent, privacy, and dataset licensing requirements. Keep personal recordings and derived private data out of Git; the `.gitignore` excludes common local data paths.

## Repository layout

```text
ai-driver-safety/
├── alerts/                 # Alert handling
├── dashboard/              # Streamlit event dashboard
├── database/               # Local SQLite persistence
├── features/               # Face features and heuristic risk scoring
├── models/                 # Optional GRU inference code and bootstrap files
├── tests/                  # Feature tests
├── training/               # Temporal model training workflow
├── utils/                  # FPS helper
├── collect_data.py         # Feature sequence collection
├── config.yaml             # Runtime configuration
├── main.py                 # Webcam application entry point
├── requirements.txt        # General Python dependencies
└── requirements-windows.txt # Windows-specific dependency pins
```

## Screenshots and demo

For a portfolio README, capture a screenshot of the application using sample/demo conditions with no identifying personal data. Save it in `assets/` and embed it here:

```markdown
![Driver Safety application](assets/driver-safety-demo.png)
```

A live hosted demo is not currently provided because this application requires local webcam access. A short screen recording can be added if it uses consented, non-sensitive footage.

## Known limitations and next steps

- The heuristic risk score uses fixed thresholds and is not clinically or operationally validated.
- The GRU checkpoint uses generated data and has no demonstrated real-driver accuracy.
- The current interface is a local webcam window plus a separate event dashboard; it is not a production deployment.
- Add real-data, participant-independent evaluation before making model performance claims.
- Add tests for alerting, persistence, and model inference; document platform-specific setup and accessibility behavior.
- Consider an explicit camera indicator and configurable retention/deletion controls before any real-world study.

## Technology

Python · OpenCV · MediaPipe · NumPy · PyTorch · scikit-learn · Streamlit · SQLite
