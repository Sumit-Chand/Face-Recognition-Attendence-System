# Face Recognition Attendance System

A college mini-project that automatically marks student attendance using
face recognition, built with **Python + OpenCV**.

## How it works

1. **Register** — capture ~60 photos of each student's face from the webcam.
2. **Train** — an LBPH (Local Binary Patterns Histogram) recognizer learns
   to tell the registered faces apart.
3. **Attendance** — the webcam feed is scanned continuously; every
   recognized student is logged **once per day** with a timestamp, into a
   CSV file. Unrecognized faces are labelled "Unknown" and ignored.

LBPH was chosen (over a deep-learning library like `face_recognition`/dlib)
because it ships inside `opencv-contrib-python` — no extra compilers or
large model downloads, so it installs reliably on any lab PC.

## Project structure

```
face_attendance_system/
├── main.py                # menu that ties everything together
├── register_student.py    # Step 1: capture face samples
├── train_model.py         # Step 2: train the recognizer
├── mark_attendance.py     # Step 3: recognize + log attendance
├── utils.py                # shared paths/constants
├── students.csv            # ID -> Name (created automatically)
├── requirements.txt
├── dataset/                 # captured face images (auto-created)
├── trainer/                 # trainer.yml, the trained model (auto-created)
└── attendance/               # Attendance_YYYY-MM-DD.csv files (auto-created)
```

## Setup

```bash
pip install -r requirements.txt
```

> **Important:** install `opencv-contrib-python`, not plain `opencv-python`.
> Only the "contrib" build includes `cv2.face`, which this project needs
> for the LBPH recognizer. If both are installed at once they can
> conflict — run `pip uninstall opencv-python opencv-contrib-python` and
> reinstall just the contrib version if you hit an `AttributeError: module
> 'cv2' has no attribute 'face'`.

## Running it

```bash
python main.py
```

Then, for each student:
1. Choose **1** to register them (look at the camera, press `q` to stop early).
2. Choose **2** to (re)train the model — do this after every new registration.
3. Choose **3** to start live attendance. A green box + name means marked;
   red "Unknown" means the face isn't recognized.
4. Choose **4** any time to print today's attendance sheet.

Each script (`register_student.py`, `train_model.py`, `mark_attendance.py`)
also runs standalone if you don't want the menu.

## Tuning

- `SAMPLES_PER_STUDENT` in `utils.py` — more samples (different angles,
  lighting) generally improve accuracy.
- `CONFIDENCE_THRESHOLD` in `utils.py` — LBPH returns a *distance*, so
  **lower is a better match**. If real students are shown as "Unknown",
  raise this value; if strangers get matched to a student, lower it.

## Troubleshooting

- **Webcam won't open on Windows**: try
  `cv2.VideoCapture(0, cv2.CAP_DSHOW)` in place of `cv2.VideoCapture(0)`.
- **Poor recognition**: register in good, even lighting, face the camera
  directly, and capture samples with slight head turns.
- **Multiple faces per person needed**: the system already stores many
  samples per student for exactly this reason.

## Possible extensions (good talking points for a viva/report)

- Swap LBPH for a deep-learning embedding model (e.g. `face_recognition`/
  dlib or a FaceNet-based model) for higher accuracy at scale.
- Add a Tkinter/PyQt GUI instead of the console menu.
- Push attendance rows to a database (SQLite/MySQL) instead of CSV.
- Add liveness detection (blink/motion check) to prevent spoofing with a photo.
- Email/SMS a daily attendance summary automatically.
