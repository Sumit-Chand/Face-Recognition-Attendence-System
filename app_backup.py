import os
import csv
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for
)

from utils import (
    DATASET_DIR,
    CASCADE_PATH,
    SAMPLES_PER_STUDENT,
    TRAINER_FILE,
    ATTENDANCE_DIR,
    CONFIDENCE_THRESHOLD,
    add_student,
    load_students
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
app.secret_key = "face_attendance_secret_key_2026"


# ============================================================
# LABEL MAPPING
# Internal LBPH label -> Real Student ID
# ============================================================

LABEL_MAP_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trainer",
    "label_map.csv"
)


def load_label_map():
    """
    Loads:
        InternalLabel,StudentID

    Example:
        1,231190101060
    """

    mapping = {}

    if not os.path.exists(LABEL_MAP_FILE):
        return mapping

    try:
        with open(
            LABEL_MAP_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                internal_label = int(row["InternalLabel"])
                student_id = int(row["StudentID"])

                mapping[internal_label] = student_id

    except Exception as e:
        print("Error loading label map:", e)

    return mapping


# ============================================================
# ATTENDANCE FILE
# ============================================================

def get_today_attendance_file():

    today = datetime.now().strftime("%Y-%m-%d")

    filename = f"Attendance_{today}.csv"

    return os.path.join(
        ATTENDANCE_DIR,
        filename
    )


def load_today_attendance():

    attendance_file = get_today_attendance_file()

    marked = {}

    if not os.path.exists(attendance_file):
        return marked

    try:

        with open(
            attendance_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                try:
                    student_id = int(row["ID"])
                    marked[student_id] = {
                        "name": row["Name"],
                        "time": row["Time"]
                    }

                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print("Error reading attendance:", e)

    return marked


def mark_attendance(student_id, student_name):

    attendance_file = get_today_attendance_file()

    os.makedirs(
        ATTENDANCE_DIR,
        exist_ok=True
    )

    already_marked = load_today_attendance()

    # Do not mark same student twice
    if student_id in already_marked:

        return {
            "success": True,
            "already_marked": True,
            "time": already_marked[student_id]["time"]
        }

    current_time = datetime.now().strftime("%H:%M:%S")

    file_exists = os.path.exists(attendance_file)

    with open(
        attendance_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "ID",
                "Name",
                "Time"
            ])

        writer.writerow([
            student_id,
            student_name,
            current_time
        ])

    return {
        "success": True,
        "already_marked": False,
        "time": current_time
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    students = load_students()

    total_students = len(students)

    today_file = get_today_attendance_file()

    present_today = 0

    if os.path.exists(today_file):

        with open(
            today_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                present_today += 1

    absent_today = max(
        total_students - present_today,
        0
    )

    return render_template(
        "dashboard.html",
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today
    )
# ============================================================
# REGISTER STUDENT
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        student_id = request.form.get(
            "student_id",
            ""
        ).strip()

        name = request.form.get(
            "name",
            ""
        ).strip()

        if not student_id:

            return "Student ID is required."

        if not student_id.isdigit():

            return "Student ID must be numeric."

        student_id = int(student_id)

        if not name:

            return "Student name is required."

        students = load_students()

        if student_id in students:

            return (
                f"Student ID {student_id} "
                "already exists."
            )

        # Store temporarily in session.
        # Student is added to CSV only after
        # successful 60 face captures.

        session["student_id"] = student_id
        session["student_name"] = name
        session["capture_count"] = 0

        return redirect(
            url_for("capture_page")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# FACE CAPTURE PAGE
# ============================================================
# ============================================================
# ATTENDANCE PAGE
# ============================================================


@app.route("/capture-page")
def capture_page():

    student_id = session.get(
        "student_id"
    )

    student_name = session.get(
        "student_name"
    )

    if (
        student_id is None
        or student_name is None
    ):

        return redirect(
            url_for("register")
        )

    return render_template(
        "capture.html",
        student_id=student_id,
        student_name=student_name,
        samples_required=SAMPLES_PER_STUDENT
    )


# ============================================================
# CAPTURE FACE IMAGE
# ============================================================

@app.route(
    "/capture",
    methods=["POST"]
)
def capture():

    student_id = session.get(
        "student_id"
    )

    student_name = session.get(
        "student_name"
    )

    count = session.get(
        "capture_count",
        0
    )

    if (
        student_id is None
        or student_name is None
    ):

        return jsonify({
            "success": False,
            "message": "Registration session expired."
        })

    if count >= SAMPLES_PER_STUDENT:

        return jsonify({
            "success": True,
            "completed": True,
            "count": count,
            "message": "Face capture already completed."
        })

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "No image received."
        })

    image_bytes = request.files[
        "image"
    ].read()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            "success": False,
            "message": "Invalid image."
        })

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    # --------------------------------------------------------
    # FACE DETECTOR
    # --------------------------------------------------------

    face_detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )

    if len(faces) == 0:

        return jsonify({
            "success": False,
            "message":
                "No face detected. "
                "Please look at the camera."
        })

    if len(faces) > 1:

        return jsonify({
            "success": False,
            "message":
                "Multiple faces detected. "
                "Only one person should be visible."
        })

    # --------------------------------------------------------
    # CROP FACE
    # --------------------------------------------------------

    x, y, w, h = faces[0]

    face_img = gray[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    count += 1

    filename = (
        f"User.{student_id}.{count}.jpg"
    )

    filepath = os.path.join(
        DATASET_DIR,
        filename
    )

    cv2.imwrite(
        filepath,
        face_img
    )

    session["capture_count"] = count

    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    if count >= SAMPLES_PER_STUDENT:

        added = add_student(
            student_id,
            student_name
        )

        session.pop(
            "student_id",
            None
        )

        session.pop(
            "student_name",
            None
        )

        session.pop(
            "capture_count",
            None
        )

        return jsonify({
            "success": True,
            "completed": True,
            "count": count,
            "message":
                "Face capture completed successfully."
        })

    return jsonify({
        "success": True,
        "completed": False,
        "count": count,
        "message":
            f"Face sample {count} captured."
    })


# ============================================================
# ATTENDANCE PAGE
# ============================================================




# ============================================================
# FACE RECOGNITION
# ============================================================

@app.route(
    "/recognize",
    methods=["POST"]
)
def recognize():

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        TRAINER_FILE
    ):

        return jsonify({
            "success": False,
            "recognized": False,
            "message":
                "Trained model not found. "
                "Please train the model first."
        })

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "recognized": False,
            "message": "No camera image received."
        })

    image_bytes = request.files[
        "image"
    ].read()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            "success": False,
            "recognized": False,
            "message": "Invalid camera image."
        })

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    # --------------------------------------------------------
    # DETECT FACE
    # --------------------------------------------------------

    face_detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(80, 80)
    )

    if len(faces) == 0:

        return jsonify({
            "success": True,
            "recognized": False,
            "message":
                "No face detected."
        })

    if len(faces) > 1:

        return jsonify({
            "success": True,
            "recognized": False,
            "message":
                "Multiple faces detected. "
                "Only one person should be visible."
        })

    # --------------------------------------------------------
    # LOAD RECOGNIZER
    # --------------------------------------------------------

    try:

        recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        recognizer.read(
            TRAINER_FILE
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "recognized": False,
            "message":
                f"Could not load recognizer: {e}"
        })

    # --------------------------------------------------------
    # LOAD LABEL MAP + STUDENTS
    # --------------------------------------------------------

    label_map = load_label_map()

    students = load_students()

    if not label_map:

        return jsonify({
            "success": False,
            "recognized": False,
            "message":
                "Label mapping file not found."
        })

    # --------------------------------------------------------
    # FACE PREDICTION
    # --------------------------------------------------------

    x, y, w, h = faces[0]

    face_img = gray[
        y:y + h,
        x:x + w
    ]

    try:

        predicted_label, distance = (
            recognizer.predict(face_img)
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "recognized": False,
            "message":
                f"Recognition error: {e}"
        })

    predicted_label = int(
        predicted_label
    )

    distance = float(
        distance
    )

    # Internal label -> Real Student ID

    real_student_id = label_map.get(
        predicted_label
    )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if real_student_id is None:

        return jsonify({
            "success": True,
            "recognized": False,
            "distance": round(distance, 2),
            "message": "Unknown face."
        })

    # --------------------------------------------------------
    # CHECK CONFIDENCE
    # --------------------------------------------------------

    if distance > CONFIDENCE_THRESHOLD:

        return jsonify({
            "success": True,
            "recognized": False,
            "distance": round(distance, 2),
            "message":
                "Face not confidently recognized."
        })

    # --------------------------------------------------------
    # CHECK STUDENT
    # --------------------------------------------------------

    if real_student_id not in students:

        return jsonify({
            "success": True,
            "recognized": False,
            "distance": round(distance, 2),
            "message":
                "Recognized ID is not registered."
        })

    student_name = students[
        real_student_id
    ]

    # --------------------------------------------------------
    # MARK ATTENDANCE
    # --------------------------------------------------------

    attendance_result = mark_attendance(
        real_student_id,
        student_name
    )

    return jsonify({
        "success": True,
        "recognized": True,
        "student_id": real_student_id,
        "name": student_name,
        "distance": round(distance, 2),
        "already_marked":
            attendance_result["already_marked"],
        "time":
            attendance_result["time"],
        "message":
            "Attendance already marked."
            if attendance_result["already_marked"]
            else "Attendance marked successfully."
    })

# ============================================================
# ATTENDANCE PAGE
# ============================================================


@app.route("/attendance")
def attendance_page():

attendance = load_today_attendance()

attendance_data = []

for student_id, data in attendance.items():
    attendance_data.append({
        "id": student_id,
        "name": data.get("name", ""),
        "time": data.get("time", "")
    })

print("ATTENDANCE PAGE DATA:", attendance_data)

return render_template(
    "attendance.html",
    attendance_data=attendance_data
)
```


# ============================================================
# ATTENDANCE HISTORY
# ============================================================

@app.route("/attendance-history")
def attendance_history():

    history_data = []

    if os.path.exists(ATTENDANCE_DIR):

        files = sorted(
            os.listdir(ATTENDANCE_DIR),
            reverse=True
        )

        for filename in files:

            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(
                ATTENDANCE_DIR,
                filename
            )

            date = filename.replace(
                "Attendance_",
                ""
            ).replace(
                ".csv",
                ""
            )

            with open(
                file_path,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    history_data.append({
                        "date": date,
                        "id": row.get("ID", ""),
                        "name": row.get("Name", ""),
                        "time": row.get("Time", "")
                    })

    return render_template(
        "attendance_history.html",
        history_data=history_data
    )

# ============================================================
# FACE RECOGNITION PAGE
# ============================================================

@app.route("/recognize-page")
def recognize_page():
    return render_template("recognize.html")
# ============================================================
# ATTENDANCE REPORTS
# ============================================================

@app.route("/attendance-reports")
def attendance_reports():

    students = load_students()

    report_data = {}

    # Create report entry for every registered student
    for student_id, student_name in students.items():

        report_data[str(student_id)] = {
            "id": student_id,
            "name": student_name,
            "present": 0,
            "absent": 0,
            "percentage": 0
        }

    total_days = 0

    # Read all attendance CSV files
    if os.path.exists(ATTENDANCE_DIR):

        files = sorted(
            os.listdir(ATTENDANCE_DIR)
        )

        for filename in files:

            if not filename.startswith("Attendance_"):
                continue

            if not filename.endswith(".csv"):
                continue

            total_days += 1

            file_path = os.path.join(
                ATTENDANCE_DIR,
                filename
            )

            present_ids = set()

            with open(
                file_path,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    student_id = row.get(
                        "ID",
                        ""
                    ).strip()

                    if student_id:
                        present_ids.add(student_id)

            # Calculate present / absent
            for student_id in students:

                student_key = str(student_id)

                if student_key in present_ids:

                    report_data[student_key]["present"] += 1

                else:

                    report_data[student_key]["absent"] += 1

    # Calculate percentage
    for student_id in report_data:

        present = report_data[student_id]["present"]

        if total_days > 0:

            report_data[student_id]["percentage"] = round(
                (present / total_days) * 100,
                2
            )

        else:

            report_data[student_id]["percentage"] = 0

    return render_template(
        "attendance_reports.html",
        reports=report_data.values(),
        total_days=total_days
    )
# ============================================================
# STUDENT MANAGEMENT
# ============================================================

@app.route("/students")
def students_page():

    students = load_students()

    return render_template(
        "students.html",
        students=students
    )

# ============================================================
# DELETE STUDENT
# ============================================================

@app.route("/delete-student", methods=["POST"])
def delete_student():

    student_id = request.form.get("student_id", "").strip()

    if not student_id:
        return "Student ID is required."

    if not student_id.isdigit():
        return "Invalid Student ID."

    student_id = int(student_id)

    students = load_students()

    if student_id not in students:
        return "Student not found."

    # --------------------------------------------------------
    # 1. Remove student from students.csv
    # --------------------------------------------------------

    rows = []

    if os.path.exists("students.csv"):

        with open(
            "students.csv",
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row.get("ID", "").strip() != str(student_id):
                    rows.append(row)

    with open(
        "students.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["ID", "Name"]
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # 2. Delete this student's face dataset
    # --------------------------------------------------------

    deleted_images = 0

    if os.path.exists(DATASET_DIR):

        for filename in os.listdir(DATASET_DIR):

            if filename.startswith(f"User.{student_id}."):

                file_path = os.path.join(
                    DATASET_DIR,
                    filename
                )

                if os.path.isfile(file_path):

                    os.remove(file_path)
                    deleted_images += 1

    # --------------------------------------------------------
    # 3. Retrain face recognition model
    # --------------------------------------------------------

    remaining_images = []

    if os.path.exists(DATASET_DIR):

        for filename in os.listdir(DATASET_DIR):

            if filename.lower().endswith(".jpg"):
                remaining_images.append(filename)

    if remaining_images:

        try:

            subprocess.run(
                [sys.executable, "train_model.py"],
                check=True
            )

        except subprocess.CalledProcessError:

            return (
                "Student deleted, but model retraining failed. "
                "Please run train_model.py manually."
            )

    else:

        # No students/dataset left.
        # Remove old model files.

        trainer_file = os.path.join(
            "trainer",
            "trainer.yml"
        )

        label_map_file = os.path.join(
            "trainer",
            "label_map.csv"
        )

        if os.path.exists(trainer_file):
            os.remove(trainer_file)

        if os.path.exists(label_map_file):
            os.remove(label_map_file)

    # --------------------------------------------------------
    # 4. Go back to Student Management
    # --------------------------------------------------------

    return redirect(url_for("students_page"))
# ============================================================
# STUDENT MANAGEMENT
# ============================================================


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       FACE ATTENDANCE WEB APPLICATION")
    print("=" * 60)
    print()
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )