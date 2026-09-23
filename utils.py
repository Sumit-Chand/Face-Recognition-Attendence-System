"""
utils.py
--------
Shared paths, constants and small helpers used by every script in this
project. Keeping them in one place means register_student.py,
train_model.py and mark_attendance.py all agree on where things live.
"""
import os
import csv
import cv2

# --- Folder / file locations (all relative to this file, so the project
#     works no matter where it's copied) ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
STUDENTS_CSV = os.path.join(BASE_DIR, "students.csv")
STUDENT_CREDENTIALS_FILE = os.path.join(BASE_DIR, "student_credentials.csv")

# Haar Cascade face detector. Bundled directly in this project (instead of
# reading from cv2.data.haarcascades) because some OpenCV builds don't ship
# that data file at the expected path, which causes a confusing
# "cv::FileStorage::Impl::open Can't open file" error at runtime.
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

if not os.path.exists(CASCADE_PATH):
    # Fallback: try the path bundled inside the installed cv2 package.
    _fallback = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(_fallback):
        CASCADE_PATH = _fallback
    else:
        raise FileNotFoundError(
            "Could not find haarcascade_frontalface_default.xml. "
            "Make sure it sits next to utils.py in the project folder."
        )

# --- Tunables -------------------------------------------------------------
SAMPLES_PER_STUDENT = 60   # how many face images to capture per student
CONFIDENCE_THRESHOLD = 70  # LBPH DISTANCE (lower = better match). Tune this:
                           # too low -> real students get marked "Unknown"
                           # too high -> strangers get matched to a student

for _dir in (DATASET_DIR, TRAINER_DIR, ATTENDANCE_DIR):
    os.makedirs(_dir, exist_ok=True)


def load_students():
    """Return {id: name} from students.csv. Creates the file if missing."""
    return {sid: data["name"] for sid, data in load_student_records().items()}


def load_student_records():
    """Return {id: {name, department, year, section, subject}} for all students."""
    students = {}
    if not os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
            f.write("ID,Name,Department,Year,Section,Subject\n")
        return students

    with open(STUDENTS_CSV, "r", newline="", encoding="utf-8") as f:
        try:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if fieldnames and "Department" not in fieldnames:
                rows = list(reader)
                normalized_rows = []
                for row in rows:
                    sid = (row.get("ID") or "").strip()
                    name = (row.get("Name") or "").strip()
                    if not sid or not name:
                        continue
                    normalized_rows.append({
                        "ID": sid,
                        "Name": name,
                        "Department": "",
                        "Year": "",
                        "Section": "",
                        "Subject": ""
                    })
                with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=["ID", "Name", "Department", "Year", "Section", "Subject"])
                    writer.writeheader()
                    writer.writerows(normalized_rows)
                return load_student_records()

            for row in reader:
                sid = (row.get("ID") or "").strip()
                name = (row.get("Name") or "").strip()
                if not sid or not name:
                    continue
                try:
                    student_id = int(sid)
                except ValueError:
                    continue

                students[student_id] = {
                    "name": name,
                    "department": (row.get("Department") or "").strip(),
                    "year": (row.get("Year") or "").strip(),
                    "section": (row.get("Section") or "").strip(),
                    "subject": (row.get("Subject") or "").strip()
                }
        except Exception:
            with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
                f.write("ID,Name,Department,Year,Section,Subject\n")
    return students


def load_student_credentials():
    """Return {student_id: password} for student login support."""
    credentials = {}

    if not os.path.exists(STUDENT_CREDENTIALS_FILE):
        return credentials

    with open(STUDENT_CREDENTIALS_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            student_id = (row.get("ID") or "").strip()
            password = (row.get("Password") or "").strip()

            if not student_id or not password:
                continue

            try:
                credentials[int(student_id)] = password
            except ValueError:
                continue

    return credentials


def save_student_password(student_id, password):
    """Store a student password for login protection."""
    credentials = load_student_credentials()
    credentials[int(student_id)] = password.strip()

    with open(STUDENT_CREDENTIALS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Password"])
        for sid, pwd in sorted(credentials.items()):
            writer.writerow([sid, pwd])


def add_student(student_id, name, password=None, department="", year="", section="", subject=""):
    """Append a new student to students.csv. Returns False if ID exists."""
    students = load_students()
    if student_id in students:
        return False

    file_exists = os.path.exists(STUDENTS_CSV)
    if not file_exists:
        with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
            f.write("ID,Name,Department,Year,Section,Subject\n")

    with open(STUDENTS_CSV, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows or rows[0] != ["ID", "Name", "Department", "Year", "Section", "Subject"]:
        normalized = []
        for row in rows[1:] if len(rows) > 1 else []:
            if not row:
                continue
            if len(row) < 2:
                continue
            normalized.append([row[0], row[1], "", "", "", ""])
        with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Department", "Year", "Section", "Subject"])
            writer.writerows(normalized)

    with open(STUDENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(student_id), name, department, year, section, subject])

    if password is not None:
        save_student_password(student_id, password)

    return True
