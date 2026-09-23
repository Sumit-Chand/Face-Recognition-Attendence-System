import os
import csv
import subprocess
import sys
from datetime import datetime
from functools import wraps

import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash
)

from utils import (
    DATASET_DIR,
    CASCADE_PATH,
    SAMPLES_PER_STUDENT,
    TRAINER_FILE,
    ATTENDANCE_DIR,
    CONFIDENCE_THRESHOLD,
    add_student,
    load_students,
    load_student_records,
    load_student_credentials,
    save_student_password
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
app.secret_key = "face_attendance_secret_key_2026"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_PASSWORDS_FILE = os.path.join(BASE_DIR, "student_credentials.csv")
TEACHER_CSV_PATHS = (
    os.path.join(BASE_DIR, "teacher.csv"),
    os.path.join(BASE_DIR, "teachers.csv")
)


def get_teacher_file_path():
    """Return the preferred CSV for teacher accounts."""
    for teacher_file in TEACHER_CSV_PATHS:
        if os.path.exists(teacher_file):
            return teacher_file
    return os.path.join(BASE_DIR, "teachers.csv")


def load_teachers():
    """Load teacher accounts from teacher.csv or teachers.csv."""
    teachers = {}

    for teacher_file in TEACHER_CSV_PATHS:
        if not os.path.exists(teacher_file):
            continue

        try:
            with open(teacher_file, "r", newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
        except Exception as exc:
            print(f"Error reading teacher file {teacher_file}: {exc}")
            continue

        if not rows:
            continue

        header = [cell.strip().lower() for cell in rows[0]]
        data_rows = rows[1:] if header and set(header) >= {"username", "password"} else rows

        for row in data_rows:
            if not row or len(row) < 2:
                continue

            username = (row[0] if len(row) > 0 else "").strip()
            password = (row[1] if len(row) > 1 else "").strip()
            name = (row[2] if len(row) > 2 else "Teacher").strip() or "Teacher"
            department = (row[3] if len(row) > 3 else "").strip()
            year = (row[4] if len(row) > 4 else "").strip()
            section = (row[5] if len(row) > 5 else "").strip()
            subject = (row[6] if len(row) > 6 else "").strip()

            if not username or not password:
                continue

            teachers[username] = {
                "password": password,
                "name": name,
                "department": department,
                "year": year,
                "section": section,
                "subject": subject
            }

    return teachers


def parse_subject_list(value):
    """Normalize a subject value into a unique list of cleaned subject names."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).replace(";", ",").replace("/", ",").replace("\n", ",").split(",")

    subjects = []
    seen = set()
    for item in raw_values:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            subjects.append(cleaned)
    return subjects


def parse_department_list(value):
    """Normalize a department value into a unique list of cleaned department names."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).replace(";", ",").replace("/", ",").replace("\n", ",").split(",")

    departments = []
    seen = set()
    for item in raw_values:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            departments.append(cleaned)
    return departments


def teacher_subjects_match(teacher_subject_value, student_subject):
    """Return True if the student subject is in the teacher's allowed subject list."""
    if not teacher_subject_value:
        return True

    allowed_subjects = parse_subject_list(teacher_subject_value)
    if not allowed_subjects:
        return True

    student_value = str(student_subject or "").strip()
    if not student_value:
        return False

    return student_value.lower() in {subject.lower() for subject in allowed_subjects}


def teacher_departments_match(teacher_department_value, student_department):
    """Return True if the student department is in the teacher's allowed department list."""
    if not teacher_department_value:
        return True

    allowed_departments = parse_department_list(teacher_department_value)
    if not allowed_departments:
        return True

    student_value = str(student_department or "").strip()
    if not student_value:
        return False

    return student_value.lower() in {dept.lower() for dept in allowed_departments}


def save_teacher(username, password, name, department="", year="", section="", subject=""):
    """Append a new teacher record to the teacher CSV file."""
    teacher_file = get_teacher_file_path()
    username = username.strip()
    password = password.strip()
    name = (name or "Teacher").strip() or "Teacher"
    department = (department or "").strip()
    year = (year or "").strip()
    section = (section or "").strip()
    subject = ", ".join(parse_subject_list(subject))

    if not username or not password:
        return False

    teachers = load_teachers()
    if username in teachers:
        return False

    file_exists = os.path.exists(teacher_file)
    with open(teacher_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Username", "Password", "Name", "Department", "Year", "Section", "Subject"])
        writer.writerow([username, password, name, department, year, section, subject])

    return True


def teacher_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("teacher_logged_in"):
            return redirect(url_for("teacher_login"))
        return view_func(*args, **kwargs)
    return wrapper


def student_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("student_logged_in"):
            return redirect(url_for("student_login"))
        return view_func(*args, **kwargs)
    return wrapper


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
                except (ValueError, KeyError):
                    continue

                subject = str(row.get("Subject", "")).strip()
                section = str(row.get("Section", "")).strip()
                teacher = str(row.get("Teacher", "")).strip()
                key = f"{student_id}|{subject}|{section}|{teacher}"

                marked[key] = {
                    "id": student_id,
                    "name": row.get("Name", ""),
                    "time": row.get("Time", ""),
                    "department": row.get("Department", ""),
                    "year": row.get("Year", ""),
                    "section": section,
                    "subject": subject,
                    "teacher": teacher
                }

    except Exception as e:
        print("Error reading attendance:", e)

    return marked


def get_student_attendance_summary(student_id):
    """Return total, present, absent, and percentage data for one student."""
    if student_id is None:
        return {
            "total_days": 0,
            "present_days": 0,
            "absent_days": 0,
            "percentage": 0.0
        }

    total_days = 0
    present_days = 0
    absent_days = 0
    student_id_str = str(student_id).strip()

    if not os.path.exists(ATTENDANCE_DIR):
        return {
            "total_days": total_days,
            "present_days": present_days,
            "absent_days": absent_days,
            "percentage": 0.0
        }

    for filename in sorted(os.listdir(ATTENDANCE_DIR)):
        if not filename.startswith("Attendance_") or not filename.endswith(".csv"):
            continue

        file_path = os.path.join(ATTENDANCE_DIR, filename)
        total_days += 1
        present = False

        try:
            with open(file_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    if str(row.get("ID", "")).strip() == student_id_str:
                        present = True
                        break
        except Exception as exc:
            print(f"Error reading attendance summary for {file_path}: {exc}")
            continue

        if present:
            present_days += 1
        else:
            absent_days += 1

    percentage = round((present_days / total_days) * 100, 2) if total_days else 0.0

    return {
        "total_days": total_days,
        "present_days": present_days,
        "absent_days": absent_days,
        "percentage": percentage
    }


def mark_attendance(student_id, student_name, department="", year="", section="", subject="", teacher_name=""):

    attendance_file = get_today_attendance_file()

    os.makedirs(
        ATTENDANCE_DIR,
        exist_ok=True
    )

    already_marked = load_today_attendance()
    attendance_key = f"{student_id}|{subject}|{section}|{teacher_name}"

    if attendance_key in already_marked:

        return {
            "success": True,
            "already_marked": True,
            "time": already_marked[attendance_key]["time"]
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
                "Time",
                "Department",
                "Year",
                "Section",
                "Subject",
                "Teacher"
            ])

        writer.writerow([
            student_id,
            student_name,
            current_time,
            department,
            year,
            section,
            subject,
            teacher_name
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


@app.route("/logout")
def logout():

    session.pop("teacher_logged_in", None)
    session.pop("teacher_name", None)
    session.pop("teacher_department", None)
    session.pop("teacher_year", None)
    session.pop("teacher_section", None)
    session.pop("teacher_subject", None)
    session.pop("student_logged_in", None)
    session.pop("student_id", None)
    session.pop("student_name", None)
    session.pop("student_department", None)
    session.pop("student_year", None)
    session.pop("student_section", None)
    session.pop("student_subject", None)

    return redirect(url_for("home"))


@app.route("/teacher/admin")
@teacher_required
def teacher_admin():
    teacher_department = session.get("teacher_department", "")
    teacher_year = session.get("teacher_year", "")
    teacher_section = session.get("teacher_section", "")
    teacher_subject = session.get("teacher_subject", "")
    teacher_allowed_departments = parse_department_list(teacher_department)
    teacher_allowed_subjects = parse_subject_list(teacher_subject)
    selected_department = request.args.get("department", teacher_department).strip()
    selected_year = request.args.get("year", teacher_year).strip()
    selected_section = request.args.get("section", teacher_section).strip()
    requested_subject = request.args.get("subject", "").strip()
    selected_subject = requested_subject if requested_subject else (teacher_allowed_subjects[0] if len(teacher_allowed_subjects) == 1 else "")
    all_students = load_student_records()

    if selected_department or selected_year or selected_section or selected_subject:
        filtered_students = {}
        for student_id, data in all_students.items():
            if teacher_allowed_departments and not selected_department:
                department_ok = teacher_departments_match(teacher_department, data.get("department", ""))
            else:
                department_ok = not selected_department or data.get("department", "") == selected_department
            year_ok = not selected_year or data.get("year", "") == selected_year
            section_ok = not selected_section or data.get("section", "") == selected_section
            if teacher_allowed_subjects and not selected_subject:
                subject_ok = teacher_subjects_match(teacher_subject, data.get("subject", ""))
            else:
                subject_ok = not selected_subject or data.get("subject", "") == selected_subject
            if department_ok and year_ok and section_ok and subject_ok:
                filtered_students[student_id] = data["name"]
    else:
        filtered_students = {student_id: data["name"] for student_id, data in all_students.items()}

    if teacher_allowed_departments and not selected_department:
        filtered_students = {
            student_id: data["name"]
            for student_id, data in all_students.items()
            if teacher_departments_match(teacher_department, data.get("department", ""))
            and teacher_subjects_match(teacher_subject, data.get("subject", ""))
            and (not selected_year or data.get("year", "") == selected_year)
            and (not selected_section or data.get("section", "") == selected_section)
        }

    total_students = len(filtered_students)
    today_file = get_today_attendance_file()
    present_today = 0

    if os.path.exists(today_file):
        with open(today_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                student_id = str(row.get("ID", "")).strip()
                if not student_id or not student_id.isdigit():
                    continue

                if selected_department or selected_year or selected_section or selected_subject:
                    student_record = all_students.get(int(student_id))
                    if student_record is None:
                        continue
                    dept_ok = not selected_department or student_record.get("department", "") == selected_department
                    year_ok = not selected_year or student_record.get("year", "") == selected_year
                    section_ok = not selected_section or student_record.get("section", "") == selected_section
                    subject_ok = not selected_subject or student_record.get("subject", "") == selected_subject
                    if not (dept_ok and year_ok and section_ok and subject_ok):
                        continue

                present_today += 1

    absent_today = max(total_students - present_today, 0)

    today_attendance = []
    today_file = get_today_attendance_file()
    if os.path.exists(today_file):
        with open(today_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                student_id = str(row.get("ID", "")).strip()
                if not student_id or not student_id.isdigit():
                    continue

                if selected_department or selected_year or selected_section or selected_subject:
                    student_record = all_students.get(int(student_id))
                    if student_record is None:
                        continue
                    dept_ok = not selected_department or student_record.get("department", "") == selected_department
                    year_ok = not selected_year or student_record.get("year", "") == selected_year
                    section_ok = not selected_section or student_record.get("section", "") == selected_section
                    subject_ok = not selected_subject or student_record.get("subject", "") == selected_subject
                    if not (dept_ok and year_ok and section_ok and subject_ok):
                        continue

                today_attendance.append({
                    "id": student_id,
                    "name": row.get("Name", ""),
                    "time": row.get("Time", "")
                })

    departments = sorted({data.get("department", "").strip() for data in all_students.values() if data.get("department", "").strip()})
    sections = sorted({data.get("section", "").strip() for data in all_students.values() if data.get("section", "").strip()})
    subjects = sorted({data.get("subject", "").strip() for data in all_students.values() if data.get("subject", "").strip()})
    valid_years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]

    return render_template(
        "teacher_admin.html",
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today,
        students=filtered_students,
        teacher_name=session.get("teacher_name", "Teacher"),
        teacher_department=selected_department,
        teacher_year=selected_year,
        teacher_section=selected_section,
        teacher_subject=selected_subject,
        today_attendance=today_attendance,
        departments=departments,
        valid_years=valid_years,
        sections=sections,
        subjects=subjects,
        selected_department=selected_department,
        selected_year=selected_year,
        selected_section=selected_section,
        selected_subject=selected_subject
    )


@app.route("/teacher/retrain-model")
@teacher_required
def retrain_model():
    try:
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
            check=True,
            cwd=BASE_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        flash("Face model retrained successfully.", "success")
    except Exception:
        flash("Model retraining failed. Please check the training data and try again.", "error")
    return redirect(url_for("teacher_admin"))


# ============================================================
# STUDENT LOGIN
# ============================================================

@app.route("/student-login", methods=["GET", "POST"])
def student_login():

    if request.method == "POST":

        student_id = request.form.get("student_id", "").strip()
        student_name = request.form.get("student_name", "").strip()
        student_password = request.form.get("student_password", "").strip()

        if not student_id or not student_name:
            return render_template(
                "student_login.html",
                error="Student ID and name are required."
            )

        if not student_id.isdigit():
            return render_template(
                "student_login.html",
                error="Student ID must be numeric."
            )

        students = load_student_records()
        student_id_int = int(student_id)
        stored_student = students.get(student_id_int)
        stored_name = stored_student["name"] if stored_student else ""
        stored_password = None

        if student_id_int in students:
            stored_password = load_student_credentials().get(student_id_int)

        if student_id_int not in students:
            return render_template(
                "student_login.html",
                error="Invalid student ID or name."
            )

        if stored_name.strip().lower() != student_name.strip().lower():
            return render_template(
                "student_login.html",
                error="Invalid student ID or name."
            )

        if stored_password is not None and student_password != stored_password:
            return render_template(
                "student_login.html",
                error="Incorrect student password."
            )

        session["student_logged_in"] = True
        session["student_id"] = student_id_int
        session["student_name"] = stored_name
        session["student_department"] = stored_student.get("department", "")
        session["student_year"] = stored_student.get("year", "")
        session["student_section"] = stored_student.get("section", "")
        session["student_subject"] = stored_student.get("subject", "")
        return redirect(url_for("student_dashboard"))

    return render_template("student_login.html")

@app.route("/student/dashboard")
@student_required
def student_dashboard():
    student_id = session.get("student_id")
    summary = get_student_attendance_summary(student_id)

    return render_template(
        "student_dashboard.html",
        student_id=student_id,
        student_name=session.get("student_name"),
        attendance_percentage=summary["percentage"],
        absent_days=summary["absent_days"],
        present_days=summary["present_days"],
        total_days=summary["total_days"]
    )


@app.route("/student/profile")
@student_required
def student_profile():
    student_id = session.get("student_id")
    student_name = session.get("student_name")
    summary = get_student_attendance_summary(student_id)

    return render_template(
        "student_profile.html",
        student_id=student_id,
        student_name=student_name,
        attendance_percentage=summary["percentage"],
        absent_days=summary["absent_days"],
        present_days=summary["present_days"],
        total_days=summary["total_days"]
    )


@app.route("/student/attendance")
@student_required
def student_attendance():

    student_id = str(session.get("student_id", ""))
    attendance_data = []

    if os.path.exists(ATTENDANCE_DIR):
        for filename in sorted(os.listdir(ATTENDANCE_DIR), reverse=True):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(ATTENDANCE_DIR, filename)

            with open(file_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    row_id = str(row.get("ID", "")).strip()
                    if row_id == student_id:
                        attendance_data.append({
                            "date": filename.replace("Attendance_", "").replace(".csv", ""),
                            "id": row.get("ID", ""),
                            "name": row.get("Name", ""),
                            "time": row.get("Time", "")
                        })

    return render_template(
        "student_attendance.html",
        attendance_data=attendance_data
    )


# ============================================================
# LOGIN ALIASES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    return redirect(url_for("teacher_login"))


@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login_alias():
    return redirect(url_for("teacher_login"))


@app.route("/student/login", methods=["GET", "POST"])
def student_login_alias():
    return redirect(url_for("student_login"))


# ============================================================
# TEACHER LOGIN
# ============================================================

@app.route("/teacher-login", methods=["GET", "POST"])
def teacher_login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        teachers = load_teachers()
        teacher = teachers.get(username)

        if teacher and teacher.get("password") == password:
            session["teacher_logged_in"] = True
            session["teacher_name"] = teacher.get("name", "Teacher")
            session["teacher_department"] = teacher.get("department", "")
            session["teacher_year"] = teacher.get("year", "")
            session["teacher_section"] = teacher.get("section", "")
            session["teacher_subject"] = teacher.get("subject", "")
            return redirect(url_for("teacher_admin"))

        return render_template(
            "teacher_login.html",
            error="Invalid username or password."
        )

    return render_template("teacher_login.html")


@app.route("/teacher-register", methods=["GET", "POST"])
def teacher_register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip()
        subject = request.form.get("subject", "").strip()

        if not username or not password or not name:
            return render_template(
                "teacher_register.html",
                error="Username, password, and name are required."
            )

        if not department or not subject:
            return render_template(
                "teacher_register.html",
                error="Department(s) and at least one subject are required."
            )

        if any(ch.isspace() for ch in username):
            return render_template(
                "teacher_register.html",
                error="Username cannot contain spaces."
            )

        teachers = load_teachers()
        if username in teachers:
            return render_template(
                "teacher_register.html",
                error="This teacher username already exists."
            )

        if save_teacher(username, password, name, department, "", "", subject):
            session["teacher_logged_in"] = True
            session["teacher_name"] = name
            session["teacher_department"] = department
            session["teacher_year"] = ""
            session["teacher_section"] = ""
            session["teacher_subject"] = subject
            return redirect(url_for("teacher_admin"))

        return render_template(
            "teacher_register.html",
            error="Could not create teacher account."
        )

    return render_template("teacher_register.html")


@app.route("/teacher-logout")
def teacher_logout():

    session.pop("teacher_logged_in", None)
    session.pop("teacher_name", None)
    session.pop("teacher_department", None)
    session.pop("teacher_year", None)

    return redirect(
        url_for("teacher_login")
    )
# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@teacher_required
def dashboard():
    return redirect(url_for("teacher_admin"))
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

        department = request.form.get(
            "department",
            ""
        ).strip()

        year = request.form.get(
            "year",
            ""
        ).strip()

        subject = request.form.get(
            "subject",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        valid_years = {
            "1st Year",
            "2nd Year",
            "3rd Year",
            "4th Year"
        }

        if not student_id:

            return "Student ID is required."

        if not student_id.isdigit():

            return "Student ID must be numeric."

        student_id = int(student_id)

        if not name:

            return "Student name is required."

        if not department:
            return "Department is required."

        if not subject:
            return "Subject is required."

        if year not in valid_years:
            return "Please select a valid current year (1st, 2nd, 3rd, or 4th year)."

        students = load_students()

        if student_id in students:

            return (
                f"Student ID {student_id} "
                "already exists."
            )

        if password:
            save_student_password(student_id, password)

        # Store temporarily in session.
        # Student is added to CSV only after
        # successful 60 face captures.

        session["student_id"] = student_id
        session["student_name"] = name
        session["student_department"] = department
        session["student_year"] = year
        session["student_section"] = ""
        session["student_subject"] = subject
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

        add_student(
            student_id,
            student_name,
            department=session.get("student_department", ""),
            year=session.get("student_year", ""),
            section=session.get("student_section", ""),
            subject=session.get("student_subject", "")
        )

        try:
            subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
                check=True,
                cwd=BASE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

        session.pop(
            "student_id",
            None
        )

        session.pop(
            "student_name",
            None
        )

        session.pop(
            "student_department",
            None
        )

        session.pop(
            "student_year",
            None
        )

        session.pop(
            "student_section",
            None
        )

        session.pop(
            "student_subject",
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
        print(
            f"RECOGNITION: internal_label={predicted_label}, "
            f"distance={distance:.2f}, "
            f"student_id={label_map.get(int(predicted_label))}"
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

    student_record = load_student_records().get(real_student_id, {})

    if session.get("teacher_logged_in"):
        teacher_department = session.get("teacher_department", "").strip()
        teacher_section = session.get("teacher_section", "").strip()
        teacher_subject = str(session.get("teacher_subject", "")).strip()

        student_department = str(student_record.get("department", "")).strip()
        student_section = str(student_record.get("section", "")).strip()
        student_subject = str(student_record.get("subject", "")).strip()

        if teacher_department and student_department and teacher_department != student_department:
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "This student does not belong to your department."
            })

        if teacher_section and student_section and teacher_section != student_section:
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "This student is not in your section."
            })

        if teacher_subject and student_subject and not teacher_subjects_match(teacher_subject, student_subject):
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "This student is not in your subject class."
            })

    # --------------------------------------------------------
    # MARK ATTENDANCE
    # --------------------------------------------------------

    attendance_result = mark_attendance(
        real_student_id,
        student_name,
        department=str(student_record.get("department", "")).strip(),
        year=str(student_record.get("year", "")).strip(),
        section=str(student_record.get("section", "")).strip(),
        subject=str(student_record.get("subject", "")).strip(),
        teacher_name=session.get("teacher_name", "")
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
    teacher_department = str(session.get("teacher_department", "")).strip()
    teacher_section = str(session.get("teacher_section", "")).strip()
    teacher_subject = str(session.get("teacher_subject", "")).strip()

    for key, data in attendance.items():
        record_department = str(data.get("department", "")).strip()
        record_section = str(data.get("section", "")).strip()
        record_subject = str(data.get("subject", "")).strip()
        record_id = str(data.get("id", str(key).split("|")[0])).strip()

        if session.get("teacher_logged_in"):
            if teacher_department and record_department and not teacher_departments_match(teacher_department, record_department):
                continue
            if teacher_section and record_section and record_section != teacher_section:
                continue
            if teacher_subject and record_subject and not teacher_subjects_match(teacher_subject, record_subject):
                continue

        attendance_data.append({
            "id": record_id,
            "name": data.get("name", ""),
            "time": data.get("time", ""),
            "department": record_department,
            "section": record_section,
            "subject": record_subject,
            "teacher": data.get("teacher", "")
        })

    print("ATTENDANCE PAGE DATA:", attendance_data)

    return render_template(
        "attendance.html",
        attendance_data=attendance_data
    )


@app.route("/take-attendance")
def take_attendance_page():
    return attendance_page()

# ============================================================
# ATTENDANCE HISTORY
# ============================================================

@app.route("/attendance-history")
@teacher_required
def attendance_history():

    selected_department = request.args.get("department", session.get("teacher_department", "")).strip()
    selected_student_year = request.args.get("student_year", session.get("teacher_year", "")).strip()
    selected_section = request.args.get("section", session.get("teacher_section", "")).strip()
    selected_subject = request.args.get("subject", session.get("teacher_subject", "")).strip()
    selected_date_year = request.args.get("date_year", "").strip()
    selected_month = request.args.get("month", "").strip()
    selected_day = request.args.get("day", "").strip()

    history_data = []
    available_years = []
    available_months = []
    available_days = []
    available_departments = []
    available_sections = []
    available_subjects = []
    available_student_years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
    all_students = load_student_records()

    if os.path.exists(ATTENDANCE_DIR):

        files = sorted(
            os.listdir(ATTENDANCE_DIR),
            reverse=True
        )

        for filename in files:

            if not filename.endswith(".csv"):
                continue

            if not filename.startswith("Attendance_"):
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

            try:
                year, month, day = date.split("-")
            except ValueError:
                continue

            if year not in available_years:
                available_years.append(year)
            if month not in available_months:
                available_months.append(month)
            if day not in available_days:
                available_days.append(day)

            if selected_date_year and year != selected_date_year:
                continue
            if selected_month and month != selected_month:
                continue
            if selected_day and day != selected_day:
                continue

            with open(
                file_path,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:
                    student_id = str(row.get("ID", "")).strip()
                    if not student_id:
                        continue
                    student_record = all_students.get(int(student_id), {}) if student_id.isdigit() else {}
                    department = str(student_record.get("department", "")).strip()
                    student_year = str(student_record.get("year", "")).strip()
                    section = str(student_record.get("section", "")).strip()
                    subject = str(student_record.get("subject", "")).strip()

                    if department and department not in available_departments:
                        available_departments.append(department)
                    if section and section not in available_sections:
                        available_sections.append(section)
                    if subject and subject not in available_subjects:
                        available_subjects.append(subject)

                    if selected_department and department != selected_department:
                        continue
                    if selected_student_year and student_year != selected_student_year:
                        continue
                    if selected_section and section != selected_section:
                        continue
                    if selected_subject and subject != selected_subject:
                        continue

                    history_data.append({
                        "date": date,
                        "id": row.get("ID", ""),
                        "name": row.get("Name", ""),
                        "time": row.get("Time", ""),
                        "department": department,
                        "year": student_year,
                        "section": section,
                        "subject": subject
                    })

    return render_template(
        "attendance_history.html",
        history_data=history_data,
        selected_department=selected_department,
        selected_student_year=selected_student_year,
        selected_section=selected_section,
        selected_subject=selected_subject,
        selected_date_year=selected_date_year,
        selected_month=selected_month,
        selected_day=selected_day,
        available_departments=sorted(available_departments),
        available_sections=sorted(available_sections),
        available_subjects=sorted(available_subjects),
        available_student_years=available_student_years,
        available_years=sorted(available_years, reverse=True),
        available_months=sorted(available_months),
        available_days=sorted(available_days)
    )

# ============================================================
# FACE RECOGNITION PAGE
# ============================================================

@app.route("/recognize-page")
def recognize_page():
    return render_template("attendance.html")

# ============================================================
# ATTENDANCE REPORTS
# ============================================================

@app.route("/attendance-reports")
@teacher_required
def attendance_reports():

    selected_department = request.args.get("department", "").strip()
    selected_year = request.args.get("year", "").strip()
    selected_section = request.args.get("section", "").strip()
    selected_subject = request.args.get("subject", "").strip()

    student_records = load_student_records()
    filtered_students = {}
    departments = []
    sections = []
    subjects = []
    valid_years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]

    for student_id, data in student_records.items():
        department = str(data.get("department", "")).strip()
        year = str(data.get("year", "")).strip()
        section = str(data.get("section", "")).strip()
        subject = str(data.get("subject", "")).strip()

        if department and department not in departments:
            departments.append(department)
        if section and section not in sections:
            sections.append(section)
        if subject and subject not in subjects:
            subjects.append(subject)

        if selected_department and department != selected_department:
            continue
        if selected_year and year != selected_year:
            continue
        if selected_section and section != selected_section:
            continue
        if selected_subject and subject != selected_subject:
            continue

        filtered_students[student_id] = {
            "name": data.get("name", ""),
            "department": department,
            "year": year,
            "section": section,
            "subject": subject,
            "present": 0,
            "absent": 0,
            "percentage": 0
        }

    report_data = {}
    for student_id, info in filtered_students.items():
        report_data[str(student_id)] = {
            "id": student_id,
            "name": info["name"],
            "department": info["department"],
            "year": info["year"],
            "section": info["section"],
            "subject": info["subject"],
            "present": 0,
            "absent": 0,
            "percentage": 0
        }

    total_days = 0

    if os.path.exists(ATTENDANCE_DIR):
        for filename in sorted(os.listdir(ATTENDANCE_DIR)):
            if not filename.startswith("Attendance_") or not filename.endswith(".csv"):
                continue

            total_days += 1
            file_path = os.path.join(ATTENDANCE_DIR, filename)
            present_ids = set()

            with open(file_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    student_id = str(row.get("ID", "")).strip()
                    if student_id:
                        present_ids.add(student_id)

            for student_id in filtered_students:
                student_key = str(student_id)
                student_info = student_records.get(student_id, {})
                student_department = str(student_info.get("department", "")).strip()
                student_year = str(student_info.get("year", "")).strip()
                student_section = str(student_info.get("section", "")).strip()
                student_subject = str(student_info.get("subject", "")).strip()

                if selected_department and student_department != selected_department:
                    continue
                if selected_year and student_year != selected_year:
                    continue
                if selected_section and student_section != selected_section:
                    continue
                if selected_subject and student_subject != selected_subject:
                    continue

                if student_key in present_ids:
                    report_data[student_key]["present"] += 1
                else:
                    report_data[student_key]["absent"] += 1

    for student_key in report_data:
        present = report_data[student_key]["present"]
        if total_days > 0:
            report_data[student_key]["percentage"] = round((present / total_days) * 100, 2)
        else:
            report_data[student_key]["percentage"] = 0

    grouped_summary = {}
    for student_key, result in report_data.items():
        group_key = (result["department"], result["year"], result["section"], result["subject"])
        if group_key not in grouped_summary:
            grouped_summary[group_key] = {
                "department": result["department"],
                "year": result["year"],
                "section": result["section"],
                "subject": result["subject"],
                "students": 0,
                "present": 0,
                "absent": 0,
                "percentage": 0
            }

        grouped_summary[group_key]["students"] += 1
        grouped_summary[group_key]["present"] += result["present"]
        grouped_summary[group_key]["absent"] += result["absent"]

    for group in grouped_summary.values():
        total_entries = group["present"] + group["absent"]
        group["percentage"] = round((group["present"] / total_entries) * 100, 2) if total_entries else 0

    return render_template(
        "attendance_reports.html",
        reports=report_data.values(),
        group_summary=sorted(grouped_summary.values(), key=lambda item: (item["department"], item["year"], item["section"], item["subject"])),
        total_days=total_days,
        selected_department=selected_department,
        selected_year=selected_year,
        selected_section=selected_section,
        selected_subject=selected_subject,
        departments=sorted(departments),
        sections=sorted(sections),
        subjects=sorted(subjects),
        years=valid_years
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
    # 1. Remove student from students.csv while preserving schema
    # --------------------------------------------------------

    students_file = os.path.join(BASE_DIR, "students.csv")
    full_fieldnames = ["ID", "Name", "Department", "Year", "Section", "Subject"]
    rows = []

    if os.path.exists(students_file):
        with open(students_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or full_fieldnames
            available_fields = [field for field in full_fieldnames if field in fieldnames]
            if not available_fields:
                available_fields = full_fieldnames

            for row in reader:
                if row.get("ID", "").strip() == str(student_id):
                    continue
                clean_row = {field: (row.get(field, "") or "") for field in available_fields}
                clean_row["ID"] = row.get("ID", "")
                clean_row["Name"] = row.get("Name", "")
                rows.append(clean_row)

    with open(students_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=full_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    credentials_file = os.path.join(BASE_DIR, "student_credentials.csv")
    if os.path.exists(credentials_file):
        credentials_rows = []
        with open(credentials_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if str(row.get("ID", "")).strip() != str(student_id):
                    credentials_rows.append(row)
        with open(credentials_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["ID", "Password"])
            writer.writeheader()
            writer.writerows(credentials_rows)

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
