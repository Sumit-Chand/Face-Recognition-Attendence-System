import os
import csv
import cv2
from datetime import datetime

from utils import (
    TRAINER_FILE,
    CASCADE_PATH,
    ATTENDANCE_DIR,
    CONFIDENCE_THRESHOLD,
    load_students
)


LABEL_MAP_FILE = os.path.join(
    os.path.dirname(TRAINER_FILE),
    "label_map.csv"
)


def get_today_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(
        ATTENDANCE_DIR,
        f"Attendance_{today}.csv"
    )


def load_marked_ids(filepath):
    marked = set()

    if os.path.exists(filepath):
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if row:
                    try:
                        marked.add(int(row[0]))
                    except ValueError:
                        pass

    return marked


def load_label_map():
    """
    Convert LBPH internal label to real student ID.

    Example:
        1 -> 231190101060
    """

    label_map = {}

    if not os.path.exists(LABEL_MAP_FILE):
        print("[ERROR] label_map.csv not found:")
        print(LABEL_MAP_FILE)
        return label_map

    with open(LABEL_MAP_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                internal_label = int(row["InternalLabel"])
                student_id = int(row["StudentID"])

                label_map[internal_label] = student_id

            except (ValueError, KeyError):
                pass

    return label_map


def mark_attendance(student_id, name, filepath, marked_ids):
    """Mark a student only once per day."""

    if student_id in marked_ids:
        return False

    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["ID", "Name", "Time"])

        writer.writerow([
            student_id,
            name,
            datetime.now().strftime("%H:%M:%S")
        ])

    marked_ids.add(student_id)

    return True


def run():

    print("=" * 60)
    print("        FACE ATTENDANCE SYSTEM")
    print("=" * 60)
    print()

    # --------------------------------------------------
    # Check model
    # --------------------------------------------------

    if not os.path.exists(TRAINER_FILE):
        print("[ERROR] Trained model not found:")
        print(TRAINER_FILE)
        return

    print("[OK] Trained model found:")
    print(TRAINER_FILE)
    print()

    # --------------------------------------------------
    # Load students
    # --------------------------------------------------

    students = load_students()

    if not students:
        print("[ERROR] No students found in students.csv")
        return

    print("[OK] Students loaded:")

    for student_id, name in students.items():
        print(f"    ID: {student_id}   Name: {name}")

    print()

    # --------------------------------------------------
    # Load label map
    # --------------------------------------------------

    label_map = load_label_map()

    if not label_map:
        print("[ERROR] No label mapping available.")
        return

    print("[OK] Label mapping loaded:")

    for internal_label, real_id in label_map.items():
        print(
            f"    Internal Label: {internal_label}"
            f"  ->  Student ID: {real_id}"
        )

    print()

    # --------------------------------------------------
    # Load recognizer
    # --------------------------------------------------

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    print("[OK] LBPH recognizer loaded.")

    # --------------------------------------------------
    # Load face detector
    # --------------------------------------------------

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)

    if face_detector.empty():
        print("[ERROR] Face detector could not be loaded.")
        return

    print("[OK] Face detector loaded.")
    print()

    # --------------------------------------------------
    # Attendance file
    # --------------------------------------------------

    filepath = get_today_file()

    marked_ids = load_marked_ids(filepath)

    print("Today's attendance file:")
    print(filepath)
    print()

    print("Already marked IDs:")

    if marked_ids:
        for student_id in sorted(marked_ids):
            print(f"    {student_id}")
    else:
        print("    None")

    print()

    # --------------------------------------------------
    # Camera
    # --------------------------------------------------

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    print("=" * 60)
    print("CAMERA STARTED")
    print("=" * 60)
    print()
    print("Look at the camera.")
    print("Press Q to quit.")
    print()
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print()
    print("Recognition output will appear below:")
    print()

    while True:

        ok, frame = cam.read()

        if not ok:
            print("[ERROR] Could not read camera frame.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            predicted_label, distance = recognizer.predict(
                gray[y:y + h, x:x + w]
            )

            print(
                f"Internal Label: {predicted_label}"
                f" | Distance: {distance:.2f}"
                f" | Threshold: {CONFIDENCE_THRESHOLD}"
            )

            # ------------------------------------------
            # Convert internal label -> real student ID
            # ------------------------------------------

            real_student_id = label_map.get(
                int(predicted_label)
            )

            if (
                real_student_id is not None
                and distance < CONFIDENCE_THRESHOLD
                and real_student_id in students
            ):

                name = students[real_student_id]

                just_marked = mark_attendance(
                    real_student_id,
                    name,
                    filepath,
                    marked_ids
                )

                if just_marked:
                    status = "Marked!"
                else:
                    status = "Present"

                label = f"{name} - {status}"

                print(
                    f"[RECOGNIZED] "
                    f"ID: {real_student_id} | "
                    f"Name: {name} | "
                    f"Distance: {distance:.2f}"
                )

                color = (0, 255, 0)

            else:

                label = "Unknown"

                print(
                    f"[UNKNOWN] Internal Label: "
                    f"{predicted_label} | "
                    f"Distance: {distance:.2f}"
                )

                color = (0, 0, 255)

            # ------------------------------------------
            # Draw rectangle + name
            # ------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow(
            "Attendance System - Press Q to quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("ATTENDANCE SYSTEM CLOSED")
    print("=" * 60)
    print()
    print("Attendance saved to:")
    print(filepath)
    print()
    print("Today's marked students:")

    if marked_ids:
        for student_id in sorted(marked_ids):
            name = students.get(student_id, "Unknown")
            print(f"    {student_id} - {name}")
    else:
        print("    None")


if __name__ == "__main__":
    run()