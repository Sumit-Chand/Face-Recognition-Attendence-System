"""
register_student.py
--------------------
STEP 1 of 3. Captures face images of a new student from the webcam and
saves them into dataset/, then records the student in students.csv.

Run:
    python register_student.py
"""
import cv2
from utils import DATASET_DIR, CASCADE_PATH, SAMPLES_PER_STUDENT, add_student, load_students


def register_student():
    students = load_students()

    while True:
        raw_id = input("Enter numeric Student ID (e.g. roll number): ").strip()
        try:
            student_id = int(raw_id)
            break
        except ValueError:
            print("Please enter a valid whole number.")

    if student_id in students:
        print(f"ID {student_id} is already registered as '{students[student_id]}'. "
              f"Choose a different ID.")
        return

    name = input("Enter Student Name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Could not access the webcam. Check the camera connection/permissions.")
        return

    print("\nLook at the camera and move your head slightly (left/right/up/down).")
    print("Capturing face samples... press 'q' to stop early.\n")

    count = 0
    while count < SAMPLES_PER_STUDENT:
        ok, frame = cam.read()
        if not ok:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            filename = f"{DATASET_DIR}/User.{student_id}.{count}.jpg"
            cv2.imwrite(filename, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_STUDENT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # only take one face per frame, even if several are visible

        cv2.imshow("Registering Face - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("No face samples were captured. Registration aborted.")
        return

    add_student(student_id, name)
    print(f"\nCaptured {count} samples for '{name}' (ID: {student_id}).")
    print("Next: run train_model.py to teach the recognizer this face.")


if __name__ == "__main__":
    register_student()
