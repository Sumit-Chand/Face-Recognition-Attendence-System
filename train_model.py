import os
import csv
import cv2
import numpy as np

from utils import DATASET_DIR, TRAINER_FILE


LABEL_MAP_FILE = os.path.join(
    os.path.dirname(TRAINER_FILE),
    "label_map.csv"
)


def get_student_ids():
    student_ids = set()

    for filename in os.listdir(DATASET_DIR):
        if filename.startswith("User.") and filename.endswith(".jpg"):
            parts = filename.split(".")

            if len(parts) >= 3:
                try:
                    student_id = int(parts[1])
                    student_ids.add(student_id)
                except ValueError:
                    pass

    return sorted(student_ids)


def get_images_and_labels():
    face_samples = []
    student_ids = []

    for filename in os.listdir(DATASET_DIR):
        if not (filename.startswith("User.") and filename.endswith(".jpg")):
            continue

        parts = filename.split(".")

        if len(parts) < 3:
            continue

        try:
            student_id = int(parts[1])
        except ValueError:
            continue

        image_path = os.path.join(DATASET_DIR, filename)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        face_samples.append(img)
        student_ids.append(student_id)

    return face_samples, student_ids


def run():
    os.makedirs(os.path.dirname(TRAINER_FILE), exist_ok=True)

    print("==============================================")
    print("       TRAINING FACE RECOGNITION MODEL")
    print("==============================================")
    print()

    face_samples, real_student_ids = get_images_and_labels()

    if not face_samples:
        print("ERROR: No training images found.")
        return

    unique_ids = sorted(set(real_student_ids))

    print(f"Students found: {len(unique_ids)}")
    for student_id in unique_ids:
        print(f"    Student ID: {student_id}")

    print()

    # OpenCV LBPH gets SMALL internal labels.
    # Real student IDs are stored separately in label_map.csv.
    real_to_internal = {
        real_id: index + 1
        for index, real_id in enumerate(unique_ids)
    }

    internal_labels = np.array(
        [real_to_internal[sid] for sid in real_student_ids],
        dtype=np.int32
    )

    print("Internal label mapping:")

    for real_id in unique_ids:
        print(
            f"    Internal {real_to_internal[real_id]}"
            f"  ->  Student ID {real_id}"
        )

    print()

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Training model, please wait...")

    recognizer.train(
        face_samples,
        internal_labels
    )

    recognizer.write(TRAINER_FILE)

    # Save mapping
    with open(LABEL_MAP_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["InternalLabel", "StudentID"])

        for real_id in unique_ids:
            writer.writerow([
                real_to_internal[real_id],
                real_id
            ])

    print()
    print("==============================================")
    print("TRAINING COMPLETE")
    print("==============================================")
    print(f"Images trained: {len(face_samples)}")
    print(f"Students: {len(unique_ids)}")
    print()
    print(f"Model saved:")
    print(TRAINER_FILE)
    print()
    print("Label map saved:")
    print(LABEL_MAP_FILE)
    print("==============================================")


def train_model():
    run()


if __name__ == "__main__":
    run()