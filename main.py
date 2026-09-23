"""
main.py
-------
Face Recognition Attendance System - menu-driven entry point.
College mini-project built with Python + OpenCV.

Run:
    python main.py
"""
import os
from register_student import register_student
from train_model import train_model
from mark_attendance import run as start_attendance, get_today_file


def view_today_attendance():
    filepath = get_today_file()
    if not os.path.exists(filepath):
        print("No attendance has been marked today yet.")
        return
    print(f"\n--- Attendance for today ({os.path.basename(filepath)}) ---")
    with open(filepath, "r") as f:
        print(f.read())


MENU = """
==============================================
 FACE RECOGNITION ATTENDANCE SYSTEM
==============================================
 1. Register New Student
 2. Train Recognition Model
 3. Start Attendance (webcam)
 4. View Today's Attendance
 5. Exit
==============================================
"""


def main():
    while True:
        print(MENU)
        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            register_student()
        elif choice == "2":
            train_model()
        elif choice == "3":
            start_attendance()
        elif choice == "4":
            view_today_attendance()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
