import cv2
import os
import datetime
import winsound
import pandas as pd

from student_database_mp import load_student_database
from face_match_mp import match_face

# ---------------- ALARM ----------------
def play_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

# ---------------- LOAD DATABASE ----------------
known_encodings, student_details = load_student_database()

# ---------------- CREATE FOLDERS ----------------
os.makedirs("logs", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

log_file = "logs/suspicious_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Time","HallTicket","Name","Action"]).to_csv(log_file, index=False)

# ---------------- VIDEO SETUP ----------------
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
recording = False
writer = None
record_start = None

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y+h, x:x+w], (100,100)).flatten()
        student = match_face(face_img, known_encodings, student_details)
        name = student["name"] if student else "Unknown"
        hall = student["hallticket"] if student else "Unknown"
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{name} ({hall})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        # Trigger alarm and record if unknown / cheating detected
        # For demo, trigger for unknown student
        if student is None and not recording:
            recording = True
            record_start = datetime.datetime.now()
            filename = f"recordings/Unknown_{record_start.strftime('%Y%m%d_%H%M%S')}.avi"
            writer = cv2.VideoWriter(filename, fourcc, 20, (frame.shape[1], frame.shape[0]))
            play_alarm()

            # Log
            df = pd.read_csv(log_file)
            df.loc[len(df)] = [datetime.datetime.now(), "Unknown", "Unknown", "Detected"]
            df.to_csv(log_file, index=False)

    # ---------------- WRITE FRAME ----------------
    if recording:
        writer.write(frame)
        if (datetime.datetime.now() - record_start).seconds > 5:
            recording = False
            writer.release()

    cv2.imshow("Offline Exam Proctoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
