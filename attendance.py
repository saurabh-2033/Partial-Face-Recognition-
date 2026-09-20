import cv2
import os
import pickle
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

PKL_FILE = "embeddings.pkl"
DETAILS_FILE = "detail.csv"
ATT_FILE = "biometric_attendance.csv"
THRESHOLD = 0.65

detector = MTCNN()
facenet = FaceNet()

with open(PKL_FILE, "rb") as f:
    db = pickle.load(f)

known_embs = np.array(db["embeddings"])
known_names = db["names"]

details_df = pd.read_csv(DETAILS_FILE) if os.path.exists(DETAILS_FILE) else None

if not os.path.exists(ATT_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(
        ATT_FILE, index=False
    )

logged_names = set()
cap = cv2.VideoCapture(0)

print("Camera started. Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = [abs(i) for i in face["box"]]
        roi = frame_rgb[
            max(0, y-10):y+h+10,
            max(0, x-10):x+w+10
        ]
        if roi.size == 0:
            continue

        roi = cv2.resize(roi, (160, 160))

        emb_orig = facenet.embeddings(
            np.expand_dims(roi, axis=0)
        )[0]
        emb_flip = facenet.embeddings(
            np.expand_dims(cv2.flip(roi, 1), axis=0)
        )[0]

        sims_orig = cosine_similarity(
            [emb_orig], known_embs
        )[0]
        sims_flip = cosine_similarity(
            [emb_flip], known_embs
        )[0]

        sims = np.maximum(sims_orig, sims_flip)
        idx = np.argmax(sims)
        score = sims[idx]

        if score > THRESHOLD:
            name = known_names[idx]

            if name not in logged_names:
                now = datetime.now()
                pd.DataFrame(
                    [[name,
                      now.strftime("%Y-%m-%d"),
                      now.strftime("%H:%M:%S")]],
                    columns=["Name", "Date", "Time"]
                ).to_csv(
                    ATT_FILE, mode="a",
                    header=False, index=False
                )
                logged_names.add(name)

            label = f"{name} ({score:.2f})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    cv2.imshow("Partial Face Recognition + Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
