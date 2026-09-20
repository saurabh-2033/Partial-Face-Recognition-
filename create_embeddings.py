import os
import cv2
import pickle
import gc
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ================= PATHS =================
DATASET_PATH = r"path/to/your/Dataset"
PKL_FILE = "embeddings.pkl"

# ================= CREATE EMBEDDINGS =================
def create_embeddings(dataset_path, output_pkl):
    detector = MTCNN()
    embedder = FaceNet()
    known_embeddings = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                results = detector.detect_faces(img_rgb)
                if results:
                    x, y, w_face, h_face = [abs(i) for i in results[0]['box']]
                    face = img_rgb[y:y+h_face, x:x+w_face]
                    if face.size > 0:
                        face = cv2.resize(face, (160, 160))
                        emb = embedder.embeddings(
                            np.expand_dims(face, axis=0)
                        )[0]
                        known_embeddings.append(emb)
                        known_names.append(person_name)
            except:
                pass

            del img, img_rgb
            gc.collect()

    with open(output_pkl, "wb") as f:
        pickle.dump(
            {"embeddings": known_embeddings, "names": known_names}, f
        )

    print(f"✅ Embeddings created: {len(known_names)} samples")

# ================= ACCURACY CHECK =================
def calculate_accuracy(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["embeddings"])
    y = np.array(data["names"])

    unique, counts = np.unique(y, return_counts=True)
    if min(counts) < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    y_pred = []
    for test_emb in X_test:
        sims = cosine_similarity([test_emb], X_train)[0]
        idx = np.argmax(sims)
        y_pred.append(y_train[idx] if sims[idx] > 0.70 else "Unknown")

    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))


create_embeddings(DATASET_PATH, PKL_FILE)
calculate_accuracy(PKL_FILE)
