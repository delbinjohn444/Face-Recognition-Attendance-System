from flask import Flask, render_template, jsonify, request
import subprocess
import sys
import os
import cv2
import time
import base64


app = Flask(__name__)

# Path to main.py
MAIN_SCRIPT = os.path.join(os.getcwd(), "main.py")


# =================================================
# TRAIN MODEL FUNCTION (BUILD FACE CACHE)
# =================================================
def train_model():

    import pickle
    import torch
    import numpy as np
    from PIL import Image
    from facenet_pytorch import MTCNN, InceptionResnetV1

    device = torch.device("cpu")

    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval()

    dataset_path = "dataset"
    embeddings = {}

    print("🔄 Training started...")

    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found")
        return

    for person in os.listdir(dataset_path):

        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        vectors = []

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            try:

                img = Image.open(img_path).convert("RGB")

                face = mtcnn(img)

                if face is None:
                    continue

                with torch.no_grad():
                    emb = model(face.unsqueeze(0))

                vectors.append(emb[0].numpy())

            except Exception as e:
                print("Image error:", e)
                continue

        if vectors:
            embeddings[person] = np.mean(vectors, axis=0)

    # Save cache file
    with open("face_cache.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print("✅ Training completed!")


# =================================================
# HOME PAGE
# =================================================
@app.route("/")
def home():
    return render_template("index.html")


# =================================================
# REGISTER PAGE
# =================================================
@app.route("/register")
def register_page():
    return render_template("register.html")


# =================================================
# OLD REGISTER ROUTE
# =================================================
@app.route("/register/<name>")
def register_student(name):

    return jsonify({
        "status": "⚠️ Use web camera registration page instead"
    })


# =================================================
# SAVE IMAGE FROM BROWSER CAMERA
# =================================================
@app.route("/save_image/<name>", methods=["POST"])
def save_image(name):

    try:

        data = request.json["img"]
        index = request.json["index"]

        folder = os.path.join("dataset", name)

        if not os.path.exists(folder):
            os.makedirs(folder)

        # Remove base64 header
        img_data = data.split(",")[1]

        img_bytes = base64.b64decode(img_data)

        path = os.path.join(folder, f"{index}.jpg")

        with open(path, "wb") as f:
            f.write(img_bytes)

        print("Saved:", path)

        return jsonify({"status": "saved"})

    except Exception as e:

        print("ERROR:", e)

        return jsonify({"status": "failed"})


# =================================================
# TRAIN MODEL ROUTE (BUTTON API)
# =================================================
@app.route("/train")
def train():

    try:

        train_model()

        return jsonify({"status": "✅ Model trained successfully"})

    except Exception as e:

        return jsonify({"status": f"❌ Training failed: {str(e)}"})


# =================================================
# START ATTENDANCE
# =================================================
@app.route("/start")
def start_attendance():

    try:

        subprocess.Popen([sys.executable, MAIN_SCRIPT])

        return jsonify({"status": "✅ Camera Started"})

    except Exception as e:

        return jsonify({"status": f"❌ Error: {str(e)}"})


# =================================================
# GET ATTENDANCE
# =================================================
@app.route("/attendance")
def get_attendance():

    data = []

    try:

        with open("attendance.csv", "r") as f:

            lines = f.readlines()[1:]

            for line in lines:
                row = line.strip().split(",")
                data.append(row)

    except:
        pass

    return jsonify(data)


# =================================================
# RESET ATTENDANCE
# =================================================
@app.route("/reset_attendance")
def reset_attendance():

    try:

        file_path = "attendance.csv"

        # Keep header only
        with open(file_path, "w") as f:
            f.write("Name,Date,Time\n")

        return jsonify({"status": "✅ Attendance reset successfully"})

    except Exception as e:

        return jsonify({"status": f"❌ Error: {str(e)}"})


# =================================================
# RELOAD FACES (OPTIONAL)
# =================================================
@app.route("/reload_faces")
def reload_faces():

    try:

        subprocess.Popen([sys.executable, "main.py", "--reload"])

        return jsonify({"status": "✅ Face database updated"})

    except Exception as e:

        return jsonify({"status": f"❌ {e}"})


# =================================================
# RUN SERVER
# =================================================
if __name__ == "__main__":

    app.run(debug=True, port=8000)