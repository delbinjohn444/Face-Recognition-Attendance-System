import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from pathlib import Path
import time
from datetime import datetime
import csv
import sys


# =====================================================
# FACE RECOGNITION SYSTEM
# =====================================================

class FaceRecognitionSystem:

    def __init__(self,
                 dataset_path="dataset",
                 confidence_threshold=0.6,
                 attendance_file="attendance.csv",
                 camera_index=0):  # Fixed: Changed default from 1 to 0 (standard webcam)

        self.dataset_path = dataset_path
        self.confidence_threshold = confidence_threshold
        self.attendance_file = attendance_file
        self.camera_index = camera_index

        # Force CPU (Stable on M1/M2/M3)
        self.device = torch.device("cpu")

        print("✅ Using CPU")

        # Face Detector
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]  # Added: Better detection thresholds
        )

        # FaceNet Model
        self.facenet = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)

        # Storage
        self.known_embeddings = []
        self.known_names = []

        self.cache_file = "face_cache.pkl"

        self.attendance_marked = set()

        self._init_attendance_file()

        print("✅ System Initialized")


    # =====================================================
    # ATTENDANCE FILE
    # =====================================================

    def _init_attendance_file(self):

        if not os.path.exists(self.attendance_file):

            with open(self.attendance_file, "w", newline="") as f:

                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])

            print("✅ Attendance file created")


    # =====================================================
    # MARK ATTENDANCE
    # =====================================================

    def mark_attendance(self, name):

        if name in self.attendance_marked:
            return

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")

        with open(self.attendance_file, "a", newline="") as f:

            writer = csv.writer(f)
            writer.writerow([name, date, time_now])

        self.attendance_marked.add(name)

        print(f"✅ Marked: {name}")


    # =====================================================
    # GET FACE EMBEDDING
    # =====================================================

    def get_embedding(self, face_img):

        try:
            if isinstance(face_img, np.ndarray):
                face_img = Image.fromarray(
                    cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                )

            face_img = face_img.resize((160, 160))

            face = np.array(face_img)

            face = torch.tensor(face).permute(2, 0, 1).float()

            face = (face - 127.5) / 128.0

            face = face.unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.facenet(face)

            emb = emb / torch.norm(emb)

            return emb.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None


    # =====================================================
    # CLEAR CACHE (AUTO)
    # =====================================================

    def clear_cache(self):

        if os.path.exists(self.cache_file):

            os.remove(self.cache_file)

            print("🔄 Cache cleared")


    # =====================================================
    # LOAD DATASET
    # =====================================================

    def load_dataset(self, force_reload=False):

        if force_reload:
            self.clear_cache()


        # Try Cache
        if not force_reload and os.path.exists(self.cache_file):

            try:
                with open(self.cache_file, "rb") as f:

                    data = pickle.load(f)

                    self.known_embeddings = data["embeddings"]
                    self.known_names = data["names"]

                print("✅ Loaded from cache")

                return
            
            except Exception as e:
                print(f"⚠️ Cache corrupted, reloading: {e}")
                self.clear_cache()


        print("\n📂 Loading dataset...")

        self.known_embeddings = []
        self.known_names = []


        if not os.path.exists(self.dataset_path):

            print("❌ Dataset folder missing")
            print(f"💡 Create folder: {self.dataset_path}")
            print(f"💡 Add subfolders for each person with their photos")

            return


        person_count = 0

        for person in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person)

            if not os.path.isdir(person_path):
                continue


            print(f"👤 {person}")

            embeddings = []


            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                # Fixed: Skip non-image files
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                img = cv2.imread(img_path)

                if img is None:
                    print(f"   ⚠️ Could not read: {img_name}")
                    continue


                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pil = Image.fromarray(rgb)


                boxes, _ = self.mtcnn.detect(pil)


                if boxes is None:
                    print(f"   ⚠️ No face detected: {img_name}")
                    continue


                x1, y1, x2, y2 = map(int, boxes[0])

                # Fixed: Add boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)

                face = rgb[y1:y2, x1:x2]


                if face.size == 0:
                    print(f"   ⚠️ Invalid face crop: {img_name}")
                    continue


                emb = self.get_embedding(face)

                if emb is not None:
                    embeddings.append(emb)


            if embeddings:

                avg = np.mean(embeddings, axis=0)

                avg = avg / np.linalg.norm(avg)

                self.known_embeddings.append(avg)

                self.known_names.append(person)

                person_count += 1

                print(f"   ✅ Added ({len(embeddings)} images)")

            else:
                print(f"   ❌ No valid images for {person}")


        # Save Cache
        if self.known_embeddings:

            try:
                with open(self.cache_file, "wb") as f:

                    pickle.dump({
                        "embeddings": self.known_embeddings,
                        "names": self.known_names
                    }, f)

                print("💾 Cache Saved")
            
            except Exception as e:
                print(f"⚠️ Could not save cache: {e}")


        print(f"\n✅ Loaded {len(self.known_names)} People\n")


    # =====================================================
    # FACE MATCHING
    # =====================================================

    def recognize(self, emb):

        if not self.known_embeddings or emb is None:
            return None, 0


        sims = np.dot(self.known_embeddings, emb)


        best = np.argmax(sims)

        score = sims[best]


        if score >= self.confidence_threshold:

            return self.known_names[best], score * 100


        return None, 0


    # =====================================================
    # CAMERA LOOP
    # =====================================================

    def run(self):


        self.camera_index=1

        cap = cv2.VideoCapture(self.camera_index)

        # Fixed: Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)


        if not cap.isOpened():

            print("❌ Camera not working")
            print(f"💡 Try changing camera_index (currently {self.camera_index})")
            print("💡 Common values: 0 (built-in), 1 (external)")

            return


        print(f"🎥 Camera {self.camera_index} Started")


        print("Press Q = Quit | R = Reset Attendance")


        last_results = []

        frame_count = 0
        skip_frames = 2  # Process every 3rd frame for performance


        while True:

            ret, frame = cap.read()

            if not ret:
                print("❌ Failed to read frame")
                break

            frame_count += 1

            # Fixed: Process faces every few frames for better performance
            if frame_count % skip_frames == 0:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pil = Image.fromarray(rgb)


                try:
                    boxes, probs = self.mtcnn.detect(pil)
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    boxes, probs = None, None


                results = []


                if boxes is not None:


                    for box, p in zip(boxes, probs):

                        if p < 0.9:
                            continue


                        x1, y1, x2, y2 = map(int, box)

                        # Fixed: Add boundary checks
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        # Fixed: Ensure valid face region
                        if x2 <= x1 or y2 <= y1:
                            continue

                        face = rgb[y1:y2, x1:x2]


                        if face.size == 0:
                            continue


                        emb = self.get_embedding(face)


                        name, conf = self.recognize(emb)


                        if name:
                            self.mark_attendance(name)


                        results.append((x1, y1, x2, y2, name, conf))


                last_results = results


            # Draw Boxes
            for x1, y1, x2, y2, name, conf in last_results:


                if name:

                    color = (0, 255, 0)
                    label = f"{name} {conf:.1f}%"

                else:

                    color = (0, 0, 255)
                    label = "Unknown"


                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


                # Fixed: Better text background for readability
                text_size = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    2
                )[0]
                
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2
                )


            # Info
            cv2.putText(
                frame,
                f"Present: {len(self.attendance_marked)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.imshow("Attendance System", frame)


            key = cv2.waitKey(1) & 0xFF


            if key == ord("q") or key == ord("Q"):
                break


            elif key == ord("r") or key == ord("R"):

                self.attendance_marked.clear()

                print("🔄 Attendance Reset")


        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System Stopped")


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 50)
    print(" FACE RECOGNITION ATTENDANCE SYSTEM ")
    print("=" * 50)


    # Parse command line arguments
    camera_index = 0
    force_reload = False

    for arg in sys.argv[1:]:
        if arg == "--reload":
            force_reload = True
        elif arg.startswith("--camera="):
            try:
                camera_index = int(arg.split("=")[1])
            except ValueError:
                print(f"⚠️ Invalid camera index: {arg}")


    system = FaceRecognitionSystem(
        camera_index=camera_index
    )


    system.load_dataset(force_reload=force_reload)


    if not system.known_embeddings:

        print("\n❌ No faces loaded")
        print("\n📝 To use this system:")
        print(f"1. Create a '{system.dataset_path}' folder")
        print("2. Create subfolders for each person (e.g., 'John', 'Sarah')")
        print("3. Add 3-5 photos of each person in their folder")
        print("4. Run the script again")

        return


    system.run()


if __name__ == "__main__":

    main()