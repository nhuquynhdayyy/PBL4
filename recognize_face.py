# recognize_face.py
from deepface import DeepFace
import cv2, numpy as np, pickle, os
from scipy.spatial.distance import cosine

# --- 1️⃣ Load model tự động từ cache ---
print("[INFO] Loading Facenet model từ DeepFace cache...")
model = DeepFace.build_model("Facenet")

# --- 2️⃣ Load cơ sở dữ liệu khuôn mặt ---
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

threshold = 0.4  # ngưỡng mặc định của Facenet

# --- 3️⃣ Hàm nhận diện khuôn mặt ---
def recognize_face(frame):
    try:
        # Lấy vector đặc trưng (embedding) từ ảnh hiện tại
        emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]['embedding']
    except:
        return None, 0.0  # nếu không detect được mặt

    best_match = None
    best_conf = 0

    # So sánh embedding hiện tại với từng học sinh trong DB
    for student_id, emb_list in face_db.items():
        for ref_emb in emb_list:
            dist = cosine(emb, ref_emb)
            conf = max(0, 1 - dist / threshold)  # chuyển khoảng cách thành độ tin cậy (0–1)
            if conf > best_conf:
                best_conf = conf
                best_match = student_id

    # Nếu độ tin cậy thấp, coi là "unknown"
    if best_conf < 0.5:
        best_match = "unknown"

    return best_match, round(best_conf, 2)

# --- 4️⃣ Nhận diện real-time ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

print("[INFO] Bắt đầu nhận diện... (Nhấn ESC để thoát)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được frame từ camera.")
        break

    frame_small = cv2.resize(frame, (480, 360))

    # --- Nhận diện khuôn mặt ---
    student_id, conf = recognize_face(frame_small)

    # --- Hiển thị kết quả trên màn hình ---
    label = f"{student_id} ({conf*100:.0f}%)"
    color = (0, 255, 0) if student_id != "unknown" else (0, 0, 255)
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition", frame)

    # --- ✅ In ra terminal mỗi lần có nhận diện ---
    if student_id != "unknown":
        print({
            "student_id": student_id,
            "confidence": conf
        })

    # --- Thoát bằng ESC ---
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

