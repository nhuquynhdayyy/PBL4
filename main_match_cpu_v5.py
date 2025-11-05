import cv2, time, numpy as np
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face  # import hàm nhận diện khuôn mặt

# --- Load model ---
hand_model = YOLO("runs/detect/train2/weights/best.pt")   # model phát hiện tay (có thể bỏ nếu chưa cần)
person_model = YOLO("yolov8n.pt")  # model detect người (pretrained)

# --- Mở camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Bộ nhớ theo dõi ---
tracks = {}  # theo dõi trạng thái của từng người
last_display = {"img": None, "label": "", "time": 0}  # hiển thị người nhận diện gần nhất

print("[INFO] Bắt đầu theo dõi... (Nhấn ESC để thoát)")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- Detect person ---
    res_person = person_model.predict(frame, imgsz=640, conf=0.3, device='cpu', verbose=False)[0]
    person_boxes = []
    for box in res_person.boxes:
        cls = int(box.cls[0])
        if person_model.names[cls] == "person":
            person_boxes.append(list(map(int, box.xyxy[0].cpu().numpy())))

    # --- Xử lý từng người ---
    for pb in person_boxes:
        x1, y1, x2, y2 = pb
        h = y2 - y1
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        key = (cx//20, cy//20)

        if key not in tracks:
            tracks[key] = {
                'heights': deque(maxlen=5),
                'baseline': h,
                'stand_frames': 0,
                'time_stand_start': None,
                'last_recog': 0,
                'confirmed_stand': False,
                'time_confirmed': 0
            }

        # cập nhật chiều cao
        tracks[key]['heights'].append(h)
        baseline = tracks[key]['baseline']
        standing = h > 1.1 * baseline  # ngưỡng đứng lên

        # cập nhật baseline khi đang ngồi
        if len(tracks[key]['heights']) == 5 and not standing:
            tracks[key]['baseline'] = np.median(tracks[key]['heights'])

        # --- xử lý thời gian đứng ---
        if standing:
            if tracks[key]['time_stand_start'] is None:
                tracks[key]['time_stand_start'] = time.time()
            tracks[key]['stand_frames'] += 1
        else:
            tracks[key]['stand_frames'] = 0
            tracks[key]['time_stand_start'] = None
            tracks[key]['confirmed_stand'] = False

        elapsed = (time.time() - tracks[key]['time_stand_start']) if tracks[key]['time_stand_start'] else 0

        # --- xác nhận STAND nếu đứng > 1.0s ---
        if standing and elapsed >= 1.0:
            if not tracks[key]['confirmed_stand']:
                tracks[key]['time_confirmed'] = time.time()
            tracks[key]['confirmed_stand'] = True

        # --- nếu đứng ổn định đủ lâu -> nhận diện ---
        if tracks[key]['confirmed_stand']:
            now = time.time()
            stable_time = now - tracks[key].get('time_confirmed', 0)

            if stable_time >= 1.0 and now - tracks[key]['last_recog'] > 5:
                fy1 = y1 + int(0.05 * (y2 - y1))
                fy2 = y1 + int(0.45 * (y2 - y1))
                fx1 = x1 + int(0.10 * (x2 - x1))
                fx2 = x2 - int(0.10 * (x2 - x1))

                # Lấy 3 frame để trung bình ảnh khuôn mặt
                face_frames = []
                for _ in range(3):
                    ok2, frame2 = cap.read()
                    if ok2:
                        face_frames.append(frame2[fy1:fy2, fx1:fx2])
                        time.sleep(0.1)

                if face_frames:
                    avg_face = np.mean(face_frames, axis=0).astype(np.uint8)
                    zoom_face = cv2.resize(avg_face, (480, 480))
                    cv2.imshow("Zoom Face", zoom_face)

                    student_id, conf = recognize_face(zoom_face)
                    print(f"[INFO] Đứng ổn định: {student_id} ({conf:.2f})")

                    if student_id and student_id != "unknown" and conf >= 0.5:
                        color = (0, 0, 255)
                        label = f"{student_id} ({conf*100:.0f}%)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        # ✅ Cập nhật overlay hiển thị kết quả
                        face_thumb = cv2.resize(zoom_face, (120, 120))
                        last_display["img"] = face_thumb
                        last_display["label"] = label
                        last_display["time"] = now

                    else:
                        color = (0, 255, 255)
                        cv2.putText(frame, "STAND", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    tracks[key]['last_recog'] = now

        # --- hiển thị khung người ---
        if tracks[key]['confirmed_stand']:
            color = (0, 255, 255)
            cv2.putText(frame, "STAND", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # --- Hiển thị overlay người nhận diện gần nhất ---
    if last_display["img"] is not None:
        thumb = last_display["img"]
        label = last_display["label"]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (160, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        frame[20:140, 20:140] = thumb
        cv2.putText(frame, label, (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Hand + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
