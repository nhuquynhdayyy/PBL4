import cv2, time, numpy as np
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face  # import hàm có sẵn

# --- Load model ---
hand_model = YOLO("runs/detect/train2/weights/best.pt")   # model phát hiện tay (vẫn load để sau có thể kết hợp)
person_model = YOLO("yolov8n.pt")  # model detect người (pretrained)

# --- Mở camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Bộ nhớ theo dõi ---
tracks = {}  # key=(cx,cy) -> {'heights','baseline','stand_frames','time_stand_start','last_recog'}

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
                'confirmed_stand': False
            }

        # cập nhật lịch sử chiều cao
        tracks[key]['heights'].append(h)
        baseline = tracks[key]['baseline']
        standing = h > 1.1 * baseline  # ngưỡng nhẹ hơn

        # --- CHỈ cập nhật baseline khi KHÔNG đứng ---
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
            tracks[key]['confirmed_stand'] = True

        # --- debug ---
        print(f"[DEBUG] h={h:.1f}, baseline={baseline:.1f}, stand_frames={tracks[key]['stand_frames']}, elapsed={elapsed:.1f}s, confirmed={tracks[key]['confirmed_stand']}")

        # --- nếu đứng ổn định -> nhận diện khuôn mặt ---
        if tracks[key]['confirmed_stand']:
            now = time.time()
            if now - tracks[key]['last_recog'] > 5:
                # --- Crop vùng mặt cận ---
                # fy1 = y1 + int(0.08 * (y2 - y1))
                # fy2 = y1 + int(0.35 * (y2 - y1))
                # fx1 = x1 + int(0.18 * (x2 - x1))
                # fx2 = x2 - int(0.18 * (x2 - x1))
                # ít zoom hơn (mở rộng vùng crop khuôn mặt)
                fy1 = y1 + int(0.05 * (y2 - y1))   # bắt đầu gần đỉnh đầu hơn
                fy2 = y1 + int(0.45 * (y2 - y1))   # kết thúc thấp hơn (xuống gần vai)
                fx1 = x1 + int(0.10 * (x2 - x1))   # giảm cắt 2 bên
                fx2 = x2 - int(0.10 * (x2 - x1))


                face_crop = frame[fy1:fy2, fx1:fx2]
                if face_crop.size > 0:
                    zoom_face = cv2.resize(face_crop, (480, 480))
                    cv2.imshow("Zoom Face", zoom_face)

                    student_id, conf = recognize_face(zoom_face)
                    print(f"[INFO] Đứng ổn định: {student_id} ({conf:.2f})")
                    cv2.putText(frame, f"{student_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tracks[key]['last_recog'] = now

        # --- hiển thị khung người ---
        if tracks[key]['confirmed_stand']:
            color = (0, 255, 255)
            cv2.putText(frame, "STAND", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Hand + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
