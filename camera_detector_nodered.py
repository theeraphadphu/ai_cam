import os
import time
import base64
from datetime import datetime
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

NODE_RED_URL = os.getenv("NODE_RED_URL", "http://localhost:1880/detection")

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo26n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))

CLASS_LABELS_TH = {
    "person": "มีคนเข้ามาในพื้นที่",
    "cat": "พบแมว",
    "dog": "พบสุนัข",
    "backpack": "พบพัสดุ/กระเป๋าวางอยู่",
    "suitcase": "พบพัสดุ/กระเป๋าเดินทางวางอยู่",
    "handbag": "พบพัสดุ/กระเป๋าวางอยู่",
}
TARGET_CLASSES = set(CLASS_LABELS_TH.keys())

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "60"))
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", "0"))  # ถ้ามีกล้องหลายตัว ลองเปลี่ยนเป็น 1, 2, ...

FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))         # รัน YOLO ทุกๆ N เฟรม (video ยังลื่นเพราะ imshow ทุกเฟรม)
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "640"))   # ย่อภาพก่อนส่งเข้าโมเดล


def send_to_node_red(image_path: str, label: str, text_th: str, confidence: float, timestamp: str) -> None:
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "label": label,
            "text_th": text_th,
            "confidence": round(confidence, 2),
            "timestamp": timestamp,
            "image_base64": img_b64,
        }
        resp = requests.post(NODE_RED_URL, json=payload, timeout=15)
        if resp.status_code == 200:
            print(f"[Node-RED] ส่งข้อมูลสำเร็จ: {label}")
        else:
            print(f"[Node-RED] ส่งข้อมูลไม่สำเร็จ: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[Node-RED] เกิดข้อผิดพลาด: {e}")


def main():
    print("กำลังโหลดโมเดล YOLO ...")
    model = YOLO(MODEL_PATH)

    print("กำลังเปิดกล้อง ...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ลองเปลี่ยนค่า WEBCAM_INDEX เป็น 1 หรือ 2")
        return

    last_alert_time: dict[str, float] = {}
    frame_count = 0
    print("เริ่มตรวจจับ... กด 'q' ในหน้าต่างวิดีโอ หรือ Ctrl+C เพื่อหยุด")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("อ่านภาพจากกล้องไม่ได้")
                break

            frame_count += 1
            run_inference = (frame_count % FRAME_SKIP == 0)

            detected_this_frame = set()
            detected_conf = {}

            if run_inference:
                h, w = frame.shape[:2]
                if w > RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / w
                    small = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
                else:
                    small = frame
                    scale = 1.0

                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = model(small_rgb, verbose=False)

                for r in results:
                    for box in r.boxes:
                        name = model.names[int(box.cls)]
                        conf = float(box.conf)

                        # แปลงพิกัดกลับเป็นสเกลของเฟรมจริงก่อนวาดกรอบ
                        x1, y1, x2, y2 = [int(v / scale) for v in box.xyxy[0]]
                        color = (0, 255, 0) if name in TARGET_CLASSES else (150, 150, 150)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame, f"{name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )

                        if name in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                            detected_this_frame.add(name)
                            detected_conf[name] = conf

            cv2.imshow("Pi Camera - Object Detector", frame)

            now = time.time()
            for name in detected_this_frame:
                last = last_alert_time.get(name, 0)
                if now - last < COOLDOWN_SECONDS:
                    continue

                last_alert_time[name] = now
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                path = SNAPSHOT_DIR / f"{name}_{int(now)}.jpg"
                cv2.imwrite(str(path), frame)

                text_th = CLASS_LABELS_TH[name]
                print(f"[แจ้งเตือน] {text_th} เวลา: {timestamp}")
                send_to_node_red(str(path), name, text_th, detected_conf[name], timestamp)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nหยุดการทำงาน")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
