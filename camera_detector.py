import time
import base64
from datetime import datetime
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

CHANNEL_ACCESS_TOKEN = "rn9NnbVsUYTIpGfv/ABIXXvEXgPSZODjj23mMaKHEBuCPE2IyNbCdX5f56CfXQ0iKhJe2xc5IZCqvFcTU7AqOf8+89auPYDB5knU4EPV9XGi0lvdF6CXI71fJUsi1lb4PkvnGDGQdS8EKMxscFZrMAdB04t89/1O/w1cDnyilFU="
USER_ID = "U7c8231354b901f1e898c13c176d86bd4"

IMGBB_API_KEY = "e9fe4c2307d22ade9a596b9566ec3f6c"  

MODEL_PATH = "yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.5

CLASS_LABELS_TH = {
    "person": "มีคนเข้ามาในพื้นที่",
    "cat": "พบแมว",
    "dog": "พบสุนัข",
    "backpack": "พบพัสดุ/กระเป๋าวางอยู่",
    "suitcase": "พบพัสดุ/กระเป๋าเดินทางวางอยู่",
    "handbag": "พบพัสดุ/กระเป๋าวางอยู่",
}
TARGET_CLASSES = set(CLASS_LABELS_TH.keys())

COOLDOWN_SECONDS = 60
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

WEBCAM_INDEX = 0  # ถ้ามีกล้องหลายตัว ลองเปลี่ยนเป็น 1, 2, ...

def upload_image_to_imgbb(path: str) -> str | None:
    if not IMGBB_API_KEY:
        return None
    try:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": IMGBB_API_KEY, "image": img_b64},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["data"]["url"]
    except Exception as e:
        print(f"[imgbb] อัปโหลดรูปไม่สำเร็จ: {e}")
        return None


def send_line_message(text: str, image_url: str | None = None) -> None:
    headers = {
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    messages = [{"type": "text", "text": text}]
    if image_url:
        messages.append(
            {"type": "image", "originalContentUrl": image_url, "previewImageUrl": image_url}
        )
    payload = {"to": USER_ID, "messages": messages}

    try:
        resp = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers=headers,
            json=payload,
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[LINE] ส่งแจ้งเตือนไม่สำเร็จ: {resp.status_code} {resp.text}")
        else:
            print("[LINE] ส่งแจ้งเตือนสำเร็จ")
    except Exception as e:
        print(f"[LINE] เกิดข้อผิดพลาด: {e}")

def main():
    print("กำลังโหลดโมเดล YOLO ...")
    model = YOLO(MODEL_PATH)

    print("กำลังเปิดเว็บแคม ...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(" เปิดเว็บแคมไม่ได้ ลองเปลี่ยนค่า WEBCAM_INDEX เป็น 1 หรือ 2")
        return

    last_alert_time: dict[str, float] = {}
    print("เริ่มตรวจจับ... กด 'q' ในหน้าต่างวิดีโอ หรือ Ctrl+C เพื่อหยุด")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("อ่านภาพจากกล้องไม่ได้")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, verbose=False)

            detected_this_frame = set()

            for r in results:
                for box in r.boxes:
                    name = model.names[int(box.cls)]
                    conf = float(box.conf)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if name in TARGET_CLASSES else (150, 150, 150)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f"{name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )

                    if name in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                        detected_this_frame.add(name)

            cv2.imshow(" Test - Object Detector", frame)

            now = time.time()
            for name in detected_this_frame:
                last = last_alert_time.get(name, 0)
                if now - last < COOLDOWN_SECONDS:
                    continue

                last_alert_time[name] = now
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                path = SNAPSHOT_DIR / f"{name}_{int(now)}.jpg"
                cv2.imwrite(str(path), frame)

                text = f" {CLASS_LABELS_TH[name]}\nเวลา: {timestamp}\n"

                url = upload_image_to_imgbb(str(path))
                send_line_message(text, url)
                print(f"[แจ้งเตือน] {text}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nหยุดการทำงาน")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()