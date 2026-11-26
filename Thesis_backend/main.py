import atexit
import signal
import cv2
import time
import threading
import os
import json
import numpy as np
import requests
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from ultralytics import YOLO
from supabase import create_client, Client

# ======================================================================
# Setup
# ======================================================================
app = Flask(__name__)
model = YOLO("best.pt")
print(f"This model can detect: {model.names}")

ESP32_CAM_URL = os.environ.get("ESP32_CAM_URL", "http://192.168.1.119:81/stream")
ENABLE_LAN_READER = os.environ.get("ENABLE_LAN_READER", "0") == "1"

# Optional shared secret so random internet traffic cannot spoof ESP32 frames.
# Set CAM_PUSH_TOKEN in both ESP32 code and Railway env vars if desired.
CAM_PUSH_TOKEN = os.environ.get("CAM_PUSH_TOKEN", "")
CAM_ONLINE_TIMEOUT = 10.0  # seconds without a pushed frame before marking camera offline
CLIENT_IDLE_TIMEOUT = 12.0  # seconds without yielding a frame before ending client

# ---------- Supabase ----------
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://jodqigqgnxiakhcdnzlw.supabase.co"
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY", "sb_publishable_aBUBA9pGcHeCqoY7roQOeQ_x75IS62V"
)
if not SUPABASE_KEY:
    print("[Supabase] WARNING: SUPABASE_KEY not set - detections will NOT be saved.")

TABLE_LIVE = "live_sessions"
TABLE_SCAN = "scheduled_scans"
SNAPSHOT_BUCKET = "session_snapshots"

supabase: Client = None
if SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[Supabase] Connected.")
    except Exception as e:
        print(f"[Supabase] Init failed: {e}")

# ---------- Tuning & Automated Lettuce-Exclusive Filtering ----------
INFER_INTERVAL = 0.15
INFER_IMGSZ = 640
JPEG_QUALITY = 80
MEMORY_TIME = 15.0
CHUNK_SIZE = 8192
IOU_MATCH_THRESHOLD = 0.15
MIN_FRAMES_TO_CONFIRM = 1
CONF_THRESHOLD = 0.25

SESSION_FLUSH_INTERVAL = 60  # seconds (optional periodic flush if enabled)
ENABLE_PERIODIC_FLUSH = os.environ.get("ENABLE_PERIODIC_FLUSH", "0") == "1"
MIN_CONFIRMED_TO_SAVE = 1

# ---------- Colors & Strict Whitelist ----------
COLOR_MAP = {
    "healthy": (0, 255, 0),
    "downy mildew": (0, 165, 255),
    "powdery mildew": (128, 0, 128),
    "bacterial leaf spot": (0, 0, 255),
}
DEFAULT_COLOR = (200, 200, 200)

VALID_CLASSES = {name.lower() for name in COLOR_MAP.keys()}


def get_class_color(class_name: str):
    return COLOR_MAP.get(class_name.strip().lower(), DEFAULT_COLOR)


def is_exclusive_lettuce(frame, box, conf) -> bool:
    """
    Lettuce-Exclusive Validation Filter:
    1. Enforces a confidence floor (0.25) to allow valid healthy & infected detections.
    2. Uses a broader HSV color space mask to verify plant foliage and
       accommodate dark greens, yellow-greens, bright highlights, and lesions.
    """
    if conf < CONF_THRESHOLD:
        return False

    try:
        x1, y1, x2, y2 = box
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 8 or box_height <= 8:
            return False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_foliage = np.array([5, 10, 10])
        upper_foliage = np.array([105, 255, 255])

        mask = cv2.inRange(hsv, lower_foliage, upper_foliage)
        foliage_pixel_count = cv2.countNonZero(mask)
        total_pixels = box_width * box_height

        foliage_ratio = foliage_pixel_count / float(total_pixels)
        if foliage_ratio < 0.01:
            return False

        return True
    except Exception:
        return True


def health_label_for(diseased: int) -> str:
    if diseased == 0:
        return "Healthy (100%)"
    if diseased <= 2:
        return "Mild Infection"
    if diseased <= 4:
        return "Moderate Infection"
    return "Severe Infection"


# ======================================================================
# Shared state
# ======================================================================
latest_frame = None
last_annotated_frame = None
frame_lock = threading.Lock()
frame_id = 0
last_frame_push_time = 0.0
pushed_frames_count = 0

model_lock = threading.Lock()

notif_lock = threading.Lock()
notifications = deque(maxlen=200)

NOTIF_FILE = "notif_state.json"


def _load_notif_counter():
    try:
        with open(NOTIF_FILE, "r") as f:
            return int(json.load(f).get("counter", 0))
    except Exception:
        return 0


notif_counter = _load_notif_counter()


def _save_notif_counter():
    try:
        with open(NOTIF_FILE, "w") as f:
            json.dump({"counter": notif_counter}, f)
    except Exception:
        pass


def is_camera_online() -> bool:
    with frame_lock:
        if latest_frame is None or last_frame_push_time == 0.0:
            return False
        return (time.time() - last_frame_push_time) <= CAM_ONLINE_TIMEOUT


# ======================================================================
# Session state
# ======================================================================
session_lock = threading.Lock()
session_active = False
session_saved = False
session_start_time = None
_shutting_down = False


# ======================================================================
# Tracker
# ======================================================================
class Track:
    __slots__ = ("track_id", "cls_votes", "box", "last_seen", "last_conf")

    def __init__(self, track_id, cls_id, conf, box, now):
        self.track_id = track_id
        self.cls_votes = {cls_id: {"count": 1, "conf_sum": float(conf)}}
        self.box = box
        self.last_seen = now
        self.last_conf = float(conf)

    def update(self, cls_id, conf, box, now):
        self.box = box
        self.last_seen = now
        self.last_conf = float(conf)
        v = self.cls_votes.get(cls_id)
        if v is None:
            self.cls_votes[cls_id] = {"count": 1, "conf_sum": float(conf)}
        else:
            v["count"] += 1
            v["conf_sum"] += float(conf)

    def total_seen(self):
        return sum(v["count"] for v in self.cls_votes.values())

    def best_class(self):
        return max(
            self.cls_votes.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["conf_sum"]),
        )[0]

    def best_confidence(self):
        _, stats = max(
            self.cls_votes.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["conf_sum"]),
        )
        return stats["conf_sum"] / stats["count"]

    def confirmed(self):
        return self.total_seen() >= MIN_FRAMES_TO_CONFIRM


tracks = []
tracks_lock = threading.Lock()
next_track_id = 1


def compute_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def tracker_update(detections, now):
    global next_track_id
    with tracks_lock:
        used = [False] * len(tracks)
        for det in detections:
            best_idx = -1
            best_iou = IOU_MATCH_THRESHOLD
            for i, tr in enumerate(tracks):
                if used[i]:
                    continue
                iou = compute_iou(tr.box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                tracks[best_idx].update(det["cls_id"], det["conf"], det["box"], now)
                used[best_idx] = True
            else:
                tracks.append(
                    Track(next_track_id, det["cls_id"], det["conf"], det["box"], now)
                )
                used.append(True)
                next_track_id += 1


# ======================================================================
# Supabase save - Live Stream Session
# ======================================================================
def _build_session_row():
    with tracks_lock:
        snapshot = list(tracks)
        raw_track_count = len(snapshot)

    confirmed = []
    for tr in snapshot:
        if not tr.confirmed():
            continue
        cls_id = tr.best_class()
        class_name = model.names.get(cls_id, f"class_{cls_id}").strip().lower()
        if class_name not in VALID_CLASSES:
            continue
        confirmed.append((tr, class_name))

    if len(confirmed) < MIN_CONFIRMED_TO_SAVE:
        return None, len(confirmed), raw_track_count

    counts = {}
    confs = []
    for tr, class_name in confirmed:
        counts[class_name] = counts.get(class_name, 0) + 1
        confs.append(tr.best_confidence())

    healthy = counts.get("healthy", 0)
    powdery = counts.get("powdery mildew", 0)
    downy = counts.get("downy mildew", 0)
    bacterial = counts.get("bacterial leaf spot", 0)
    diseased = powdery + downy + bacterial

    with session_lock:
        start_ts = session_start_time
    duration = int(time.time() - start_ts) if start_ts else 0

    row = {
        "overall_health": health_label_for(diseased),
        "overall_confidence": round(sum(confs) / len(confs), 4),
        "healthy_count": healthy,
        "powdery_mildew_count": powdery,
        "downy_mildew_count": downy,
        "bacterial_leaf_spot_count": bacterial,
        "total_diseases_detected": diseased,
        "total_unique_plants": healthy + diseased,
        "session_duration_seconds": duration,
    }
    return row, len(confirmed), raw_track_count


def _capture_annotated_snapshot():
    """Returns a copy of the latest frame with bounding boxes & summary drawn."""
    snapshot_frame = None
    with frame_lock:
        if latest_frame is not None:
            snapshot_frame = latest_frame.copy()
        elif last_annotated_frame is not None:
            return last_annotated_frame.copy()

    if snapshot_frame is None:
        return None

    draw_tracks_and_summary(snapshot_frame)
    with frame_lock:
        globals()["last_annotated_frame"] = snapshot_frame.copy()
    return snapshot_frame


def save_session_to_supabase(force: bool = False):
    if supabase is None:
        print("[Supabase] Not configured, cannot save session.")
        return False

    row, n_confirmed, raw_count = _build_session_row()
    if row is None and not force:
        print(
            f"[Session] Skipping save: {n_confirmed} confirmed out of {raw_count} raw tracks."
        )
        return False

    if row is None and force:
        with session_lock:
            start_ts = session_start_time
        duration = int(time.time() - start_ts) if start_ts else 0
        row = {
            "overall_health": health_label_for(0),
            "overall_confidence": 0.0,
            "healthy_count": 0,
            "powdery_mildew_count": 0,
            "downy_mildew_count": 0,
            "bacterial_leaf_spot_count": 0,
            "total_diseases_detected": 0,
            "total_unique_plants": 0,
            "session_duration_seconds": duration,
        }

    snapshot_frame = _capture_annotated_snapshot()
    image_url = None

    if snapshot_frame is not None:
        try:
            success, encoded_img = cv2.imencode(
                ".jpg", snapshot_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if success:
                file_bytes = encoded_img.tobytes()
                file_name = f"session_{int(time.time())}.jpg"

                supabase.storage.from_(SNAPSHOT_BUCKET).upload(
                    path=file_name,
                    file=file_bytes,
                    file_options={"content-type": "image/jpeg"},
                )
                image_url = supabase.storage.from_(SNAPSHOT_BUCKET).get_public_url(
                    file_name
                )
                print(f"[Supabase] Snapshot with bounding boxes uploaded: {image_url}")
        except Exception as e:
            print(f"[Supabase] Image upload failed: {type(e).__name__}: {e}")
    else:
        print("[Supabase] No frame available for bounding box snapshot.")

    row["image_url"] = image_url

    try:
        supabase.table(TABLE_LIVE).insert(row).execute()
        print(
            f"[Session] Saved to {TABLE_LIVE}: status={row['overall_health']} "
            f"healthy={row['healthy_count']} diseases={row['total_diseases_detected']} "
            f"image_url={bool(image_url)}"
        )
        return True
    except Exception as e:
        print(f"[Supabase] Session insert failed: {type(e).__name__}: {e}")
        return False


def _reset_session_tracks_locked():
    global tracks, next_track_id, session_start_time
    with tracks_lock:
        tracks = []
        next_track_id = 1
    session_start_time = None


# ======================================================================
# Session lifecycle & exit handlers
# ======================================================================
def on_client_connected():
    global session_active, session_saved, session_start_time
    with session_lock:
        if not session_active:
            session_active = True
            session_saved = False
            session_start_time = time.time()
            _reset_session_tracks_locked()
            print("[Session] Started. Tracker cleared.")


def on_client_disconnected(last_client: bool):
    global session_active, session_saved
    if not last_client:
        return

    with session_lock:
        if not session_active or session_saved:
            return
        session_saved = True
        session_active = False

    print("[Session] Stream ended. Saving session with bounding boxes to Supabase...")
    save_session_to_supabase(force=False)
    with session_lock:
        _reset_session_tracks_locked()
    print("[Session] Tracker reset. Ready for next session.")


def flush_active_session_on_exit(reason: str = "exit"):
    """Ensures any active or unsaved tracked detections are saved when app exits."""
    global session_active, session_saved, _shutting_down
    with session_lock:
        if _shutting_down and session_saved:
            return
        _shutting_down = True
        with tracks_lock:
            has_tracks = len(tracks) > 0
        if not session_active and not has_tracks:
            return
        session_active = False
        session_saved = True

    print(f"[Session] Application {reason} detected. Saving session to Supabase...")
    save_session_to_supabase(force=False)
    with session_lock:
        _reset_session_tracks_locked()


def _handle_signal(signum, _frame):
    flush_active_session_on_exit(reason=f"signal {signum}")
    raise SystemExit(0)


atexit.register(flush_active_session_on_exit, "atexit")
for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _handle_signal)
    except Exception:
        pass


def session_flush_thread():
    global session_start_time
    while True:
        time.sleep(SESSION_FLUSH_INTERVAL)
        if not ENABLE_PERIODIC_FLUSH:
            continue
        with session_lock:
            if not session_active:
                continue
            saved = save_session_to_supabase(force=False)
            if saved:
                _reset_session_tracks_locked()
                session_start_time = time.time()
                print("[Session] Periodic flush complete. Window restarted.")


# ======================================================================
# Reader thread (Optional LAN stream reader; use /push_frame over internet)
# ======================================================================
def reader_thread():
    global latest_frame, frame_id, last_frame_push_time
    session = requests.Session()
    backoff = 1

    while True:
        try:
            print(f"[Reader] Connecting to {ESP32_CAM_URL}")
            resp = session.get(
                ESP32_CAM_URL,
                stream=True,
                timeout=(3, 5),
                headers={"Cache-Control": "no-cache"},
            )
            resp.raise_for_status()
            print("[Reader] Connected.")
            backoff = 1
            buf = bytearray()

            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                buf.extend(chunk)

                last_jpg = None
                while True:
                    a = buf.find(b"\xff\xd8")
                    if a == -1:
                        buf.clear()
                        break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b == -1:
                        if a > 0:
                            del buf[:a]
                        break
                    last_jpg = bytes(buf[a : b + 2])
                    del buf[: b + 2]

                if len(buf) > 4_000_000:
                    buf.clear()

                if last_jpg is not None:
                    frame = cv2.imdecode(
                        np.frombuffer(last_jpg, np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        with frame_lock:
                            latest_frame = frame
                            frame_id += 1
                            last_frame_push_time = time.time()

        except Exception as e:
            print(f"[Reader] Connection dropped: {e}. Reconnecting in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)


# ======================================================================
# Inference thread
# ======================================================================
def extract_box(det):
    try:
        if hasattr(det, "xyxyxyxy") and det.xyxyxyxy is not None:
            pts = det.xyxyxyxy[0].cpu().numpy()
            return (
                int(np.min(pts[:, 0])),
                int(np.min(pts[:, 1])),
                int(np.max(pts[:, 0])),
                int(np.max(pts[:, 1])),
            )
        if hasattr(det, "xyxy") and det.xyxy is not None:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            return (x1, y1, x2, y2)
    except Exception:
        pass
    return None


def inference_thread():
    last_infer = 0.0
    last_seen_frame = -1
    while True:
        now = time.time()
        if now - last_infer < INFER_INTERVAL:
            time.sleep(0.01)
            continue

        frame = None
        fid = -1
        with frame_lock:
            if latest_frame is not None:
                fid = frame_id
                frame = latest_frame.copy()

        if frame is None or fid == last_seen_frame:
            time.sleep(0.02)
            continue

        last_seen_frame = fid
        last_infer = now

        try:
            with model_lock:
                results = model.predict(
                    source=frame, conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, verbose=False
                )
            r0 = results[0]

            if r0.boxes is not None and len(r0.boxes) > 0:
                dets = list(r0.boxes)
            elif hasattr(r0, "obb") and r0.obb is not None and len(r0.obb) > 0:
                dets = list(r0.obb)
            else:
                dets = []

            clean = []
            for det in dets:
                cls_id = int(det.cls[0])
                cls_name = model.names.get(cls_id, "").strip().lower()

                if cls_name not in VALID_CLASSES:
                    continue

                box = extract_box(det)
                if box is None:
                    continue

                conf = float(det.conf[0]) if det.conf is not None else 0.0

                if not is_exclusive_lettuce(frame, box, conf):
                    continue

                clean.append({
                    "cls_id": cls_id,
                    "conf": conf,
                    "box": box,
                })

            tracker_update(clean, now)

            if clean:
                annotated = frame.copy()
                draw_tracks_and_summary(annotated)
                with frame_lock:
                    globals()["last_annotated_frame"] = annotated

        except Exception as e:
            print(f"[Infer] {type(e).__name__}: {e}")


# ======================================================================
# Drawing & Total Diseases Screen Overlay
# ======================================================================
def draw_tracks_and_summary(frame):
    now = time.time()
    with tracks_lock:
        snapshot = list(tracks)

    # 1. Draw bounding boxes
    for tr in snapshot:
        age = now - tr.last_seen
        if age > MEMORY_TIME:
            continue
        cls_id = tr.best_class()
        name = model.names.get(cls_id, "").strip().lower()

        if name not in VALID_CLASSES:
            continue

        if not is_exclusive_lettuce(frame, tr.box, tr.best_confidence()):
            continue

        x1, y1, x2, y2 = tr.box
        color = get_class_color(name)
        display_name = model.names.get(cls_id, str(cls_id))
        if age < 0.5:
            label = f"#{tr.track_id} {display_name} {tr.last_conf:.2f}"
        else:
            label = f"#{tr.track_id} {display_name} "
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # 2. Compute counts for the summary banner
    counts = {}
    for tr in snapshot:
        if not tr.confirmed():
            continue
        name = model.names.get(tr.best_class(), "").strip().lower()
        if name not in VALID_CLASSES:
            continue
        if not is_exclusive_lettuce(frame, tr.box, tr.best_confidence()):
            continue
        counts[name] = counts.get(name, 0) + 1

    powdery = counts.get("powdery mildew", 0)
    downy = counts.get("downy mildew", 0)
    bacterial = counts.get("bacterial leaf spot", 0)
    diseased_total = powdery + downy + bacterial

    # 3. Draw summary bar with total diseases only, positioned at Y=80
    summary_text = f"total diseses : {diseased_total}"

    cv2.putText(
        frame, summary_text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4
    )
    cv2.putText(
        frame,
        summary_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


# ======================================================================
# Stream generator
# ======================================================================
active_clients = 0
clients_lock = threading.Lock()


def generate_frames():
    global active_clients

    with clients_lock:
        active_clients += 1
        is_first = active_clients == 1
    if is_first:
        on_client_connected()
    print(f"[Stream] Client connected. Active: {active_clients}")

    last_yielded = -1
    last_active_at = time.time()

    try:
        while True:
            now = time.time()
            frame = None
            fid = last_yielded

            with frame_lock:
                if latest_frame is not None and frame_id != last_yielded:
                    fid = frame_id
                    frame = latest_frame.copy()

            if frame is None:
                if (now - last_active_at) >= CLIENT_IDLE_TIMEOUT:
                    print(
                        f"[Stream] Client inactive for {CLIENT_IDLE_TIMEOUT:.0f}s. Ending stream."
                    )
                    break
                time.sleep(0.02)
                continue

            draw_tracks_and_summary(frame)
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                time.sleep(0.02)
                continue

            last_yielded = fid
            last_active_at = time.time()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    except GeneratorExit:
        print("[Stream] Client closed connection.")
    except Exception as e:
        print(f"[Stream] Client stream error ({type(e).__name__}): {e}")
    finally:
        with clients_lock:
            active_clients = max(0, active_clients - 1)
            was_last = active_clients == 0
        print(f"[Stream] Client disconnected. Active: {active_clients}")

        if was_last:
            on_client_disconnected(last_client=True)


# ======================================================================
# Scheduled-scan analysis (For Notifications / Periodic Scans)
# ======================================================================
def analyze_image(frame):
    with model_lock:
        results = model.predict(
            source=frame, conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, verbose=False
        )
    r0 = results[0]

    if r0.boxes is not None and len(r0.boxes) > 0:
        dets = list(r0.boxes)
    elif hasattr(r0, "obb") and r0.obb is not None and len(r0.obb) > 0:
        dets = list(r0.obb)
    else:
        dets = []

    counts, confs = {}, []
    for det in dets:
        cls_id = int(det.cls[0])
        name = model.names.get(cls_id, "").strip().lower()
        conf = float(det.conf[0]) if det.conf is not None else 0.0
        print(f"[Analyze] Raw det: class={name} ({cls_id}) conf={conf:.3f}")

        if name not in VALID_CLASSES:
            continue

        box = extract_box(det)
        if box is None:
            continue

        if not is_exclusive_lettuce(frame, box, conf):
            print(f"[Analyze] Filtered out by is_exclusive_lettuce: {name} conf={conf:.3f}")
            continue

        counts[name] = counts.get(name, 0) + 1
        confs.append(conf)

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return counts, avg_conf


def save_scan_to_supabase(counts, avg_conf):
    healthy = counts.get("healthy", 0)
    powdery = counts.get("powdery mildew", 0)
    downy = counts.get("downy mildew", 0)
    bacterial = counts.get("bacterial leaf spot", 0)
    diseased = powdery + downy + bacterial
    total = healthy + diseased

    row = {
        "overall_health": health_label_for(diseased),
        "overall_confidence": round(avg_conf, 4),
        "healthy_count": healthy,
        "powdery_mildew_count": powdery,
        "downy_mildew_count": downy,
        "bacterial_leaf_spot_count": bacterial,
        "total_diseases_detected": diseased,
        "total_unique_plants": total,
    }

    if supabase is None:
        print("[Supabase] Not configured, scan not saved.")
        return row

    if total == 0:
        print("[Scan] No valid lettuce detected, nothing saved.")
        return row

    try:
        supabase.table(TABLE_SCAN).insert(row).execute()
        print(
            f"[Scan] Saved to {TABLE_SCAN}: {row['overall_health']} | "
            f"healthy={healthy} diseased={diseased}"
        )
    except Exception as e:
        print(f"[Supabase] Scan insert failed: {type(e).__name__}: {e}")
    return row


def push_notification(row):
    global notif_counter
    healthy = row["healthy_count"]
    diseased = row["total_diseases_detected"]
    total = healthy + diseased

    if total == 0:
        title, msg = "Lettuce scan", "No lettuce detected in the latest capture."
    elif diseased == 0:
        title = "All lettuce healthy"
        msg = f"{total} plant(s) scanned, all healthy."
    else:
        parts = []
        if row["downy_mildew_count"]:
            parts.append(f"{row['downy_mildew_count']} downy mildew")
        if row["powdery_mildew_count"]:
            parts.append(f"{row['powdery_mildew_count']} powdery mildew")
        if row["bacterial_leaf_spot_count"]:
            parts.append(f"{row['bacterial_leaf_spot_count']} bacterial leaf spot")
        title = f"{diseased} diseased lettuce detected"
        msg = (
            f"{healthy} healthy, {diseased} diseased ({', '.join(parts)}). "
            f"Status: {row['overall_health']}"
        )

    with notif_lock:
        notif_counter += 1
        notifications.append({
            "id": notif_counter,
            "title": title,
            "message": msg,
            "healthy": healthy,
            "diseased": diseased,
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        _save_notif_counter()
        print(f"[Notify] #{notif_counter} queued: {title}")


# ======================================================================
# Routes
# ======================================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lettuce Disease Detection</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; background: #121212; color: #ffffff; }
        h1 { margin-top: 30px; }
        img { border: 4px solid #444; border-radius: 8px; margin-top: 20px; max-width: 95%; }
    </style>
</head>
<body>
    <h1>ESP32-CAM Lettuce Detection</h1>
    <img src="/video_feed" alt="YOLO Video Stream">
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/push_frame", methods=["POST"])
def push_frame():
    """Receives live JPEG frames pushed over the internet by the ESP32-CAM."""
    global latest_frame, frame_id, last_frame_push_time, pushed_frames_count

    if CAM_PUSH_TOKEN:
        token = request.headers.get("X-Cam-Token", "")
        if token != CAM_PUSH_TOKEN:
            return jsonify({"error": "unauthorized"}), 401

    data = request.get_data()
    if not data:
        return jsonify({"error": "empty body"}), 400

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad image"}), 400

    with frame_lock:
        latest_frame = frame
        frame_id += 1
        last_frame_push_time = time.time()
        pushed_frames_count += 1
        count = pushed_frames_count

    if count % 50 == 1:
        print(f"[Push] Received frame #{count} ({frame.shape[1]}x{frame.shape[0]})")

    return jsonify({"ok": True, "frame_id": frame_id})


@app.route("/cam_status")
def cam_status():
    with frame_lock:
        last_push = last_frame_push_time
        total_pushed = pushed_frames_count
    age = round(time.time() - last_push, 2) if last_push > 0 else None
    return jsonify({
        "online": is_camera_online(),
        "last_push_age_seconds": age,
        "total_frames_pushed": total_pushed,
    })


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_data()
    if not data:
        return jsonify({"error": "empty body"}), 400

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad image"}), 400

    try:
        counts, avg_conf = analyze_image(frame)

        total_lettuce = sum(counts.values())
        if total_lettuce == 0:
            print(
                "[Upload Error] Rejected: Non-lettuce or invalid image content detected."
            )
            return jsonify({
                "error": "Invalid image: No valid lettuce detected. This system is strictly exclusive for lettuce crops and lettuce diseases only."
            }), 400

        row = save_scan_to_supabase(counts, avg_conf)
        push_notification(row)
        print(f"[Upload] Scan done: {counts} avg_conf={avg_conf:.3f}")
        return jsonify({"ok": True, "result": row})
    except Exception as e:
        print(f"[Upload] {type(e).__name__}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications")
@app.route("/notifications")
def notifications_api():
    try:
        since = int(request.args.get("since", -1))
    except ValueError:
        since = -1

    with notif_lock:
        latest = notif_counter
        if since < 0:
            items = [notifications[-1]] if notifications else []
        else:
            items = [n for n in notifications if n["id"] > since]

    return jsonify({"latest_id": latest, "items": items})


@app.route("/api/counts")
@app.route("/counts")
def counts_api():
    with tracks_lock:
        snapshot = list(tracks)

    per_class = {}
    for tr in snapshot:
        if not tr.confirmed():
            continue
        name = model.names.get(tr.best_class(), "").strip().lower()

        if name not in VALID_CLASSES:
            continue

        per_class[name] = per_class.get(name, 0) + 1

    total = sum(per_class.values())
    healthy = per_class.get("healthy", 0)
    diseased = total - healthy

    return jsonify({
        "camera_online": is_camera_online(),
        "per_class": per_class,
        "total_unique_plants": total,
        "healthy": healthy,
        "diseased": diseased,
    })


@app.route("/api/history")
@app.route("/history")
def history():
    if supabase is None:
        return jsonify({"error": "supabase not configured"})

    source = request.args.get("source", "live").lower()
    table = {"live": TABLE_LIVE, "scan": TABLE_SCAN}.get(source)
    if table is None:
        return jsonify({"error": "source must be 'live' or 'scan'"}), 400

    try:
        limit = int(request.args.get("limit", 50))
        res = (
            supabase.table(table)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return jsonify({"source": source, "records": res.data or []})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/session/force_save", methods=["POST"])
def force_save_session():
    force = request.args.get("force", "0") == "1"
    ok = save_session_to_supabase(force=force)
    return jsonify({"saved": ok})


# Start background threads once when module loads
_bg_threads_started = False
_bg_threads_lock = threading.Lock()


def ensure_background_threads():
    global _bg_threads_started
    with _bg_threads_lock:
        if not _bg_threads_started:
            threading.Thread(target=inference_thread, daemon=True).start()
            threading.Thread(target=session_flush_thread, daemon=True).start()
            _bg_threads_started = True


ensure_background_threads()

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    ensure_background_threads()
    if ENABLE_LAN_READER:
        threading.Thread(target=reader_thread, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    print(f"[Main] Background threads started. Serving on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
