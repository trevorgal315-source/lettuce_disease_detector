import cv2
import time
import threading
import os
import json
import numpy as np
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

# Optional shared secret so random internet traffic cannot spoof ESP32 frames.
CAM_PUSH_TOKEN = os.environ.get("CAM_PUSH_TOKEN", "")
CAM_ONLINE_TIMEOUT = 10.0  # seconds without a pushed frame before marking camera offline

# ---------- Supabase ----------
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://isslbortjspiazoslkdu.supabase.co"
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY", "sb_secret_omRLJyMKoDkAhPKeTmsAtg_Yb05T48f"
)
if not SUPABASE_KEY:
    print("[Supabase] WARNING: SUPABASE_KEY not set - detections will NOT be saved.")

TABLE_LIVE = "live_sessions"
TABLE_SCAN = "scheduled_scans"

supabase: Client = None
if SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[Supabase] Connected.")
    except Exception as e:
        print(f"[Supabase] Init failed: {e}")

# ---------- Tuning, Camera Height & Filtering ----------
INFER_INTERVAL = float(os.environ.get("INFER_INTERVAL", "0.5"))
INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ", "512"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.15"))
MIN_FOLIAGE_RATIO = float(os.environ.get("MIN_FOLIAGE_RATIO", "0.005"))
MIN_BOX_PX = int(os.environ.get("MIN_BOX_PX", "5"))

# Camera height & scale adaptation when camera mount height changes
CAMERA_HEIGHT_CM = float(os.environ.get("CAMERA_HEIGHT_CM", "45.0"))
REF_BOX_HEIGHT_RATIO = float(os.environ.get("REF_BOX_HEIGHT_RATIO", "0.35"))
MIN_HEIGHT_RATIO = float(os.environ.get("MIN_HEIGHT_RATIO", "0.02"))
FAR_ZOOM_MARGIN = float(os.environ.get("FAR_ZOOM_MARGIN", "0.0"))

# Bounding box smoothing factor (0.15 = very smooth, 0.5 = faster response)
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", "0.25"))
DEBUG_DETECTIONS = os.environ.get("DEBUG_DETECTIONS", "1") == "1"

JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "75"))
MEMORY_TIME = 15.0
IOU_MATCH_THRESHOLD = 0.15
MIN_FRAMES_TO_CONFIRM = 1

SESSION_FLUSH_INTERVAL = 60  # seconds
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


def normalize_class_name(raw_name: str) -> str:
    return raw_name.replace("_", " ").strip().lower()


def get_class_color(class_name: str):
    return COLOR_MAP.get(normalize_class_name(class_name), DEFAULT_COLOR)


def compute_height_metrics(box, frame_shape):
    """
    Computes bounding-box height in pixels, relative height ratio in the frame,
    and an estimated height in cm calibrated against CAMERA_HEIGHT_CM and REF_BOX_HEIGHT_RATIO.
    """
    h, w = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = box
    box_h = max(1.0, float(y2 - y1))
    box_w = max(1.0, float(x2 - x1))
    height_ratio = box_h / max(1.0, float(h))
    ref_ratio = max(0.01, REF_BOX_HEIGHT_RATIO)
    est_height_cm = round((height_ratio / ref_ratio) * (CAMERA_HEIGHT_CM * 0.25), 1)
    return {
        "box_width_px": int(round(box_w)),
        "box_height_px": int(round(box_h)),
        "height_ratio": round(height_ratio, 4),
        "est_height_cm": est_height_cm,
    }


def prepare_inference_view(frame):
    """
    Optional center crop when camera is mounted high/far; returns (view, ox, oy).
    """
    margin = max(0.0, min(0.4, FAR_ZOOM_MARGIN))
    if margin <= 0.0:
        return frame, 0, 0

    h, w, _ = frame.shape
    x1 = int(w * margin)
    y1 = int(h * margin)
    x2 = int(w * (1.0 - margin))
    y2 = int(h * (1.0 - margin))

    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return frame, 0, 0

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame, 0, 0

    return cropped, x1, y1


def check_lettuce_box(frame, box, conf):
    """
    Validates box dimensions, confidence, height ratio, and broad foliage HSV range.
    Returns (True, "ok", metrics) or (False, reason, metrics).
    """
    metrics = compute_height_metrics(box, frame.shape)

    if conf < CONF_THRESHOLD:
        return False, f"low_conf({conf:.2f}<{CONF_THRESHOLD:.2f})", metrics

    if metrics["height_ratio"] < MIN_HEIGHT_RATIO:
        return (
            False,
            f"too_small_height({metrics['height_ratio']:.4f}<{MIN_HEIGHT_RATIO:.4f})",
            metrics,
        )

    try:
        x1, y1, x2, y2 = box
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= MIN_BOX_PX or box_height <= MIN_BOX_PX:
            return False, f"small_box({box_width}x{box_height})", metrics

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "empty_crop", metrics

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Broad green / yellow-green / pale / diseased-lesion plant range
        lower_foliage = np.array([5, 8, 8])
        upper_foliage = np.array([115, 255, 255])

        mask = cv2.inRange(hsv, lower_foliage, upper_foliage)
        foliage_pixel_count = cv2.countNonZero(mask)
        total_pixels = float(box_width * box_height)
        foliage_ratio = foliage_pixel_count / total_pixels

        if foliage_ratio < MIN_FOLIAGE_RATIO:
            return (
                False,
                f"low_foliage({foliage_ratio:.4f}<{MIN_FOLIAGE_RATIO:.4f})",
                metrics,
            )

        return True, "ok", metrics
    except Exception as e:
        return True, f"fallback({e})", metrics


def is_exclusive_lettuce(frame, box, conf) -> bool:
    ok, _, _ = check_lettuce_box(frame, box, conf)
    return ok


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


# ======================================================================
# Smooth Tracker
# ======================================================================
class Track:
    __slots__ = (
        "track_id",
        "cls_votes",
        "box",
        "smooth_box",
        "last_seen",
        "last_conf",
        "smooth_conf",
        "height_info",
    )

    def __init__(self, track_id, cls_id, conf, box, now, height_info=None):
        self.track_id = track_id
        self.cls_votes = {cls_id: {"count": 1, "conf_sum": float(conf)}}
        self.box = box
        self.smooth_box = tuple(float(v) for v in box)
        self.last_seen = now
        self.last_conf = float(conf)
        self.smooth_conf = float(conf)
        self.height_info = height_info or {}

    def update(self, cls_id, conf, box, now, height_info=None):
        self.box = box
        self.last_seen = now
        self.last_conf = float(conf)
        if height_info:
            self.height_info = height_info

        alpha = max(0.05, min(1.0, SMOOTH_ALPHA))
        self.smooth_box = tuple(
            (1.0 - alpha) * s + alpha * float(c)
            for s, c in zip(self.smooth_box, box)
        )
        self.smooth_conf = (1.0 - alpha) * self.smooth_conf + alpha * float(conf)

        v = self.cls_votes.get(cls_id)
        if v is None:
            self.cls_votes[cls_id] = {"count": 1, "conf_sum": float(conf)}
        else:
            v["count"] += 1
            v["conf_sum"] += float(conf)

    def get_smoothed_box(self):
        # Blend slightly toward latest raw box when stationary to prevent lag
        alpha = max(0.05, min(1.0, SMOOTH_ALPHA))
        blended = [
            (1.0 - alpha) * s + alpha * float(r)
            for s, r in zip(self.smooth_box, self.box)
        ]
        self.smooth_box = tuple(blended)
        return tuple(int(round(v)) for v in blended)

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
                tracks[best_idx].update(
                    det["cls_id"],
                    det["conf"],
                    det["box"],
                    now,
                    det.get("height_info"),
                )
                used[best_idx] = True
            else:
                tracks.append(
                    Track(
                        next_track_id,
                        det["cls_id"],
                        det["conf"],
                        det["box"],
                        now,
                        det.get("height_info"),
                    )
                )
                used.append(True)
                next_track_id += 1


# ======================================================================
# Supabase save - Live Stream Session
# ======================================================================
def _build_session_row():
    global session_start_time

    with tracks_lock:
        snapshot = list(tracks)

    confirmed = []
    for tr in snapshot:
        if not tr.confirmed():
            continue
        cls_id = tr.best_class()
        class_name = normalize_class_name(model.names.get(cls_id, f"class_{cls_id}"))
        if class_name not in VALID_CLASSES:
            continue
        confirmed.append((tr, class_name))

    if len(confirmed) < MIN_CONFIRMED_TO_SAVE:
        return None, 0

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

    duration = int(time.time() - session_start_time) if session_start_time else 0

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
    return row, len(confirmed)


def save_session_to_supabase():
    if supabase is None:
        print("[Supabase] Not configured, cannot save session.")
        return False

    row, n_confirmed = _build_session_row()
    if row is None:
        print(f"[Session] Only {n_confirmed} confirmed plant(s) - skipping save.")
        return False

    snapshot_frame = None
    with frame_lock:
        if latest_frame is not None:
            snapshot_frame = latest_frame.copy()

    image_url = None
    if snapshot_frame is not None:
        try:
            draw_tracks_and_summary(snapshot_frame)

            success, encoded_img = cv2.imencode(
                ".jpg", snapshot_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if success:
                file_bytes = encoded_img.tobytes()
                file_name = f"session_{int(time.time())}.jpg"

                supabase.storage.from_("session_snapshots").upload(
                    path=file_name,
                    file=file_bytes,
                    file_options={"content-type": "image/jpeg"},
                )

                public_url_res = supabase.storage.from_(
                    "session_snapshots"
                ).get_public_url(file_name)
                image_url = public_url_res
                print(f"[Supabase] Snapshot with bounding boxes uploaded: {image_url}")
        except Exception as e:
            print(f"[Supabase] Image upload failed: {e}")

    row["image_url"] = image_url

    try:
        supabase.table(TABLE_LIVE).insert(row).execute()
        print(f"[Session] Saved to {TABLE_LIVE} with image and boxes.")
        return True
    except Exception as e:
        print(f"[Supabase] Session insert failed: {type(e).__name__}: {e}")
        return False


# ======================================================================
# Session lifecycle
# ======================================================================
def on_client_connected():
    global session_active, session_saved, tracks, next_track_id, session_start_time
    with session_lock:
        if not session_active:
            session_active = True
            session_saved = False
            session_start_time = time.time()
            with tracks_lock:
                tracks = []
                next_track_id = 1
            print("[Session] Started. Tracker cleared.")


def on_client_disconnected(last_client: bool):
    global session_active, session_saved, tracks, next_track_id, session_start_time
    if not last_client:
        return

    with session_lock:
        if not session_active or session_saved:
            return
        session_saved = True
        session_active = False

    print("[Session] Stream ended. Saving session to Supabase...")
    save_session_to_supabase()

    with tracks_lock:
        tracks = []
        next_track_id = 1
    session_start_time = None
    print("[Session] Tracker reset. Ready for next session.")


def session_flush_thread():
    global session_start_time, next_track_id
    while True:
        time.sleep(SESSION_FLUSH_INTERVAL)
        with session_lock:
            if not session_active:
                continue
            saved = save_session_to_supabase()
            if saved:
                with tracks_lock:
                    tracks.clear()
                    next_track_id = 1
                session_start_time = time.time()
                print("[Session] Flush complete. Window restarted.")


# ======================================================================
# Inference thread
# ======================================================================
def extract_box(det, offset_x=0, offset_y=0):
    try:
        if hasattr(det, "xyxyxyxy") and det.xyxyxyxy is not None:
            pts = det.xyxyxyxy[0].cpu().numpy()
            return (
                int(np.min(pts[:, 0])) + offset_x,
                int(np.min(pts[:, 1])) + offset_y,
                int(np.max(pts[:, 0])) + offset_x,
                int(np.max(pts[:, 1])) + offset_y,
            )
        if hasattr(det, "xyxy") and det.xyxy is not None:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
    except Exception:
        pass
    return None


def inference_thread():
    last_infer = 0.0
    last_seen_frame = -1
    while True:
        now = time.time()
        if now - last_infer < INFER_INTERVAL:
            time.sleep(0.05)
            continue

        frame = None
        fid = -1
        with frame_lock:
            if latest_frame is not None:
                fid = frame_id
                frame = latest_frame.copy()

        if frame is None or fid == last_seen_frame:
            time.sleep(0.05)
            continue

        last_seen_frame = fid
        last_infer = now

        try:
            infer_view, ox, oy = prepare_inference_view(frame)
            with model_lock:
                results = model.predict(
                    source=infer_view,
                    conf=CONF_THRESHOLD,
                    imgsz=INFER_IMGSZ,
                    verbose=False,
                )
            r0 = results[0]

            if r0.boxes is not None and len(r0.boxes) > 0:
                dets = list(r0.boxes)
            elif hasattr(r0, "obb") and r0.obb is not None and len(r0.obb) > 0:
                dets = list(r0.obb)
            else:
                dets = []

            clean = []
            rejected_reasons = []
            for det in dets:
                cls_id = int(det.cls[0])
                raw_name = model.names.get(cls_id, f"class_{cls_id}")
                cls_name = normalize_class_name(raw_name)
                conf = float(det.conf[0]) if det.conf is not None else 0.0

                if cls_name not in VALID_CLASSES:
                    rejected_reasons.append(f"{raw_name}({conf:.2f}):invalid_class")
                    continue

                box = extract_box(det, ox, oy)
                if box is None:
                    rejected_reasons.append(f"{raw_name}({conf:.2f}):no_box")
                    continue

                ok, reason, metrics = check_lettuce_box(frame, box, conf)
                if not ok:
                    rejected_reasons.append(f"{cls_name}({conf:.2f}):{reason}")
                    continue

                clean.append({
                    "cls_id": cls_id,
                    "conf": conf,
                    "box": box,
                    "height_info": metrics,
                })

            if DEBUG_DETECTIONS and fid % 8 == 1:
                kept_summary = [
                    f"{normalize_class_name(model.names.get(d['cls_id'], str(d['cls_id'])))}({d['conf']:.2f},h={d['height_info'].get('box_height_px', 0)}px)"
                    for d in clean
                ]
                print(
                    f"[Infer fid={fid} cam_h={CAMERA_HEIGHT_CM}cm] kept={kept_summary} rejected={rejected_reasons}"
                )

            tracker_update(clean, now)

        except Exception as e:
            print(f"[Infer] {type(e).__name__}: {e}")


# ======================================================================
# Drawing & Total Diseases Screen Overlay
# ======================================================================
def draw_tracks_and_summary(frame):
    now = time.time()
    with tracks_lock:
        snapshot = list(tracks)

    # 1. Draw smooth bounding boxes for active tracks
    for tr in snapshot:
        age = now - tr.last_seen
        if age > MEMORY_TIME:
            continue
        cls_id = tr.best_class()
        raw_name = model.names.get(cls_id, str(cls_id))
        name = normalize_class_name(raw_name)

        if name not in VALID_CLASSES:
            continue

        x1, y1, x2, y2 = tr.get_smoothed_box()
        color = get_class_color(name)
        display_name = raw_name.replace("_", " ").strip()
        h_px = tr.height_info.get("box_height_px", max(1, y2 - y1))

        if age < 1.0:
            label = f"#{tr.track_id} {display_name} {tr.smooth_conf:.2f} H:{h_px}px"
        else:
            label = f"#{tr.track_id} {display_name} H:{h_px}px"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    # 2. Compute counts for the summary banner from active confirmed tracks
    counts = {}
    for tr in snapshot:
        if (now - tr.last_seen) > MEMORY_TIME or not tr.confirmed():
            continue
        name = normalize_class_name(model.names.get(tr.best_class(), ""))
        if name not in VALID_CLASSES:
            continue
        counts[name] = counts.get(name, 0) + 1

    powdery = counts.get("powdery mildew", 0)
    downy = counts.get("downy mildew", 0)
    bacterial = counts.get("bacterial leaf spot", 0)
    diseased_total = powdery + downy + bacterial

    # 3. Draw summary bar with total diseases, positioned at Y=80
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
    try:
        while True:
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    fid = frame_id
                    if fid != last_yielded:
                        frame = latest_frame.copy()
                        last_yielded = fid

            if frame is None:
                time.sleep(0.05)
                continue

            draw_tracks_and_summary(frame)

            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.05)

    finally:
        with clients_lock:
            active_clients -= 1
            was_last = active_clients == 0
        print(f"[Stream] Client disconnected. Active: {active_clients}")

        if was_last:
            on_client_disconnected(last_client=True)


# ======================================================================
# Scheduled-scan analysis (For Notifications / Periodic Scans)
# ======================================================================
def analyze_image(frame):
    infer_view, ox, oy = prepare_inference_view(frame)
    with model_lock:
        results = model.predict(
            source=infer_view, conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, verbose=False
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
        name = normalize_class_name(model.names.get(int(det.cls[0]), ""))
        if name not in VALID_CLASSES:
            continue

        box = extract_box(det, ox, oy)
        if box is None:
            continue

        conf = float(det.conf[0]) if det.conf is not None else 0.0

        if not is_exclusive_lettuce(frame, box, conf):
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
        "camera_height_cm": CAMERA_HEIGHT_CM,
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
    now = time.time()
    with tracks_lock:
        snapshot = list(tracks)

    per_class = {}
    heights = []
    for tr in snapshot:
        if (now - tr.last_seen) > MEMORY_TIME or not tr.confirmed():
            continue
        name = normalize_class_name(model.names.get(tr.best_class(), ""))

        if name not in VALID_CLASSES:
            continue

        per_class[name] = per_class.get(name, 0) + 1
        if tr.height_info:
            heights.append(tr.height_info)

    total = sum(per_class.values())
    healthy = per_class.get("healthy", 0)
    diseased = total - healthy

    return jsonify({
        "camera_online": is_camera_online(),
        "camera_height_cm": CAMERA_HEIGHT_CM,
        "per_class": per_class,
        "total_unique_plants": total,
        "healthy": healthy,
        "diseased": diseased,
        "heights": heights,
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
    ok = save_session_to_supabase()
    return jsonify({"saved": ok})


# Start background inference thread once when module loads
_infer_thread_started = False
_infer_thread_lock = threading.Lock()


def ensure_inference_thread():
    global _infer_thread_started
    with _infer_thread_lock:
        if not _infer_thread_started:
            threading.Thread(target=inference_thread, daemon=True).start()
            _infer_thread_started = True


ensure_inference_thread()

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    ensure_inference_thread()
    port = int(os.environ.get("PORT", 5000))
    print(f"[Main] Inference thread started. Serving on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
