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
from werkzeug.exceptions import HTTPException
from ultralytics import YOLO
from supabase import create_client, Client

# ======================================================================
# Setup
# ======================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # reject uploads > 10 MB
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")  # e.g. best.onnx for faster CPU
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[Model] FATAL: could not load '{MODEL_PATH}': {type(e).__name__}: {e}")
    raise
print(f"This model can detect: {model.names}")

ESP32_CAM_URL = os.environ.get("ESP32_CAM_URL", "http://192.168.1.119:81/stream")
ENABLE_LAN_READER = os.environ.get("ENABLE_LAN_READER", "0") == "1"

# Optional shared secret so random internet traffic cannot spoof ESP32 frames.
# Set CAM_PUSH_TOKEN in both ESP32 code and Railway env vars if desired.
CAM_PUSH_TOKEN = os.environ.get("CAM_PUSH_TOKEN", "")
CAM_ONLINE_TIMEOUT = 10.0  # seconds without a pushed frame before marking camera offline
CLIENT_IDLE_TIMEOUT = 12.0  # seconds without yielding a frame before ending client

# How long to keep the session (and its disease total) alive after the last
# viewer disconnects. A viewer who reconnects inside this window (page refresh,
# network blip) continues the same totals; after it, the session is saved and
# the next stream starts from zero. 0 = save and reset immediately.
SESSION_GRACE_SECONDS = float(os.environ.get("SESSION_GRACE_SECONDS", "15"))

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
INFER_INTERVAL = float(os.environ.get("INFER_INTERVAL", "0.0"))  # 0 = run as fast as the model allows
INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ", "480"))  # 416/320 = faster, 640 = more detail
JPEG_QUALITY = 75
MEMORY_TIME = 15.0
CHUNK_SIZE = 8192
IOU_MATCH_THRESHOLD = 0.15
MIN_FRAMES_TO_CONFIRM = int(os.environ.get("MIN_FRAMES_TO_CONFIRM", "3"))  # must be seen N times

# ---------- Curly-lettuce-only strict filter (all tunable via env vars) ----------
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.40"))  # model-level floor
CLASS_MIN_CONF = {
    "healthy": float(os.environ.get("MIN_CONF_HEALTHY", "0.60")),
    "downy mildew": float(os.environ.get("MIN_CONF_DISEASE", "0.50")),
    "powdery mildew": float(os.environ.get("MIN_CONF_DISEASE", "0.50")),
    "bacterial leaf spot": float(os.environ.get("MIN_CONF_DISEASE", "0.50")),
}
MIN_FRAME_SIDE = 64                      # reject tiny / corrupt images
MIN_SCENE_GREEN_RATIO = float(os.environ.get("MIN_SCENE_GREEN_RATIO", "0.05"))  # whole frame must contain some plant green
MIN_BOX_AREA_RATIO = float(os.environ.get("MIN_BOX_AREA_RATIO", "0.01"))        # box too tiny = noise
MAX_BOX_AREA_RATIO = float(os.environ.get("MAX_BOX_AREA_RATIO", "0.90"))        # box covering everything = not a plant
MIN_ASPECT, MAX_ASPECT = 0.4, 2.5        # a lettuce head is roughly round, not a thin strip
MIN_GREEN_HEALTHY = float(os.environ.get("MIN_GREEN_HEALTHY", "0.40"))          # healthy = mostly green leaf
MIN_GREEN_DISEASED = float(os.environ.get("MIN_GREEN_DISEASED", "0.15"))        # diseased = lesions allowed, leaf still green
MIN_EDGE_DENSITY = float(os.environ.get("MIN_EDGE_DENSITY", "0.04"))            # curly/ruffled leaves have many edges

SESSION_FLUSH_INTERVAL = 60  # seconds (optional periodic flush if enabled)
ENABLE_PERIODIC_FLUSH = os.environ.get("ENABLE_PERIODIC_FLUSH", "0") == "1"
MIN_CONFIRMED_TO_SAVE = 1

# ---------- Box smoothing / display tuning ----------
SMOOTH_DURATION = 0.12   # seconds a box takes to glide to its newest position
BOX_EMA_ALPHA = 0.8      # 0..1, higher = follows new detections faster, lower = steadier
CONF_EMA_ALPHA = 0.3     # smoothing for the confidence number in the label
FADE_IN_SECONDS = 0.12   # new boxes fade in instead of popping
FADE_OUT_SECONDS = 2.0   # boxes fade out over the last seconds of MEMORY_TIME
STALE_AFTER = 1.0        # seconds without a fresh detection -> thinner "remembered" box
DEDUPE_IOU = 0.70        # overlapping detections above this are merged (keeps best conf)
ANNOTATED_SNAPSHOT_EVERY = 2.0  # seconds between cached annotated frames

# ---------- Latency compensation (motion prediction) ----------
# Boxes come from a frame that is a few hundred ms old. We estimate each box's
# velocity and project it forward to "now", so boxes sit on the plant in the
# LIVE video and keep moving smoothly between detections.
MAX_EXTRAPOLATE = 0.50   # max seconds to project ahead
VEL_EMA_ALPHA = 0.5      # smoothing of the velocity estimate
VEL_DEADZONE = 20.0      # px/s below this is treated as "not moving" (kills jitter drift)
VEL_MAX = 500.0          # px/s sanity cap

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


GREEN_LO = np.array([25, 35, 25])
GREEN_HI = np.array([95, 255, 255])

_last_reject_log = 0.0


def _log_reject(reason: str):
    """Rate-limited log so rejected detections don't flood the console."""
    global _last_reject_log
    now = time.time()
    if now - _last_reject_log >= 2.0:
        _last_reject_log = now
        print(f"[Filter] Rejected: {reason}")


def is_valid_frame(frame) -> bool:
    """Guards against corrupt / empty / wrongly shaped images."""
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] >= MIN_FRAME_SIDE
        and frame.shape[1] >= MIN_FRAME_SIDE
    )


def frame_has_plant_scene(frame) -> bool:
    """Scene gate: if the whole frame has almost no plant-green, there can't be
    lettuce in it, so skip detection entirely (fail closed)."""
    try:
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
        ratio = cv2.countNonZero(mask) / float(mask.size)
        if ratio < MIN_SCENE_GREEN_RATIO:
            _log_reject(f"scene has no plant foliage (green {ratio:.3f})")
            return False
        return True
    except Exception:
        return False


def validate_lettuce_detection(frame, box, conf, class_name):
    """
    Strict curly-lettuce validation, applied once per detection. Returns
    (ok, reason). Every check must pass, and any error rejects the detection.
      1. per-class confidence floor (healthy needs more certainty)
      2. box size and shape sanity
      3. green foliage ratio (stricter for healthy than for diseased)
      4. edge density, which separates ruffled/curly leaves from smooth objects
    """
    try:
        min_conf = CLASS_MIN_CONF.get(class_name, 1.0)
        if conf < min_conf:
            return False, f"{class_name} conf {conf:.2f} < {min_conf:.2f}"

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 8 or bh <= 8:
            return False, "box too small"

        area_ratio = (bw * bh) / float(w * h)
        if area_ratio < MIN_BOX_AREA_RATIO:
            return False, f"box area {area_ratio:.3f} too small"
        if area_ratio > MAX_BOX_AREA_RATIO:
            return False, f"box area {area_ratio:.3f} too large"

        aspect = bw / float(bh)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            return False, f"aspect {aspect:.2f} not plant-like"

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "empty crop"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.countNonZero(cv2.inRange(hsv, GREEN_LO, GREEN_HI)) / float(bw * bh)
        min_green = MIN_GREEN_HEALTHY if class_name == "healthy" else MIN_GREEN_DISEASED
        if green < min_green:
            return False, f"{class_name} green {green:.2f} < {min_green:.2f}"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = 160.0 / max(bw, bh)
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 130)
        edge_density = cv2.countNonZero(edges) / float(edges.size)
        if edge_density < MIN_EDGE_DENSITY:
            return False, f"not curly: edge density {edge_density:.3f} < {MIN_EDGE_DENSITY:.3f}"

        return True, "ok"
    except Exception as e:
        return False, f"validation error {type(e).__name__}: {e}"


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
latest_frame_ts = 0.0  # when the newest frame arrived (capture time)
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
_end_timer = None
_end_timer_lock = threading.Lock()


# ======================================================================
# Tracker
# ======================================================================
class Track:
    __slots__ = (
        "track_id", "cls_votes", "box", "last_seen", "last_conf",
        "first_seen", "smooth_conf", "anim",
    )

    def __init__(self, track_id, cls_id, conf, box, now, frame_ts):
        self.track_id = track_id
        self.cls_votes = {cls_id: {"count": 1, "conf_sum": float(conf)}}
        self.box = box  # raw last detection (used for matching)
        self.last_seen = now
        self.last_conf = float(conf)
        self.first_seen = now
        self.smooth_conf = float(conf)
        fb = tuple(float(v) for v in box)
        # anim = (start_box, target_box, start_time, velocity(px/s), frame_ts)
        # Display position is a pure function of time, so every viewer and every
        # saved snapshot sees the same smooth box.
        self.anim = (fb, fb, now, (0.0, 0.0), frame_ts)

    @staticmethod
    def _offset(vel, now, frame_ts):
        """How far the plant has probably moved since the detection frame."""
        dt = now - frame_ts
        dt = 0.0 if dt < 0.0 else MAX_EXTRAPOLATE if dt > MAX_EXTRAPOLATE else dt
        dx, dy = vel[0] * dt, vel[1] * dt
        return (dx, dy, dx, dy)

    def current_box(self, now):
        """Smoothly interpolated + motion-compensated box for drawing at `now`."""
        prev, target, t0, vel, fts = self.anim
        t = (now - t0) / SMOOTH_DURATION
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        t = t * t * (3.0 - 2.0 * t)  # smoothstep easing
        off = self._offset(vel, now, fts)
        return tuple(
            int(round(p + (q - p) * t + o)) for p, q, o in zip(prev, target, off)
        )

    def update(self, cls_id, conf, box, now, frame_ts):
        shown = self.current_box(now)  # where the box is drawn right now
        _, old_target, _, old_vel, old_fts = self.anim

        # --- velocity estimate from consecutive raw detections ---
        vel = old_vel
        dt = frame_ts - old_fts
        if 0.02 < dt < 1.0:
            ocx = (self.box[0] + self.box[2]) / 2.0
            ocy = (self.box[1] + self.box[3]) / 2.0
            ncx = (box[0] + box[2]) / 2.0
            ncy = (box[1] + box[3]) / 2.0
            vx = max(-VEL_MAX, min(VEL_MAX, (ncx - ocx) / dt))
            vy = max(-VEL_MAX, min(VEL_MAX, (ncy - ocy) / dt))
            vx = (1.0 - VEL_EMA_ALPHA) * old_vel[0] + VEL_EMA_ALPHA * vx
            vy = (1.0 - VEL_EMA_ALPHA) * old_vel[1] + VEL_EMA_ALPHA * vy
            if (vx * vx + vy * vy) ** 0.5 < VEL_DEADZONE:
                vx = vy = 0.0
            vel = (vx, vy)
        elif dt >= 1.0:
            vel = (0.0, 0.0)  # track was lost for a while: don't extrapolate

        new_target = tuple(
            BOX_EMA_ALPHA * float(r) + (1.0 - BOX_EMA_ALPHA) * o
            for r, o in zip(box, old_target)
        )
        # Start gliding from the position currently on screen (minus the new
        # offset that gets added back on top) -> no visual jumps.
        off_new = self._offset(vel, now, frame_ts)
        start = tuple(float(s) - o for s, o in zip(shown, off_new))
        self.anim = (start, new_target, now, vel, frame_ts)

        self.box = box
        self.last_seen = now
        self.last_conf = float(conf)
        self.smooth_conf = (
            (1.0 - CONF_EMA_ALPHA) * self.smooth_conf + CONF_EMA_ALPHA * float(conf)
        )
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


def dedupe_detections(dets):
    """Drops duplicate overlapping boxes (e.g. same plant tagged twice) so only
    one clean box is shown per plant. Keeps the highest-confidence one."""
    kept = []
    for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
        if all(compute_iou(d["box"], k["box"]) < DEDUPE_IOU for k in kept):
            kept.append(d)
    return kept


def tracker_update(detections, now, frame_ts=None):
    """Global best-first matching: the strongest overlaps are paired first, so
    boxes keep the same ID/identity instead of swapping between neighbours."""
    global next_track_id
    if frame_ts is None:
        frame_ts = now
    with tracks_lock:
        pairs = []
        for di, det in enumerate(detections):
            for ti, tr in enumerate(tracks):
                iou = compute_iou(tr.box, det["box"])
                if iou > IOU_MATCH_THRESHOLD:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)

        used_d, used_t = set(), set()
        for _iou, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            det = detections[di]
            tracks[ti].update(det["cls_id"], det["conf"], det["box"], now, frame_ts)
            used_d.add(di)
            used_t.add(ti)

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tracks.append(
                Track(next_track_id, det["cls_id"], det["conf"], det["box"], now, frame_ts)
            )
            next_track_id += 1


def compute_class_counts(snapshot):
    """
    Counts every confirmed track by class. This does NOT depend on track age
    or on the current frame content, so the total never drops just because a
    plant is out of view or its box has faded. Tracks were already validated
    as lettuce at detection time in inference_thread().
    """
    counts = {}
    for tr in snapshot:
        if not tr.confirmed():
            continue
        name = model.names.get(tr.best_class(), "").strip().lower()
        if name not in VALID_CLASSES:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


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
def _cancel_end_timer():
    global _end_timer
    with _end_timer_lock:
        if _end_timer is not None:
            _end_timer.cancel()
            _end_timer = None


def on_client_connected():
    """Called when the FIRST viewer connects. If the previous viewer left only
    moments ago (grace period still running), the same session continues with
    its totals. Otherwise a brand-new session starts with every count at zero."""
    global session_active, session_saved, session_start_time

    with _end_timer_lock:
        grace_pending = _end_timer is not None
    with session_lock:
        resumable = grace_pending and session_active and not session_saved
    if resumable:
        _cancel_end_timer()
        print("[Session] Viewer reconnected within grace period. Continuing totals.")
        return

    _cancel_end_timer()

    # Save any leftover, unsaved session from the previous stream first.
    with session_lock:
        leftover = session_active and not session_saved
        if leftover:
            session_saved = True
            session_active = False
    if leftover:
        print("[Session] Leftover session found. Saving before starting a new one.")
        save_session_to_supabase(force=False)

    # Always start fresh: clear tracks, IDs back to #1, totals back to 0.
    with session_lock:
        _reset_session_tracks_locked()      # sets session_start_time = None
        session_active = True
        session_saved = False
        session_start_time = time.time()    # set AFTER the reset
    print("[Session] New stream started. Total diseases reset to 0.")


def _finalize_session():
    """Saves and resets the session once nobody is watching."""
    global session_active, session_saved, _end_timer
    with _end_timer_lock:
        _end_timer = None

    with clients_lock:
        if active_clients > 0:
            return

    with session_lock:
        if not session_active or session_saved:
            return
        session_saved = True
        session_active = False

    print("[Session] Saving session with bounding boxes to Supabase...")
    save_session_to_supabase(force=False)
    with session_lock:
        _reset_session_tracks_locked()
    print("[Session] Tracker reset. Ready for next session.")


def on_client_disconnected(last_client: bool):
    global _end_timer
    if not last_client:
        return

    with session_lock:
        if not session_active or session_saved:
            return

    _cancel_end_timer()

    # No grace period: save and reset right away so the next stream starts at 0.
    if SESSION_GRACE_SECONDS <= 0:
        print("[Session] Last viewer left. Saving and resetting now.")
        threading.Thread(target=_finalize_session, daemon=True).start()
        return

    print(
        f"[Session] Last viewer left. Keeping totals for {SESSION_GRACE_SECONDS:.0f}s "
        f"before saving."
    )
    timer = threading.Timer(SESSION_GRACE_SECONDS, _finalize_session)
    timer.daemon = True
    with _end_timer_lock:
        _end_timer = timer
    timer.start()


def flush_active_session_on_exit(reason: str = "exit"):
    """Ensures any active or unsaved tracked detections are saved when app exits."""
    global session_active, session_saved, _shutting_down
    _cancel_end_timer()
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
    global latest_frame, frame_id, last_frame_push_time, latest_frame_ts
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
                            latest_frame_ts = last_frame_push_time

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


def detect_lettuce(frame):
    """Runs the model and returns ONLY detections that pass the strict curly-lettuce
    filter. Used by both the live stream and /upload so they behave identically.
    Never raises on bad input: returns [] instead."""
    if not is_valid_frame(frame):
        return []
    if not frame_has_plant_scene(frame):
        return []

    with model_lock:
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=INFER_IMGSZ,
            verbose=False,
            agnostic_nms=True,  # one box per plant even if classes disagree
            max_det=30,
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
        ok, reason = validate_lettuce_detection(frame, box, conf, cls_name)
        if not ok:
            _log_reject(reason)
            continue

        clean.append({"cls_id": cls_id, "name": cls_name, "conf": conf, "box": box})

    return dedupe_detections(clean)


def inference_thread():
    last_infer = 0.0
    last_seen_frame = -1
    last_annotated_at = 0.0
    consecutive_errors = 0
    while True:
        now = time.time()
        if now - last_infer < INFER_INTERVAL:
            time.sleep(0.01)
            continue

        frame = None
        fid = -1
        frame_ts = 0.0
        with frame_lock:
            if latest_frame is not None:
                fid = frame_id
                frame = latest_frame  # no copy needed: inference never modifies it
                frame_ts = latest_frame_ts

        if frame is None or fid == last_seen_frame:
            time.sleep(0.005)
            continue

        last_seen_frame = fid
        last_infer = now

        try:
            clean = detect_lettuce(frame)
            consecutive_errors = 0

            # Only track while a livestream session is active. When nobody is
            # watching, nothing is counted, so a new stream always starts at 0.
            with session_lock:
                active = session_active
            if active:
                tracker_update(clean, time.time(), frame_ts)

                # Cache an annotated frame only occasionally (saves CPU = less lag)
                if clean and (now - last_annotated_at) >= ANNOTATED_SNAPSHOT_EVERY:
                    annotated = frame.copy()
                    draw_tracks_and_summary(annotated)
                    with frame_lock:
                        globals()["last_annotated_frame"] = annotated
                    last_annotated_at = time.time()

        except Exception as e:
            consecutive_errors += 1
            print(f"[Infer] {type(e).__name__}: {e} (error #{consecutive_errors})")
            time.sleep(min(0.2 * consecutive_errors, 3.0))  # back off instead of spinning


# ======================================================================
# Drawing & Total Diseases Screen Overlay
# ======================================================================
def draw_tracks_and_summary(frame):
    now = time.time()
    with tracks_lock:
        snapshot = list(tracks)

    # 1. Draw bounding boxes: smooth glide, fade in/out, readable labels
    fh, fw = frame.shape[:2]
    for tr in snapshot:
        age = now - tr.last_seen
        if age > MEMORY_TIME:
            continue
        if not tr.confirmed():
            continue
        cls_id = tr.best_class()
        name = model.names.get(cls_id, "").strip().lower()
        if name not in VALID_CLASSES:
            continue

        # fade in when new, fade out as the memory expires
        fade_in = min(1.0, (now - tr.first_seen) / FADE_IN_SECONDS)
        remaining = MEMORY_TIME - age
        fade_out = 1.0 if remaining >= FADE_OUT_SECONDS else max(0.0, remaining / FADE_OUT_SECONDS)
        alpha = min(fade_in, fade_out)
        if alpha < 0.08:
            continue

        x1, y1, x2, y2 = tr.current_box(now)
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh - 1))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        base = get_class_color(name)
        color = tuple(int(c * alpha) for c in base)
        thickness = 3 if age <= STALE_AFTER else 2  # thinner = remembered, not live

        display_name = model.names.get(cls_id, str(cls_id))
        label = f"#{tr.track_id} {display_name} {tr.smooth_conf * 100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        ly2 = y1 if y1 - th - baseline - 6 >= 0 else y1 + th + baseline + 6
        ly1 = ly2 - th - baseline - 6
        lx2 = min(fw - 1, x1 + tw + 8)
        cv2.rectangle(frame, (x1, ly1), (lx2, ly2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, ly2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. Summary counts: every confirmed track, regardless of age or frame content.
    counts = compute_class_counts(snapshot)

    powdery = counts.get("powdery mildew", 0)
    downy = counts.get("downy mildew", 0)
    bacterial = counts.get("bacterial leaf spot", 0)
    diseased_total = powdery + downy + bacterial

    # 3. Draw summary bar with total diseases, positioned at Y=80
    summary_text = f"Total diseases: {diseased_total}"

    cv2.putText(
        frame, summary_text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        summary_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
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
    else:
        _cancel_end_timer()
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
    dets = detect_lettuce(frame)
    counts, confs = {}, []
    for d in dets:
        print(f"[Analyze] Accepted: class={d['name']} conf={d['conf']:.3f}")
        counts[d["name"]] = counts.get(d["name"], 0) + 1
        confs.append(d["conf"])

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


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """404, 405, 413 (file too large), etc. are returned as clean JSON."""
    return jsonify({"error": e.name, "message": e.description}), e.code


@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    """Any unhandled error: log it, return JSON, never crash the server."""
    print(f"[Error] Unhandled {type(e).__name__}: {e}")
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong while processing the request.",
    }), 500


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/push_frame", methods=["POST"])
def push_frame():
    """Receives live JPEG frames pushed over the internet by the ESP32-CAM."""
    global latest_frame, frame_id, last_frame_push_time, pushed_frames_count, latest_frame_ts

    if CAM_PUSH_TOKEN:
        token = request.headers.get("X-Cam-Token", "")
        if token != CAM_PUSH_TOKEN:
            return jsonify({"error": "unauthorized"}), 401

    data = request.get_data()
    if not data:
        return jsonify({"error": "empty body"}), 400

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if not is_valid_frame(frame):
        return jsonify({"error": "bad or too small image"}), 400

    with frame_lock:
        latest_frame = frame
        frame_id += 1
        last_frame_push_time = time.time()
        latest_frame_ts = last_frame_push_time
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
    if not is_valid_frame(frame):
        return jsonify({"error": "Invalid image: file is corrupt, empty, or too small."}), 400

    try:
        counts, avg_conf = analyze_image(frame)

        total_lettuce = sum(counts.values())
        if total_lettuce == 0:
            print(
                "[Upload Error] Rejected: no curly lettuce detected."
            )
            return jsonify({
                "error": "Invalid image: no curly lettuce detected. This system only analyzes curly lettuce (healthy, downy mildew, powdery mildew, bacterial leaf spot)."
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

    per_class = compute_class_counts(snapshot)

    total = sum(per_class.values())
    healthy = per_class.get("healthy", 0)
    diseased = total - healthy

    return jsonify({
        "camera_online": is_camera_online(),
        "per_class": per_class,
        "total_unique_plants": total,
        "healthy": healthy,
        "diseased": diseased,
        "total_diseases": diseased,  # same number shown on the video overlay
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


@app.route("/health")
def health():
    with session_lock:
        active = session_active
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "camera_online": is_camera_online(),
        "session_active": active,
    })


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
