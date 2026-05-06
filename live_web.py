#!/usr/bin/env python3
# Live dual-camera web viewer — both RTSP cams stream in parallel in the
# background; the UI shows one at a time. Cam switches are instant because
# both streams are already warm; only HD mode-change restarts a stream.
# Usage: python3 live_web.py  (or: source .env && python3 live_web.py)
# version 21.04.2026

import os
import time
import asyncio
import mimetypes
import logging
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

DEBUG = False

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# Silence noisy third-party loggers
for _noisy in ("uvicorn", "uvicorn.access", "fastapi", "asyncio", "multipart"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
# uvicorn.error emits ERROR-level tracebacks on shutdown — block entirely
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)


class _StderrFilter:
    """Drop noisy lines that libraries write directly to stderr."""
    _DROP = (
        "pipe closed by peer",
        "os.write(pipe, data) raised exception",
        "Task was destroyed but it is pending",
    )

    def __init__(self, stream):
        self._s = stream

    def write(self, text: str):
        if not any(p in text for p in self._DROP):
            self._s.write(text)

    def flush(self):
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


import time as _time

class _StdoutFilter:
    """Suppress noisy multi-line output from insightface and ONNX Runtime.

    Triggering on a known bad line starts a 2-second blackout — so all follow-up
    writes (model names, float arrays, blank lines) are also dropped regardless of
    how many separate write() calls the original print() was split into.
    """
    _TRIGGERS = (
        "Applied providers:",
        "find model:",
        ".insightface",
        "set det-size:",
        "pipe closed by peer",
        "os.write(pipe, data) raised",
    )

    def __init__(self, stream):
        self._s = stream
        self._suppress_until = 0.0

    def write(self, text: str):
        now = _time.monotonic()
        if any(p in text for p in self._TRIGGERS):
            self._suppress_until = now + 2.0
            return
        if now < self._suppress_until:
            return
        self._s.write(text)

    def flush(self):
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


import sys as _sys
_sys.stderr = _StderrFilter(_sys.stderr)
_sys.stdout = _StdoutFilter(_sys.stdout)

_LOCAL_ENV  = Path(__file__).parent / ".env"
_CONFIG_FILE = Path(__file__).parent / "config.json"

CAM_DEFAULT_PORT    = 554
CAM_DEFAULT_PATH_SD = "/stream2"
CAM_DEFAULT_PATH_HD = "/stream1"

# ── CSS named colors (W3C/CSS4 full list) ────────────────────────────────────
_CSS_NAMED_COLORS: set[str] = {
    "aliceblue","antiquewhite","aqua","aquamarine","azure","beige","bisque","black",
    "blanchedalmond","blue","blueviolet","brown","burlywood","cadetblue","chartreuse",
    "chocolate","coral","cornflowerblue","cornsilk","crimson","cyan","darkblue",
    "darkcyan","darkgoldenrod","darkgray","darkgreen","darkgrey","darkkhaki",
    "darkmagenta","darkolivegreen","darkorange","darkorchid","darkred","darksalmon",
    "darkseagreen","darkslateblue","darkslategray","darkslategrey","darkturquoise",
    "darkviolet","deeppink","deepskyblue","dimgray","dimgrey","dodgerblue","firebrick",
    "floralwhite","forestgreen","fuchsia","gainsboro","ghostwhite","gold","goldenrod",
    "gray","green","greenyellow","grey","honeydew","hotpink","indianred","indigo",
    "ivory","khaki","lavender","lavenderblush","lawngreen","lemonchiffon","lightblue",
    "lightcoral","lightcyan","lightgoldenrodyellow","lightgray","lightgreen","lightgrey",
    "lightpink","lightsalmon","lightseagreen","lightskyblue","lightslategray",
    "lightslategrey","lightsteelblue","lightyellow","lime","limegreen","linen",
    "magenta","maroon","mediumaquamarine","mediumblue","mediumorchid","mediumpurple",
    "mediumseagreen","mediumslateblue","mediumspringgreen","mediumturquoise",
    "mediumvioletred","midnightblue","mintcream","mistyrose","moccasin","navajowhite",
    "navy","oldlace","olive","olivedrab","orange","orangered","orchid","palegoldenrod",
    "palegreen","paleturquoise","palevioletred","papayawhip","peachpuff","peru","pink",
    "plum","powderblue","purple","rebeccapurple","red","rosybrown","royalblue",
    "saddlebrown","salmon","sandybrown","seagreen","seashell","sienna","silver",
    "skyblue","slateblue","slategray","slategrey","snow","springgreen","steelblue",
    "tan","teal","thistle","tomato","turquoise","violet","wheat","white","whitesmoke",
    "yellow","yellowgreen",
}

import re as _re

def _valid_color(color: str) -> bool:
    """Return True if color is a valid CSS color (named, hex, rgb, hsl)."""
    c = color.strip().lower()
    if c in _CSS_NAMED_COLORS:
        return True
    if _re.match(r'^#([0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$', c):
        return True
    if _re.match(r'^rgba?\s*\(', c):
        return True
    if _re.match(r'^hsla?\s*\(', c):
        return True
    return False


_CONFIG_EXAMPLE = Path(__file__).parent / "config.example.json"

def _default_config() -> dict:
    """Return default config — from config.example.json if present, else built-in."""
    if _CONFIG_EXAMPLE.exists():
        import json
        try:
            return json.loads(_CONFIG_EXAMPLE.read_text())
        except Exception:
            pass
    return {
        "family_dir": "family",
        "box_size": 75,
        "box_offset_x": 15,
        "box_offset_y": 0,
        "identity_colors": {},
        "cameras": {},
    }


def _load_config() -> dict:
    """Load config.json; create from template if missing."""
    import json
    if not _CONFIG_FILE.exists():
        default = _default_config()
        _CONFIG_FILE.write_text(json.dumps(default, indent=2) + "\n")
        print(f"Created config.json from template.")
        return default
    try:
        return json.loads(_CONFIG_FILE.read_text())
    except Exception as e:
        print(_clr("ERROR: ", "1;31") + f"config.json is invalid: {e}")
        raise SystemExit(1)


def _save_config(cfg: dict) -> None:
    import json
    _CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")


_PLACEHOLDER_VALUES = {"192.168.1.100", "admin", "yourpassword", "janedoe",
                       "johndoe1", "johndoe2"}

def _clr(text: str, code: str) -> str:
    """Wrap text in ANSI color if stdout is a real terminal."""
    if _sys.__stdout__ and _sys.__stdout__.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def _check_placeholder_config(cfg: dict) -> list[str]:
    """Return list of fields that still contain example/template placeholder values."""
    found = []
    for name, cam in cfg.get("cameras", {}).items():
        if cam.get("host") in _PLACEHOLDER_VALUES:
            found.append(f"  cameras.{name}.host = '{cam['host']}' (example value)")
        if cam.get("user") in _PLACEHOLDER_VALUES:
            found.append(f"  cameras.{name}.user = '{cam['user']}' (example value)")
        if cam.get("password") in _PLACEHOLDER_VALUES:
            found.append(f"  cameras.{name}.password = '{cam['password']}' (example value)")
    for person in cfg.get("identity_colors", {}):
        if person.lower() in _PLACEHOLDER_VALUES:
            found.append(f"  identity_colors.{person} (example name — replace with real name or remove if no photos yet)")
    return found


def _strip_placeholder_cams(cfg: dict) -> None:
    cameras = cfg.get("cameras", {})
    for name in [n for n, c in list(cameras.items())
                 if c.get("host") in _PLACEHOLDER_VALUES
                 or c.get("user") in _PLACEHOLDER_VALUES
                 or c.get("password") in _PLACEHOLDER_VALUES]:
        del cameras[name]

def _strip_placeholder_persons(cfg: dict) -> None:
    colors = cfg.get("identity_colors", {})
    for name in [n for n in list(colors) if n.lower() in _PLACEHOLDER_VALUES]:
        del colors[name]


def _validate_config(cfg: dict) -> list[str]:
    """Return list of error strings; empty = valid."""
    errors = []
    for name, color in cfg.get("identity_colors", {}).items():
        if not _valid_color(color):
            errors.append(f"  Person '{name}': '{color}' is not a valid CSS color")
    return errors


_SERVER_PORT  = 8766
# Paths resolved after config loads — see below where _cfg is defined.
_LOG_FILE     = Path("/tmp/live_web.log")
_MONITOR_LOG  = Path("/tmp/cam_monitor.log")


def _server_running() -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", _SERVER_PORT)) == 0


def _kill_server(silent: bool = False) -> bool:
    """Kill process on port 8766. Returns True if something was killed."""
    import subprocess
    result = subprocess.run(["lsof", "-t", f"-i:{_SERVER_PORT}", "-sTCP:LISTEN"],
                            capture_output=True, text=True)
    pid = result.stdout.strip()
    if pid:
        subprocess.run(["kill", "-9", pid], capture_output=True)
        time.sleep(1)
        return True
    return False


def _start_background() -> None:
    """Start live_web.py in background, logging to _LOG_FILE."""
    import subprocess
    script   = Path(__file__).resolve()
    _LOG_FILE.write_text("")  # truncate so tail reads fresh output
    cmd = f"nohup python3 {script} --background > {_LOG_FILE} 2>&1 &"
    subprocess.Popen(["bash", "-c", cmd], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _tail_until_live(timeout: float = 45.0) -> None:
    """Print startup banner then tail log until all cams report LIVE."""
    cam_count = len(CAMS)
    W = 38
    def _row(text): return f"║  {text:<{W - 2}}║"
    print("╔" + "═" * W + "╗")
    print("║" + "RTSP CAM VIEWER  READY".center(W) + "║")
    print("╠" + "═" * W + "╣")
    print(_row(f"Cameras : {cam_count}"))
    for k, v in CAMS.items():
        print(_row(f"  {k}  →  {v['host']}"))
    print("╠" + "═" * W + "╣")
    print(_row(f"http://localhost:{_SERVER_PORT}"))
    print("╚" + "═" * W + "╝")
    print()
    live_seen = 0
    deadline  = time.time() + timeout
    _LOG_FILE.touch(exist_ok=True)
    with open(_LOG_FILE) as f:
        while time.time() < deadline:
            line = f.readline()
            if line:
                if "LIVE" in line or "ERROR: " in line:
                    print(line, end="", flush=True)
                if "LIVE" in line:
                    live_seen += 1
                if live_seen >= cam_count:
                    break
            else:
                time.sleep(0.1)


def _restart_server() -> None:
    """Kill server and restart in background (used after config changes)."""
    _kill_server()
    _start_background()


def _load_local_env() -> None:
    """Load key=value pairs from local .env into os.environ (won't overwrite existing)."""
    if not _LOCAL_ENV.exists():
        return
    with open(_LOCAL_ENV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _build_cams_from_config() -> dict:
    """Build CAMS dict from config.json cameras section.
    Falls back to CAM1_*/CAM2_* env vars for backward compatibility.
    Prompts interactively if no cameras found anywhere."""
    import getpass, json

    cfg = _load_config()
    cameras: dict = cfg.get("cameras", {})

    # Legacy fallback: read from env vars and migrate into config
    if not cameras:
        host1 = os.getenv("CAM1_HOST") or os.getenv("TAPO_HOST", "")
        if host1:
            cameras["cam1"] = {
                "host": host1,
                "user": os.getenv("CAM1_USER") or os.getenv("TAPO_USER", ""),
                "password": os.getenv("CAM1_PASSWORD") or os.getenv("TAPO_PASSWORD", ""),
            }
            host2 = os.getenv("CAM2_HOST") or os.getenv("TAPO_HOST_2", "")
            if host2:
                cameras["cam2"] = {
                    "host": host2,
                    "user": os.getenv("CAM2_USER") or os.getenv("TAPO_USER_2", "") or cameras["cam1"]["user"],
                    "password": os.getenv("CAM2_PASSWORD") or os.getenv("TAPO_PASSWORD_2", "") or cameras["cam1"]["password"],
                }
            cfg["cameras"] = cameras
            _save_config(cfg)

    # Interactive prompt if still empty
    if not cameras:
        print("\n── RTSP Camera Setup ─────────────────────────────────")
        print("No cameras configured. Add at least one camera.\n")
        name = input("  Camera name (e.g. home, front): ").strip() or "cam1"
        host = input("  IP address: ").strip()
        user = input("  Username: ").strip()
        pwd  = getpass.getpass("  Password: ").strip()
        cameras[name] = {"host": host, "user": user, "password": pwd}
        cfg["cameras"] = cameras
        _save_config(cfg)
        print(f"  Saved. Run --add-cam to add more cameras.")
        print("─────────────────────────────────────────────────────\n")

    # Build CAMS dict — fallback to first cam's credentials if user/password blank
    first = next(iter(cameras.values()), {})
    result = {}
    for name, c in cameras.items():
        result[name] = {
            "host":     c.get("host", ""),
            "user":     c.get("user", "") or first.get("user", ""),
            "password": c.get("password", "") or first.get("password", ""),
            "port":     int(c.get("port", CAM_DEFAULT_PORT)),
            "path_sd":  c.get("path_sd", CAM_DEFAULT_PATH_SD),
            "path_hd":  c.get("path_hd", CAM_DEFAULT_PATH_HD),
            "label":    name.upper(),
            "ip":       c.get("host", ""),
        }
    return result


_load_local_env()
CAMS = _build_cams_from_config()


UPLOAD_DIR = Path("/tmp/rtsp_cam_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_uploaded_videos: dict[str, dict[str, str | float]] = {}

STREAM_QUALITY_BY_MODE = {
    "standard": "VGA",
    "hd": "HD",
}

# ── state ─────────────────────────────────────────────────────────────────────
# Both cams run continuously at "standard" quality in background threads.
# The browser hides whichever one isn't active. HD is enabled on-demand per-cam
# and restarts only that cam's thread, leaving the other untouched.
_active_cam:      str = next(iter(CAMS), "cam1")
_state_lock           = threading.Lock()
_cam_threads:     dict[str, threading.Thread] = {}
_cam_stop_events: dict[str, threading.Event]  = {}
_cam_modes:       dict[str, str]              = {}   # cam -> "standard" | "hd"
_cam_statuses:    dict[str, str]              = {}   # cam -> "connecting" | "live" | "offline"
_shutdown_requested    = threading.Event()
_signal_count          = 0
_prev_signal_handlers: dict[int, object]      = {}
_stderr_devnull        = None
_detection_enabled     = True
_detection_lock        = threading.Lock()
_source_mode           = "rtsp"   # "rtsp" | "webcam" | "upload"
_source_lock           = threading.Lock()
_cam_visible_names:    dict[str, frozenset] = {}  # cam -> current named tracks
_bg_detect_interval    = 2.0   # seconds between background pipeline runs per cam


def _write_monitor_log(line: str) -> None:
    import time as _time
    ts = _time.strftime("%H:%M:%S")
    try:
        _MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _MONITOR_LOG.open("a") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass


def _clear_monitor_log() -> None:
    try:
        _MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        _MONITOR_LOG.write_text("")
    except Exception:
        pass


def _log_cam_state() -> None:
    cam = _active_cam
    cfg = CAMS.get(cam, {})
    label = cfg.get("label", cam)
    ip    = cfg.get("ip", cfg.get("host", ""))
    hd    = "hd" if _cam_modes.get(cam) == "hd" else "sd"
    src   = _source_mode
    det   = "on" if _detection_enabled else "off"
    if src == "rtsp":
        cam_str = f"{label} ({ip})"
    elif src == "webcam":
        cam_str = "webcam"
    else:
        cam_str = "upload"
    _write_monitor_log(f"STATE | cam: {cam_str} | hd: {hd} | detect: {det}")


def _silence_stderr() -> None:
    """Redirect stderr to /dev/null once shutdown starts."""
    global _stderr_devnull
    try:
        if getattr(_sys.stderr, "name", None) == os.devnull:
            return
        if _stderr_devnull is None or _stderr_devnull.closed:
            _stderr_devnull = open(os.devnull, "w")
        _sys.stderr = _stderr_devnull
    except Exception:
        pass


def _request_shutdown() -> None:
    """Fan out stop requests immediately so Ctrl+C feels responsive."""
    _shutdown_requested.set()
    for evt in list(_cam_stop_events.values()):
        try:
            evt.set()
        except Exception:
            pass


def _join_camera_threads(total_timeout: float = 2.5) -> None:
    """Give camera threads a short shared window to exit, then move on."""
    _request_shutdown()
    deadline = time.time() + total_timeout
    for t in list(_cam_threads.values()):
        if not t or not t.is_alive():
            continue
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        t.join(timeout=remaining)
    for cam_key in CAMS:
        _cam_statuses[cam_key] = "offline"


def _graceful_signal_handler(sig, frame):
    """First Ctrl+C requests a clean stop; second one exits immediately."""
    global _signal_count
    _signal_count += 1
    _silence_stderr()
    _request_shutdown()

    if _signal_count > 1:
        raise SystemExit(130)

    import signal as _signal
    prev = _prev_signal_handlers.get(sig, _signal.SIG_DFL)
    if prev in (None, _signal.SIG_IGN, _signal.SIG_DFL, _graceful_signal_handler):
        raise KeyboardInterrupt()
    if callable(prev):
        prev(sig, frame)
    else:
        raise KeyboardInterrupt()


def _install_signal_handlers() -> None:
    """Install our handler early, then re-chain it after uvicorn replaces signals."""
    import signal as _signal
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        current = _signal.getsignal(sig)
        if current is _graceful_signal_handler:
            continue
        _prev_signal_handlers[sig] = current
        _signal.signal(sig, _graceful_signal_handler)


def build_url(host: str, user: str, password: str, stream: str,
              port: int = CAM_DEFAULT_PORT,
              path_sd: str = CAM_DEFAULT_PATH_SD,
              path_hd: str = CAM_DEFAULT_PATH_HD) -> str:
    from urllib.parse import quote
    u = quote(user, safe="")
    p = quote(password, safe="")
    path = path_hd if stream == "hd" else path_sd
    return f"rtsp://{u}:{p}@{host}:{port}{path}"

_feeds = {}

class _LiveFeed:
    def __init__(self, url: str, cam_key: str, is_hd: bool = False):
        import cv2
        self.cv2 = cv2
        self.url = url
        self.cam_key = cam_key
        self.is_hd = is_hd
        self.frame_jpeg = None
        self.latest_frame = None
        self.frame_ready = threading.Condition()
        self.fps = 0.0
        self._frame_ts = 0.0
        self.error_count = 0

    def run(self, stop_evt: threading.Event) -> None:
        is_hd = self.is_hd
        # Match test_rtsp.py exactly — no extra FFmpeg flags, no BUFFERSIZE tweak.
        # Aggressive flags (nobuffer/low_delay/framedrop) destabilize HD streams.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        while not stop_evt.is_set():
            cap = self.cv2.VideoCapture(self.url, self.cv2.CAP_FFMPEG)
            cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                log.warning(f"[{self.cam_key}] could not open RTSP. Retrying...")
                time.sleep(2)
                continue

            # Drain stale frames buffered during cap.open().
            # SD: per-frame timing — buffered frames decode fast (<5ms), live blocks >25ms.
            # HD: wall-clock drain — raw HD frames all decode slowly so timing can't
            #     distinguish buffered from live. Read for 3s to flush the buffer.
            drained = 0
            if is_hd:
                drain_end = time.perf_counter() + 3.0
                while not stop_evt.is_set() and time.perf_counter() < drain_end:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    drained += 1
            else:
                cap.read()  # absorb decoder-init latency
                while not stop_evt.is_set():
                    t = time.perf_counter()
                    ret, _ = cap.read()
                    elapsed = time.perf_counter() - t
                    if not ret or elapsed > 0.025:
                        break
                    drained += 1

            import time as _t
            _sys.__stdout__.write(f"  [{_t.strftime('%H:%M:%S')}]  {self.cam_key.upper()}  LIVE  (drained {drained} stale frames)\n")
            _sys.__stdout__.flush()
            _cam_statuses[self.cam_key] = "live"
            frames = 0
            t0 = time.time()
            self.error_count = 0

            while not stop_evt.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.error_count += 1
                    if self.error_count > 30:
                        log.warning(f"[{self.cam_key}] lost RTSP stream, reconnecting...")
                        break
                    time.sleep(0.01)
                    continue

                self.error_count = 0
                # HD: resize BEFORE encoding. Encoding 2304×1296 takes 30-80ms,
                # blocking cap.read() and causing FFmpeg TCP socket backup →
                # frames pile up → slow motion + black flash.
                # Resize to 1280px first so both MJPEG and pipeline work on a
                # smaller frame — encode takes ~5ms instead of 50ms.
                if is_hd:
                    ph, pw = frame.shape[:2]
                    scale = 1280 / pw
                    frame = self.cv2.resize(
                        frame, (1280, int(ph * scale)),
                        interpolation=self.cv2.INTER_AREA)
                ok, buf = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.frame_ready:
                        self.latest_frame = frame
                        self.frame_jpeg = buf.tobytes()
                        self._frame_ts = time.time()
                        self.frame_ready.notify_all()
                    
                    frames += 1
                    dt = time.time() - t0
                    if dt >= 1.0:
                        self.fps = frames / dt
                        frames = 0
                        t0 = time.time()
            
            cap.release()
            if not stop_evt.is_set():
                time.sleep(1)

def _run_rtsp(cam_key: str, stop_evt: threading.Event, mode_label: str):
    cfg = CAMS[cam_key]
    mode_key = mode_label.lower()
    stream = "hd" if mode_key == "hd" else "sd"
    url = build_url(cfg["host"], cfg["user"], cfg["password"], stream,
                    port=cfg.get("port", CAM_DEFAULT_PORT),
                    path_sd=cfg.get("path_sd", CAM_DEFAULT_PATH_SD),
                    path_hd=cfg.get("path_hd", CAM_DEFAULT_PATH_HD))
    feed = _LiveFeed(url, cam_key, is_hd=(mode_key == "hd"))
    _feeds[cam_key] = feed
    feed.run(stop_evt)

def _run_standard(cam_key: str, stop_evt: threading.Event):
    _run_rtsp(cam_key, stop_evt, "standard")

def _run_hd(cam_key: str, stop_evt: threading.Event):
    _run_rtsp(cam_key, stop_evt, "hd")


def _cam_thread(cam_key: str, mode: str, stop_evt: threading.Event):
    """Retry loop with exponential backoff. Dispatches to standard or hd runner."""
    delay = 5
    try:
        while not stop_evt.is_set():
            try:
                _cam_statuses[cam_key] = "connecting"
                if mode == "hd":
                    _run_hd(cam_key, stop_evt)
                else:
                    _run_standard(cam_key, stop_evt)
                if stop_evt.is_set():
                    break
                delay = 5
            except Exception as e:
                _cam_statuses[cam_key] = "offline"
                for _ in range(delay * 10):
                    if stop_evt.is_set():
                        break
                    time.sleep(0.1)
                delay = min(delay * 2, 60)
    finally:
        _cam_statuses[cam_key] = "offline"


def _start_camera(cam_key: str, mode: str = "standard"):
    """Start or restart a single camera's stream thread. Other cams are untouched."""
    with _state_lock:
        if cam_key not in CAMS or not CAMS[cam_key]["host"]:
            raise ValueError(f"camera {cam_key} not configured")
        if mode not in ("standard", "hd"):
            raise ValueError(f"unknown mode {mode}")

        existing     = _cam_threads.get(cam_key)
        existing_evt = _cam_stop_events.get(cam_key)
        if existing and existing.is_alive():
            log.info(f"[{cam_key}] stopping current thread before restart...")
            if existing_evt:
                existing_evt.set()
            existing.join(timeout=15)
            if existing.is_alive():
                log.warning(f"[{cam_key}] previous thread did not exit cleanly")

        stop_evt = threading.Event()
        t = threading.Thread(
            target=_cam_thread, args=(cam_key, mode, stop_evt), daemon=True
        )
        t.start()
        _cam_threads[cam_key]     = t
        _cam_stop_events[cam_key] = stop_evt
        _cam_modes[cam_key]       = mode
        _cam_statuses[cam_key]    = "connecting"


def _bg_detect_loop() -> None:
    """Background face-detection loop for inactive cameras (log-only).

    Uses InsightFace only — no YOLO, no tracking — so it never contends with
    the active-cam pipeline. Results go to the log and monitor file only;
    they have no effect on the UI display or track state.
    """
    import time as _time
    # Wait until at least one cam is live to avoid competing with startup.
    while not _shutdown_requested.is_set():
        if any(s == "live" for s in _cam_statuses.values()):
            break
        _shutdown_requested.wait(timeout=2.0)

    while not _shutdown_requested.is_set():
        try:
            for cam in list(CAMS.keys()):
                if _shutdown_requested.is_set():
                    break
                if cam == _active_cam:
                    continue   # active cam is handled by browser polling
                if _cam_statuses.get(cam) != "live":
                    continue
                frame = _get_latest_frame(cam)
                if frame is None:
                    continue
                if not _face_app_loaded.is_set():
                    continue   # InsightFace still loading — skip, don't block
                try:
                    dets = _detect_faces_in_frame(frame, cam, emit_log=False)
                except Exception as exc:
                    log.warning(f"[bg-detect][{cam}] face-detect error: {exc}")
                    continue
                known = [d for d in dets if d.get("name") and d["name"] != "Unknown"]
                names = frozenset(d["name"] for d in known)
                prev  = _cam_visible_names.get(cam)
                if names != prev:
                    _cam_visible_names[cam] = names
                    if names:
                        score_map = {d["name"]: d.get("recog_score") or 0.0 for d in known}
                        parts = [f"{n} ({round(score_map[n]*100)}%)" for n in sorted(names)]
                        _write_monitor_log(f"DETECT | {cam}: {', '.join(parts)}")
                if known:
                    best = max(known, key=lambda d: d.get("recog_score") or 0.0)
                    _log_recog(cam, best["name"], best.get("recog_score") or 0.0)
                else:
                    _log_recog(cam, None)
        except Exception as exc:
            log.warning(f"[bg-detect] unexpected error: {exc}")
        _shutdown_requested.wait(timeout=_bg_detect_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _active_cam
    _install_signal_handlers()
    started: list[str] = []
    for cam_key, cfg in CAMS.items():
        if _shutdown_requested.is_set():
            break
        if cfg["host"]:
            await asyncio.to_thread(_start_camera, cam_key, "standard")
            started.append(cam_key)
    if started and not _shutdown_requested.is_set():
        _active_cam = started[0]

    if not _shutdown_requested.is_set():
        # ── clean startup banner ──────────────────────────────────────────
        print()
        W = 38
        def _row(text): return f"║  {text:<{W - 2}}║"
        print("╔" + "═" * W + "╗")
        print("║" + "RTSP CAM VIEWER  READY".center(W) + "║")
        print("╠" + "═" * W + "╣")
        print(_row(f"Cameras : {len(started)}"))
        for cam_key in started:
            print(_row(f"  {cam_key}  →  {CAMS[cam_key]['host']}"))
        print("╠" + "═" * W + "╣")
        print(_row("http://localhost:8766"))
        print("╚" + "═" * W + "╝")
        print()

        _clear_monitor_log()
        _log_cam_state()

        # Pre-warm YOLO and InsightFace in parallel background threads so both
        # models are ready as soon as possible. Pipeline calls return [] / skip
        # face recognition until each model signals ready via its Event, keeping
        # the thread pool free and the MJPEG stream smooth during startup.
        threading.Thread(target=_get_yolo,     daemon=True, name="yolo-warmup").start()
        threading.Thread(target=_get_face_app, daemon=True, name="insightface-warmup").start()

        # Background detection loop — keeps running pipeline on inactive cams so
        # all detections appear in the log regardless of which cam is shown in UI.
        threading.Thread(target=_bg_detect_loop, daemon=True, name="bg-detect").start()

    try:
        yield
    finally:
        _join_camera_threads(total_timeout=2.0)


app = FastAPI(lifespan=lifespan)



from fastapi.responses import StreamingResponse

async def mjpeg_generator(cam_key: str):
    feed = _feeds.get(cam_key)
    if not feed:
        return
    last_ts = 0.0
    while True:
        ts = feed._frame_ts
        if ts > last_ts:
            jpg = feed.frame_jpeg
            if jpg:
                last_ts = ts
                yield (b"--FRAME\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                       jpg + b"\r\n")
        else:
            await asyncio.sleep(0.01)

@app.get("/mjpeg/{cam}")
async def mjpeg_stream(cam: str):
    if cam not in CAMS:
        raise HTTPException(404, "unknown camera")
    return StreamingResponse(mjpeg_generator(cam), media_type="multipart/x-mixed-replace; boundary=FRAME")

@app.get("/state")
async def state():
    return {
        "active": _active_cam,
        "cameras": {
            k: {
                "label":     v["label"],
                "ip":        v["ip"],
                "available": bool(v["host"]),
                "mode":      _cam_modes.get(k, "standard"),
                "status":    _cam_statuses.get(k, "offline"),
            }
            for k, v in CAMS.items()
        },
    }


@app.post("/switch/{cam}")
async def switch(cam: str, mode: str = "standard"):
    global _active_cam
    if cam not in CAMS:
        raise HTTPException(404, "unknown camera")
    if not CAMS[cam]["host"]:
        raise HTTPException(400, "camera not configured")
    if mode not in ("standard", "hd"):
        raise HTTPException(400, "unknown mode")

    prev_cam  = _active_cam
    _active_cam = cam
    mode_changed = (_cam_modes.get(cam) != mode)

    # Pure cam switches (same mode) are free — both RTSP feeds run continuously
    # in background threads. Only a real mode change costs a stream-thread
    # restart, and it only touches THIS cam's thread.
    if mode_changed:
        await asyncio.to_thread(_start_camera, cam, mode)

    if cam != prev_cam or mode_changed:
        _log_cam_state()

    return {"active": cam, "mode": mode, "mode_changed": mode_changed}


@app.post("/detection/{state}")
async def set_detection(state: str):
    global _detection_enabled
    if state not in ("on", "off"):
        raise HTTPException(400, "use 'on' or 'off'")
    with _detection_lock:
        _detection_enabled = (state == "on")
    _log_cam_state()
    return {"detection": _detection_enabled}


@app.post("/source/{mode}")
async def set_source(mode: str):
    global _source_mode
    if mode not in ("rtsp", "webcam", "upload"):
        raise HTTPException(400, "use 'rtsp', 'webcam', or 'upload'")
    with _source_lock:
        _source_mode = mode
    _log_cam_state()
    return {"source": _source_mode}


# ── InsightFace (server-side face detection) ──────────────────────────────────
_face_app        = None
_face_app_lock   = threading.Lock()
_face_app_loaded = threading.Event()   # set once InsightFace is ready; lets pipeline skip the block

_cfg = _load_config()
# Resolve log paths from config (overrides the defaults set at module top)
_LOG_FILE    = Path(_cfg.get("server_log",  "/tmp/live_web.log"))
_MONITOR_LOG = Path(_cfg.get("monitor_log", "/tmp/cam_monitor.log"))
_strictness = max(1, min(10, int(_cfg.get("identity_strictness", 5))))
_cfg_errors = _validate_config(_cfg)
if _cfg_errors:
    print(_clr("ERROR: ", "1;31") + "config.json contains invalid color values:")
    for _e in _cfg_errors:
        print(_e)
    print("Fix config.json (run: python3 live_web.py --edit-config) and try again.")
    raise SystemExit(1)


_family_dir_raw = _cfg.get("family_dir", "family")
FAMILY_DIR = (Path(_family_dir_raw) if Path(_family_dir_raw).is_absolute()
              else Path(__file__).parent / _family_dir_raw)
# All recognition thresholds scale with identity_strictness (1=loose → 10=strict).
# _strictness is set earlier from config; these must be defined after it.
RECOG_THRESH  = round(0.50 + (_strictness - 1) * 0.016, 3)  # 1→0.50  5→0.56  10→0.64
RECOG_MARGIN  = round(0.06 + (_strictness - 1) * 0.009, 3)  # 1→0.06  5→0.10  10→0.14
DISPLAY_RECOG_SCORE_MIN  = RECOG_THRESH
DISPLAY_RECOG_MARGIN_MIN = RECOG_MARGIN
_face_db: dict[str, list] = {}          # {name: [normed_embedding, ...]}

# Per-camera (or 'upload') mask regions in NORMALIZED frame coords (x1, y1, x2, y2).
# Any face whose center falls inside a region is discarded before identity matching.
# Used to ignore the round mirror in cam1 which reflects people already in frame.
_MIRROR_MASKS: dict[str, list[tuple[float, float, float, float]]] = {
    "cam1":   [(0.340, 0.220, 0.515, 0.560)],
    # Test-uploaded videos from the same camera get the same mask.
    "upload": [(0.340, 0.220, 0.515, 0.560)],
}

# Terminal recognition log state — per camera
# Rules: log name once when identified, re-log every 5 min while active,
#        log Unknown only if no known face seen for 10 min straight.
_RECOG_LOG_INTERVAL  = 5 * 60    # re-announce known name every 5 min
_RECOG_UNKNOWN_AFTER = 10 * 60   # declare Unknown after 10 min with no known face
_recog: dict[str, dict] = {}     # {cam: {locked_name, locked_at, last_logged_at, last_known_at}}


def _get_face_app():
    """Lazy-init InsightFace FaceAnalysis. Returns None if not installed."""
    global _face_app
    if _face_app is not None:
        return _face_app
    with _face_app_lock:
        if _face_app is not None:
            return _face_app
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _face_app = app
            _build_face_db(app)
            _face_app_loaded.set()   # signal: face recognition is now available
        except ImportError:
            log.error("insightface not installed — run: pip3 install insightface --break-system-packages")
        except Exception as e:
            log.error(f"InsightFace init error: {e}")
    return _face_app


def _build_face_db(fa) -> None:
    """Load all reference photos from family/ and store normed embeddings per person."""
    import numpy as np
    if not FAMILY_DIR.exists():
        log.warning(f"Family directory not found: {FAMILY_DIR}")
        return
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for person_dir in sorted(FAMILY_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        embeddings = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            if not _CV2_OK:
                continue
            img = _cv2.imread(str(img_path))
            if img is None:
                continue
            faces = fa.get(img)
            if not faces:
                log.warning(f"No face found in {img_path.name} — skipping")
                continue
            # Take the most confident face in the reference photo
            best = max(faces, key=lambda f: float(f.det_score))
            emb  = getattr(best, "normed_embedding", None)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            _face_db[name] = embeddings
        else:
            pass  # silently skip — person has no usable face embeddings


def _identify_face(face) -> tuple[str, float, float]:
    """Compare a detected face embedding against the DB. Returns (name, best_score, margin)."""
    import numpy as np
    emb = getattr(face, "normed_embedding", None)
    if emb is None or not _face_db:
        return "Unknown", 0.0, 0.0
    best_name   = "Unknown"
    best_score  = 0.0
    second_best = 0.0
    for name, ref_embs in _face_db.items():
        scores = [float(np.dot(emb, r)) for r in ref_embs]
        score  = max(scores)
        if score > best_score:
            second_best = best_score
            best_score  = score
            best_name   = name
        elif score > second_best:
            second_best = score
    margin = best_score - second_best
    if best_score < RECOG_THRESH or margin < RECOG_MARGIN:
        return "Unknown", best_score, margin
    return best_name, best_score, margin


def _get_latest_frame(cam_key: str):
    feed = _feeds.get(cam_key)
    if feed:
        return feed.latest_frame
    return None


def _purge_old_uploads(max_age_sec: float = 6 * 60 * 60) -> None:
    """Best-effort cleanup for stale test uploads."""
    now = time.time()
    stale_ids = [
        upload_id
        for upload_id, meta in list(_uploaded_videos.items())
        if now - float(meta.get("created_at", 0.0)) > max_age_sec
    ]
    for upload_id in stale_ids:
        meta = _uploaded_videos.pop(upload_id, None)
        if not meta:
            continue
        try:
            Path(str(meta["path"])).unlink(missing_ok=True)
        except Exception:
            pass


def _store_uploaded_video(file_obj, filename: str | None) -> dict[str, str | float]:
    """Persist a test-history clip so backend face detection can read frames from it."""
    _purge_old_uploads()
    upload_id = secrets.token_hex(8)
    suffix = Path(filename or "history.mp4").suffix.lower() or ".mp4"
    path = UPLOAD_DIR / f"{upload_id}{suffix}"
    with path.open("wb") as dst:
        shutil.copyfileobj(file_obj, dst)
    meta = {
        "id": upload_id,
        "path": str(path),
        "name": Path(filename or path.name).name,
        "created_at": time.time(),
    }
    _uploaded_videos[upload_id] = meta
    return meta


def _get_uploaded_video(upload_id: str) -> dict[str, str | float] | None:
    meta = _uploaded_videos.get(upload_id)
    if not meta:
        return None
    path = Path(str(meta["path"]))
    if not path.exists():
        _uploaded_videos.pop(upload_id, None)
        return None
    return meta


def _get_video_frame_at(video_path: Path, pos_sec: float):
    """Decode one frame at the requested timestamp from an uploaded clip."""
    if not _CV2_OK:
        return None
    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        pos_sec = max(0.0, float(pos_sec or 0.0))
        cap.set(_cv2.CAP_PROP_POS_MSEC, pos_sec * 1000.0)
        ret, frame = cap.read()
        if not ret and pos_sec > 0:
            fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, int(pos_sec * fps))
                ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def _log_recog(cam: str, name: str | None, score: float = 0.0) -> None:
    """Terminal recognition logger. Logs a name on first sighting, re-logs every
    5 min, and prints Unknown only after 10 min of no known face."""
    import time as _time
    now  = _time.time()
    ts   = _time.strftime("%H:%M:%S")
    st   = _recog.setdefault(cam, {"locked_name": None, "locked_at": 0,
                                    "last_logged_at": 0, "last_known_at": 0})
    if name:
        st["last_known_at"] = now
        changed = st["locked_name"] != name
        if changed:
            st["locked_name"]    = name
            st["locked_at"]      = now
            st["last_logged_at"] = now
            # Store the first (best) score for this name — reused for all repeat logs
            # so it matches the GUI locked_score rather than fluctuating crop scores.
            if score:
                st["locked_score"] = score
        logged_score = st.get("locked_score", score)
        s = f" match {round(logged_score * 100)}%" if logged_score else ""
        if changed:
            print(f"  [{ts}]  {name}{s}  →  {cam.upper()}")
        elif now - st["last_logged_at"] >= _RECOG_LOG_INTERVAL:
            st["last_logged_at"] = now
            print(f"  [{ts}]  {name}{s}  →  {cam.upper()}")
    else:
        if st["locked_name"] and now - st["last_known_at"] >= _RECOG_UNKNOWN_AFTER:
            st["locked_name"]    = None
            st["locked_score"]   = 0.0
            st["last_logged_at"] = now
            print(f"  [{ts}]  Unknown  →  {cam.upper()}")


def _detect_faces_in_frame(
    frame,
    source_key: str,
    *,
    emit_log: bool,
) -> list[dict]:
    """Shared InsightFace pipeline for both live RTSP frames and uploaded test videos."""
    fa = _get_face_app()
    if fa is None or frame is None:
        if emit_log:
            _log_recog(source_key, None, None, None)
        return []

    faces = fa.get(frame)
    h, w = frame.shape[:2]
    # Minimum face area: drop tiny hallucinated faces (paintings, mirror reflections,
    # distant clutter) — require at least ~1.5% of the shorter frame dimension on each side.
    min_side = max(18, int(min(h, w) * 0.035))
    min_area = min_side * min_side
    # Per-camera mask regions (normalized x1,y1,x2,y2). Faces whose center falls inside
    # any region are discarded — used to ignore the mirror in cam1 which reflects
    # the same person already visible in the frame.
    masks = _MIRROR_MASKS.get(source_key, [])
    results = []
    best_display_known = None  # (face, name, recog_score) — best strong match for log

    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox
        fw_box = max(0.0, fx2 - fx1)
        fh_box = max(0.0, fy2 - fy1)
        if fw_box * fh_box < min_area or fw_box < min_side or fh_box < min_side:
            continue
        cx_n = ((fx1 + fx2) / 2) / w
        cy_n = ((fy1 + fy2) / 2) / h
        in_mask = any(mx1 <= cx_n <= mx2 and my1 <= cy_n <= my2
                      for (mx1, my1, mx2, my2) in masks)
        if in_mask:
            continue
        name, recog_score, recog_margin = _identify_face(face)
        results.append({
            "bbox": face.bbox.tolist(),
            "score": round(float(face.det_score), 3),
            "recog_score": round(float(recog_score), 3),
            "recog_margin": round(float(recog_margin), 3),
            "fw": w, "fh": h,
            "cx": float((face.bbox[0] + face.bbox[2]) / 2),
            "cy": float((face.bbox[1] + face.bbox[3]) / 2),
            "name": name,
        })
        if name != "Unknown" and (
            recog_score >= DISPLAY_RECOG_SCORE_MIN and
            recog_margin >= DISPLAY_RECOG_MARGIN_MIN and
            (best_display_known is None or recog_score > best_display_known[2])
        ):
            best_display_known = (face, name, recog_score)

    # Same-name dedup: for each non-Unknown name, keep only the largest bbox area.
    # Suppresses secondary detections of the same person (mirror reflections, etc.)
    # without affecting multi-person scenes.
    by_name: dict[str, dict] = {}
    unnamed: list[dict] = []
    for r in results:
        n = r.get("name")
        if not n or n == "Unknown":
            unnamed.append(r)
            continue
        bx = r["bbox"]
        area = max(0.0, bx[2] - bx[0]) * max(0.0, bx[3] - bx[1])
        prev = by_name.get(n)
        if prev is None:
            by_name[n] = r
        else:
            pbx = prev["bbox"]
            parea = max(0.0, pbx[2] - pbx[0]) * max(0.0, pbx[3] - pbx[1])
            if area > parea:
                by_name[n] = r
    results = list(by_name.values()) + unnamed

    if emit_log:
        if best_display_known:
            _, name, recog_score = best_display_known
            _log_recog(source_key, name, recog_score)
        else:
            _log_recog(source_key, None)

    return results


# ─── YOLOv11-pose (+ BoT-SORT tracker) ────────────────────────────────────────
_yolo_model = None
_yolo_lock  = threading.Lock()    # guards initialization only
_yolo_ready = threading.Event()   # set once YOLO is loaded; pipeline checks this non-blocking

def _get_yolo() -> object | None:
    """Lazy-load Ultralytics YOLOv11-pose. Returns None if library missing."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model if _yolo_model is not False else None
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model if _yolo_model is not False else None
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolo11n-pose.pt")
            log.info("YOLOv11-pose model loaded")
            _yolo_ready.set()
        except Exception as e:
            log.error("ultralytics not available: %s", e)
            _yolo_model = False
    return _yolo_model if _yolo_model else None


def _run_yolo_pose(frame, source_key: str, track: bool) -> list[dict]:
    """Returns per-person detections with bbox + nose keypoint + optional track_id."""
    if frame is None:
        return []
    model = _get_yolo()
    if model is None:
        return []
    h, w = frame.shape[:2]
    try:
        if track:
            results = model.track(
                frame, persist=True, tracker="botsort.yaml",
                classes=[0], conf=round(0.30 + (max(1, min(10, int(_cfg.get("detection_sensitivity", 5)))) - 1) * 0.033, 3), imgsz=640, verbose=False,
            )
        else:
            results = model.predict(
                frame, classes=[0], conf=round(0.30 + (max(1, min(10, int(_cfg.get("detection_sensitivity", 5)))) - 1) * 0.033, 3), imgsz=640, verbose=False,
            )
    except Exception as e:
        log.warning("[yolo] inference failed: %s", e)
        return []
    out: list[dict] = []
    if not results:
        return out
    r = results[0]
    boxes = getattr(r, "boxes", None)
    kpts  = getattr(r, "keypoints", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
    ids = None
    if track and getattr(boxes, "id", None) is not None:
        ids = boxes.id.cpu().numpy().astype(int).tolist()
    kxy = kpts.xy.cpu().numpy() if kpts is not None and kpts.xy is not None else None
    kconf = kpts.conf.cpu().numpy() if kpts is not None and getattr(kpts, "conf", None) is not None else None
    # COCO pose keypoint order: 0 nose, 1 leye, 2 reye, 3 lear, 4 rear, 5 lshoulder, 6 rshoulder
    for i, (bb, cf) in enumerate(zip(xyxy, confs)):
        x1, y1, x2, y2 = bb.tolist()
        entry: dict = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": round(float(cf), 3),
            "fw": w, "fh": h,
        }
        if ids is not None:
            entry["track_id"] = ids[i]
        if kxy is not None:
            pts = kxy[i]
            pconf = kconf[i] if kconf is not None else None
            def kp(idx):
                p = pts[idx]
                c = float(pconf[idx]) if pconf is not None else 1.0
                if c < 0.25:
                    return None
                return [float(p[0]), float(p[1]), round(c, 2)]
            entry["nose"]  = kp(0)
            entry["leye"]  = kp(1)
            entry["reye"]  = kp(2)
            entry["lear"]  = kp(3)
            entry["rear"]  = kp(4)
            entry["lsh"]   = kp(5)
            entry["rsh"]   = kp(6)
        out.append(entry)
    return out


@app.get("/pose/{cam}")
async def pose_cam(cam: str, track: int = 0):
    if cam not in CAMS:
        raise HTTPException(404, "unknown camera")
    def _run() -> list[dict]:
        frame = _get_latest_frame(cam)
        return _run_yolo_pose(frame, cam, bool(track))
    result = await asyncio.to_thread(_run)
    return {"detections": result}


@app.get("/pose/upload/{upload_id}")
async def pose_upload(upload_id: str, t: float = 0.0, track: int = 0):
    meta = _get_uploaded_video(upload_id)
    if not meta:
        raise HTTPException(404, "upload not found")
    def _run() -> list[dict]:
        frame = _get_video_frame_at(Path(str(meta["path"])), t)
        return _run_yolo_pose(frame, f"upload:{upload_id}", bool(track))
    result = await asyncio.to_thread(_run)
    return {"detections": result}


# ─── Unified pipeline: YOLO-pose + BoT-SORT + InsightFace per track ──────────
# Persistent per-source, per-track identity state.
_track_state: dict[str, dict[int, dict]] = {}
_GALLERY_KEY = "__gallery__"  # {name: [best embeddings]}  (shared across tracks)
_gallery: dict[str, list] = {}

_TRACK_STATE_TTL_SEC = 30.0

# identity_strictness: 1 (loose, fast to name) → 10 (strict, needs high confidence)
# Drives VOTE_MIN, VOTE_SCORE_MIN, and VOTE_MARGIN_MIN linearly.
_VOTE_WINDOW     = 8
_VOTE_MIN        = max(1, round(1 + (_strictness - 1) * 0.44))  # 1→1  5→3  10→5
_VOTE_SCORE_MIN  = round(0.44 + (_strictness - 1) * 0.016, 3)  # 1→0.44  5→0.50  10→0.59
_VOTE_MARGIN_MIN = round(0.05 + (_strictness - 1) * 0.011, 3)  # 1→0.05  5→0.09  10→0.15

# Override thresholds: allow re-locking a track when a DIFFERENT person is
# detected with very high confidence. Handles BoT-SORT track swaps (when two
# people are close, the tracker sometimes re-assigns a track_id to the wrong body;
# the old locked name then rides the wrong track until this override fires).
_OVERRIDE_SCORE_MIN  = 0.60   # face score must be this strong to override a lock
_OVERRIDE_MARGIN_MIN = 0.12   # must beat runner-up by this much (avoids ambiguous overrides)

# Lock expiry: if a track has been locked to a name but no face has been
# detected in it for this many seconds, clear the lock. Prevents a stale
# Yaniv lock from riding Chalie's body after a BoT-SORT track swap where
# Chalie's face is not visible (so override/spatial checks can't fire).
_LOCK_FACE_TTL_SEC = 4.0

# High-confidence grace period: if a lock is established at >= this score,
# keep showing the name for _GRACE_DURATION seconds even when the face is
# not visible (person moved / turned away). During that window we keep
# capturing crops for re-evaluation. At expiry we check if the crops
# confirmed the name — if yes, re-lock with a fresh grace; if not, clear.
_GRACE_SCORE_MIN    = 0.80   # locked_score must reach this to activate grace
_GRACE_DURATION     = 60.0   # seconds to hold the name after high-confidence lock
_GRACE_CONFIRM_MIN  = 2      # min confirming crops needed during grace to extend

# Head box as fraction of frame width — stays same canvas size in SD and HD.
# box_size in config = SD-pixel reference width of the head box (default 73).
# box_offset_x / box_offset_y = shift in SD pixels (positive = right/down).
FIXED_HEAD_RATIO    = _cfg.get("box_size", 73) / 1280
_BOX_OFFSET_X_RATIO = _cfg.get("box_offset_x", 0) / 1280
_BOX_OFFSET_Y_RATIO = _cfg.get("box_offset_y", 0) / 1280
_GALLERY_PER_NAME    = 8
_REID_SCORE_MIN      = 0.55


def _head_crop_from_det(frame, det) -> tuple:
    """Return (crop, bbox_on_frame) for the head region, or (None, None)."""
    pts = [det.get(k) for k in ("nose", "leye", "reye", "lear", "rear")]
    pts = [p for p in pts if p]
    if not pts:
        return None, None
    nose = det.get("nose") or pts[0]
    cx, cy = float(nose[0]), float(nose[1])
    if det.get("lear") and det.get("rear"):
        import math
        ear_w = math.hypot(det["lear"][0] - det["rear"][0],
                           det["lear"][1] - det["rear"][1])
        side = max(80.0, ear_w * 2.4)
    else:
        bb = det["bbox"]
        side = max(80.0, (bb[2] - bb[0]) * 0.7)
    side = min(side, 400.0)
    h, w = frame.shape[:2]
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side * 0.55))   # bias upward — face is above nose
    x2 = min(w, int(cx + side / 2))
    y2 = min(h, int(cy + side * 0.55))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None, None
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _cosine(a, b) -> float:
    import numpy as np
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(na, nb))


def _reid_from_gallery(emb) -> tuple[str | None, float]:
    best_name, best_score = None, 0.0
    for name, embs in _gallery.items():
        for e in embs:
            s = _cosine(emb, e)
            if s > best_score:
                best_score = s
                best_name = name
    return (best_name, best_score) if best_score >= _REID_SCORE_MIN else (None, best_score)


def _add_to_gallery(name: str, emb) -> None:
    lst = _gallery.setdefault(name, [])
    lst.append(emb)
    if len(lst) > _GALLERY_PER_NAME:
        lst.pop(0)


def _pipeline_step(frame, source_key: str) -> list[dict]:
    """YOLO-pose + BoT-SORT tracking with per-track InsightFace identity.
    Returns one entry per track with stable head position and resolved name."""
    import time as _time, numpy as np
    if frame is None:
        return []
    # Non-blocking: if YOLO isn't loaded yet return [] immediately so the
    # thread pool doesn't saturate with blocked threads — which stalls the
    # MJPEG stream and the whole UI. YOLO loads in one background warmup
    # thread; pipeline results appear automatically once it's ready.
    if not _yolo_ready.is_set():
        return []
    dets = _run_yolo_pose(frame, source_key, track=True)
    if not dets:
        return []
    # Non-blocking: skip InsightFace face recognition until it finishes loading.
    # YOLO boxes display immediately; names appear once InsightFace is ready.
    fa = _get_face_app() if _face_app_loaded.is_set() else None
    store = _track_state.setdefault(source_key, {})
    now   = _time.time()
    out: list[dict] = []
    seen_ids: set[int] = set()

    for det in dets:
        tid = det.get("track_id")
        if tid is None:
            continue
        seen_ids.add(tid)
        state = store.get(tid) or {
            "votes": [],            # [(name, score, margin, ts)]
            "locked_name": None,
            "locked_score": 0.0,
            "last_face_ts": 0.0,
            "last_seen_ts": now,
            "emb_buf": [],          # recent embeddings (for gallery)
            "grace_until": 0.0,    # timestamp when grace period ends (0 = none)
            "grace_votes": [],     # crops captured during grace for re-evaluation
        }
        state["last_seen_ts"] = now

        # Try to identify from a head crop (only if a face is plausibly visible —
        # nose or both eyes detected). Runs every poll but is cheap on a tiny crop.
        name, recog_score, recog_margin = None, 0.0, 0.0
        _anchor_kp = det.get("nose") or det.get("leye") or det.get("reye")
        _bb_body   = det["bbox"]
        _bw = _bb_body[2] - _bb_body[0]
        _bh = _bb_body[3] - _bb_body[1]
        # Guard: keypoint must lie within (or very near) this body's own bbox.
        # YOLO-pose sometimes cross-assigns a neighbour's nose to this detection;
        # that nose lands outside the body box — skip face recognition entirely.
        _kp_in_body = _anchor_kp is not None and (
            _bb_body[0] - _bw * 0.35 <= float(_anchor_kp[0]) <= _bb_body[2] + _bw * 0.35 and
            _bb_body[1] - _bh * 0.20 <= float(_anchor_kp[1]) <= _bb_body[3] + _bh * 0.10
        )
        if fa is not None and _kp_in_body:
            crop, _bb = _head_crop_from_det(frame, det)
            if crop is not None:
                try:
                    faces = fa.get(crop)
                except Exception:
                    faces = []
                if faces:
                    face = max(faces, key=lambda f: float(f.det_score))
                    # Reject faces that are off-center in the crop — they leaked
                    # in from an adjacent person whose keypoints YOLO misassigned.
                    # The intended face should be near the center of the crop.
                    ch, cw = crop.shape[:2]
                    fb = face.bbox  # [x1,y1,x2,y2] within crop
                    fcx = (fb[0] + fb[2]) / 2
                    fcy = (fb[1] + fb[3]) / 2
                    off_x = abs(fcx - cw / 2) / (cw / 2)  # 0=center, 1=edge
                    off_y = abs(fcy - ch / 2) / (ch / 2)
                    face_is_centered = off_x < 0.55 and off_y < 0.55
                    if float(face.det_score) >= 0.50 and face_is_centered:
                        emb = getattr(face, "normed_embedding", None)
                        nm, sc, mg = _identify_face(face)
                        name, recog_score, recog_margin = nm, float(sc), float(mg)
                        # Only confirm the lock timestamp when the face matches
                        # the locked name. If it returns Unknown or a different
                        # name, the TTL clock keeps ticking — ensures a wrong
                        # lock clears even when a face IS visible in the crop.
                        if nm == state.get("locked_name"):
                            state["last_face_ts"] = now
                        # Vote buffer (rolling window)
                        state["votes"].append((nm, recog_score, recog_margin, now))
                        if len(state["votes"]) > _VOTE_WINDOW:
                            state["votes"] = state["votes"][-_VOTE_WINDOW:]
                        # Accumulate grace re-evaluation crops while in grace period
                        if state.get("grace_until", 0) > now:
                            state.setdefault("grace_votes", []).append(
                                (nm, recog_score, recog_margin, now))
                        # Gallery: add strong embeddings of locked identities
                        if emb is not None and nm != "Unknown" and recog_score >= 0.50:
                            state.setdefault("emb_buf", []).append(emb)
                            if len(state["emb_buf"]) > 4:
                                state["emb_buf"].pop(0)

        # Grace period expiry: evaluate crops collected during the grace window.
        # If enough confirmed the locked name → re-lock with fresh grace.
        # Otherwise → clear the lock.
        if state.get("locked_name") and state.get("grace_until", 0) and now >= state["grace_until"]:
            grace_votes = state.get("grace_votes", [])
            confirmed = [v for v in grace_votes
                         if v[0] == state["locked_name"]
                         and v[1] >= _VOTE_SCORE_MIN and v[2] >= _VOTE_MARGIN_MIN]
            if len(confirmed) >= _GRACE_CONFIRM_MIN:
                best_sc = max(v[1] for v in confirmed)
                state["locked_score"] = best_sc
                # high confidence again → start another grace window
                if best_sc >= _GRACE_SCORE_MIN:
                    state["grace_until"] = now + _GRACE_DURATION
                else:
                    state["grace_until"] = 0.0
            else:
                state["locked_name"]  = None
                state["locked_score"] = 0.0
                state["votes"]        = []
                state["grace_until"]  = 0.0
            state["grace_votes"] = []

        # Lock TTL: clear if no face confirmed this identity within TTL seconds.
        # Skipped during an active grace period — grace handles the decision.
        if (state["locked_name"]
                and not state.get("grace_until", 0)
                and now - state.get("last_face_ts", 0) > _LOCK_FACE_TTL_SEC):
            state["locked_name"]  = None
            state["locked_score"] = 0.0
            state["votes"]        = []

        # Lock logic: pick majority name among strong votes
        if not state["locked_name"]:
            strong = [v for v in state["votes"]
                      if v[0] != "Unknown" and v[1] >= _VOTE_SCORE_MIN and v[2] >= _VOTE_MARGIN_MIN]
            if len(strong) >= _VOTE_MIN:
                counts: dict[str, list[float]] = {}
                for nm, sc, _mg, _t in strong:
                    counts.setdefault(nm, []).append(sc)
                winner, scores = max(counts.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
                if len(scores) >= _VOTE_MIN:
                    state["locked_name"]  = winner
                    state["locked_score"] = sum(scores) / len(scores)
                    state["grace_votes"]  = []
                    if state["locked_score"] >= _GRACE_SCORE_MIN:
                        state["grace_until"] = now + _GRACE_DURATION
                    # seed gallery with best recent embeddings
                    for e in state.get("emb_buf", []):
                        _add_to_gallery(winner, e)

        # Re-ID: if unlocked and gallery has candidates, try cross-track match
        if (not state["locked_name"]) and state.get("emb_buf"):
            cand_name, cand_score = _reid_from_gallery(state["emb_buf"][-1])
            if cand_name:
                state["locked_name"]  = cand_name
                state["locked_score"] = cand_score

        # Track-swap override: if already locked but current face read is a DIFFERENT
        # person with very strong confidence, re-lock. Fixes BoT-SORT track swaps where
        # two close people cause the tracker to reassign a track_id to the wrong body —
        # the old name would otherwise ride the wrong body for the entire track lifetime.
        if (state["locked_name"] and name and name != "Unknown"
                and name != state["locked_name"]
                and recog_score >= _OVERRIDE_SCORE_MIN
                and recog_margin >= _OVERRIDE_MARGIN_MIN):
            state["locked_name"]  = name
            state["locked_score"] = recog_score
            state["votes"]        = [(name, recog_score, recog_margin, now)]
            state["grace_votes"]  = []
            state["grace_until"]  = (now + _GRACE_DURATION
                                     if recog_score >= _GRACE_SCORE_MIN else 0.0)

        store[tid] = state

        # Head square from keypoints, anchored in this priority:
        #   1) nose / eye  (face visible)
        #   2) shoulder midpoint + upward offset (face turned away)
        #   3) top of bbox  (weakest — only if bbox is clearly person-shaped)
        import math
        head = None
        # Head CENTER from keypoints — best available anchor, priority: nose/eye → shoulders → bbox.
        # Box SIZE scales with frame width (FIXED_HEAD_RATIO) — same canvas size in SD and HD.
        _side = det["fw"] * FIXED_HEAD_RATIO   # scales with SD/HD frame width
        anchor = det.get("nose") or det.get("leye") or det.get("reye")
        if anchor is not None:
            cx, cy = float(anchor[0]), float(anchor[1])
            cy -= _side * 0.10   # shift up so nose sits in lower third
        elif det.get("lsh") and det.get("rsh"):
            lsh, rsh = det["lsh"], det["rsh"]
            sh_w = math.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1])
            cx   = (lsh[0] + rsh[0]) / 2
            cy   = (lsh[1] + rsh[1]) / 2 - max(50.0, sh_w * 0.55) * 1.10
        else:
            bb = det["bbox"]
            bw = bb[2] - bb[0]
            bh = bb[3] - bb[1]
            if bh < 1.4 * bw or det.get("score", 0) < 0.65:
                continue
            cx = (bb[0] + bb[2]) / 2
            cy = bb[1] + _side * 0.55
        fw = det["fw"]
        cx += fw * _BOX_OFFSET_X_RATIO
        cy += fw * _BOX_OFFSET_Y_RATIO
        head = {"cx": float(cx), "cy": float(cy), "side": float(_side)}

        out.append({
            "track_id":    tid,
            "bbox":        det["bbox"],
            "score":       det["score"],
            "fw":          det["fw"],
            "fh":          det["fh"],
            "head":        head,
            "name":        state["locked_name"],
            "recog_score": round(state["locked_score"], 3) if state["locked_name"] else None,
            "pending":     name if (name and name != "Unknown" and not state["locked_name"]) else None,
            "pending_score": round(recog_score, 3) if (recog_score and not state["locked_name"]) else None,
        })

    # GC stale tracks
    for tid in list(store.keys()):
        if tid not in seen_ids and (now - store[tid].get("last_seen_ts", now)) > _TRACK_STATE_TTL_SEC:
            del store[tid]

    # ── InsightFace fallback + track-swap conflict resolution ─────────────────
    # 1) If two YOLO tracks claim the same locked name, use InsightFace face
    #    positions to find which track is spatially consistent with the actual
    #    face — reset the lock on the impostor.
    # 2) If InsightFace sees a known person but no YOLO track is locked to them,
    #    emit a synthetic track so the box stays on the face.
    if fa is not None and frame is not None:
        h_f, w_f = frame.shape[:2]
        _side_f = w_f * FIXED_HEAD_RATIO
        if_dets = _detect_faces_in_frame(frame, source_key, emit_log=False)

        # Build a map: name → face center (from IF), for strongly identified faces
        face_centers: dict[str, tuple[float, float]] = {}
        for ifd in if_dets:
            iname = ifd.get("name")
            if not iname or iname == "Unknown":
                continue
            bx1, by1, bx2, by2 = ifd["bbox"]
            face_centers[iname] = ((bx1 + bx2) / 2, (by1 + by2) / 2)

        # Conflict resolution: if multiple YOLO tracks share the same locked name,
        # keep only the one whose head is closest to where IF actually sees that face.
        # The others had their track_id swapped by BoT-SORT — clear their lock.
        from collections import defaultdict
        name_to_tracks: dict[str, list[int]] = defaultdict(list)
        for entry in out:
            if entry.get("name"):
                name_to_tracks[entry["name"]].append(entry["track_id"])

        reset_tids: set[int] = set()
        for locked_name, tids in name_to_tracks.items():
            if len(tids) < 2:
                continue
            fc = face_centers.get(locked_name)
            if fc is None:
                continue  # IF didn't see this face clearly — can't resolve
            fx, fy = fc
            best_tid, best_dist = None, float("inf")
            for entry in out:
                if entry["track_id"] not in tids:
                    continue
                hd = entry.get("head") or {}
                dist = ((hd.get("cx", 0) - fx) ** 2 + (hd.get("cy", 0) - fy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_tid = dist, entry["track_id"]
            for tid in tids:
                if tid != best_tid:
                    reset_tids.add(tid)

        # Single-track conflict: locked name but IF places that face far away.
        # Threshold: if head center is more than 1.5× head-side from the IF face, bust the lock.
        _SWAP_DIST_THRESH = 1.5
        for entry in out:
            tid = entry["track_id"]
            if tid in reset_tids:
                continue
            lname = entry.get("name")
            if not lname or lname not in face_centers:
                continue
            fx, fy = face_centers[lname]
            hd = entry.get("head") or {}
            side = hd.get("side", _side_f)
            dist = ((hd.get("cx", 0) - fx) ** 2 + (hd.get("cy", 0) - fy) ** 2) ** 0.5
            if dist > side * _SWAP_DIST_THRESH:
                reset_tids.add(tid)

        # Apply resets to track state and output list
        for tid in reset_tids:
            if tid in store:
                store[tid]["locked_name"]  = None
                store[tid]["locked_score"] = 0.0
                store[tid]["votes"]        = []
        for entry in out:
            if entry["track_id"] in reset_tids:
                entry["name"]        = None
                entry["recog_score"] = None

        yolo_locked = {entry["name"] for entry in out if entry.get("name")}
        for ifd in if_dets:
            iname = ifd.get("name")
            if not iname or iname == "Unknown" or iname in yolo_locked:
                continue
            bx1, by1, bx2, by2 = ifd["bbox"]
            fcx = (bx1 + bx2) / 2
            fcy = (by1 + by2) / 2 - _side_f * 0.10
            # Stable negative ID per name — never collides with positive BoT-SORT IDs
            fake_id = -(abs(hash(iname)) % 9000 + 1000)
            out.append({
                "track_id":    fake_id,
                "bbox":        ifd["bbox"],
                "score":       round(float(ifd["score"]), 3),
                "fw":          w_f,
                "fh":          h_f,
                "head":        {"cx": float(fcx), "cy": float(fcy), "side": float(_side_f)},
                "name":        iname,
                "recog_score": ifd.get("recog_score"),
                "pending":     None,
            })

    # Deduplicate: if the same name appears on multiple tracks (e.g. one YOLO
    # track + one InsightFace synthetic), keep only the entry with the highest
    # recog_score. Never show two boxes with the same name simultaneously.
    seen_names: dict[str, int] = {}   # name → index in out of best entry so far
    dedup_out: list[dict] = []
    for entry in out:
        name = entry.get("name")
        if not name:
            dedup_out.append(entry)
            continue
        score = entry.get("recog_score") or 0.0
        if name not in seen_names:
            seen_names[name] = len(dedup_out)
            dedup_out.append(entry)
        else:
            existing_idx = seen_names[name]
            existing_score = dedup_out[existing_idx].get("recog_score") or 0.0
            if score > existing_score:
                seen_names[name] = len(dedup_out)
                dedup_out[existing_idx] = None   # mark old as removed
                dedup_out.append(entry)
    out = [e for e in dedup_out if e is not None]

    return out


@app.get("/pipeline/{cam}")
async def pipeline_cam(cam: str):
    if cam not in CAMS:
        raise HTTPException(404, "unknown camera")
    def _run() -> list[dict]:
        frame = _get_latest_frame(cam)
        return _pipeline_step(frame, cam)
    result = await asyncio.to_thread(_run)
    names = frozenset(t["name"] for t in result if t.get("name"))
    prev  = _cam_visible_names.get(cam)
    if names != prev:
        _cam_visible_names[cam] = names
        if names:
            score_map = {t["name"]: t.get("recog_score") or 0.0 for t in result if t.get("name")}
            parts = [f"{n} ({round(score_map[n]*100)}%)" for n in sorted(names)]
            _write_monitor_log(f"DETECT | {cam}: {', '.join(parts)}")
    return {"tracks": result}


@app.post("/pipeline/frame")
async def pipeline_frame(frame: UploadFile = File(...)):
    """Run the pipeline on a single JPEG frame from a browser webcam."""
    import numpy as np, cv2
    raw = await frame.read()
    if not raw:
        return {"tracks": []}
    def _run() -> list[dict]:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        return _pipeline_step(img, "webcam")
    result = await asyncio.to_thread(_run)
    return {"tracks": result}


@app.get("/pipeline/upload/{upload_id}")
async def pipeline_upload(upload_id: str, t: float = 0.0):
    meta = _get_uploaded_video(upload_id)
    if not meta:
        raise HTTPException(404, "upload not found")
    def _run() -> list[dict]:
        frame = _get_video_frame_at(Path(str(meta["path"])), t)
        return _pipeline_step(frame, f"upload:{upload_id}")
    result = await asyncio.to_thread(_run)
    return {"tracks": result}


@app.get("/detections/{cam}")
async def detections(cam: str):
    """Run InsightFace on the latest RTSP frame for this camera."""
    if cam not in CAMS:
        raise HTTPException(404, "unknown camera")

    def _run() -> list[dict]:
        frame = _get_latest_frame(cam)
        return _detect_faces_in_frame(frame, cam, emit_log=True)

    result = await asyncio.to_thread(_run)
    return {"detections": result}


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """Store a test-history clip so the existing detection flow can inspect it."""
    if not video.filename:
        raise HTTPException(400, "missing filename")

    def _run() -> dict[str, str | float]:
        return _store_uploaded_video(video.file, video.filename)

    try:
        meta = await asyncio.to_thread(_run)
    finally:
        await video.close()

    return {
        "id": meta["id"],
        "name": meta["name"],
    }


@app.get("/detections/upload/{upload_id}")
async def upload_detections(upload_id: str, t: float = 0.0):
    """Run the same InsightFace pipeline against a frame from an uploaded video."""
    meta = _get_uploaded_video(upload_id)
    if not meta:
        raise HTTPException(404, "upload not found")

    def _run() -> list[dict]:
        frame = _get_video_frame_at(Path(str(meta["path"])), t)
        return _detect_faces_in_frame(frame, f"upload:{upload_id}", emit_log=False)

    result = await asyncio.to_thread(_run)
    return {"detections": result}


def _identity_colors_js() -> str:
    """Build the IDENTITY_COLORS JS object body from config."""
    import json
    cfg = _load_config()
    colors = cfg.get("identity_colors", {})
    lines = [f"  {json.dumps(name.lower())}: {json.dumps(color)}," for name, color in colors.items()]
    lines.append("  _default:'#FFD700'")
    return "\n" + "\n".join(lines) + "\n"


def _cams_js() -> str:
    """Inject CAMS as a JS object: {name: {ip, label}}"""
    import json
    entries = [f"  {json.dumps(k)}: {{ip: {json.dumps(v['ip'])}, label: {json.dumps(v['label'])}}}"
               for k, v in CAMS.items()]
    return "{\n" + ",\n".join(entries) + "\n}"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (
        HTML
        .replace("__CAMS_JS__", _cams_js())
        .replace("__ACTIVE_CAM__", _active_cam)
        .replace("__IDENTITY_COLORS__", _identity_colors_js())
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Secure Monitor</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #07070f;
    --surface:   #0c0c1a;
    --surface2:  #11111f;
    --surface3:  #16162a;
    --border:    #1c1c38;
    --border2:   #2a2a50;
    --accent:    #00c8ff;
    --accent2:   #7b2fff;
    --live:      #ff3b3b;
    --offline:   #ff6b35;
    --connecting:#f0b429;
    --text:      #e2e8f0;
    --muted:     #4a5568;
    --success:   #10b981;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; overflow-x: hidden; }

  /* ── Header ── */
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
  }

  .logo { display: flex; align-items: center; gap: 10px; }
  .logo svg { width: 28px; height: 28px; }
  .logo-text {
    font-size: 15px; font-weight: 700; letter-spacing: 3px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }

  .header-center { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  #clock { font-size: 22px; font-weight: 300; letter-spacing: 4px; color: var(--text); font-variant-numeric: tabular-nums; }
  .date-str { font-size: 11px; color: var(--muted); letter-spacing: 2px; }

  .sys-status {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; letter-spacing: 2px; color: var(--muted);
  }
  .sys-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: pulse-sys 2s ease-in-out infinite;
  }
  @keyframes pulse-sys { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  /* ── Toolbar with camera switcher + HD toggle ── */
  .toolbar {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 14px;
    padding: 18px 28px 0;
    flex-wrap: wrap;
  }

  .cam-tabs {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
  }

  /* HD toggle button */
  .mode-btn {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--muted);
    font-family: inherit;
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    cursor: pointer;
    transition: all 0.25s;
  }
  .mode-btn:hover:not(.active) {
    background: var(--surface2);
    color: var(--text);
    border-color: var(--border2);
  }
  .mode-btn.active {
    background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(123,47,255,0.2));
    color: var(--accent);
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(0,200,255,0.15);
  }
  .mode-btn-icon {
    width: 14px; height: 14px;
    display: inline-block;
    opacity: 0.7;
  }
  .mode-btn.active .mode-btn-icon { opacity: 1; }

  .cam-dropdown {
    position: relative;
    min-width: 280px;
    user-select: none;
  }
  .cam-dropdown-btn {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(123,47,255,0.08));
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 12px 16px 12px 20px;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
  }
  .cam-dropdown-btn:hover, .cam-dropdown.open .cam-dropdown-btn {
    background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(123,47,255,0.15));
    border-color: var(--accent);
    box-shadow: 0 0 16px rgba(0,200,255,0.15);
  }
  .cam-dropdown-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
    flex-shrink: 0;
  }
  .cam-dropdown-label {
    flex: 1;
    font-size: 13px; font-weight: 700; letter-spacing: 2px;
    color: var(--text);
  }
  .cam-dropdown-arrow {
    font-size: 10px; color: var(--muted);
    transition: transform 0.2s;
    flex-shrink: 0;
  }
  .cam-dropdown.open .cam-dropdown-arrow { transform: rotate(180deg); }
  .cam-dropdown-list {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    left: 0; right: 0;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 12px;
    overflow: hidden;
    z-index: 999;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  .cam-dropdown.open .cam-dropdown-list { display: block; }
  .cam-dropdown-item {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 20px;
    cursor: pointer;
    font-size: 13px; font-weight: 600; letter-spacing: 1px;
    color: var(--muted);
    transition: all 0.15s;
    border-bottom: 1px solid var(--border);
  }
  .cam-dropdown-item:last-child { border-bottom: none; }
  .cam-dropdown-item:hover {
    background: rgba(0,200,255,0.08);
    color: var(--text);
  }
  .cam-dropdown-item.active {
    color: var(--accent);
    background: rgba(0,200,255,0.06);
  }
  .cam-dropdown-item-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: all 0.15s;
  }
  .cam-dropdown-item.active .cam-dropdown-item-dot {
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
  }
  .cam-dropdown-item-ip {
    font-size: 11px; font-weight: 400; letter-spacing: 1px;
    color: var(--muted); font-family: monospace;
    margin-left: auto;
  }

  /* ── Main stage ── */
  main {
    padding: 20px 28px 28px;
    display: flex;
    justify-content: center;
  }

  .stage {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
    max-width: 100%;
  }
  .stage.live {
    border-color: rgba(0,200,255,0.25);
    box-shadow: 0 0 40px rgba(0,200,255,0.05), 0 8px 32px rgba(0,0,0,0.4);
  }
  .stage.offline    { border-color: rgba(255,107,53,0.3); }
  .stage.connecting { border-color: rgba(240,180,41,0.3); }

  /* info bar */
  .info-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }
  .info-left { display: flex; align-items: center; gap: 18px; }
  .info-name { font-size: 14px; font-weight: 700; letter-spacing: 2px; }
  .info-ip   { font-size: 11px; color: var(--muted); font-family: monospace; letter-spacing: 1px; }

  .live-badge {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
  }
  .live-badge.live       { background: rgba(255,59,59,0.15);  color: var(--live);       border: 1px solid rgba(255,59,59,0.4); }
  .live-badge.offline    { background: rgba(255,107,53,0.15); color: var(--offline);    border: 1px solid rgba(255,107,53,0.4); }
  .live-badge.connecting { background: rgba(240,180,41,0.15); color: var(--connecting); border: 1px solid rgba(240,180,41,0.4); }

  .pulse-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .live-badge.live .pulse-dot { animation: blink 1.2s ease-in-out infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

  /* video area — sized by JS to match active video's natural aspect ratio */
  .video-wrap {
    position: relative;
    background: #000;
    overflow: hidden;
    width: 960px;
    height: 540px;
    max-width: 100%;
  }
  .cam-video {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    display: block;
  }

  /* scanline overlay */
  .video-wrap::after {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 5;
  }

  /* loading overlay */
  .loading-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 18px;
    background: rgba(7,7,15,0.85);
    backdrop-filter: blur(4px);
    transition: opacity 0.4s;
    z-index: 10;
  }
  .loading-overlay.hidden { opacity: 0; pointer-events: none; }

  .spinner {
    width: 48px; height: 48px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-label { font-size: 13px; letter-spacing: 4px; color: var(--muted); font-weight: 600; }
  .loading-sub   { font-size: 11px; letter-spacing: 2px; color: var(--muted); opacity: 0.7; }

  /* footer bar */
  .stage-footer {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px;
    background: var(--surface2);
    border-top: 1px solid var(--border);
  }
  .status-text { font-size: 11px; letter-spacing: 1px; color: var(--muted); }
  .status-text.live       { color: var(--success); }
  .status-text.offline    { color: var(--offline); }
  .status-text.connecting { color: var(--connecting); }

  .footer-right { display: flex; gap: 12px; align-items: center; }
  .tag {
    font-size: 10px; letter-spacing: 1px; color: var(--muted);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 3px 8px;
  }

  footer { text-align: center; padding: 20px; font-size: 11px; color: var(--muted); letter-spacing: 1px; border-top: 1px solid var(--border); }

  /* Detection overlay canvas — above scanlines (z5), below loading overlay (z10) */
  #detect-canvas {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 6;
  }

  @media (max-width: 700px) {
    header { padding: 10px 14px; }
    .logo-text { font-size: 12px; letter-spacing: 2px; }
    #clock { font-size: 16px; }
    .toolbar { padding: 12px; }
    .cam-dropdown { min-width: 200px; }
    .cam-dropdown-label { font-size: 11px; }
    main { padding: 14px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="14" cy="14" r="13" stroke="url(#g1)" stroke-width="1.5"/>
      <circle cx="14" cy="14" r="6" fill="url(#g2)"/>
      <circle cx="14" cy="14" r="2.5" fill="#fff" opacity="0.9"/>
      <line x1="14" y1="1" x2="14" y2="5" stroke="url(#g1)" stroke-width="1.5" stroke-linecap="round"/>
      <line x1="14" y1="23" x2="14" y2="27" stroke="url(#g1)" stroke-width="1.5" stroke-linecap="round"/>
      <line x1="1" y1="14" x2="5" y2="14" stroke="url(#g1)" stroke-width="1.5" stroke-linecap="round"/>
      <line x1="23" y1="14" x2="27" y2="14" stroke="url(#g1)" stroke-width="1.5" stroke-linecap="round"/>
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop stop-color="#00c8ff"/><stop offset="1" stop-color="#7b2fff"/>
        </linearGradient>
        <linearGradient id="g2" x1="8" y1="8" x2="20" y2="20" gradientUnits="userSpaceOnUse">
          <stop stop-color="#00c8ff" stop-opacity="0.3"/><stop offset="1" stop-color="#7b2fff" stop-opacity="0.3"/>
        </linearGradient>
      </defs>
    </svg>
    <span class="logo-text">SECURE MONITOR</span>
  </div>

  <div class="header-center">
    <div id="clock">00:00:00</div>
    <div class="date-str" id="date-str"></div>
  </div>

  <div class="sys-status">
    <span class="sys-dot"></span>
    <span>SYSTEM ONLINE</span>
  </div>
</header>

<div class="toolbar">
  <div class="cam-dropdown" id="cam-dropdown">
    <div class="cam-dropdown-btn" id="cam-dropdown-btn">
      <span class="cam-dropdown-dot"></span>
      <span class="cam-dropdown-label" id="cam-dropdown-label">—</span>
      <span class="cam-dropdown-arrow">▼</span>
    </div>
    <div class="cam-dropdown-list" id="cam-dropdown-list"></div>
  </div>

  <button class="mode-btn" id="detect-btn" title="Toggle AI detection — highlights persons and cats">
    <svg class="mode-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 8V5h3M16 3h3v3M8 21H5v-3M21 16v3h-3"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
    <span id="detect-label">DETECTION · ON</span>
  </button>

  <button class="mode-btn" id="webcam-btn" title="Use this computer's built-in webcam — zero delay, ideal for testing detection">
    <svg class="mode-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="6" width="14" height="12" rx="2"/>
      <path d="m22 8-6 4 6 4V8z"/>
    </svg>
    <span id="webcam-label">WEBCAM</span>
  </button>
  <select id="webcam-picker" style="display:none;background:#0a0a0a;color:#0ff;border:1px solid #0ff;padding:4px 8px;font-family:inherit;font-size:11px;"></select>

  <button class="mode-btn" id="history-btn" title="Load a recorded camera clip for AI testing">
    <svg class="mode-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M14 3v5a1 1 0 0 0 1 1h5"/>
      <path d="M6 8V6a2 2 0 0 1 2-2h6l6 6v8a2 2 0 0 1-2 2h-3"/>
      <rect x="3" y="11" width="10" height="8" rx="2"/>
      <path d="m8 15 3-2v4Z"/>
    </svg>
    <span id="history-label">TEST VIDEO</span>
  </button>



<button class="mode-btn" id="mode-btn" title="Toggle between standard 720p and full-resolution HD">
    <svg class="mode-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
    <span id="mode-label">TRY HD MODE</span>
  </button>

  <button class="mode-btn" id="fs-btn" title="Toggle fullscreen">
    <svg class="mode-btn-icon" id="fs-icon-expand" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
      <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
    </svg>
    <svg class="mode-btn-icon" id="fs-icon-compress" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:none">
      <polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>
      <line x1="10" y1="14" x2="3" y2="21"/><line x1="21" y1="3" x2="14" y2="10"/>
    </svg>
    <span id="fs-label">FULLSCREEN</span>
  </button>

  <button class="mode-btn" id="live-btn" title="Return to the live camera streams" style="display:none;">
    <svg class="mode-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="m15 18-6-6 6-6"/>
      <path d="M21 12H9"/>
    </svg>
    <span id="live-label">BACK TO LIVE</span>
  </button>
</div>

<input type="file" id="history-file" accept="video/*" style="display:none;">

<main>
  <div class="stage connecting" id="stage">
    <div class="info-bar">
      <div class="info-left">
        <span class="info-name" id="info-name">CAMERA 01</span>
        <span class="info-ip"   id="info-ip">__CAM1_IP__</span>
      </div>
      <div class="live-badge connecting" id="badge">
        <span class="pulse-dot"></span>
        <span id="badge-text">CONNECTING</span>
      </div>
    </div>

    <div class="video-wrap" id="video-wrap">
      <!-- cam video elements injected by populateCamDropdown() -->
      <video id="video-upload" class="cam-video" muted playsinline controls preload="metadata" style="display:none;"></video>
      <video id="video-webcam" class="cam-video" muted playsinline autoplay style="display:none;"></video>
      <canvas id="detect-canvas"></canvas>
      <div class="loading-overlay" id="loading">
        <div class="spinner"></div>
        <span class="loading-label" id="loading-label">CONNECTING</span>
        <span class="loading-sub"   id="loading-sub">buffering stream...</span>
      </div>
    </div>

    <div class="stage-footer">
      <span class="status-text connecting" id="status-text">Waiting for stream...</span>
      <div class="footer-right">
        <span class="tag">RTSP</span>
        <span class="tag" id="tag-quality">720P</span>
        <span class="tag" id="tag-latency">~ 1s</span>
      </div>
    </div>
  </div>
</main>

<footer>SECURE MONITOR &nbsp;·&nbsp; RTSP DUAL CAM &nbsp;·&nbsp; RTSP · MJPEG</footer>

<script>
// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toTimeString().split(' ')[0];
  document.getElementById('date-str').textContent =
    now.toLocaleDateString('en-GB', {weekday:'long', year:'numeric', month:'long', day:'numeric'}).toUpperCase();
}
updateClock(); setInterval(updateClock, 1000);

// ── State ────────────────────────────────────────────────────────────────────
const CAM_META = __CAMS_JS__;
const CAM_KEYS = Object.keys(CAM_META);

let currentCam       = '__ACTIVE_CAM__';
let currentMode      = 'standard';
let playbackMode     = 'live';
let uploadSession    = null;
let uploadObjectUrl  = null;
const camMode        = {};  // backend-known mode per cam — populated on load
const players        = {};   // cam -> { hls, video }
let pendingModeCam   = null; // cam currently mid mode-switch (owns the overlay)
let statePollTimer   = null;
const READY_POLL_MS  = 500;

function videoEl(cam) { return document.getElementById('video-' + cam); }
function uploadVideoEl() { return document.getElementById('video-upload'); }
function webcamVideoEl() { return document.getElementById('video-webcam'); }
function activeVideoEl() {
  if (playbackMode === 'upload') return uploadVideoEl();
  if (playbackMode === 'webcam') return webcamVideoEl();
  return videoEl(currentCam);
}
function sleep(ms)    { return new Promise(resolve => setTimeout(resolve, ms)); }

// Cached canvas refs — looked up once; the element never changes after DOM load
const _detectCanvas = document.getElementById('detect-canvas');
const _detectCtx    = _detectCanvas ? _detectCanvas.getContext('2d') : null;
const MOTION_W      = 96;
const MOTION_H      = 54;
const _motionCanvas = document.createElement('canvas');
_motionCanvas.width = MOTION_W;
_motionCanvas.height = MOTION_H;
const _motionCtx    = _motionCanvas.getContext('2d', { willReadFrequently: true });
let _motionPrevGray = null;
let _motionSource   = null;

function _currentMotionSourceKey() {
  return playbackMode === 'upload'
    ? (uploadSession ? 'upload:' + uploadSession.id : 'upload')
    : currentCam;
}

function _resetMotionState() {
  _motionPrevGray = null;
  _motionSource   = null;
}

function _sampleMotion(video, sourceKey) {
  if (!_motionCtx || !video || !(video.videoWidth || video.naturalWidth) || !(video.videoHeight || video.naturalHeight)) return null;
  if (_motionSource !== sourceKey) {
    _motionPrevGray = null;
    _motionSource = sourceKey;
  }

  _motionCtx.drawImage(video, 0, 0, MOTION_W, MOTION_H);
  const data = _motionCtx.getImageData(0, 0, MOTION_W, MOTION_H).data;
  const gray = new Uint8Array(MOTION_W * MOTION_H);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = (data[i] * 38 + data[i + 1] * 75 + data[i + 2] * 15) >> 7;
  }

  if (!_motionPrevGray) {
    _motionPrevGray = gray;
    return null;
  }

  const mask = new Uint8Array(gray.length);
  let changed = 0;
  for (let i = 0; i < gray.length; i++) {
    if (Math.abs(gray[i] - _motionPrevGray[i]) >= MOTION_PIXEL_DELTA) {
      mask[i] = 1;
      changed += 1;
    }
  }
  _motionPrevGray = gray;
  return { mask, w: MOTION_W, h: MOTION_H, changed };
}

function _motionRatioForBBox(motion, bbox, video) {
  if (!motion?.mask || !video || !(video.videoWidth || video.naturalWidth) || !(video.videoHeight || video.naturalHeight)) return 0;
  const [bx, by, bw, bh] = bbox;
  const x1 = Math.max(0, Math.floor((bx / (video.videoWidth || video.naturalWidth)) * motion.w));
  const y1 = Math.max(0, Math.floor((by / (video.videoHeight || video.naturalHeight)) * motion.h));
  const x2 = Math.min(motion.w, Math.ceil(((bx + bw) / (video.videoWidth || video.naturalWidth)) * motion.w));
  const y2 = Math.min(motion.h, Math.ceil(((by + bh) / (video.videoHeight || video.naturalHeight)) * motion.h));
  if (x2 <= x1 || y2 <= y1) return 0;
  let area = 0, changed = 0;
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      area += 1;
      changed += motion.mask[y * motion.w + x];
    }
  }
  return area ? (changed / area) : 0;
}

function _extractMotionBlobs(motion) {
  if (!motion?.mask) return [];
  if (motion.blobs) return motion.blobs;

  const labels = new Int16Array(motion.mask.length);
  const blobs = [];
  let blobId = 0;

  for (let i = 0; i < motion.mask.length; i++) {
    if (!motion.mask[i] || labels[i]) continue;
    const stack = [i];
    const pixels = [];
    blobId += 1;
    labels[i] = blobId;
    let x1 = motion.w, y1 = motion.h, x2 = 0, y2 = 0;
    while (stack.length) {
      const idx = stack.pop();
      pixels.push(idx);
      const x = idx % motion.w;
      const y = (idx / motion.w) | 0;
      if (x < x1) x1 = x;
      if (y < y1) y1 = y;
      if (x + 1 > x2) x2 = x + 1;
      if (y + 1 > y2) y2 = y + 1;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= motion.w || ny >= motion.h) continue;
          const nidx = ny * motion.w + nx;
          if (!motion.mask[nidx] || labels[nidx]) continue;
          labels[nidx] = blobId;
          stack.push(nidx);
        }
      }
    }
    if (pixels.length < 4) {
      pixels.forEach(idx => { labels[idx] = 0; });
      blobId -= 1;
      continue;
    }
    blobs.push({ id: blobId, x1, y1, x2, y2, pixels: pixels.length });
  }

  motion.labels = labels;
  motion.blobs = blobs;
  return blobs;
}

function _motionInfoForBBox(motion, bbox, video) {
  if (!motion?.mask || !video || !(video.videoWidth || video.naturalWidth) || !(video.videoHeight || video.naturalHeight))
    return { ratio: 0, blobId: null, blobRatio: 0, blob: null };

  _extractMotionBlobs(motion);
  const [bx, by, bw, bh] = bbox;
  const x1 = Math.max(0, Math.floor((bx / (video.videoWidth || video.naturalWidth)) * motion.w));
  const y1 = Math.max(0, Math.floor((by / (video.videoHeight || video.naturalHeight)) * motion.h));
  const x2 = Math.min(motion.w, Math.ceil(((bx + bw) / (video.videoWidth || video.naturalWidth)) * motion.w));
  const y2 = Math.min(motion.h, Math.ceil(((by + bh) / (video.videoHeight || video.naturalHeight)) * motion.h));
  if (x2 <= x1 || y2 <= y1) return { ratio: 0, blobId: null, blobRatio: 0, blob: null };

  const counts = new Map();
  let area = 0;
  let changed = 0;
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      area += 1;
      const id = motion.labels[y * motion.w + x];
      if (!id) continue;
      changed += 1;
      counts.set(id, (counts.get(id) || 0) + 1);
    }
  }
  if (!area || !changed) return { ratio: 0, blobId: null, blobRatio: 0, blob: null };

  let bestBlobId = null;
  let bestCount = 0;
  counts.forEach((count, id) => {
    if (count > bestCount) {
      bestCount = count;
      bestBlobId = id;
    }
  });

  const blob = motion.blobs.find(b => b.id === bestBlobId) || null;
  return {
    ratio: changed / area,
    blobId: bestBlobId,
    blobRatio: bestCount / area,
    blob,
  };
}

// ── Fit video container to active video's natural aspect ratio ───────────────
const CHROME_HEIGHT = 280;
const PAGE_MARGIN   = 56;

function fitStage() {
  const video = activeVideoEl();
  const wrap  = document.getElementById('video-wrap');
  const stage = document.getElementById('stage');
  if (!video || !wrap || !stage) return;

  const vw = (video.videoWidth || video.naturalWidth)  || 1280;
  const vh = (video.videoHeight || video.naturalHeight) ||  720;
  const ratio = vw / vh;

  const isFs = !!document.fullscreenElement;
  const maxH = isFs ? window.innerHeight : window.innerHeight - CHROME_HEIGHT;
  const maxW = isFs ? window.innerWidth  : Math.min(window.innerWidth - PAGE_MARGIN, 1800);

  let w = maxW, h = maxW / ratio;
  if (h > maxH) { h = maxH; w = maxH * ratio; }
  w = Math.floor(w); h = Math.floor(h);
  wrap.style.width  = w + 'px';
  wrap.style.height = h + 'px';
  stage.style.width = w + 'px';
  // keep detect canvas pixel dimensions in sync
  if (_detectCanvas) { _detectCanvas.width = w; _detectCanvas.height = h; }
}

fitStage();
window.addEventListener('resize', fitStage);
[...CAM_KEYS.map(videoEl), uploadVideoEl()].forEach(v => {
  if (v) v.addEventListener(v.tagName === 'IMG' ? 'load' : 'loadedmetadata', () => {
    const isCurrentLive = CAM_KEYS.includes(v.id.replace('video-', '')) && v === videoEl(currentCam);
    if (playbackMode === 'upload' || isCurrentLive) fitStage();
  });
});

// ── Loading overlay ──────────────────────────────────────────────────────────
function showLoading(label, sub) {
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('loading-label').textContent = label || 'CONNECTING';
  document.getElementById('loading-sub').textContent   = sub   || 'buffering stream...';
}
function hideLoading() {
  document.getElementById('loading').classList.add('hidden');
}

async function fetchStateSnapshot() {
  try {
    const r = await fetch('/state', { cache: 'no-store' });
    return await r.json();
  } catch (_) {
    return null;
  }
}

async function waitForCamReady(cam, mode, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const d = await fetchStateSnapshot();
    const c = d && d.cameras && d.cameras[cam];
    if (c && c.mode === mode && c.status === 'live') return true;
    await sleep(READY_POLL_MS);
  }
  return false;
}

// ── UI helpers ───────────────────────────────────────────────────────────────
function updateTabs(cam) {
  const label = document.getElementById('cam-dropdown-label');
  const meta  = CAM_META[cam];
  if (label && meta) {
    label.textContent = meta.label + (meta.ip ? '  ·  ' + meta.ip : '');
  }
  document.querySelectorAll('.cam-dropdown-item').forEach(el => {
    el.classList.toggle('active', el.dataset.cam === cam);
  });
}

function populateCamDropdown() {
  const list = document.getElementById('cam-dropdown-list');
  const btn  = document.getElementById('cam-dropdown-btn');
  const dd   = document.getElementById('cam-dropdown');
  if (!list || !btn || !dd) return;

  // Build list items
  CAM_KEYS.forEach(k => {
    const meta = CAM_META[k];
    camMode[k] = camMode[k] || 'standard';
    const item = document.createElement('div');
    item.className = 'cam-dropdown-item' + (k === currentCam ? ' active' : '');
    item.dataset.cam = k;
    const dot = document.createElement('span');
    dot.className = 'cam-dropdown-item-dot';
    const name = document.createElement('span');
    name.textContent = meta.label;
    const ip = document.createElement('span');
    ip.className = 'cam-dropdown-item-ip';
    ip.textContent = meta.ip || '';
    item.appendChild(dot); item.appendChild(name); item.appendChild(ip);
    item.addEventListener('click', () => {
      if (playbackMode === 'upload') return;
      dd.classList.remove('open');
      switchTo(k);
    });
    list.appendChild(item);
  });

  // Toggle open/close on button click
  btn.addEventListener('click', () => {
    if (playbackMode === 'upload') return;
    dd.classList.toggle('open');
  });

  // Close on outside click
  document.addEventListener('click', e => {
    if (!dd.contains(e.target)) dd.classList.remove('open');
  });

  // Set initial label
  updateTabs(currentCam);

  // Inject <img> video elements for each camera into video-wrap
  const wrap = document.getElementById('video-wrap');
  if (wrap) {
    CAM_KEYS.forEach((k, i) => {
      const img = document.createElement('img');
      img.id = 'video-' + k;
      img.className = 'cam-video';
      img.style.cssText = 'object-fit:contain;' + (i === 0 ? '' : 'display:none;');
      wrap.insertBefore(img, wrap.firstChild);
    });
  }
}

function updateCameraInfo(cam) {
  if (playbackMode === 'upload' && uploadSession) {
    document.getElementById('info-name').textContent = 'TEST VIDEO';
    document.getElementById('info-ip').textContent   = uploadSession.name || 'local upload';
    return;
  }
  const meta = CAM_META[cam];
  if (!meta) return;
  document.getElementById('info-name').textContent = meta.name;
  document.getElementById('info-ip').textContent   = meta.ip;
}

function applyStatus(status) {
  const stage = document.getElementById('stage');
  const badge = document.getElementById('badge');
  const btext = document.getElementById('badge-text');
  const stEl  = document.getElementById('status-text');
  const statusClass = status === 'playback' ? 'live' : status;

  ['live','offline','connecting'].forEach(c => {
    stage.classList.remove(c); badge.classList.remove(c); stEl.classList.remove(c);
  });
  stage.classList.add(statusClass);
  badge.classList.add(statusClass);
  stEl.classList.add(statusClass);

  const BADGE = { live:'LIVE', offline:'OFFLINE', connecting:'CONNECTING', playback:'PLAYBACK' };
  const TEXT  = { live:'Stream active', offline:'Camera offline', connecting:'Waiting for stream...', playback:'Recorded video active' };
  btext.textContent = BADGE[status] || status.toUpperCase();
  stEl.textContent  = TEXT[status]  || status;

  // Overlay: a mode switch on the currently visible cam owns the overlay UI
  // (showing its own custom label). Otherwise, reflect live/connecting/offline.
  if (playbackMode !== 'upload' && pendingModeCam && pendingModeCam === currentCam) return;
  if (status === 'live' || status === 'playback') hideLoading();
  else showLoading(
    status.toUpperCase(),
    status === 'offline' ? 'camera offline' : 'buffering stream...'
  );
}

function updateModeButton() {
  const btn   = document.getElementById('mode-btn');
  const label = document.getElementById('mode-label');
  const tag   = document.getElementById('tag-quality');
  const latency = document.getElementById('tag-latency');
  if (playbackMode === 'upload') {
    btn.disabled = true;
    btn.classList.remove('active');
    label.textContent = 'LIVE ONLY';
    tag.textContent = 'FILE';
    latency.textContent = 'LOCAL';
    return;
  }
  btn.disabled = false;
  latency.textContent = '~ 1s';
  if (currentMode === 'hd') {
    btn.classList.add('active');
    label.textContent = 'HD MODE · ACTIVE';
    tag.textContent = 'FULL HD';
  } else {
    btn.classList.remove('active');
    label.textContent = 'SWITCH TO HD';
    tag.textContent = '720P';
  }
}

// ── Per-cam MJPEG player ─────────────────────────────────────────────────────

function destroyPlayer(cam) {
  const p = players[cam];
  if (p && p.hls) { try { p.hls.destroy(); } catch (_){} }
  delete players[cam];
  const v = videoEl(cam);
  if (v) {
    if (v.tagName === 'VIDEO') {
      try { v.pause(); } catch (_){} 
      v.removeAttribute('src'); v.load();
    } else {
      v.removeAttribute('src');
    }
  }
}

function initPlayerForCam(cam) {
  const img = videoEl(cam);
  if (!img) return;
  destroyPlayer(cam);
  img.src = '/mjpeg/' + cam + '?t=' + Date.now();
  players[cam] = { hls: null, video: img };
}


// ── Visible-cam management ───────────────────────────────────────────────────
function showCam(cam) {
  if (playbackMode !== 'live') fetch('/source/rtsp', { method: 'POST' }).catch(() => {});
  playbackMode = 'live';
  CAM_KEYS.forEach(c => {
    const v = videoEl(c);
    if (!v) return;
    v.style.display = (c === cam ? 'block' : 'none');
  });
  const uploadVideo = uploadVideoEl();
  if (uploadVideo) uploadVideo.style.display = 'none';
  currentCam  = cam;
  currentMode = camMode[cam] || 'standard';
  updateTabs(cam);
  updateCameraInfo(cam);
  updateModeButton();
  _applyDetectionForMode(currentMode);
  document.getElementById('history-label').textContent = 'TEST VIDEO';
  document.getElementById('live-btn').style.display = 'none';
  fitStage();
}

async function enterUploadMode(file) {
  if (!file) return;
  const form = new FormData();
  form.append('video', file);

  showLoading('UPLOADING VIDEO', 'sending test clip to backend...');
  const r = await fetch('/upload-video', { method: 'POST', body: form });
  if (!r.ok) throw new Error('upload failed');
  const meta = await r.json();

  const uploadVideo = uploadVideoEl();
  if (!uploadVideo) return;

  clearDetectCanvas();
  if (uploadObjectUrl) URL.revokeObjectURL(uploadObjectUrl);
  uploadObjectUrl = URL.createObjectURL(file);
  uploadSession = { id: meta.id, name: meta.name || file.name };
  playbackMode = 'upload';
  fetch('/source/upload', { method: 'POST' }).catch(() => {});

  CAM_KEYS.forEach(c => {
    const v = videoEl(c);
    if (v) v.style.display = 'none';
  });
  uploadVideo.pause();
  uploadVideo.src = uploadObjectUrl;
  uploadVideo.currentTime = 0;
  uploadVideo.style.display = 'block';
  updateCameraInfo(currentCam);
  updateModeButton();
  document.getElementById('history-label').textContent = 'TEST VIDEO · ACTIVE';
  document.getElementById('live-btn').style.display = '';
  fitStage();
  applyStatus('playback');
  try { await uploadVideo.play(); } catch (_) {}
}

async function returnToLive() {
  if (playbackMode !== 'upload') return;
  const uploadVideo = uploadVideoEl();
  clearDetectCanvas();
  playbackMode = 'live';
  uploadSession = null;
  if (uploadVideo) {
    try { uploadVideo.pause(); } catch (_) {}
    uploadVideo.removeAttribute('src');
    uploadVideo.load();
    uploadVideo.style.display = 'none';
  }
  if (uploadObjectUrl) {
    URL.revokeObjectURL(uploadObjectUrl);
    uploadObjectUrl = null;
  }
  showCam(currentCam);
  const d = await fetchStateSnapshot();
  const c = d && d.cameras && d.cameras[currentCam];
  if (c) applyStatus(c.status || 'connecting');
}

// ── Switch cam (instant — no stream restart) ─────────────────────────────────
async function switchTo(cam) {
  if (playbackMode === 'upload') return;
  if (cam === currentCam) return;
  if (!players[cam]) return;  // cam not configured; tab should be disabled anyway
  // Increment generation so any inflight poll for the old camera is discarded on arrival.
  _camGeneration++;
  yoloTracks.clear();
  visualTracks.clear();
  showCam(cam);
  // Notify backend which cam is "active" (no-op on the stream threads when
  // the mode hasn't changed — this just updates _active_cam server-side).
  try {
    await fetch('/switch/' + cam + '?mode=' + currentMode, { method: 'POST' });
  } catch (_) {}
  const d = await fetchStateSnapshot();
  const c = d && d.cameras && d.cameras[cam];
  if (c) applyStatus(c.status || 'connecting');
}

// ── Toggle HD for the currently visible cam ──────────────────────────────────
function _applyDetectionForMode(mode) {
  const btn   = document.getElementById('detect-btn');
  const label = document.getElementById('detect-label');
  if (mode === 'hd') {
    // Force detection off — YOLO holds GIL and freezes HD stream
    if (detectEnabled) { detectEnabled = false; updateDetectButton(); _pipelineStop(); }
    btn.disabled = true;
    btn.title    = 'Detection disabled in HD mode';
    label.textContent = 'DETECTION · OFF';
    btn.style.opacity = '0.35';
    btn.style.cursor  = 'not-allowed';
  } else {
    btn.disabled = false;
    btn.title    = 'Toggle AI detection';
    btn.style.opacity = '';
    btn.style.cursor  = '';
    updateDetectButton();
  }
}

async function toggleMode() {
  if (playbackMode === 'upload') return;
  if (pendingModeCam) return;
  const cam  = currentCam;
  const next = currentMode === 'hd' ? 'standard' : 'hd';
  pendingModeCam = cam;

  applyStatus('connecting');
  showLoading(
    next === 'hd' ? 'ENABLING HD' : 'BACK TO STANDARD',
    'restarting ' + cam.toUpperCase() + ' at ' + next.toUpperCase() + '...'
  );

  destroyPlayer(cam);

  currentMode   = next;
  camMode[cam]  = next;
  updateModeButton();
  _applyDetectionForMode(next);

  try {
    await fetch('/switch/' + cam + '?mode=' + next, { method: 'POST' });
  } catch (e) {
    console.warn('mode switch failed', e);
  }

  const ready = await waitForCamReady(cam, next, next === 'hd' ? 30000 : 15000);
  showLoading(
    'CONNECTING',
    ready
      ? (next === 'hd' ? 'buffering HD stream...' : 'buffering stream...')
      : 'stream is taking longer than expected...'
  );
  initPlayerForCam(cam);
  pendingModeCam = null;
  // After release, let the next poll (or live MANIFEST_PARSED) clear the overlay.
  const d = await fetchStateSnapshot();
  const c = d && d.cameras && d.cameras[currentCam];
  if (c) applyStatus(c.status || 'connecting');
}

// ── Button wiring ────────────────────────────────────────────────────────────
populateCamDropdown();

document.getElementById('mode-btn').addEventListener('click', toggleMode);
document.getElementById('history-btn').addEventListener('click', () => {
  document.getElementById('history-file').click();
});
document.getElementById('live-btn').addEventListener('click', () => {
  returnToLive().catch(e => console.warn('return-to-live failed', e));
});
document.getElementById('history-file').addEventListener('change', async (event) => {
  const file = event.target.files && event.target.files[0];
  event.target.value = '';
  if (!file) return;
  try {
    await enterUploadMode(file);
  } catch (e) {
    console.warn('history upload failed', e);
    if (playbackMode !== 'upload') {
      hideLoading();
      const d = await fetchStateSnapshot();
      const c = d && d.cameras && d.cameras[currentCam];
      if (c) applyStatus(c.status || 'connecting');
    }
  }
});

// ── State polling ────────────────────────────────────────────────────────────
const _lastCamStatus = {};  // track previous status per cam to detect transitions

async function pollState() {
  const d = await fetchStateSnapshot();
  if (!d || !d.cameras) return;
  CAM_KEYS.forEach(c => {
    const camData = d.cameras[c];
    if (!camData) return;
    if (camData.mode) camMode[c] = camData.mode;

    const newStatus  = camData.status || 'connecting';
    const prevStatus = _lastCamStatus[c] || 'connecting';

    // When a cam just became live, check if the MJPEG img has a src loaded.
    // Reinitialise the player if the img went blank (e.g. server restart).
    if (newStatus === 'live' && prevStatus !== 'live') {
      const video = videoEl(c);
      const stuck = !video || video.readyState < 2 || video.paused || video.ended;
      if (stuck) {
        console.log('[state]', c, 'just went live — reinitialising player');
        initPlayerForCam(c);
      }
    }

    _lastCamStatus[c] = newStatus;
  });

  if (playbackMode === 'upload') return;
  const active = d.cameras[currentCam];
  if (active) applyStatus(active.status || 'connecting');
}

function startStatePolling() {
  if (statePollTimer !== null) return;
  pollState();
  statePollTimer = setInterval(pollState, 3000);
}

// ── AI Detection (COCO-SSD · multi-person) ───────────────────────────────────
//
const CLASS_MAP = {
  person: { conf: 0.50 },
};

// Each tracked person gets a unique color — PERSON_0 is always blue (slot 0)
const PERSON_COLORS = ['#00c8ff','#ff4455','#44ff88','#ffaa00','#cc44ff','#ff8822','#00ffcc','#ff44cc'];
//                      ^^^^ slot 0 = blue, permanently reserved for PERSON_0
const _colorSlots  = {};
const _colorFree   = new Set([1, 2, 3, 4, 5, 6, 7]);  // slot 0 reserved for PERSON_0
const UNKNOWN_BOX_COLOR = '#ffd400';
const IDENTIFIED_BOX_COLORS = {
  // Add name: color entries here to assign fixed colors per identity
  // e.g.  Alice: '#00c8ff',
};
const IDENTIFIED_FALLBACK_COLOR = '#44ff88';

function _assignColor(label) {
  if (_colorSlots[label] !== undefined) return PERSON_COLORS[_colorSlots[label]];
  if (label === 'PERSON_0') {
    _colorSlots[label] = 0;
    return PERSON_COLORS[0];
  }
  const slot = _colorFree.size ? [..._colorFree][0] : (1 + Object.keys(_colorSlots).length % (PERSON_COLORS.length - 1));
  _colorFree.delete(slot);
  _colorSlots[label] = slot;
  return PERSON_COLORS[slot];
}
function _freeColor(label) {
  const slot = _colorSlots[label];
  if (slot !== undefined) {
    if (slot !== 0) _colorFree.add(slot);  // never put slot 0 back — reserved for PERSON_0
    delete _colorSlots[label];
  }
}

const DETECT_MS          = 200;
const BBOX_SHRINK        = 0.02;
const DETECT_TTL         = 2000;  // raw detection freshness window
const TRACKED_MIN_SCORE  = 0.58;  // existing tracked/ghost boxes may survive on weaker readings
const NEW_BOX_MIN_SCORE  = 0.78;  // brand-new generic boxes need higher confidence to avoid wall-art false positives
const FACE_SEED_MIN_SCORE = 0.55; // face-derived fallback boxes may start from this confidence
const BODY_CONFIRM_HITS  = 3;     // body-only tracks need repeat hits before they render
const BODY_CONFIRM_MS    = 2500;  // repeated body hits must happen within this window
const MOTION_CONFIRM_HITS = 3;    // body-only tracks also need repeated movement evidence
const MOTION_HOLD_MS     = 4000;  // keep a moved person confirmed for a short quiet window
const MOTION_PIXEL_DELTA = 26;    // pixel delta in the low-res motion grid
const BODY_MOTION_RATIO  = 0.012; // existing tracks count as moving above this ratio
const NEW_BOX_MOTION_RATIO = 0.024; // new generic boxes require stronger motion
const NEW_BOX_BLOB_RATIO = 0.55;  // dominant motion blob must explain most of the box's motion
const NEW_BOX_BLOB_CENTER_FRAC = 0.42; // dominant motion blob center must stay near the box center
const SMOOTH_ALPHA       = 0.35;  // max position alpha (reached at SMOOTH_SCALE px delta)
const SMOOTH_ALPHA_SZ    = 0.06;  // size EMA — very slow so box doesn't resize every frame
const SMOOTH_MIN         = 0.03;  // min alpha for tiny movements — suppresses detection noise
const SMOOTH_SCALE       = 70;    // px delta at which alpha reaches SMOOTH_ALPHA
const RENDER_MS          = 40;    // dedicated render loop, decoupled from detection
const IF_POLL_MS         = 300;
const FACE_TO_BODY_H     = 5.0;   // body height ≈ 5× face height
const FACE_TO_BODY_W     = 4.5;   // body width  ≈ 4.5× face width (covers seated/arms-out)
const HEAD_PAD           = 0.14;  // extend top of box upward by this fraction of bbox height
const IF_LIVE_MS         = 6000;  // face match stays valid for 6s (person may face away briefly)
const IF_STRIKES_NORMAL  = 15;    // ~4.5s before evicting unconfirmed box
const IF_STRIKES_COMPETE = 5;     // ~1.5s when another box just confirmed
const BOX_COAST_MS       = 4500;  // keep last good body box visible through short detector dropouts
const BOX_FACE_HOLD_MS   = 2000;  // hold after last face match; >this → track hidden to avoid ghost on empty space

let detectEnabled  = localStorage.getItem('detectEnabled') !== 'off';
let detectModel    = null;
let detectInflight = false;
let detectTimer    = null;
const smoothedBoxes = {};   // active: label -> { x, y, w, h, score, color, ts, ...identity }
const ghostBoxes    = {};   // expired but still matchable: label -> { ...same, ghostTs }
const GHOST_TTL     = 10000; // ms a ghost survives before the label is fully released

function updateDetectButton() {
  const btn   = document.getElementById('detect-btn');
  const label = document.getElementById('detect-label');
  if (!btn || !label) return;
  if (detectEnabled) {
    btn.classList.add('active');
    label.textContent = 'DETECTION · ON';
  } else {
    btn.classList.remove('active');
    label.textContent = 'DETECTION · OFF';
    clearDetectCanvas();
  }
}

function clearDetectCanvas() {
  if (_detectCanvas) _detectCtx.clearRect(0, 0, _detectCanvas.width, _detectCanvas.height);
  _resetMotionState();
  [...Object.keys(smoothedBoxes), ...Object.keys(ghostBoxes)].forEach(k => {
    _freeColor(k);
    delete _savedIdentities[k];
    delete smoothedBoxes[k];
    delete ghostBoxes[k];
  });
}

const NAME_LOCK_MS     = 5000;   // first vote at 5s; extend 5s more if still Unknown
const NAME_LOCK_EXT    = 10000;  // extended window when model isn't sure yet
const NAME_SHOW_MIN_VOTES = 1;   // show a name on the first strong face read
const DEMO_REFRESH_MS  = 10000;  // refresh gender/age title from recent readings every 10s
const NAME_SCORE_MIN   = 0.43;   // match Python RECOG_THRESH so no valid recognition is ignored
const NAME_MARGIN_MIN  = 0.03;
const FACE_TRACK_MAX_SCORE = 2.2; // lower = stricter face-to-body attachment

// Shorthand for "all PERSON_ keys in a tracking dict"
function personKeys(obj) { return Object.keys(obj).filter(k => k.startsWith('PERSON_')); }

// NMS: suppress lower-confidence boxes that heavily overlap a higher-confidence detection.
// Slightly looser suppression so two nearby people survive more often.
function nmsPersons(dets) {
  const suppressed = new Set();
  for (let i = 0; i < dets.length; i++) {
    if (suppressed.has(i)) continue;
    const [ax, ay, aw, ah] = dets[i].bbox;
    const aArea = aw * ah;
    for (let j = i + 1; j < dets.length; j++) {
      if (suppressed.has(j)) continue;
      const [bx, by, bw, bh] = dets[j].bbox;
      const ix    = Math.max(0, Math.min(ax+aw, bx+bw) - Math.max(ax, bx));
      const iy    = Math.max(0, Math.min(ay+ah, by+bh) - Math.max(ay, by));
      const inter = ix * iy;
      const iou   = inter / (aArea + bw*bh - inter);
      if (iou > 0.28 || inter / Math.min(aArea, bw*bh) > 0.50)
        suppressed.add(j);
    }
  }
  return dets.filter((_, i) => !suppressed.has(i));
}

// Per-label identity — persists after box expires so name shows immediately on re-detection
const _savedIdentities = {};
function _getSavedIdentity(label) {
  return _savedIdentities[label] ?? (_savedIdentities[label] = {
    lockedName: null, nameLocked: false, nameBuf: [], pendingName: null, nameExtended: false,
    lockedGender: null, lockedAge: null, genderLocked: false, genderBuf: [], lastGenderRefreshTs: 0,
    lastBodyTs: 0, lastFaceTs: 0, lastMotionTs: 0, lastMotionBlobId: null,
    bodyHits: 0, faceHits: 0, motionHits: 0, personConfirmed: false, bornTs: 0,
    ifBbox: null,
  });
}

function _displayNameForBox(box) {
  if (!box) return null;
  return box.nameLocked ? box.lockedName : (box.pendingName ?? null);
}

function _renderColorForBox(box) {
  const name = _displayNameForBox(box);
  if (!name || name === 'Unknown' || name === 'PERSON') return UNKNOWN_BOX_COLOR;
  return IDENTIFIED_BOX_COLORS[name] || IDENTIFIED_FALLBACK_COLOR;
}

function _hasActiveDisplayName(name) {
  if (!name || name === 'Unknown' || name === 'PERSON') return false;
  const now = Date.now();
  return personKeys(smoothedBoxes).some(label => {
    const box = smoothedBoxes[label];
    if (!box) return false;
    const display = _displayNameForBox(box);
    const freshFace = !!(box.lastIfMatchTs && (now - box.lastIfMatchTs) < BOX_FACE_HOLD_MS);
    return display === name && (box.personConfirmed || freshFace);
  });
}

function _strongFaceName(det) {
  if (!det || det.name === 'Unknown') return 'Unknown';
  if ((det.recog_score ?? 0) < NAME_SCORE_MIN) return 'Unknown';
  if ((det.recog_margin ?? 0) < NAME_MARGIN_MIN) return 'Unknown';
  return det.name;
}

function _boxIou(a, b) {
  if (!a || !b) return 0;
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  if (!inter) return 0;
  return inter / Math.max(1, (a.w * a.h) + (b.w * b.h) - inter);
}

function _boxCenterInside(a, b) {
  if (!a || !b) return false;
  const cx = a.x + a.w / 2;
  const cy = a.y + a.h / 2;
  return cx >= b.x && cx <= (b.x + b.w) && cy >= b.y && cy <= (b.y + b.h);
}

function _trackPriority(box, now = Date.now()) {
  if (!box) return -Infinity;
  let p = (box.score || 0) * 100;
  const faceFresh = !!(box.lastIfMatchTs && (now - box.lastIfMatchTs) < BOX_FACE_HOLD_MS);
  const named = !!_displayNameForBox(box);
  if (faceFresh) p += 350;
  if (box.nameLocked) p += 220;
  else if (named) p += 110;
  p += (box.motionHits || 0) * 25;
  p += (box.faceHits || 0) * 20;
  p += (box.bodyHits || 0) * 10;
  if (box.bornTs) p += Math.min(120, (now - box.bornTs) / 250);
  return p;
}

function _pruneCompetingTracks(now = Date.now()) {
  const labels = personKeys(smoothedBoxes);
  const losers = new Set();

  for (let i = 0; i < labels.length; i++) {
    const aLabel = labels[i];
    const a = smoothedBoxes[aLabel];
    if (!a || losers.has(aLabel)) continue;
    for (let j = i + 1; j < labels.length; j++) {
      const bLabel = labels[j];
      const b = smoothedBoxes[bLabel];
      if (!b || losers.has(bLabel)) continue;

      const aName = _displayNameForBox(a);
      const bName = _displayNameForBox(b);
      const aFace = !!(a.lastIfMatchTs && (now - a.lastIfMatchTs) < BOX_FACE_HOLD_MS);
      const bFace = !!(b.lastIfMatchTs && (now - b.lastIfMatchTs) < BOX_FACE_HOLD_MS);
      const sameBlob = !!(a.lastMotionBlobId && b.lastMotionBlobId && a.lastMotionBlobId === b.lastMotionBlobId);
      const iou = _boxIou(a, b);
      const centerInside = _boxCenterInside(a, b) || _boxCenterInside(b, a);
      const centerDist = Math.hypot((a.x + a.w / 2) - (b.x + b.w / 2), (a.y + a.h / 2) - (b.y + b.h / 2));
      const close = centerDist < Math.max(36, Math.min(a.w, b.w) * 0.82);
      const competing = sameBlob ? (iou > 0.03 || centerInside || close)
                                 : (iou > 0.22 || (centerInside && (aFace !== bFace)) || (!aFace && !bFace && !aName && !bName && (iou > 0.12 || close)));
      if (!competing) continue;

      if (aName && bName && aName !== bName && !sameBlob) continue;

      const pa = _trackPriority(a, now);
      const pb = _trackPriority(b, now);
      const clearWinner = sameBlob || Math.abs(pa - pb) >= 70 || (aFace !== bFace) || (!!aName !== !!bName);
      if (!clearWinner) continue;
      losers.add(pa >= pb ? bLabel : aLabel);
    }
  }

  losers.forEach(label => {
    const box = smoothedBoxes[label];
    if (!box) return;
    ghostBoxes[label] = { ...box, ghostTs: now - (GHOST_TTL - 1200) };
    delete smoothedBoxes[label];
    _clearBoxIdentity(label, ghostBoxes[label]);
  });
}

function _clearBoxIdentity(label, box) {
  if (box) {
    box.lockedName   = null;
    box.nameLocked   = false;
    box.pendingName  = null;
    box.nameBuf      = [];
    box.nameExtended = false;
  }
  const si = _savedIdentities[label];
  if (si) {
    si.lockedName   = null;
    si.nameLocked   = false;
    si.pendingName  = null;
    si.nameBuf      = [];
    si.nameExtended = false;
  }
}

function _pickNameWinner(a, b) {
  const af = a?.lastIfMatchTs || 0, bf = b?.lastIfMatchTs || 0;
  if (af !== bf) return af > bf ? a : b;
  const al = a?.nameLocked ? 1 : 0, bl = b?.nameLocked ? 1 : 0;
  if (al !== bl) return al > bl ? a : b;
  const as = a?.score || 0, bs = b?.score || 0;
  if (as !== bs) return as > bs ? a : b;
  const at = a?.ts || 0, bt = b?.ts || 0;
  return at >= bt ? a : b;
}

function _enforceUniqueNames() {
  const winners = {};
  Object.entries(smoothedBoxes).forEach(([label, box]) => {
    const name = _displayNameForBox(box);
    if (!name || name === 'Unknown' || name === 'PERSON') return;
    const prev = winners[name];
    winners[name] = prev ? _pickNameWinner(prev, { label, ...box }) : { label, ...box };
  });

  Object.entries(smoothedBoxes).forEach(([label, box]) => {
    const name = _displayNameForBox(box);
    if (!name || name === 'Unknown' || name === 'PERSON') return;
    const winner = winners[name];
    if (winner && winner.label !== label) _clearBoxIdentity(label, box);
  });
}

function _isTrackConfirmed(box, now = Date.now()) {
  if (!box) return false;
  const faceFresh = !!(box.lastFaceTs && (now - box.lastFaceTs) < BOX_FACE_HOLD_MS);
  const bodyFresh = !!(box.lastBodyTs && (now - box.lastBodyTs) < BODY_CONFIRM_MS);
  const motionFresh = !!(box.lastMotionTs && (now - box.lastMotionTs) < MOTION_HOLD_MS);
  if (faceFresh && (box.faceHits || 0) >= 1) return true;
  if (bodyFresh && motionFresh &&
      (box.bodyHits || 0) >= BODY_CONFIRM_HITS &&
      (box.motionHits || 0) >= MOTION_CONFIRM_HITS) return true;
  return !!box.personConfirmed && (
    faceFresh ||
    (bodyFresh && motionFresh) ||
    !!(box.lastIfMatchTs && (now - box.lastIfMatchTs) < BOX_FACE_HOLD_MS)
  );
}

function _syncTrackMeta(label) {
  const box = smoothedBoxes[label];
  if (!box) return;
  box.personConfirmed = _isTrackConfirmed(box);
  const si = _getSavedIdentity(label);
  Object.assign(si, {
    lastBodyTs: box.lastBodyTs || 0,
    lastFaceTs: box.lastFaceTs || 0,
    lastMotionTs: box.lastMotionTs || 0,
    lastMotionBlobId: box.lastMotionBlobId ?? null,
    bodyHits: box.bodyHits || 0,
    faceHits: box.faceHits || 0,
    motionHits: box.motionHits || 0,
    personConfirmed: !!box.personConfirmed,
    bornTs: box.bornTs || 0,
  });
}

function _markTrackObserved(label, source) {
  const now = Date.now();
  const box = smoothedBoxes[label];
  if (!box) return;
  if (!box.bornTs) box.bornTs = now;
  if (source === 'body') {
    const recent = box.lastBodyTs && (now - box.lastBodyTs) < BODY_CONFIRM_MS;
    box.bodyHits = recent ? (box.bodyHits || 0) + 1 : 1;
    box.lastBodyTs = now;
  } else if (source === 'face') {
    const recent = box.lastFaceTs && (now - box.lastFaceTs) < BOX_FACE_HOLD_MS;
    box.faceHits = recent ? (box.faceHits || 0) + 1 : 1;
    box.lastFaceTs = now;
  } else if (source === 'motion') {
    const recent = box.lastMotionTs && (now - box.lastMotionTs) < MOTION_HOLD_MS;
    box.motionHits = recent ? (box.motionHits || 0) + 1 : 1;
    box.lastMotionTs = now;
  }
  _syncTrackMeta(label);
}

function _setTrackMotionBlob(label, blobId) {
  const box = smoothedBoxes[label];
  if (!box) return;
  box.lastMotionBlobId = blobId ?? null;
  const si = _getSavedIdentity(label);
  si.lastMotionBlobId = box.lastMotionBlobId;
}

function blendSmoothed(label, x, y, w, h, score, color, age = null, gender = null, name = null) {
  const now  = Date.now();
  const prev = smoothedBoxes[label];

  // When box is new (prev null) restore last known identity so name shows immediately
  const idSrc = prev ?? _getSavedIdentity(label);

  // --- name: only show after repeated strong face reads; lock after majority vote ---
  let lockedName   = idSrc.lockedName   ?? null;
  let nameLocked   = idSrc.nameLocked   ?? false;
  let nameBuf      = idSrc.nameBuf      ?? [];
  let pendingName  = idSrc.pendingName  ?? null;
  let nameExtended = idSrc.nameExtended ?? false;

  if (!nameLocked && name !== null) {
    const nameWindow = nameExtended ? NAME_LOCK_EXT : NAME_LOCK_MS;
    nameBuf = [...nameBuf, { name, ts: now }]
                .filter(e => now - e.ts < nameWindow + 1000);
    const tally = {};
    nameBuf.forEach(e => { tally[e.name] = (tally[e.name] || 0) + 1; });
    const leader = Object.entries(tally).sort((a, b) => b[1] - a[1])[0] || null;
    const winner = leader ? leader[0] : null;
    const winnerVotes = leader ? leader[1] : 0;
    if (winner && winner !== 'Unknown' && winnerVotes >= NAME_SHOW_MIN_VOTES) {
      pendingName = winner;
    } else {
      // Only clear the name if we haven't seen a real name in the last 3s.
      // This prevents a single weak/occluded frame from wiping the displayed identity.
      const hasRecentRealVote = nameBuf.some(e => e.name !== 'Unknown' && now - e.ts < 3000);
      if (!hasRecentRealVote) pendingName = null;
    }
    const span = nameBuf.length > 1 ? now - nameBuf[0].ts : 0;
    if (span >= nameWindow) {
      if (winner !== 'Unknown') {
        lockedName   = winner;
        nameLocked   = true;
        pendingName  = winner;
        nameBuf      = [];
        nameExtended = false;
      } else if (!nameExtended) {
        nameExtended = true;  // give extra 5s — keep nameBuf, don't reset
      } else {
        nameBuf      = [];    // gave 10s total, still Unknown — reset and try again
        nameExtended = false;
        // pendingName kept — don't wipe a name we already showed
      }
    }
  }

  // --- gender/age: seed immediately, then refresh from recent readings every 10s ---
  let lockedGender        = idSrc.lockedGender        ?? null;
  let lockedAge           = idSrc.lockedAge           ?? null;
  let genderLocked        = idSrc.genderLocked        ?? false;
  let genderBuf           = idSrc.genderBuf           ?? [];
  let lastGenderRefreshTs = idSrc.lastGenderRefreshTs ?? 0;
  let lastBodyTs          = idSrc.lastBodyTs          ?? 0;
  let lastFaceTs          = idSrc.lastFaceTs          ?? 0;
  let lastMotionTs        = idSrc.lastMotionTs        ?? 0;
  let lastMotionBlobId    = idSrc.lastMotionBlobId    ?? null;
  let bodyHits            = idSrc.bodyHits            ?? 0;
  let faceHits            = idSrc.faceHits            ?? 0;
  let motionHits          = idSrc.motionHits          ?? 0;
  let personConfirmed     = idSrc.personConfirmed     ?? false;
  let bornTs              = idSrc.bornTs              ?? 0;

  if (gender !== null || age !== null) {
    genderBuf = [...genderBuf, { gender, age, ts: now }]
                  .filter(e => now - e.ts < DEMO_REFRESH_MS + 1000);

    const shouldRefreshDemo =
      genderBuf.length > 0 &&
      (!lastGenderRefreshTs || (now - lastGenderRefreshTs) >= DEMO_REFRESH_MS);

    if (shouldRefreshDemo) {
      const genders = genderBuf.map(e => e.gender).filter(g => g != null);
      if (genders.length) {
        const males = genders.filter(g => g === 1).length;
        lockedGender = males >= genders.length / 2 ? 1 : 0;
      }
      const ages = genderBuf.map(e => e.age).filter(a => a != null);
      if (ages.length) {
        lockedAge = Math.round(ages.reduce((s, a) => s + a, 0) / ages.length);
      }
      genderLocked = lockedGender != null || lockedAge != null;
      lastGenderRefreshTs = now;
    }
  }

  if (prev) {
    // Adaptive alpha: scale 0..1 by how far the new detection is from current position.
    // Tiny deltas (model noise) barely move the box; large deltas (real walking) follow fully.
    const _a = (d) => SMOOTH_MIN + (SMOOTH_ALPHA - SMOOTH_MIN) * Math.min(Math.abs(d) / SMOOTH_SCALE, 1);
    const dx = x - prev.x, dy = y - prev.y;
    smoothedBoxes[label] = {
      x:      prev.x + dx * _a(dx),
      y:      prev.y + dy * _a(dy),
      w:      prev.w + (w - prev.w) * SMOOTH_ALPHA_SZ,
      h:      prev.h + (h - prev.h) * SMOOTH_ALPHA_SZ,
      score, color, ts: now,
      lockedName, nameLocked, nameBuf, pendingName, nameExtended,
      lockedGender, lockedAge, genderLocked, genderBuf, lastGenderRefreshTs,
      lastBodyTs, lastFaceTs, lastMotionTs, lastMotionBlobId, bodyHits, faceHits, motionHits, personConfirmed, bornTs,
      // preserve IF validation state from previous entry
      ifConfirmed:   prev.ifConfirmed   ?? false,
      lastIfMatchTs: prev.lastIfMatchTs ?? 0,
      ifStrikes:     prev.ifStrikes     ?? 0,
      ifBbox:        prev.ifBbox        ?? null,
    };
  } else {
    smoothedBoxes[label] = {
      x, y, w, h, score, color, ts: now,
      lockedName, nameLocked, nameBuf, pendingName, nameExtended,
      lockedGender, lockedAge, genderLocked, genderBuf, lastGenderRefreshTs,
      lastBodyTs, lastFaceTs, lastMotionTs, lastMotionBlobId, bodyHits, faceHits, motionHits, personConfirmed, bornTs: bornTs || now,
      // inherit from ghost/saved on resurrection, else start fresh
      ifConfirmed:   idSrc.ifConfirmed   ?? false,
      lastIfMatchTs: idSrc.lastIfMatchTs ?? 0,
      ifStrikes:     0,
      ifBbox:        idSrc.ifBbox        ?? null,
    };
  }

  // Persist identity fields so a future box creation restores them from _savedIdentities
  const _si = _getSavedIdentity(label);
  const _box = smoothedBoxes[label];
  Object.assign(_si, {
    lockedName, nameLocked, nameBuf, pendingName, nameExtended,
    lockedGender, lockedAge, genderLocked, genderBuf, lastGenderRefreshTs,
    lastBodyTs, lastFaceTs, lastMotionTs, lastMotionBlobId, bodyHits, faceHits, motionHits, personConfirmed, bornTs: bornTs || _box?.bornTs || now,
    ifConfirmed:   _box?.ifConfirmed   ?? _si.ifConfirmed   ?? false,
    lastIfMatchTs: _box?.lastIfMatchTs ?? _si.lastIfMatchTs ?? 0,
    ifBbox:        _box?.ifBbox        ?? _si.ifBbox        ?? null,
  });
}

// Convert any CSS color to rgba() with given opacity (0–1).
// Hex-appending only works for 6-digit hex; named colors need canvas parsing.
const _colorAlphaCache = {};
function colorWithAlpha(color, alpha) {
  const key = color + alpha;
  if (_colorAlphaCache[key]) return _colorAlphaCache[key];
  // Fast path: 6-digit hex
  if (/^#[0-9a-fA-F]{6}$/.test(color)) {
    const r = parseInt(color.slice(1,3),16), g = parseInt(color.slice(3,5),16), b = parseInt(color.slice(5,7),16);
    return (_colorAlphaCache[key] = `rgba(${r},${g},${b},${alpha})`);
  }
  // Universal path: use an offscreen canvas to resolve the color
  const tmp = document.createElement('canvas'); tmp.width = tmp.height = 1;
  const c = tmp.getContext('2d'); c.fillStyle = color; c.fillRect(0,0,1,1);
  const d = c.getImageData(0,0,1,1).data;
  return (_colorAlphaCache[key] = `rgba(${d[0]},${d[1]},${d[2]},${alpha})`);
}

// Draw one tactical box from pre-computed canvas coords
function drawBox(ctx, canvas, label, x, y, w, h, score, color, lockedAge = null, lockedGender = null, genderLocked = false, lockedName = null, nameLocked = false, pendingName = null, recogScore = null) {
  ctx.fillStyle = colorWithAlpha(color, 0.094);
  ctx.fillRect(x, y, w, h);

  ctx.shadowColor = color;
  ctx.shadowBlur  = 12;

  const br = Math.min(Math.max(w, h) * 0.28, 70, Math.min(w * 0.45, h * 0.45));
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'square';
  ctx.beginPath();
  ctx.moveTo(x,       y + br); ctx.lineTo(x,       y);      ctx.lineTo(x + br,  y);
  ctx.moveTo(x+w-br,  y);      ctx.lineTo(x+w,     y);      ctx.lineTo(x+w,     y + br);
  ctx.moveTo(x+w,     y+h-br); ctx.lineTo(x+w,     y+h);    ctx.lineTo(x+w-br,  y+h);
  ctx.moveTo(x+br,    y+h);    ctx.lineTo(x,       y+h);    ctx.lineTo(x,       y+h-br);
  ctx.stroke();

  ctx.shadowBlur = 0;

  const nameStr = nameLocked ? lockedName : (pendingName ?? label);
  // When a real identity is displayed, show recognition score (identity confidence)
  // instead of detection score (which is just "I see a face here, 99%").
  const hasIdentity = nameStr && nameStr !== 'PERSON' && nameStr !== label;
  const scoreForPill = hasIdentity && recogScore != null ? recogScore : score;
  const pct     = Math.round(scoreForPill * 100);
  const gStr    = lockedGender != null ? (lockedGender === 1 ? 'M' : 'F') : null;
  const ageStr  = lockedAge != null ? lockedAge + 'y' : null;
  const txt     = [nameStr, gStr, ageStr, pct + '%'].filter(Boolean).join(' · ');
  ctx.font  = 'bold 11px "Segoe UI", monospace';
  const tw  = ctx.measureText(txt).width;
  const ph  = 20, pw = tw + 16;
  const lx  = Math.min(x, canvas.width - pw - 2);
  const ly  = Math.max(y - ph - 4, 2);
  ctx.fillStyle = colorWithAlpha(color, 0.867);
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(lx, ly, pw, ph, 4);
  else ctx.rect(lx, ly, pw, ph);
  ctx.fill();
  ctx.fillStyle = '#000';
  ctx.fillText(txt, lx + 8, ly + 14);
}

async function runDetection() {
  return;  // COCO-SSD removed — YOLO + InsightFace is the only pipeline now
  // eslint-disable-next-line no-unreachable
  if (yoloMode !== 'off') return;
  if (detectInflight || !detectEnabled || !detectModel) return;
  detectInflight = true;
  try {
    const canvas = _detectCanvas;
    if (!canvas) return;

    const camVideos = playbackMode === 'upload'
      ? [{ cam: 'upload', video: uploadVideoEl() }]
          .filter(({ video }) => video && (video.paused === false || video.paused === undefined) && (video.readyState !== undefined ? video.readyState >= 2 : video.complete) && (video.videoWidth || video.naturalWidth))
      : CAM_KEYS
          .filter(c => players[c])
          .map(c => ({ cam: c, video: videoEl(c) }))
          .filter(({ video }) => video && (video.paused === false || video.paused === undefined) && (video.readyState !== undefined ? video.readyState >= 2 : video.complete) && (video.videoWidth || video.naturalWidth));

    const allResults = await Promise.all(
      camVideos.map(({ cam, video }) =>
        detectModel.detect(video).then(preds => ({ cam, video, preds })).catch(() => null)
      )
    );

    const activeResult = playbackMode === 'upload'
      ? allResults.find(r => r?.cam === 'upload')
      : allResults.find(r => r?.cam === currentCam);
    let activeHasPerson = false;

    if (activeResult) {
      const { video, preds } = activeResult;
      const sx = canvas.width  / (video.videoWidth || video.naturalWidth);
      const sy = canvas.height / (video.videoHeight || video.naturalHeight);
      const motion = _sampleMotion(video, _currentMotionSourceKey());

      // Collect all person detections — filter by min confidence + min box area
      const MIN_BOX_FRAC = 0.04;  // box must cover at least 4% of frame area
      const frameArea    = (video.videoWidth || video.naturalWidth) * (video.videoHeight || video.naturalHeight);
      const rawDets = preds
        .filter(p => CLASS_MAP[p.class] && p.score >= TRACKED_MIN_SCORE)
        .filter(p => (p.bbox[2] * p.bbox[3]) / frameArea >= MIN_BOX_FRAC)
        .sort((a, b) => b.score - a.score);

      const personDets = nmsPersons(rawDets).map(p => ({
        ...p,
        cx: (p.bbox[0] + p.bbox[2] / 2) * sx,
        cy: (p.bbox[1] + p.bbox[3] / 2) * sy,
      }));

      // Greedy nearest-neighbor: match each detection to its closest tracked or ghost label
      const MAX_MATCH_DIST  = 250;   // canvas px — normal tracking window
      const MAX_GHOST_DIST  = 350;   // canvas px — ghost resurrection window
      const trackedLabels   = personKeys(smoothedBoxes);
      const ghostLabels     = personKeys(ghostBoxes);
      // Labels that InsightFace has confirmed contain a real person
      const confirmedLabels = trackedLabels.filter(l => smoothedBoxes[l]?.ifConfirmed);
      const usedLabels      = new Set();
      const blobAssignments = new Map();

      for (const p of personDets) {
        const motionInfo = _motionInfoForBBox(motion, p.bbox, video);
        const motionRatio = motionInfo.ratio;
        const movingNow = motionRatio >= BODY_MOTION_RATIO;
        const motionBlobId = motionInfo.blobId;
        const motionBlob = motionInfo.blob;
        const blobCx = motionBlob ? (((motionBlob.x1 + motionBlob.x2) / 2) / motion.w) * canvas.width : null;
        const blobCy = motionBlob ? (((motionBlob.y1 + motionBlob.y2) / 2) / motion.h) * canvas.height : null;
        const blobCenterDist = (blobCx == null || blobCy == null)
          ? Infinity
          : Math.hypot(p.cx - blobCx, p.cy - blobCy);
        const blobCenterMax = Math.max(42, Math.min(p.bbox[2] * sx, p.bbox[3] * sy) * NEW_BOX_BLOB_CENTER_FRAC);
        // 1) Try to match to an active tracked label within normal distance
        let bestLabel = null, bestDist = MAX_MATCH_DIST;
        for (const lbl of trackedLabels) {
          if (usedLabels.has(lbl)) continue;
          const s = smoothedBoxes[lbl];
          const d = Math.hypot(p.cx - (s.x + s.w / 2), p.cy - (s.y + s.h / 2));
          if (d < bestDist) { bestDist = d; bestLabel = lbl; }
        }

        // 2) If no active match, try to resurrect a ghost (same person came back)
        if (!bestLabel) {
          let ghostBest = null, ghostDist = MAX_GHOST_DIST;
          for (const lbl of ghostLabels) {
            if (usedLabels.has(lbl)) continue;
            const g = ghostBoxes[lbl];
            const d = Math.hypot(p.cx - (g.x + g.w / 2), p.cy - (g.y + g.h / 2));
            if (d < ghostDist) { ghostDist = d; ghostBest = lbl; }
          }
          if (ghostBest) {
            smoothedBoxes[ghostBest] = { ...ghostBoxes[ghostBest] };
            delete ghostBoxes[ghostBest];
            bestLabel = ghostBest;
          }
        }

        // 3) Camera-pan fallback: if still no match and exactly ONE confirmed person
        //    exists, this detection must be that person at their new screen position.
        //    Reuse their label (same color, same identity) — don't create a new box.
        //    Only bypass distance when there's one confirmed person; with multiple
        //    confirmed people we can't know which one jumped.
        if (!bestLabel) {
          const freeConfirmed = confirmedLabels.filter(l => !usedLabels.has(l));
          if (freeConfirmed.length === 1 && trackedLabels.length === 1 && personDets.length === 1) {
            bestLabel = freeConfirmed[0];
            // Jump the box immediately — don't EMA-smooth a pan-sized leap
            if (smoothedBoxes[bestLabel]) {
              smoothedBoxes[bestLabel].x = p.cx - p.bbox[2] / 2 * sx;
              smoothedBoxes[bestLabel].y = p.cy - p.bbox[3] / 2 * sy;
            }
          }
        }

        // 4) Truly new person
        if (!bestLabel) {
          if (p.score < NEW_BOX_MIN_SCORE) continue;  // suppress weak new generic boxes such as wall art / reflections
          if (motionRatio < NEW_BOX_MOTION_RATIO) continue;  // static art / mirrors should not seed new body-only boxes
          if (!motionBlobId) continue;
          if (motionInfo.blobRatio < NEW_BOX_BLOB_RATIO) continue;
          if (blobCenterDist > blobCenterMax) continue;
          if (blobAssignments.has(motionBlobId)) continue;
          let i = 0;
          while (smoothedBoxes[`PERSON_${i}`] || ghostBoxes[`PERSON_${i}`] || usedLabels.has(`PERSON_${i}`)) i++;
          bestLabel = `PERSON_${i}`;
        }
        usedLabels.add(bestLabel);
        if (motionBlobId) blobAssignments.set(motionBlobId, bestLabel);

        const color = _assignColor(bestLabel);
        const [bx, by, bw, bh] = p.bbox;
        const dx  = bw * BBOX_SHRINK;
        const pad = bh * HEAD_PAD;
        blendSmoothed(bestLabel,
          (bx + dx) * sx, Math.max(0, (by - pad) * sy),
          (bw - dx * 2) * sx, (bh + pad) * sy,
          p.score, color,
        );
        _setTrackMotionBlob(bestLabel, motionBlobId);
        _markTrackObserved(bestLabel, 'body');
        if (movingNow) _markTrackObserved(bestLabel, 'motion');
      }
      activeHasPerson = usedLabels.size > 0;
    }

    // Other cams: if active cam missed people but another cam sees them,
    // refresh their TTL so boxes stay visible (don't reposition — wrong coord space)
    if (playbackMode !== 'upload' && !activeHasPerson) {
      const otherSeesPerson = allResults.some(r => {
        if (!r || r.cam === currentCam) return false;
        return r.preds.some(p => CLASS_MAP[p.class] && p.score >= CLASS_MAP[p.class].conf);
      });
      if (otherSeesPerson) {
        const now = Date.now();
        personKeys(smoothedBoxes).forEach(lbl => { smoothedBoxes[lbl].ts = now; });
      }
    }
  } catch (e) {
    console.warn('[detect]', e);
  } finally {
    detectInflight = false;
  }
}

async function loadDetectModel() {
  // COCO-SSD removed. Kept as a no-op so existing call sites don't blow up.
  return;
}

// ── Unified pipeline: YOLO-pose + BoT-SORT + InsightFace (server-side) ────────
// Everything below drives a single endpoint (/pipeline/...). Server returns a
// list of track objects with head position and resolved identity. The client
// EMA-smooths head position per track id for butter-smooth movement.
let yoloMode       = 'pipeline';  // 'off' | 'pipeline'
let yoloTimer      = null;
let yoloInflight   = false;
let _camGeneration = 0;           // incremented on every cam switch — stale poll responses are discarded
const YOLO_POLL_MS = 100;       // ~10 fps server polls
const YOLO_SMOOTH  = 0.55;      // EMA α — position smoothing per sample
const YOLO_VEL_SMOOTH = 0.30;   // EMA α for velocity
const YOLO_VEL_DEADBAND = 0.08; // px/ms — ignore keypoint jitter this small, keeps box still when not moving
const YOLO_HOLD_MS = 2500;      // long hold survives occlusions / missed polls
const YOLO_PREDICT_MS = 60;     // velocity-extrapolate cap — short enough that jitter doesn't drift the box
const IDENTITY_COLORS = {__IDENTITY_COLORS__};
function _colorFor(name) {
  if (!name || name === 'Unknown') return IDENTITY_COLORS._default;
  return IDENTITY_COLORS[name.toLowerCase()] || IDENTITY_COLORS._default;
}
const yoloTracks   = new Map(); // id → { x,y,w,h, ts, score, name, recog }  — detection targets (10fps)
const visualTracks = new Map(); // id → { x,y,w,h, opacity, name, ... }        — animated visual state (60fps rAF)
const VIS_LERP     = 0.18;      // position lerp per rAF frame — box chases head smoothly
const VIS_FADE_IN  = 0.18;      // opacity lerp in  — box appears quickly
const VIS_FADE_OUT = 0.018;     // opacity lerp out — slow fade (~4s to disappear)

function _headSquareFromDet(det) {
  // Prefer nose + eyes/ears to anchor on the face; fall back to bbox top band.
  const pts = ['nose','leye','reye','lear','rear','lsh','rsh']
    .map(k => det[k]).filter(Boolean);
  if (pts.length === 0) return null;
  const nose = det.nose ?? pts[0];
  const cx = nose[0];
  const cy = nose[1];
  // Estimate head size: ear-to-ear distance if both ears visible, else bbox width / 5.
  let side;
  if (det.lear && det.rear) {
    side = Math.hypot(det.lear[0] - det.rear[0], det.lear[1] - det.rear[1]) * 2.0;
  } else {
    const [x1, y1, x2, y2] = det.bbox;
    side = (x2 - x1) * 0.55;
  }
  side = Math.max(40, Math.min(side, 260));
  return { cx, cy, side };
}

async function pollYolo() {
  if (yoloInflight || yoloMode === 'off' || !detectEnabled) return;
  yoloInflight = true;
  const myGeneration = _camGeneration;  // snapshot — if camera switches mid-fetch, we discard
  try {
    let res = null;
    if (playbackMode === 'webcam') {
      const v = webcamVideoEl();
      if (!v || !(v.videoWidth || v.naturalWidth)) return;
      const cap = document.createElement('canvas');
      const W = 640, H = Math.round(640 * (v.videoHeight || v.naturalHeight) / (v.videoWidth || v.naturalWidth));
      cap.width = W; cap.height = H;
      cap.getContext('2d').drawImage(v, 0, 0, W, H);
      const blob = await new Promise(r => cap.toBlob(r, 'image/jpeg', 0.7));
      if (!blob) return;
      const fd = new FormData(); fd.append('frame', blob, 'f.jpg');
      res = await fetch('/pipeline/frame', { method: 'POST', body: fd })
              .then(r => r.ok ? r.json() : null).catch(() => null);
    } else {
      let url;
      if (playbackMode === 'upload') {
        const v = uploadVideoEl();
        if (!uploadSession || !v || v.readyState < 2) return;
        const t = Number.isFinite(v.currentTime) ? v.currentTime : 0;
        url = '/pipeline/upload/' + uploadSession.id + '?t=' + encodeURIComponent(t.toFixed(3));
      } else {
        url = '/pipeline/' + currentCam;
      }
      res = await fetch(url, { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null);
    }
    if (myGeneration !== _camGeneration) return;  // camera switched while fetching — discard
    const tracks = res?.tracks ?? [];
    const canvas = _detectCanvas;
    const video  = activeVideoEl();
    if (!canvas || !video || !(video.videoWidth || video.naturalWidth)) return;
    const sx = canvas.width  / (tracks[0]?.fw || (video.videoWidth || video.naturalWidth));
    const sy = canvas.height / (tracks[0]?.fh || (video.videoHeight || video.naturalHeight));
    const now = Date.now();
    const seen = new Set();
    tracks.forEach(tr => {
      const id = tr.track_id;
      if (id == null || !tr.head) return;
      seen.add(id);
      const cx = tr.head.cx * sx;
      const cy = tr.head.cy * sy;
      const sw = tr.head.side * Math.min(sx, sy);
      const sh = sw;               // square box — consistent size for all persons
      const tx = cx - sw / 2;
      const ty = cy - sh / 2;
      const prev = yoloTracks.get(id);
      const jump = prev ? Math.hypot(tx - prev.x, ty - prev.y) : 0;
      const snap = !prev || jump > sw * 1.2;     // only snap on huge teleports
      const a = snap ? 1 : YOLO_SMOOTH;
      const nx = snap ? tx : prev.x * (1 - a) + tx * a;
      const ny = snap ? ty : prev.y * (1 - a) + ty * a;
      const dt = prev ? Math.max(1, now - prev.ts) : 1;
      const rawVx = prev ? (nx - prev.x) / dt : 0;  // px / ms
      const rawVy = prev ? (ny - prev.y) / dt : 0;
      const vα = YOLO_VEL_SMOOTH;
      // Deadband: ignore sub-threshold velocity so keypoint jitter doesn't drift the box when still
      const dvx = Math.abs(rawVx) < YOLO_VEL_DEADBAND ? 0 : rawVx;
      const dvy = Math.abs(rawVy) < YOLO_VEL_DEADBAND ? 0 : rawVy;
      const vx = prev ? (prev.vx ?? 0) * (1 - vα) + dvx * vα : 0;
      const vy = prev ? (prev.vy ?? 0) * (1 - vα) + dvy * vα : 0;
      yoloTracks.set(id, {
        x: nx,
        y: ny,
        w: snap ? sw : prev.w * (1 - a) + sw * a,
        h: snap ? sh : prev.h * (1 - a) + sh * a,
        vx, vy,
        ts: now,
        score: tr.score ?? 0,
        name: tr.name ?? null,
        recog: tr.recog_score ?? null,
        pending: tr.pending ?? null,
      });
    });
    for (const [id, t] of yoloTracks) {
      if (!seen.has(id) && (now - t.ts) > YOLO_HOLD_MS) yoloTracks.delete(id);
    }

    // Name dedup: if the same name appears on multiple tracks (stale cache + new
    // server entry, or track-swap remnant), immediately drop all but the highest
    // recog score. Never render two boxes with the same name simultaneously.
    const byName = new Map(); // name → {id, recog}
    for (const [id, t] of yoloTracks) {
      if (!t.name) continue;
      const existing = byName.get(t.name);
      if (!existing) { byName.set(t.name, {id, recog: t.recog ?? 0}); continue; }
      // keep the higher score, drop the other
      if ((t.recog ?? 0) > existing.recog) {
        yoloTracks.delete(existing.id);
        visualTracks.delete(existing.id);
        byName.set(t.name, {id, recog: t.recog ?? 0});
      } else {
        yoloTracks.delete(id);
        visualTracks.delete(id);
      }
    }

    // Deduplicate: if two tracks have head centers within 80% of box width,
    // they're the same person — drop the lower-confidence one.
    const alive = [...yoloTracks.entries()];
    for (let i = 0; i < alive.length; i++) {
      const [idA, a] = alive[i];
      if (!yoloTracks.has(idA)) continue;
      for (let j = i + 1; j < alive.length; j++) {
        const [idB, b] = alive[j];
        if (!yoloTracks.has(idB)) continue;
        const dist = Math.hypot(a.x + a.w/2 - (b.x + b.w/2), a.y + a.h/2 - (b.y + b.h/2));
        if (dist < a.w * 1.5) {
          // Same person — keep the one with a locked name, else higher score
          const aWins = (a.name && !b.name) || (!a.name && !b.name && a.score >= b.score);
          const loser = aWins ? idB : idA;
          yoloTracks.delete(loser);
          visualTracks.delete(loser);
        }
      }
    }
  } catch (e) {
    console.warn('[pipeline]', e);
  } finally {
    yoloInflight = false;
  }
}

function _vLerp(a, b, t) { return a + (b - a) * t; }

function drawYoloBoxes() {
  const now = Date.now();
  const ctx = _detectCtx;
  const canvas = _detectCanvas;

  // Step 1 — advance visual state toward current detection targets (rAF = ~60fps)
  yoloTracks.forEach((target, id) => {
    const age    = now - target.ts;
    const fading = age > YOLO_HOLD_MS * 0.85;
    let vis = visualTracks.get(id);
    if (!vis) {
      // New track ID — check if an orphaned visual track is nearby (BoT-SORT re-ID).
      // If so, inherit its state (including opacity=1) so there's no blink.
      let best = null, bestDist = Infinity;
      visualTracks.forEach((v, vid) => {
        if (yoloTracks.has(vid)) return;          // only orphaned entries
        const d = Math.hypot(v.x - target.x, v.y - target.y);
        if (d < target.w * 2.5 && d < bestDist) { bestDist = d; best = { vid, v }; }
      });
      if (best) {
        vis = best.v;
        visualTracks.delete(best.vid);            // transfer, don't duplicate
      } else {
        vis = { x: target.x, y: target.y, w: target.w, h: target.h, opacity: 0 };
      }
      visualTracks.set(id, vis);
    }
    vis._orphanedAt = null;                       // track is active — clear grace timer
    vis.x = _vLerp(vis.x, target.x, VIS_LERP);
    vis.y = _vLerp(vis.y, target.y, VIS_LERP);
    vis.w = _vLerp(vis.w, target.w, VIS_LERP);
    vis.h = _vLerp(vis.h, target.h, VIS_LERP);
    vis.opacity = _vLerp(vis.opacity, fading ? 0 : 1, fading ? VIS_FADE_OUT : VIS_FADE_IN);
    vis.name    = target.name;
    vis.recog   = target.recog;
    vis.score   = target.score;
    vis.pending = target.pending;
  });

  // Step 2 — handle orphaned visual tracks (no longer in yoloTracks)
  // Grace period: stay fully opaque for 1 second before fading — covers brief detection gaps.
  // Exception: if ALL tracks vanished simultaneously (camera moved / scene cut),
  // bypass the grace period and fade immediately so old positions don't linger.
  const allOrphaned = yoloTracks.size === 0 && visualTracks.size > 0;
  visualTracks.forEach((vis, id) => {
    if (!yoloTracks.has(id)) {
      if (!vis._orphanedAt) vis._orphanedAt = now;
      const orphanMs = now - vis._orphanedAt;
      const gracePeriod = allOrphaned ? 0 : 1000;  // no grace when camera moves
      if (orphanMs > gracePeriod) {
        const fadeRate = allOrphaned ? 0.12 : VIS_FADE_OUT;  // fast fade on camera move
        vis.opacity = _vLerp(vis.opacity, 0, fadeRate);
        if (vis.opacity < 0.01) visualTracks.delete(id);
      }
    }
  });

  // Step 3 — render from smooth visual state
  visualTracks.forEach((vis) => {
    if (vis.opacity < 0.01) return;
    ctx.globalAlpha = vis.opacity;
    const displayName = vis.name || vis.pending || null;   // use pending for color while voting
    const color = _colorFor(displayName);
    const label = (vis.name && vis.name !== 'Unknown') ? vis.name : 'Person';
    drawBox(ctx, canvas, label, vis.x, vis.y, vis.w, vis.h,
            vis.score || 0, color,
            null, null, false, vis.name || null, !!vis.name, vis.pending || null, vis.recog);
    ctx.globalAlpha = 1;
  });
}

function _pipelineStart() {
  if (!yoloTimer) yoloTimer = setInterval(pollYolo, YOLO_POLL_MS);
}
function _pipelineStop() {
  if (yoloTimer) { clearInterval(yoloTimer); yoloTimer = null; }
  yoloTracks.clear();
  visualTracks.clear();
}

// ── Local webcam test mode ────────────────────────────────────────────────
let _webcamStream = null;
let _prevPlaybackMode = null;
async function _listWebcams() {
  try { await navigator.mediaDevices.getUserMedia({ video: true }); } catch(e){}
  const devs = await navigator.mediaDevices.enumerateDevices();
  return devs.filter(d => d.kind === 'videoinput');
}
async function _startWebcam(deviceId) {
  if (_webcamStream) _webcamStream.getTracks().forEach(t => t.stop());
  _webcamStream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 1280, height: 720 },
    audio: false,
  });
  const v = webcamVideoEl();
  v.srcObject = _webcamStream;
  await v.play().catch(()=>{});
  _prevPlaybackMode = playbackMode;
  playbackMode = 'webcam';
  [...CAM_KEYS.map(k => 'video-' + k), 'video-upload'].forEach(id => { const el=document.getElementById(id); if(el) el.style.display='none'; });
  v.style.display = '';
  if (typeof fitStage === 'function') fitStage();
  if (!detectEnabled) { detectEnabled = true; updateDetectButton(); }
  _pipelineStop(); _pipelineStart();
  fetch('/source/webcam', { method: 'POST' }).catch(() => {});
  document.getElementById('webcam-label').textContent = 'WEBCAM · ON';
}
function _stopWebcam() {
  if (_webcamStream) { _webcamStream.getTracks().forEach(t => t.stop()); _webcamStream = null; }
  const v = webcamVideoEl(); if (v) { v.srcObject = null; v.style.display='none'; }
  fetch('/source/rtsp', { method: 'POST' }).catch(() => {});
  playbackMode = _prevPlaybackMode || 'live';
  if (playbackMode === 'live') {
    const cv = videoEl(currentCam); if (cv) cv.style.display='';
  }
  yoloTracks.clear();
  document.getElementById('webcam-label').textContent = 'WEBCAM';
  document.getElementById('webcam-picker').style.display = 'none';
}
document.getElementById('webcam-btn').addEventListener('click', async () => {
  if (playbackMode === 'webcam') { _stopWebcam(); return; }
  const picker = document.getElementById('webcam-picker');
  const cams = await _listWebcams();
  if (cams.length === 0) { alert('No webcam detected'); return; }
  if (cams.length === 1) { await _startWebcam(cams[0].deviceId); return; }
  while (picker.firstChild) picker.removeChild(picker.firstChild);
  const ph = document.createElement('option');
  ph.value = ''; ph.textContent = 'pick camera';
  picker.appendChild(ph);
  cams.forEach((c, i) => {
    const o = document.createElement('option');
    o.value = c.deviceId;
    o.textContent = c.label || ('Camera ' + (i + 1));
    picker.appendChild(o);
  });
  picker.style.display = '';
  picker.onchange = async () => {
    if (!picker.value) return;
    picker.style.display = 'none';
    await _startWebcam(picker.value);
  };
});

document.getElementById('detect-btn').addEventListener('click', () => {
  detectEnabled = !detectEnabled;
  localStorage.setItem('detectEnabled', detectEnabled ? 'on' : 'off');
  fetch('/detection/' + (detectEnabled ? 'on' : 'off'), { method: 'POST' }).catch(() => {});
  updateDetectButton();
  if (detectEnabled) {
    _pipelineStart();
  } else {
    _pipelineStop();
  }
});

updateDetectButton();

// ── Render loop — requestAnimationFrame (~60fps, GPU-synced) ─────────────────
// yolo path:   reads visualTracks (smoothly lerped toward yoloTracks targets)
// legacy path: reads smoothedBoxes (COCO-SSD/InsightFace — inactive but kept)
function _renderFrame() {
  requestAnimationFrame(_renderFrame);
  if (!_detectCanvas) return;
  _detectCtx.clearRect(0, 0, _detectCanvas.width, _detectCanvas.height);
  if (!detectEnabled) return;
  if (yoloMode !== 'off') { drawYoloBoxes(); return; }
  const now     = Date.now();
  _enforceUniqueNames();
  _pruneCompetingTracks(now);
  // Collect draw candidates first, then suppress overlapping ones so only one
  // box remains visible per person (no duplicates on the final canvas).
  const candidates = [];
  Object.entries(smoothedBoxes).forEach(([label, s]) => {
    const ageMs = now - s.ts;
    const faceFresh = !!(s.lastIfMatchTs && (now - s.lastIfMatchTs) < BOX_FACE_HOLD_MS);
    const holdMs = faceFresh ? BOX_FACE_HOLD_MS : Math.max(DETECT_TTL, BOX_COAST_MS);
    s.personConfirmed = _isTrackConfirmed(s, now);
    if (ageMs >= holdMs) {
      ghostBoxes[label] = { ...s, ghostTs: now };
      delete smoothedBoxes[label];
      return;
    }
    if (!s.personConfirmed) return;
    const headFresh = s.ifBbox && s.lastIfMatchTs && (now - s.lastIfMatchTs) < BOX_FACE_HOLD_MS;
    const bodyFresh = !!(s.lastBodyTs && (now - s.lastBodyTs) < DETECT_TTL);
    if (!headFresh || !bodyFresh) return;
    // Stale-face guard: the face bbox must sit inside the current COCO body
    // bbox (expanded slightly). Otherwise the face position is stale relative
    // to the moving body and would draw on back/shoulder/empty area.
    const faceCx = s.ifBbox.x + s.ifBbox.w / 2;
    const faceCy = s.ifBbox.y + s.ifBbox.h / 2;
    const pad = Math.max(s.w, s.h) * 0.15;
    const insideBody =
      faceCx >= s.x - pad && faceCx <= s.x + s.w + pad &&
      faceCy >= s.y - pad && faceCy <= s.y + s.h + pad;
    if (!insideBody) return;
    candidates.push({ label, s });
  });

  // IoU-based NMS among candidates — if two head boxes overlap, keep the one
  // with the higher recognition score (named > unknown), then most recent face match.
  const iou = (a, b) => {
    const ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
    const ix2 = Math.min(a.x + a.w, b.x + b.w), iy2 = Math.min(a.y + a.h, b.y + b.h);
    const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const ua = a.w * a.h + b.w * b.h - inter;
    return ua > 0 ? inter / ua : 0;
  };
  const rank = (c) => {
    const namedBonus = (c.s.nameLocked || c.s.pendingName) ? 1000 : 0;
    const rs = (c.s.recogScore ?? 0) * 100;
    const recency = c.s.lastIfMatchTs || 0;
    return namedBonus + rs + recency / 1e9;
  };
  candidates.sort((a, b) => rank(b) - rank(a));
  const drawn = [];
  for (const c of candidates) {
    const overlaps = drawn.some(d => iou(c.s.ifBbox, d.s.ifBbox) > 0.3);
    if (overlaps) continue;
    drawn.push(c);
    const s = c.s;
    drawBox(_detectCtx, _detectCanvas, 'PERSON',
            s.ifBbox.x, s.ifBbox.y, s.ifBbox.w, s.ifBbox.h,
            s.score, _renderColorForBox(s),
            s.lockedAge, s.lockedGender, s.genderLocked, s.lockedName, s.nameLocked, s.pendingName, s.recogScore);
  }
  // Fully release ghosts that have been gone longer than GHOST_TTL
  Object.entries(ghostBoxes).forEach(([label, g]) => {
    if ((now - g.ghostTs) >= GHOST_TTL) {
      delete ghostBoxes[label];
      delete _savedIdentities[label];
      _freeColor(label);
    }
  });
}
requestAnimationFrame(_renderFrame);

// ── InsightFace polling — identity only, COCO-SSD owns the box position ──────
// InsightFace NEVER moves the box. It only writes name / gender / age into the
// existing smoothedBoxes entry that COCO-SSD created. This eliminates the
// jumpiness caused by two models giving different body-coord estimates.
let ifTimer = null;

function blendIdentityOnly(label, age, gender, name) {
  const prev = smoothedBoxes[label];
  if (!prev) return;
  blendSmoothed(label, prev.x, prev.y, prev.w, prev.h, prev.score, prev.color, age, gender, name);
}

function _nextPersonLabel(usedLabels = new Set()) {
  let i = 0;
  while (smoothedBoxes[`PERSON_${i}`] || ghostBoxes[`PERSON_${i}`] || usedLabels.has(`PERSON_${i}`)) i++;
  return `PERSON_${i}`;
}

function _computeHeadBbox(det, canvas, video) {
  const sx = canvas.width  / (det.fw || (video.videoWidth || video.naturalWidth));
  const sy = canvas.height / (det.fh || (video.videoHeight || video.naturalHeight));
  const [x1, y1, x2, y2] = det.bbox;
  const fw = x2 - x1, fh = y2 - y1;
  const pad = 0.12;
  const side = Math.max(fw, fh) * (1 + pad * 2);
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  return {
    x: Math.max(0, (cx - side / 2) * sx),
    y: Math.max(0, (cy - side / 2) * sy),
    w: Math.min(side * sx, canvas.width),
    h: Math.min(side * sy, canvas.height),
  };
}

function _faceToBodyBox(det, canvas, video) {
  const sx = canvas.width  / (det.fw || (video.videoWidth || video.naturalWidth));
  const sy = canvas.height / (det.fh || (video.videoHeight || video.naturalHeight));
  const [x1, y1, x2, y2] = det.bbox;
  const faceW = x2 - x1, faceH = y2 - y1;
  const bodyW  = faceW * FACE_TO_BODY_W;
  const bodyH  = faceH * FACE_TO_BODY_H;
  const bodyCx = (x1 + x2) / 2;
  const bodyX1 = Math.max(0, bodyCx - bodyW / 2);
  const bodyY1 = Math.max(0, y1 - faceH * HEAD_PAD * FACE_TO_BODY_H);
  const dx = bodyW * BBOX_SHRINK;
  const x = (bodyX1 + dx) * sx;
  const y = bodyY1 * sy;
  const w = Math.max(24, (bodyW - dx * 2) * sx);
  const h = Math.max(24, bodyH * sy);
  return { x, y, w, h, cx: x + w / 2, cy: y + h / 2 };
}

function _faceTrackScore(det, box, sx, sy) {
  if (!box) return Infinity;
  const fcx = det.cx * sx;
  const fcy = det.cy * sy;
  const bx = box.x, by = box.y, bw = box.w, bh = box.h;
  if (!bw || !bh) return Infinity;
  const xMin = bx - bw * 0.18;
  const xMax = bx + bw * 1.18;
  const yMin = by - bh * 0.12;
  const yMax = by + bh * 0.72;
  if (fcx < xMin || fcx > xMax || fcy < yMin || fcy > yMax) return Infinity;
  const nx = Math.abs(fcx - (bx + bw * 0.5)) / Math.max(bw * 0.42, 1);
  const ny = Math.abs(fcy - (by + bh * 0.24)) / Math.max(bh * 0.26, 1);
  return Math.hypot(nx, ny * 1.25);
}

function _upsertFaceFallbackBox(det, canvas, video, usedLabels = new Set()) {
  if (!video || !(video.videoWidth || video.naturalWidth)) return null;

  const body = _faceToBodyBox(det, canvas, video);
  const sx = canvas.width  / (det.fw || (video.videoWidth || video.naturalWidth));
  const sy = canvas.height / (det.fh || (video.videoHeight || video.naturalHeight));

  let bestLabel = null, bestScore = FACE_TRACK_MAX_SCORE;
  // Name-anchored migration: a known face always claims its existing track,
  // no matter how far it moved. Eliminates ghost duplicates on fast motion.
  const detName = _strongFaceName ? _strongFaceName(det) : (det.name || 'Unknown');
  if (detName && detName !== 'Unknown') {
    const detCx = ((det.bbox[0] + det.bbox[2]) / 2) * sx;
    const detCy = ((det.bbox[1] + det.bbox[3]) / 2) * sy;
    const MAX_NAME_JUMP = Math.min(canvas.width, canvas.height) * 0.40;
    const withinJump = (prev) => {
      if (!prev) return true;
      const pcx = prev.x + prev.w / 2;
      const pcy = prev.y + prev.h / 2;
      return Math.hypot(detCx - pcx, detCy - pcy) <= MAX_NAME_JUMP;
    };
    for (const lbl of personKeys(smoothedBoxes)) {
      if (usedLabels.has(lbl)) continue;
      if (_displayNameForBox(smoothedBoxes[lbl]) !== detName) continue;
      if (!withinJump(smoothedBoxes[lbl].ifBbox)) continue;
      bestLabel = lbl; bestScore = 0; break;
    }
    if (!bestLabel) {
      for (const lbl of personKeys(ghostBoxes)) {
        if (usedLabels.has(lbl)) continue;
        if (_displayNameForBox(ghostBoxes[lbl]) !== detName) continue;
        if (!withinJump(ghostBoxes[lbl].ifBbox)) continue;
        smoothedBoxes[lbl] = { ...ghostBoxes[lbl] };
        delete ghostBoxes[lbl];
        bestLabel = lbl; bestScore = 0; break;
      }
    }
  }
  if (!bestLabel) {
    for (const lbl of personKeys(smoothedBoxes)) {
      if (usedLabels.has(lbl)) continue;
      const s = smoothedBoxes[lbl];
      const score = _faceTrackScore(det, s, sx, sy);
      if (score < bestScore) { bestScore = score; bestLabel = lbl; }
    }
  }

  if (!bestLabel) {
    let ghostBest = null, ghostScore = FACE_TRACK_MAX_SCORE;
    for (const lbl of personKeys(ghostBoxes)) {
      if (usedLabels.has(lbl)) continue;
      const g = ghostBoxes[lbl];
      const score = _faceTrackScore(det, g, sx, sy);
      if (score < ghostScore) { ghostScore = score; ghostBest = lbl; }
    }
    if (ghostBest) {
      smoothedBoxes[ghostBest] = { ...ghostBoxes[ghostBest] };
      delete ghostBoxes[ghostBest];
      bestLabel = ghostBest;
    }
  }

  if (!bestLabel) {
    if ((det.score ?? 0) < FACE_SEED_MIN_SCORE) return null;
    if (_hasActiveDisplayName(det.name)) return null;
    bestLabel = _nextPersonLabel(usedLabels);
  }
  usedLabels.add(bestLabel);

  blendSmoothed(
    bestLabel,
    body.x, body.y, body.w, body.h,
    det.score,
    _assignColor(bestLabel),
    det.age ?? null, det.gender ?? null, det.name ?? null,
  );

  if (smoothedBoxes[bestLabel]) {
    smoothedBoxes[bestLabel].ifConfirmed   = true;
    smoothedBoxes[bestLabel].lastIfMatchTs = Date.now();
    smoothedBoxes[bestLabel].ifStrikes     = 0;
    smoothedBoxes[bestLabel].ifBbox        = _computeHeadBbox(det, canvas, video);
    smoothedBoxes[bestLabel].recogScore    = det.recog_score ?? null;
  }
  _markTrackObserved(bestLabel, 'face');
  return bestLabel;
}

async function pollInsightFace() {
  if (yoloMode !== 'off') return;
  if (!_detectCanvas) return;
  const canvas = _detectCanvas;
  try {
    const perCam = {};
    const activeVideo = activeVideoEl();
    let activeDets = [];

    if (playbackMode === 'upload') {
      if (!uploadSession || !activeVideo || activeVideo.readyState < 2 || !(activeVideo.videoWidth || activeVideo.naturalWidth)) return;
      const t = Number.isFinite(activeVideo.currentTime) ? activeVideo.currentTime : 0;
      const res = await fetch(
        '/detections/upload/' + uploadSession.id + '?t=' + encodeURIComponent(t.toFixed(3)),
        { cache: 'no-store' }
      ).then(r => r.ok ? r.json() : null).catch(() => null);
      activeDets = res?.detections?.filter(d => d.score >= 0.52) || [];
    } else {
      const activeCams = CAM_KEYS.filter(c => players[c]);
      const results = await Promise.all(
        activeCams.map(cam =>
          fetch('/detections/' + cam, { cache: 'no-store' })
            .then(r => r.ok ? r.json().then(d => ({ cam, d })) : null)
            .catch(() => null)
        )
      );
      results.forEach(res => {
        if (!res || !res.d?.detections?.length) return;
        perCam[res.cam] = res.d.detections.filter(d => d.score >= 0.52);
      });
      activeDets = perCam[currentCam] || [];
    }

    // Prioritise active cam. For each face from active cam, match to the nearest
    // tracked COCO box and attach identity. Fall back to other cams if active
    // cam returned no faces.
    const cocoLabels = personKeys(smoothedBoxes);

    if (activeDets.length && cocoLabels.length) {
      const video = activeVideo;
      if (video && (video.videoWidth || video.naturalWidth)) {
        const sx = canvas.width  / (activeDets[0]?.fw || (video.videoWidth || video.naturalWidth));
        const sy = canvas.height / (activeDets[0]?.fh || (video.videoHeight || video.naturalHeight));
        // Max distance a face center can be from a COCO box center to qualify as a match.
        // Prevents assigning identity to a box on the opposite side of the frame.
        const MAX_IF_MATCH = Math.min(canvas.width, canvas.height) * 0.25;
        const usedLabels    = new Set();
        const unmatchedDets = [];
        const knownWinners  = {};

        activeDets.forEach(det => {
          const strongName = _strongFaceName(det);
          if (!det || strongName === 'Unknown') return;
          const rank = det.recog_score ?? det.score ?? 0;
          const prev = knownWinners[strongName];
          const prevRank = prev ? (prev.recog_score ?? prev.score ?? 0) : -1;
          if (!prev || rank > prevRank) knownWinners[strongName] = det;
        });

        for (const det of activeDets) {
          const strongName = _strongFaceName(det);
          const safeName =
            strongName !== 'Unknown' && knownWinners[strongName] !== det
              ? 'Unknown'
              : strongName;
          let bestLabel = null, bestScore = FACE_TRACK_MAX_SCORE;
          // Name-anchored match: if this face has a known identity and that
          // identity already owns a track, migrate to that track — but cap the
          // jump distance so a bad recognition can't teleport the label across
          // the frame onto an unrelated person/object.
          if (safeName && safeName !== 'Unknown') {
            const detCx = ((det.bbox[0] + det.bbox[2]) / 2) * sx;
            const detCy = ((det.bbox[1] + det.bbox[3]) / 2) * sy;
            const MAX_NAME_JUMP = Math.min(canvas.width, canvas.height) * 0.40;
            for (const lbl of cocoLabels) {
              if (usedLabels.has(lbl)) continue;
              if (_displayNameForBox(smoothedBoxes[lbl]) !== safeName) continue;
              const s = smoothedBoxes[lbl];
              const prev = s.ifBbox;
              if (prev) {
                const pcx = prev.x + prev.w / 2;
                const pcy = prev.y + prev.h / 2;
                const dist = Math.hypot(detCx - pcx, detCy - pcy);
                if (dist > MAX_NAME_JUMP) continue;
              }
              bestLabel = lbl;
              bestScore = 0;
              break;
            }
          }
          if (!bestLabel) {
            for (const lbl of cocoLabels) {
              if (usedLabels.has(lbl)) continue;
              const s = smoothedBoxes[lbl];
              const score = _faceTrackScore(det, s, sx, sy);
              if (score < bestScore) { bestScore = score; bestLabel = lbl; }
            }
          }
          if (bestLabel && bestScore < FACE_TRACK_MAX_SCORE) {
            usedLabels.add(bestLabel);
            smoothedBoxes[bestLabel].ifConfirmed   = true;
            smoothedBoxes[bestLabel].lastIfMatchTs = Date.now();
            smoothedBoxes[bestLabel].ifStrikes     = 0;
            smoothedBoxes[bestLabel].ifBbox        = _computeHeadBbox(det, canvas, video);
            smoothedBoxes[bestLabel].recogScore    = det.recog_score ?? null;
            blendIdentityOnly(bestLabel, det.age ?? null, det.gender ?? null, safeName ?? null);
            _markTrackObserved(bestLabel, 'face');
          } else {
            unmatchedDets.push({ ...det, name: safeName });
          }
        }

        // If InsightFace sees extra faces that COCO didn't turn into their own body boxes,
        // synthesize a fallback body box so the second person can still get a label/title.
        unmatchedDets.forEach(det => {
          if (_hasActiveDisplayName(det.name)) return;
          _upsertFaceFallbackBox(det, canvas, video, usedLabels);
        });

        // Dual-model gate with rolling confirmation window.
        //
        // A box is "live-confirmed" if InsightFace matched a face there within the
        // last IF_LIVE_MS. Live-confirmed boxes are immune to strikes — person may
        // just be facing away temporarily.
        //
        // When another box WAS just confirmed this poll, competing boxes get fast
        // strikes (IF_STRIKES_COMPETE × 300ms ≈ 1.5s). This rapidly clears the
        // stale box created by a camera pan without waiting for a long timeout.
        //
        // Single-box frames are always immune — no competition, no false positive risk.
        const IF_LIVE_MS          = 6000;   // face match stays valid for 6s
        const IF_STRIKES_NORMAL   = 15;     // ~4.5s — initial confirmation window
        const IF_STRIKES_COMPETE  = 5;      // ~1.5s — when another box just confirmed
        const anyJustConfirmed    = usedLabels.size > 0;
        const multiBox            = cocoLabels.length > 1;

        cocoLabels.forEach(lbl => {
          const box = smoothedBoxes[lbl];
          if (!box) return;
          // Only one box in frame → immune (person may be facing away)
          if (!multiBox) { box.ifStrikes = 0; return; }

          const freshMatch  = box.lastIfMatchTs && (Date.now() - box.lastIfMatchTs) < IF_LIVE_MS;
          // If another box was JUST confirmed this poll, this box must compete —
          // even a previously-confirmed box gets fast-struck so a camera pan clears quickly.
          // If no box was confirmed this poll, give the normal grace period.
          if (freshMatch && !anyJustConfirmed) return;  // immune: face was recent, no competition

          const threshold = anyJustConfirmed ? IF_STRIKES_COMPETE : IF_STRIKES_NORMAL;
          const strikes   = (box.ifStrikes || 0) + 1;
          if (strikes >= threshold) {
            ghostBoxes[lbl] = { ...box, ghostTs: Date.now() - (GHOST_TTL - 2000) };
            delete smoothedBoxes[lbl];
          } else {
            box.ifStrikes = strikes;
          }
        });
        return;
      }
    }

    // No COCO boxes yet, or active cam has no faces — let InsightFace seed body boxes directly.
    if (!cocoLabels.length) {
      const fallbackDets = playbackMode === 'upload'
        ? activeDets
        : activeDets.length
        ? activeDets
        : Object.entries(perCam)
            .filter(([cam]) => cam !== currentCam)
            .flatMap(([, dets]) => dets);
      if (!fallbackDets.length) return;

      const video = activeVideo;
      if (!video || !(video.videoWidth || video.naturalWidth)) return;

      const usedLabels = new Set();
      fallbackDets
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .forEach(det => {
          _upsertFaceFallbackBox(det, canvas, video, usedLabels);
        });
    }
  } catch (e) {
    console.warn('[insightface]', e);
  }
}

// ── Bootstrap ────────────────────────────────────────────────────────────────
async function bootstrapPlayers() {
  const d = await fetchStateSnapshot();
  if (d) {
    currentCam = d.active || currentCam;
    CAM_KEYS.forEach(c => {
      if (d.cameras && d.cameras[c]) camMode[c] = d.cameras[c].mode || 'standard';
    });
    currentMode = camMode[currentCam] || 'standard';
  }
  showCam(currentCam);

  // Start MJPEG players for every configured cam so both RTSP feeds are warm
  // in the background — that's what makes cam switches feel instant.
  CAM_KEYS.forEach(cam => {
    const available = d && d.cameras && d.cameras[cam] && d.cameras[cam].available;
    if (available) initPlayerForCam(cam);
  });

  const active = d && d.cameras && d.cameras[currentCam];
  if (active) applyStatus(active.status || 'connecting');

  startStatePolling();
  _applyDetectionForMode(currentMode);
  if (detectEnabled) _pipelineStart();
}
bootstrapPlayers();

// ── Fullscreen ───────────────────────────────────────────────────────────────
const fsBtn      = document.getElementById('fs-btn');
const fsExpand   = document.getElementById('fs-icon-expand');
const fsCompress = document.getElementById('fs-icon-compress');
const fsLabel    = document.getElementById('fs-label');
const fsTarget   = document.getElementById('stage'); // fullscreen the video stage

function updateFsButton() {
  const isFs = !!document.fullscreenElement;
  fsExpand.style.display   = isFs ? 'none'  : '';
  fsCompress.style.display = isFs ? ''      : 'none';
  fsLabel.textContent      = isFs ? 'EXIT FULLSCREEN' : 'FULLSCREEN';
  fsBtn.classList.toggle('active', isFs);
}

fsBtn.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    fsTarget.requestFullscreen().catch(e => console.warn('fullscreen:', e));
  } else {
    document.exitFullscreen();
  }
});

document.addEventListener('fullscreenchange', () => {
  updateFsButton();
  // Resize canvas/stage to fill screen when entering fullscreen
  setTimeout(fitStage, 100);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    import json

    class _Fmt(argparse.RawDescriptionHelpFormatter):
        def __init__(self, *a, **kw):
            super().__init__(*a, max_help_position=28, width=100, **kw)

    parser = argparse.ArgumentParser(
        description="RTSP dual-cam viewer with AI detection",
        formatter_class=_Fmt,
        epilog="""
Camera management:
  python3 live_web.py --add-cam home 192.168.1.10 admin mypass
  python3 live_web.py --add-cam home 192.168.1.10 admin mypass --port 8554 --path-sd /live/main --path-hd /live/sub
  python3 live_web.py --remove-cam home
  python3 live_web.py --list-cams

Person management:
  python3 live_web.py --add-person Alice --color cyan
  python3 live_web.py --remove-person Alice
  python3 live_web.py --list-persons

Other:
  python3 live_web.py --set-family-dir /path/to/photos
  python3 live_web.py --edit-config
""",
    )
    parser.add_argument("--stop", action="store_true",
                        help="Stop the running server (kills process on port 8766)")
    parser.add_argument("--restart", action="store_true",
                        help="Restart server in background, log to /tmp/live_web.log")
    parser.add_argument("--background", action="store_true",
                        help=argparse.SUPPRESS)  # internal flag — used by --restart
    parser.add_argument("--rebuild-faces", action="store_true",
                        help="Re-extract embeddings from face photos and exit")
    parser.add_argument("--set-family-dir", metavar="PATH",
                        help="Set the face photos directory in config.json (permanent)")
    parser.add_argument("--family-dir", metavar="PATH", type=str, default=None,
                        help="Override face photos directory for this run only (does not update config.json)")

    # Camera management
    parser.add_argument("--add-cam", nargs="+", metavar=("NAME", "HOST"),
                        help="Add/update a camera: NAME HOST [USER] [PASSWORD]")
    parser.add_argument("--port", type=int, default=CAM_DEFAULT_PORT,
                        help=f"RTSP port for --add-cam (default: {CAM_DEFAULT_PORT})")
    parser.add_argument("--path-sd", default=CAM_DEFAULT_PATH_SD,
                        help=f"SD stream path for --add-cam (default: {CAM_DEFAULT_PATH_SD})")
    parser.add_argument("--path-hd", default=CAM_DEFAULT_PATH_HD,
                        help=f"HD stream path for --add-cam (default: {CAM_DEFAULT_PATH_HD})")
    parser.add_argument("--remove-cam", metavar="NAME",
                        help="Remove a camera from config.json")
    parser.add_argument("--list-cams", action="store_true",
                        help="List all configured cameras")

    # Person management
    parser.add_argument("--add-person", metavar="NAME",
                        help="Add or update a person's box color in config.json")
    parser.add_argument("--color", metavar="COLOR",
                        help="CSS color for --add-person (name, hex, rgb, hsl)")
    parser.add_argument("--remove-person", metavar="NAME",
                        help="Remove a person from config.json")
    parser.add_argument("--list-persons", action="store_true",
                        help="List all persons and colors in config.json")
    parser.add_argument("--set-box-size", type=int, metavar="PX",
                        help="Set detection box size in SD pixels (default 73, larger = bigger box)")
    parser.add_argument("--set-box-offset-x", type=int, metavar="PX",
                        help="Shift box left/right in SD pixels (negative = left, positive = right, default 0)")
    parser.add_argument("--set-box-offset-y", type=int, metavar="PX",
                        help="Shift box up/down in SD pixels (negative = up, positive = down, default 0)")
    parser.add_argument("--edit-config", action="store_true",
                        help="Open config.json in $EDITOR (validates on save)")
    parser.add_argument("--reset-config", action="store_true",
                        help="Reset config.json to default template (deletes all current settings)")

    args = parser.parse_args()

    # ── Server control ────────────────────────────────────────────────────────

    if args.stop:
        _kill_server()
        raise SystemExit(0)

    if args.restart:
        _placeholders = _check_placeholder_config(_cfg)
        if _placeholders:
            print(_clr("ERROR: ", "1;31") + "config.json still has example values. Update before starting:")
            for _p in _placeholders: print("  " + _clr(_p.strip(), "1;33"))
            print("\n  Run: " + _clr("python3 live_web.py --edit-config", "1;36"))
            raise SystemExit(1)
        _kill_server()
        _start_background()
        _tail_until_live()
        raise SystemExit(0)

    # ── Config sub-commands (all exit after completion) ───────────────────────

    if args.list_cams:
        cfg = _load_config()
        cameras = cfg.get("cameras", {})
        if not cameras:
            print("No cameras configured.")
        else:
            print(f"{'NAME':<16} {'HOST':<18} {'PORT':<6} {'USER':<16} {'PATH_SD':<12} PATH_HD")
            print("-" * 80)
            for name, c in cameras.items():
                port    = c.get("port", CAM_DEFAULT_PORT)
                path_sd = c.get("path_sd", CAM_DEFAULT_PATH_SD)
                path_hd = c.get("path_hd", CAM_DEFAULT_PATH_HD)
                print(f"  {name:<14} {c.get('host',''):<18} {port:<6} {c.get('user',''):<16} {path_sd:<12} {path_hd}")
        raise SystemExit(0)

    if args.add_cam:
        parts = args.add_cam
        if len(parts) < 2:
            print(_clr("ERROR: ", "1;31") + "--add-cam requires at least NAME and HOST")
            print("  Usage: --add-cam NAME HOST [USER] [PASSWORD]")
            raise SystemExit(1)
        cam_name = parts[0]
        cam_host = parts[1]
        cam_user = parts[2] if len(parts) > 2 else ""
        cam_pass = parts[3] if len(parts) > 3 else ""
        cfg = _load_config()
        entry: dict = {
            "host": cam_host, "user": cam_user, "password": cam_pass,
            "port": args.port,
            "path_sd": args.path_sd,
            "path_hd": args.path_hd,
        }
        cfg.setdefault("cameras", {})[cam_name] = entry
        _strip_placeholder_cams(cfg)
        _save_config(cfg)
        extras = ""
        if args.port != CAM_DEFAULT_PORT: extras += f"  port={args.port}"
        if args.path_sd != CAM_DEFAULT_PATH_SD: extras += f"  path_sd={args.path_sd}"
        if args.path_hd != CAM_DEFAULT_PATH_HD: extras += f"  path_hd={args.path_hd}"
        print(f"Added camera: {cam_name}  {cam_host}  user={cam_user or '(none)'}{extras}")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.remove_cam:
        cfg = _load_config()
        cameras = cfg.get("cameras", {})
        if args.remove_cam not in cameras:
            print(f"'{args.remove_cam}' not found in config.json")
            raise SystemExit(1)
        if len(cameras) == 1:
            print(_clr("WARNING: ", "1;33") + "this is the only camera — removing it will leave no cameras configured.")
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            if confirm != "yes":
                print("Aborted.")
                raise SystemExit(0)
        del cameras[args.remove_cam]
        _save_config(cfg)
        print(f"Removed camera: {args.remove_cam}")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.list_persons:
        cfg = _load_config()
        fdir = cfg.get("family_dir", "family")
        colors = cfg.get("identity_colors", {})
        print(f"family_dir : {fdir}")
        if colors:
            print("persons    :")
            for name, color in colors.items():
                print(f"  {name:20s}  {color}")
        else:
            print("persons    : (none)")
        raise SystemExit(0)

    if args.set_box_size is not None:
        if args.set_box_size < 10 or args.set_box_size > 400:
            print(_clr("ERROR: ", "1;31") + "box_size must be between 10 and 400")
            raise SystemExit(1)
        cfg = _load_config()
        cfg["box_size"] = args.set_box_size
        _save_config(cfg)
        print(f"box_size set to: {args.set_box_size}px  (ratio: {args.set_box_size}/1280 = {args.set_box_size/1280:.4f})")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.set_box_offset_x is not None:
        if args.set_box_offset_x < -500 or args.set_box_offset_x > 500:
            print(_clr("ERROR: ", "1;31") + "box_offset_x must be between -500 and 500")
            raise SystemExit(1)
        cfg = _load_config()
        cfg["box_offset_x"] = args.set_box_offset_x
        _save_config(cfg)
        direction = "right" if args.set_box_offset_x > 0 else "left" if args.set_box_offset_x < 0 else "center"
        print(f"box_offset_x set to: {args.set_box_offset_x}px ({direction})")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.set_box_offset_y is not None:
        if args.set_box_offset_y < -500 or args.set_box_offset_y > 500:
            print(_clr("ERROR: ", "1;31") + "box_offset_y must be between -500 and 500")
            raise SystemExit(1)
        cfg = _load_config()
        cfg["box_offset_y"] = args.set_box_offset_y
        _save_config(cfg)
        direction = "down" if args.set_box_offset_y > 0 else "up" if args.set_box_offset_y < 0 else "center"
        print(f"box_offset_y set to: {args.set_box_offset_y}px ({direction})")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.set_family_dir:
        cfg = _load_config()
        cfg["family_dir"] = args.set_family_dir
        _save_config(cfg)
        print(f"family_dir set to: {args.set_family_dir}")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.add_person:
        if not args.color:
            print(_clr("ERROR: ", "1;31") + "--add-person requires --color")
            raise SystemExit(1)
        color = args.color.strip()
        if not _valid_color(color):
            print(_clr("ERROR: ", "1;31") + f"'{color}' is not a valid CSS color.")
            print("Use a CSS name (e.g. cyan, hotpink), hex (#00e5ff), rgb(), or hsl().")
            raise SystemExit(1)
        cfg = _load_config()
        person_dir = FAMILY_DIR / args.add_person
        if not person_dir.exists():
            person_dir.mkdir(parents=True)
            print(f"Created folder: {person_dir}")
            print(f"  Add face photos there, then run: python3 live_web.py --rebuild-faces")
        else:
            photos = [f for f in person_dir.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
            if not photos:
                print(_clr("WARNING: ", "1;33") + f"{person_dir} exists but has no photos — add images and run " + _clr("--rebuild-faces", "1;36"))
        cfg.setdefault("identity_colors", {})[args.add_person] = color
        _strip_placeholder_persons(cfg)
        _save_config(cfg)
        print(f"Added: {args.add_person} -> {color}")
        raise SystemExit(0)

    if args.remove_person:
        cfg = _load_config()
        colors = cfg.setdefault("identity_colors", {})
        if args.remove_person not in colors:
            print(f"'{args.remove_person}' not found in config.json")
            raise SystemExit(1)
        del colors[args.remove_person]
        _save_config(cfg)
        print(f"Removed: {args.remove_person}")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.reset_config:
        print(_clr("WARNING: ", "1;33") + "This will DELETE all current config (cameras, persons, colors, settings).")
        confirm = input("Type 'yes' to confirm reset: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            raise SystemExit(0)
        default = _default_config()
        _save_config(default)
        print("config.json reset to default template.")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    if args.edit_config:
        import subprocess, tempfile, shutil as _shutil
        editor = os.environ.get("EDITOR") or _shutil.which("nano") or _shutil.which("vi") or "vi"
        editor_args = editor.split()  # handle "subl -w" style editors
        # Write current config to a temp file so we can validate before overwriting
        cfg_text = _CONFIG_FILE.read_text() if _CONFIG_FILE.exists() else json.dumps(
            {"family_dir": "family", "identity_colors": {}}, indent=2)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(cfg_text)
            tmp_path = tmp.name
        subprocess.run(editor_args + [tmp_path])
        # Validate
        try:
            new_cfg = json.loads(Path(tmp_path).read_text())
        except json.JSONDecodeError as e:
            print(_clr("ERROR: ", "1;31") + f"JSON syntax error — {e}")
            print("Config NOT saved.")
            Path(tmp_path).unlink(missing_ok=True)
            raise SystemExit(1)
        errors = _validate_config(new_cfg)
        if errors:
            print(_clr("ERROR: ", "1;31") + "invalid color values in edited config:")
            for e in errors:
                print(e)
            print("Config NOT saved. Re-run --edit-config to fix.")
            Path(tmp_path).unlink(missing_ok=True)
            raise SystemExit(1)
        _CONFIG_FILE.write_text(json.dumps(new_cfg, indent=2) + "\n")
        Path(tmp_path).unlink(missing_ok=True)
        print("Config saved.")
        if _server_running():
            print("Restart to apply: python3 live_web.py --restart")
        raise SystemExit(0)

    # ── Normal server startup ─────────────────────────────────────────────────

    if args.family_dir:
        globals()["FAMILY_DIR"] = Path(args.family_dir)

    if args.rebuild_faces:
        print("Building face database from:", FAMILY_DIR)
        fa = _get_face_app()
        if fa is None:
            print(_clr("ERROR: ", "1;31") + "InsightFace not available.")
        else:
            print(f"Loaded {len(_face_db)} person(s):")
            for name, embs in _face_db.items():
                print(f"  {name}: {len(embs)} embedding(s)")
    elif args.background:
        # Running as background server process — start uvicorn directly
        try:
            _install_signal_handlers()
            uvicorn.run(app, host="0.0.0.0", port=_SERVER_PORT, log_level="warning", access_log=False)
        except KeyboardInterrupt:
            pass
        finally:
            _join_camera_threads(total_timeout=1.0)
            _silence_stderr()
    else:
        _placeholders = _check_placeholder_config(_cfg)
        if _placeholders:
            print(_clr("ERROR: ", "1;31") + "config.json still has example values. Update before starting:")
            for _p in _placeholders: print("  " + _clr(_p.strip(), "1;33"))
            print("\n  Run: " + _clr("python3 live_web.py --edit-config", "1;36"))
            raise SystemExit(1)
        if _server_running():
            _kill_server()
        _start_background()
        _tail_until_live()
        raise SystemExit(0)
