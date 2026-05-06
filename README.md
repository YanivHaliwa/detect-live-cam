# RTSP Dual-Cam Viewer

Live multi-camera web viewer with AI body detection, face recognition, and identity tracking.

## Features

- **Multi-camera RTSP streaming** — up to N cameras in parallel, instant switching with no reconnect delay
- **MJPEG video feed** — low-latency stream served directly from the server
- **YOLO-pose + BoT-SORT tracking** — body detection and stable person tracking across frames
- **InsightFace identity recognition** — names known people from a reference photo database
- **Animated overlay boxes** — canvas drawn at ~25fps, detection pipeline at ~10fps
- **Background detection on inactive cams** — all cameras run face detection even when not displayed; results appear in logs
- **HD / SD switching** — toggle between high-resolution and standard streams per camera
- **Webcam / test video mode** — run the pipeline on a local webcam or uploaded video clip
- **Configurable identity strictness** — tune how many strong face reads are required before locking a name
- **Configurable detection sensitivity** — tune YOLO confidence threshold to reduce false positives
- **High-confidence grace period** — names locked at ≥80% are held for 60 s while re-evaluation continues
- **Track-swap protection** — stale identity locks expire automatically; spatial conflict detection busts wrong assignments
- **Duplicate name guard** — the same name can never appear on two boxes simultaneously
- **Monitor log** — compact event log (detection events with confidence %) written separately from the server log

---

## Requirements

### System packages

```bash
sudo apt install libgl1 libglib2.0-0 libgomp1
```

These are required by OpenCV and ONNX Runtime. Most Linux desktop systems already have them.

### Python packages

```bash
pip3 install -r requirements.txt --break-system-packages
```

### YOLO model

Download the pose model and place it in the project directory:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

---

## Setup

1. Copy the example config and fill in your camera details:
   ```bash
   cp config.example.json config.json
   ```

2. Add face reference photos (one subfolder per person):
   ```
   family/
     Alice/
       photo1.jpg
       photo2.jpg
     Bob/
       front.png
   ```

3. Build the face database:
   ```bash
   python3 live_web.py --rebuild-faces
   ```

4. Start the server:
   ```bash
   python3 live_web.py --restart
   ```
   Then open `http://localhost:8766/`.

---

## Configuration (`config.json`)

All settings live in `config.json`. Copy from `config.example.json` to start.

| Key | Default | Description |
|---|---|---|
| `family_dir` | `"family"` | Path to face reference photos (relative or absolute) |
| `identity_strictness` | `1` | 1–10: how many strong face reads needed to lock a name. 1 = fast/loose, 10 = strict |
| `detection_sensitivity` | `7` | 1–10: YOLO body detection sensitivity. 1 = catches everything, 10 = only clear detections |
| `monitor_log` | `"cam_monitor.log"` | Path for the compact detection event log. Relative paths resolve from the project folder |
| `server_log` | `"/tmp/live_web.log"` | Path for the full server log |
| `box_size` | `90` | Head box size in pixels (SD frame reference) |
| `box_offset_x` | `15` | Horizontal offset of the head box in pixels |
| `box_offset_y` | `0` | Vertical offset of the head box in pixels |
| `identity_colors` | `{}` | Per-person box colors. Accepts CSS names, hex, rgb(), hsl() |

### Camera entries (`cameras`)

Each key under `cameras` is the camera's display name:

```json
"cameras": {
  "cam1": {
    "host": "192.168.1.100",
    "user": "admin",
    "password": "yourpassword",
    "port": 554,
    "path_sd": "/stream2",
    "path_hd": "/stream1"
  }
}
```

Cameras can also be configured via environment variables (`CAM1_HOST`, `CAM1_USER`, `CAM1_PASSWORD`, `CAM2_HOST`, etc.) which override `config.json`.

---

## CLI Commands

Full help:

```
$ python3 live_web.py --help

usage: live_web.py [-h] [--stop] [--restart] [--rebuild-faces] [--set-family-dir PATH]
                   [--family-dir PATH] [--add-cam NAME [HOST ...]] [--port PORT]
                   [--path-sd PATH_SD] [--path-hd PATH_HD] [--remove-cam NAME]
                   [--list-cams] [--add-person NAME] [--color COLOR]
                   [--remove-person NAME] [--list-persons] [--set-box-size PX]
                   [--set-box-offset-x PX] [--set-box-offset-y PX]
                   [--edit-config] [--reset-config]

options:
  --stop                    Stop the running server
  --restart                 Restart server in background
  --rebuild-faces           Re-extract embeddings from face photos and exit
  --add-cam NAME HOST ...   Add/update a camera: NAME HOST [USER] [PASSWORD]
  --remove-cam NAME         Remove a camera from config.json
  --list-cams               List all configured cameras
  --add-person NAME         Add or update a person's box color
  --color COLOR             CSS color for --add-person (name, hex, rgb, hsl)
  --remove-person NAME      Remove a person from config.json
  --list-persons            List all persons and their colors
  --set-box-size PX         Set detection box size in SD pixels
  --set-box-offset-x PX     Shift box left/right in SD pixels
  --set-box-offset-y PX     Shift box up/down in SD pixels
  --set-family-dir PATH     Set face photos directory in config.json
  --edit-config             Open config.json in $EDITOR (validates on save)
  --reset-config            Reset config.json to defaults
```

### Common examples

```bash
# Start / stop / restart
python3 live_web.py --restart
python3 live_web.py --stop

# Rebuild face database after adding/changing photos
python3 live_web.py --rebuild-faces

# List cameras
$ python3 live_web.py --list-cams
NAME             HOST               PORT   USER             PATH_SD      PATH_HD
--------------------------------------------------------------------------------
  cam1           192.168.1.100      554    admin            /stream2     /stream1
  cam2           192.168.1.101      554    admin            /stream2     /stream1

# Add / remove a camera
python3 live_web.py --add-cam office 192.168.1.50 admin mypass
python3 live_web.py --add-cam office 192.168.1.50 admin mypass --port 8554 --path-sd /live/main --path-hd /live/sub
python3 live_web.py --remove-cam office

# List persons and colors
$ python3 live_web.py --list-persons
family_dir : family
persons    :
  Alice                 deepskyblue
  Bob                   hotpink

# Add / remove a person
python3 live_web.py --add-person Alice --color cyan
python3 live_web.py --add-person Bob --color '#ff4dd2'
python3 live_web.py --remove-person Alice

# Tune box position if head box is misaligned
python3 live_web.py --set-box-size 80
python3 live_web.py --set-box-offset-x 10
python3 live_web.py --set-box-offset-y -5

# Edit config directly
python3 live_web.py --edit-config
```

---

## Log Files

### Server log (`server_log`)

Full server output — camera connection events, model loading, recognition events:

```
[10:41:34] HOME  LIVE  (drained 14 stale frames)
[10:41:36] HOME2 LIVE  (drained 17 stale frames)
  [10:42:11]  Alice match 87%  →  HOME
  [10:44:03]  Bob match 82%  →  HOME
  [10:47:11]  Alice match 87%  →  HOME
```

### Monitor log (`monitor_log`)

Compact event log — one line per detection change, with confidence:

```
[10:41:34] STATE | cam: HOME (SD) | hd: off | detect: on
[10:42:11] DETECT | home: Alice (87%)
[10:44:03] DETECT | home: Bob (82%), Alice (87%)
[10:44:58] DETECT | home: Alice (87%)
[10:52:01] DETECT | home2: Bob (79%)
```

A new DETECT line is written only when the set of visible named people changes.

---

## Identity Tuning

Two config knobs control recognition accuracy:

**`identity_strictness` (1–10)**

Controls how many consistent strong face reads are needed before a name is locked onto a track.

| Value | Votes needed | Min score | Min margin |
|---|---|---|---|
| 1 | 1 | 44% | 5% |
| 5 | 3 | 50% | 9% |
| 10 | 5 | 58% | 15% |

Use a lower value in well-lit environments with clear frontal faces. Use a higher value if you see wrong names appearing.

**`detection_sensitivity` (1–10)**

Controls YOLO's confidence threshold for body detection.

| Value | YOLO conf | Effect |
|---|---|---|
| 1 | 0.30 | Catches partially visible / distant people; more false positives |
| 5 | 0.43 | Balanced |
| 7 | 0.50 | Recommended for indoor cameras |
| 10 | 0.60 | Only clear, unoccluded detections |

---

## Face Database

Place reference photos in subfolders under `family_dir`. The subfolder name is the displayed identity:

```
family/
  Alice/
    alice_front.jpg
    alice_side.jpg
  Bob/
    bob_001.png
```

After adding or changing photos, rebuild the database:

```bash
python3 live_web.py --rebuild-faces
```

Multiple photos per person improves accuracy. Clear frontal photos in similar lighting to the camera give the best results.

---

## API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Main web UI |
| `/mjpeg/{cam}` | GET | MJPEG video stream |
| `/state` | GET | Camera status JSON |
| `/switch/{cam}` | POST | Switch active camera or change HD/SD mode |
| `/pipeline/{cam}` | GET | YOLO + face recognition tracks for current frame |
| `/detection/{state}` | POST | Enable or disable detection (`on` / `off`) |
| `/upload-video` | POST | Upload a test video clip |

---

Created by [Yaniv Haliwa](https://github.com/YanivHaliwa) for security testing
