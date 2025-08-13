### Ball-Centered Reframing (16:9 → 9:16)

Keeps the ball centered in vertical video, even when it moves fast and appears blurry. Powered by YOLOv8, prediction-guided search, blur template learning, tiled fallback detection, ByteTrack, optical flow bridging, and adaptive smoothing.

### Features

- Three-pass pipeline: learn → detect-every-frame → render
- Prediction-guided ROI/full-frame/tiled detection for fast motion
- Blur Template Bank: learns blurred appearances to re-detect mid-video
- Tiled detection fallback for tiny/blurred balls
- Optical flow bridging for short misses
- Strict centering with border safety to keep the ball in-frame near edges
- Adaptive smoothing for stable and responsive output

### Requirements

- Python 3.9+ and ffmpeg
- Node.js 18+ and npm
- macOS (MPS), Linux/Windows (CUDA) or CPU-only

### Start the project (one-time setup)

```bash
cd /Users/hoangdo/Own/upwork/ball-centered

# 1) Python env
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# 2) Node CLI
npm install
npm run build
```

### Quick Start (Max Accuracy)

- macOS with Apple GPU (MPS):

```bash
node dist/cli.js \
  -i /Users/hoangdo/Own/upwork/ball-centered/input.mp4 \
  -o /Users/hoangdo/Own/upwork/ball-centered/output.mp4 \
  --max-accuracy \
  --device mps \
  --use-ffmpeg
```

- NVIDIA GPU:

```bash
node dist/cli.js -i /abs/path/input.mp4 -o /abs/path/output.mp4 \
  --max-accuracy --device cuda --use-ffmpeg
```

- CPU-only (slower):

```bash
node dist/cli.js -i input.mp4 -o output.mp4 --max-accuracy --device cpu
```

### Best example command (most accurate and stable)

```bash
node dist/cli.js -i input.mp4 -o output.mp4 \
  --max-accuracy \
  --device mps \
  --use-ffmpeg \
  --target-class "tennis ball"
```

- Omit `--target-class` if unknown.

### Presets

- **--max-accuracy**: Enables three-pass, per-frame detection, prediction-guided search, blur templates, tiled fallback, strict-centering, imgsz 1536, low conf (0.12), and ffmpeg pipe.
- **--quality**: yolov8x, imgsz 960, smoothing and profiling enabled.
- **--fast**: yolov8n, aggressive intervals and ffmpeg pipe.

### Parameters reference

Required

- **-i, --input <path>**: Input video path.
- **-o, --output <path>**: Output video path.

Device/Model

- **--device cpu|mps|cuda**: Select inference device (CPU, Apple MPS, or NVIDIA CUDA).
- **--model <path|name>**: YOLO model (e.g., `yolov8x.pt`).
- **--imgsz <int>**: YOLO input size (e.g., `960`/`1280`/`1536`). Larger = more accurate on tiny/blurred balls.
- **--conf <float>**: Detection confidence (lower finds smaller/blurry balls). Typical: `0.12–0.25`.
- **--backend yolo|yolo-bytetrack**: Tracking backend (used when not forcing per-frame detection).
- **--tracker auto|mosse|kcf|csrt**: OpenCV tracker type (non full-detect mode).

Pipeline control

- **--max-accuracy**: Turn on all high-accuracy features and sensible defaults (recommended).
- **--three-pass**: Learn appearance → detect-every-frame → render.
- **--full-detect**: Force per-frame full-frame detection.
- **--detect-every-frame**: Always run detector (ignores tracker interval).
- **--det-interval <int>**: Baseline detection interval when using YOLO+tracker.
- **--no-tta-recovery**: Disable heavier recovery (keep disabled for max recovery).
- **--profile**: Periodic performance stats.

Blur/tiny ball robustness

- **--tiled-detect**: Enable tiled detection fallback.
- **--tile-size <int>**: Tile size in pixels (e.g., `512`).
- **--tile-overlap <int>**: Overlap in pixels (e.g., `128`).
- Blur Template Bank: learned automatically from accepted detections.

Centering/smoothing

- **--strict-center**: Center on the ball; relax near borders to keep it in frame.
- **--sticky-window**: Keep previous center until ball leaves a small bound.
- **--center-bound <int>**: Half-width (px) for sticky mode (default `40`).
- **--smooth <int>**: Trajectory smoothing window (typical: `15–21`).
- Motion constraints (used when not strict-center):
  - **--max-move <int>**: Max crop center shift per frame.
  - **--max-accel <int>**: Max change per frame (acceleration clamp).
  - **--deadband <int>**: Ignore tiny movements.
  - **--jerk <int>**: Limit change in acceleration per frame.
- **--margin <int>**: Horizontal safety margin to keep ball inside crop.

Detection targets/ROI

- **--target-class <name>**: Force specific class (e.g., `"tennis ball"`).
- **--no-appearance**: Disable color histogram re-scoring.
- **--roi, --roi-size <int>**: Base ROI width for ROI detection.
- **--bootstrap-frames <int>**: Initial frames to scan for appearance learning.

Output

- **--use-ffmpeg**: Use ffmpeg pipe (hardware encoder where available).
- **--height <int>**: Output height (default: input height).
- **--fps <number>**: Output FPS (default: input FPS).

Misc

- **--quality**, **--fast**: Alternative presets.
- **--verbose** (reserved), **--dry-run** (reserved): For future logging/simulation.

### Tips

- Ball is tiny/very blurry: add `--imgsz 1536 --conf 0.12 --tiled-detect --tile-size 512 --tile-overlap 128`.
- Too jittery: increase `--smooth` to 17 or 21.
- Ball near edges cropped: increase `--margin 80`.
- Known sport/ball: add `--target-class "tennis ball"`.

### Troubleshooting

- Ultralytics not found: activate venv and `pip install -r requirements.txt`.
- MPS/CUDA not used: set `--device mps|cuda|cpu` explicitly.
- Still missing frames mid-video: ensure `--max-accuracy` or enable `--tiled-detect` and increase `--imgsz`.

### License

MIT
