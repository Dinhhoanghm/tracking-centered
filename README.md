# Ball-Centered Reframing (16:9 → 9:16)

Node.js CLI that reframes horizontal sports footage to vertical while keeping the ball centered. Backed by Ultralytics YOLOv8 with ByteTrack, multi-tracker fallback, Kalman smoothing, and motion-constrained crop planning.

## Quickstart

```bash
# 1) Install Python deps in a virtualenv (Mac/Linux):
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# 2) Install Node deps and build CLI
npm install && npm run build

# 3) Run on a short clip (20s recommended)
node dist/cli.js -i input.mp4 -o output.mp4 --quality
# or use the bin after build
ball-reframe -i input.mp4 -o output.mp4 --quality
```

## Recommended presets

- Fast: `--fast` uses `yolov8n.pt`, aggressive detection interval, and ffmpeg pipe.
- Quality: `--quality` uses `yolov8x.pt`, higher `imgsz`, stronger smoothing, and profiling.

Examples:

```bash
ball-reframe -i game.mp4 -o game_vertical.mp4 --fast
ball-reframe -i highlights.mp4 -o highlights_vertical.mp4 --quality --imgsz 960
```

## Three-pass, strict-centering workflow (ball always centered)

```bash
# Pass 1: learn appearance and class across the whole video
# Pass 2: detect the ball on every frame (high accuracy)
# Pass 3: render 9:16 with ball strictly in the center, smooth and stable
ball-reframe -i input.mp4 -o output.mp4 \
  --three-pass \
  --full-detect \
  --strict-center \
  --backend yolo-bytetrack \
  --imgsz 960 \
  --det-interval 1 \
  --use-ffmpeg \
  --profile
```

Tips:

- For very fast balls, consider `--imgsz 960` and `--det-interval 1`.
- You can enforce a class with `--target-class "tennis ball"`.

## Docker

```bash
npm run docker:build
# Mount current dir and run
docker run --rm -v "$PWD:/work" -w /work ball-reframe -- -i input.mp4 -o output.mp4 --quality
```

## Notes

- macOS: MPS will be auto-selected if available. Use `--device cpu|mps|cuda` to override.
- Models: place `yolov8n.pt`, `yolov8s.pt`, or `yolov8x.pt` in the repo or reference an absolute path.
