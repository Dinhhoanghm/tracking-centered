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

### Setup

```bash
cd /Users/hoangdo/Own/upwork/ball-centered
# Python env
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# Node CLI
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

### Presets

- `--max-accuracy` (recommended): Enables three-pass, per-frame detection, prediction-guided search, blur templates, tiled fallback, strict centering, imgsz 1536, low conf.
- `--quality`: yolov8x, imgsz 960, smoothing and profiling enabled.
- `--fast`: yolov8n, aggressive intervals and ffmpeg pipe.

### Key Flags

- Input/Output: `-i`, `-o`
- Device: `--device cpu|mps|cuda`
- Model/size: `--model yolov8x.pt`, `--imgsz 960|1280|1536`, `--conf 0.12–0.25`
- Three-pass: `--three-pass` (learn → detect-every-frame → render)
- Detection: `--full-detect` (per-frame), `--detect-every-frame`
- Prediction-guided (built-in): used automatically for ROI/full-frame/tiled
- Tiled fallback: `--tiled-detect`, `--tile-size 512`, `--tile-overlap 128`
- Blur template learning: built-in; improves re-detection on blurry frames
- Centering: `--strict-center` (with border safety), or `--sticky-window`
- Smoothing: `--smooth 11|17|21`
- Class constraint: `--target-class "tennis ball"`
- Backend: `--backend yolo|yolo-bytetrack` (default varies by preset)
- Encoding: `--use-ffmpeg` (hardware encoder where available)

### Example Commands

- High accuracy with known class:

```bash
node dist/cli.js -i input.mp4 -o output.mp4 \
  --max-accuracy --device mps --use-ffmpeg \
  --target-class "tennis ball"
```

- Manual configuration (equivalent to max accuracy):

```bash
node dist/cli.js -i input.mp4 -o output.mp4 \
  --three-pass --full-detect --detect-every-frame \
  --strict-center \
  --tiled-detect --tile-size 512 --tile-overlap 128 \
  --imgsz 1536 --conf 0.12 \
  --device mps --use-ffmpeg
```

### Tips

- If the ball is tiny/very blurry: raise `--imgsz` to 1536, lower `--conf` to ~0.12, enable `--tiled-detect` with `--tile-size 512 --tile-overlap 128`.
- If output feels jittery: increase `--smooth` to 17 or 21.
- If centering feels too strict near edges: increase `--margin 80`.

### Troubleshooting

- Ultralytics not found: activate venv and `pip install -r requirements.txt`.
- MPS/CUDA not used: set `--device mps|cuda|cpu` explicitly.
- Still missing mid-video frames: ensure `--max-accuracy` or enable `--tiled-detect` and increase `--imgsz`.

### License

MIT
