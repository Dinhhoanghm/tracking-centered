#!/usr/bin/env node

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

function printHelp() {
  console.log(
    `\nUsage: ball-reframe -i <input.mp4> -o <output.mp4> [options]\n\nOptions:\n  -i, --input <path>        Input video path (required)\n  -o, --output <path>       Output video path (required)\n  --model <path|name>       YOLO model (default: yolov8s.pt)\n  --height <int>            Output height (default: keep input)\n  --fps <number>            Override output FPS (default: keep input)\n  --margin <int>            Horizontal margin around ball in px (default: 60)\n  --smooth <int>            Savitzky-Golay window size (odd int, default: 11)\n  --max-move <int>          Max px shift per frame to avoid jumps (default: 80)\n  --max-accel <int>         Max change in shift per frame (acceleration clamp, default: 40)\n  --conf <float>            Detection confidence threshold (default: 0.2)\n  --device <cpu|mps|cuda>   Inference device (default: auto)\n  --det-interval <int>      Run YOLO every N frames (default: 1)\n  --auto-det                Adapt detection interval when tracking is stable\n  --imgsz <int>             YOLO input size (e.g. 640/960)\n  --tracker <auto|mosse|kcf|csrt>  Choose OpenCV tracker (default: auto)\n  --roi <int>               ROI width in px for YOLO detection around predicted center (default: 480)\n  --kf-q <float>            Kalman filter process noise (default: 0.05)\n  --kf-r <float>            Kalman filter measurement noise (default: 4.0)\n  --backend <yolo|yolo-bytetrack>  Tracking backend (default: yolo-bytetrack)\n  --bootstrap-frames <int>  Scan N frames at start to lock ball (default: 24)\n  --bootstrap-imgsz <int>   YOLO imgsz during bootstrap (default: 960)\n  --bootstrap-conf <float>  YOLO conf during bootstrap (default: 0.15)\n  -h, --help                Show help\n`
  );
}

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "-i":
      case "--input":
        args.input = next();
        break;
      case "-o":
      case "--output":
        args.output = next();
        break;
      case "--model":
        args.model = next();
        break;
      case "--height":
        args.height = next();
        break;
      case "--fps":
        args.fps = next();
        break;
      case "--margin":
        args.margin = next();
        break;
      case "--smooth":
        args.smooth = next();
        break;
      case "--max-move":
        args.maxMove = next();
        break;
      case "--max-accel":
        args.maxAccel = next();
        break;
      case "--conf":
        args.conf = next();
        break;
      case "--device":
        args.device = next();
        break;
      case "--det-interval":
        args.detInterval = next();
        break;
      case "--auto-det":
        args.autoDet = true;
        break;
      case "--imgsz":
        args.imgsz = next();
        break;
      case "--tracker":
        args.tracker = next();
        break;
      case "--roi":
        args.roi = next();
        break;
      case "--kf-q":
        args.kfQ = next();
        break;
      case "--kf-r":
        args.kfR = next();
        break;
      case "--backend":
        args.backend = next();
        break;
      case "-h":
      case "--help":
        args.help = true;
        break;
      default:
        console.warn(`Unknown arg: ${a}`);
    }
  }
  return args;
}

(async function main() {
  const args = parseArgs(process.argv);
  if (args.help || !args.input || !args.output) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(args.output);
  if (!fs.existsSync(inputPath)) {
    console.error(`Input not found: ${inputPath}`);
    process.exit(1);
  }
  const py = process.env.VENV_PY || path.resolve(".venv/bin/python");

  const scriptPath = path.resolve(__dirname, "scripts", "reframe.py");
  const pyArgs = [scriptPath, "-i", inputPath, "-o", outputPath];
  if (args.model) pyArgs.push("--model", args.model);
  if (args.height) pyArgs.push("--height", String(args.height));
  if (args.fps) pyArgs.push("--fps", String(args.fps));
  if (args.margin) pyArgs.push("--margin", String(args.margin));
  if (args.smooth) pyArgs.push("--smooth", String(args.smooth));
  if (args.maxMove) pyArgs.push("--max-move", String(args.maxMove));
  if (args.maxAccel) pyArgs.push("--max-accel", String(args.maxAccel));
  if (args.conf) pyArgs.push("--conf", String(args.conf));
  if (args.device) pyArgs.push("--device", String(args.device));
  if (args.detInterval) pyArgs.push("--det-interval", String(args.detInterval));
  if (args.autoDet) pyArgs.push("--auto-det");
  if (args.imgsz) pyArgs.push("--imgsz", String(args.imgsz));
  if (args.tracker) pyArgs.push("--tracker", String(args.tracker));
  if (args.roi) pyArgs.push("--roi", String(args.roi));
  if (args.kfQ) pyArgs.push("--kf-q", String(args.kfQ));
  if (args.kfR) pyArgs.push("--kf-r", String(args.kfR));
  if (args.backend) pyArgs.push("--backend", String(args.backend));

  console.log("Running:", py, pyArgs.join(" "));
  const proc = spawn(py, pyArgs, { stdio: "inherit" });
  proc.on("exit", (code) => process.exit(code));
})();
