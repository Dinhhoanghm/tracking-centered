#!/usr/bin/env node

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

function printHelp() {
  console.log(
    `\nUsage: ball-reframe -i <input.mp4> -o <output.mp4> [options]\n\nOptions:\n  -i, --input <path>        Input video path (required)\n  -o, --output <path>       Output video path (required)\n  --model <path|name>       YOLO model (default: yolov8n.pt)\n  --height <int>            Output height (default infer from input, multiple of 2)\n  --fps <number>            Override output FPS (default: keep input)\n  --margin <int>            Horizontal margin around ball in px (default: 60)\n  --smooth <int>            Savitzky-Golay window size (odd int, default: 15)\n  --max-move <int>          Max px shift per frame to avoid jumps (default: 80)\n  --max-accel <int>         Max change in shift per frame (acceleration clamp, default: 40)\n  --deadband <int>          Ignore micro movements (default: 4)\n  --conf <float>            Detection confidence threshold (default: 0.2)\n  --device <cpu|mps|cuda>   Inference device (default: auto)\n  --det-interval <int>      Baseline detection interval for YOLO backend (default: 2)\n  --imgsz <int>             YOLO input size (e.g. 640/960)\n  --tracker <auto|mosse|kcf|csrt>  Choose OpenCV tracker (default: auto)\n  --roi, --roi-size <int>   ROI width in px for YOLO detection around predicted center (default: 480)\n  --bootstrap-frames <int>  Initial frames to scan for bootstrap (default: 48)\n  --target-class <name>     Force a specific class (e.g., "tennis ball")\n  --no-appearance           Disable appearance re-scoring\n  --backend <yolo|yolo-bytetrack>  Tracking backend (default: yolo-bytetrack)\n  --use-ffmpeg              Use ffmpeg pipe for faster encoding\n  --profile                 Print periodic profiling info\n  --no-tta-recovery         Disable heavy TTA full-frame recoveries\n  --jerk <int>              Limit change in acceleration per frame (default: 0)\n  --fast                    Use fast preset\n  --quality                 Use quality preset\n  --strict-center           Always center crop on smoothed ball x; bypass motion constraints.\n  -h, --help                Show help\n`
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
      case "--deadband":
        args.deadband = next();
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
      case "--imgsz":
        args.imgsz = next();
        break;
      case "--tracker":
        args.tracker = next();
        break;
      case "--roi":
      case "--roi-size":
        args.roi = next();
        break;
      case "--bootstrap-frames":
        args.bootstrapFrames = next();
        break;
      case "--target-class":
        args.targetClass = next();
        break;
      case "--no-appearance":
        args.noAppearance = true;
        break;
      case "--backend":
        args.backend = next();
        break;
      case "--use-ffmpeg":
        args.useFfmpeg = true;
        break;
      case "--profile":
        args.profile = true;
        break;
      case "--no-tta-recovery":
        args.noTtaRecovery = true;
        break;
      case "--jerk":
        args.jerk = next();
        break;
      case "--fast":
        args.fast = true;
        break;
      case "--quality":
        args.quality = true;
        break;
      case "--sticky-window":
        args.stickyWindow = true;
        break;
      case "--center-bound":
        args.centerBound = next();
        break;
      case "--detect-every-frame":
        args.detectEveryFrame = true;
        break;
      case "--full-detect":
        args.fullDetect = true;
        break;
      case "--three-pass":
        args.threePass = true;
        break;
      case "--learn-stride":
        args.learnStride = next();
        break;
      case "--strict-center":
        args.strictCenter = true;
        break;
      case "-h":
      case "--help":
        args.help = true;
        break;
      case "--enhanced-bootstrap":
        args.enhancedBootstrap = true;
        break;
      case "--detection-confidence-boost":
        args.detectionConfidenceBoost = parseFloat(next());
        break;
      case "--stability-frames":
        args.stabilityFrames = parseInt(next());
        break;
      case "--prediction-lookahead":
        args.predictionLookahead = parseInt(next());
        break;
      case "--multi-roi-detect":
        args.multiRoiDetect = true;
        break;
      case "--ultra-quality":
        args.ultraQuality = true;
        break;
      default:
        console.warn(`Unknown arg: ${a}`);
    }
  }
  return args;
}

(function main() {
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

  // Preset resolution
  if (args.fast && args.quality) {
    console.warn("Both --fast and --quality provided. Using --quality.");
    args.fast = false;
  }
  if (args.fast) {
    args.model = args.model || "yolov8n.pt";
    args.detInterval = args.detInterval || 1;
    args.useFfmpeg = args.useFfmpeg || true;
  } else if (args.quality) {
    args.model = args.model || "yolov8x.pt";
    args.imgsz = args.imgsz || 960;
    args.detInterval = args.detInterval || 1;
    args.useFfmpeg = args.useFfmpeg || true;
  }

  const scriptPath = path.resolve(__dirname, "scripts", "reframe.py");
  const pyArgs = [scriptPath, "-i", inputPath, "-o", outputPath];
  if (args.model) pyArgs.push("--model", args.model);
  if (args.height) pyArgs.push("--height", String(args.height));
  if (args.fps) pyArgs.push("--fps", String(args.fps));
  if (args.margin) pyArgs.push("--margin", String(args.margin));
  if (args.smooth) pyArgs.push("--smooth", String(args.smooth));
  if (args.maxMove) pyArgs.push("--max-move", String(args.maxMove));
  if (args.maxAccel) pyArgs.push("--max-accel", String(args.maxAccel));
  if (args.deadband) pyArgs.push("--deadband", String(args.deadband));
  if (args.conf) pyArgs.push("--conf", String(args.conf));
  if (args.device) pyArgs.push("--device", String(args.device));
  if (args.detInterval) pyArgs.push("--det-interval", String(args.detInterval));
  if (args.imgsz) pyArgs.push("--imgsz", String(args.imgsz));
  if (args.tracker) pyArgs.push("--tracker", String(args.tracker));
  if (args.roi) pyArgs.push("--roi", String(args.roi));
  if (args.bootstrapFrames)
    pyArgs.push("--bootstrap-frames", String(args.bootstrapFrames));
  if (args.targetClass) pyArgs.push("--target-class", String(args.targetClass));
  if (args.noAppearance) pyArgs.push("--no-appearance");
  if (args.backend) pyArgs.push("--backend", String(args.backend));
  if (args.profile) pyArgs.push("--profile");
  if (args.noTtaRecovery) pyArgs.push("--no-tta-recovery");
  if (args.useFfmpeg) pyArgs.push("--use-ffmpeg");
  if (args.jerk) pyArgs.push("--jerk", String(args.jerk));
  if (args.stickyWindow) pyArgs.push("--sticky-window");
  if (args.centerBound) pyArgs.push("--center-bound", String(args.centerBound));
  if (args.detectEveryFrame) pyArgs.push("--detect-every-frame");
  if (args.fullDetect) pyArgs.push("--full-detect");
  if (args.threePass) pyArgs.push("--three-pass");
  if (args.learnStride) pyArgs.push("--learn-stride", String(args.learnStride));
  if (args.strictCenter) pyArgs.push("--strict-center");

  console.log("Running:", py, pyArgs.join(" "));
  const proc = spawn(py, pyArgs, { stdio: "inherit" });
  proc.on("exit", (code) => process.exit(code));
})();
