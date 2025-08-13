#!/usr/bin/env node
import { spawn } from "child_process";
import * as path from "path";
import * as fs from "fs";

interface ProcessingArgs {
  input?: string;
  output?: string;
  model?: string;
  device?: string;
  height?: number;
  fps?: number;
  margin?: number;
  smooth?: number;
  maxMove?: number;
  maxAccel?: number;
  deadband?: number;
  conf?: number;
  bootstrapFrames?: number;
  roiSize?: number;
  tracker?: string;
  backend?: string;
  fast?: boolean;
  quality?: boolean;
  verbose?: boolean;
  dryRun?: boolean;
  help?: boolean;
  targetClass?: string;
  noAppearance?: boolean;
  profile?: boolean;
  noTtaRecovery?: boolean;
  imgsz?: number;
  useFfmpeg?: boolean;
  detInterval?: number;
  jerk?: number;
  strictCenter?: boolean; // added
  stickyWindow?: boolean;
  centerBound?: number;
  detectEveryFrame?: boolean;
  fullDetect?: boolean;
  threePass?: boolean;
  learnStride?: number;
}

class ArgsParser {
  static printHelp() {
    console.log(`
🏀 Optimized Sports Ball Reframing Tool

Usage: ball-reframe -i <input.mp4> -o <output.mp4> [options]

Required:
  -i, --input <path>        Input video path
  -o, --output <path>       Output video path

Performance Options:
  --model <name>            YOLO model (yolov8n.pt=fastest, yolov8s.pt=balanced, yolov8x.pt=best)
  --device <cpu|cuda|mps>   Processing device (auto-detected by default)
  --fast                    Use fastest settings (yolov8n + optimized params)
  --quality                 Use highest quality settings (yolov8x + enhanced params)

Output Options:
  --height <int>            Output height (default: keep input height)
  --fps <number>            Override output FPS (default: keep input FPS)
  --margin <int>            Margin around ball in pixels (default: 60)

Smoothing Options:
  --smooth <int>            Trajectory smoothing window (default: 11)
  --max-move <int>          Max pixel movement per frame (default: 80)
  --max-accel <int>         Max acceleration per frame (default: 40)
  --deadband <int>          Ignore micro movements (default: 4)
  --jerk <int>              Limit change in acceleration per frame (default: 0)

Detection/Tracking Options:
  --conf <float>            Detection confidence threshold (default: 0.2)
  --bootstrap-frames <int>  Initial detection frames (default: 48)
  --roi-size <int>          Detection ROI width (default: 480)
  --imgsz <int>             YOLO input size (e.g., 640/960)
  --target-class <name>     Force a specific class (e.g., "tennis ball", "basketball")
  --no-appearance           Disable appearance (color histogram) re-scoring
  --tracker <auto|mosse|kcf|csrt>  Tracker type (default: auto)
  --backend <yolo|yolo-bytetrack>  Tracking backend (default: yolo-bytetrack)
  --det-interval <int>      Baseline detection interval for YOLO backend (default: 2)
  --use-ffmpeg              Use ffmpeg pipe for faster video encoding
  --strict-center           Always center crop on smoothed ball x; bypass motion constraints.

Advanced:
  --profile                 Print periodic profiling info (FPS/misses)
  --no-tta-recovery         Disable heavy TTA full-frame recoveries for speed
  --verbose                 Show detailed progress
  --dry-run                 Show command without running

Examples:
  # Basic usage
  ball-reframe -i soccer.mp4 -o soccer_vertical.mp4

  # Fast processing
  ball-reframe -i game.mp4 -o output.mp4 --fast

  # High quality
  ball-reframe -i highlights.mp4 -o reframed.mp4 --quality

  # Custom settings
  ball-reframe -i input.mp4 -o output.mp4 --model yolov8s.pt --device mps --smooth 15 --imgsz 960
`);
  }

  static parse(argv: string[]) {
    const args: ProcessingArgs = {};
    for (let i = 2; i < argv.length; i++) {
      const a = argv[i];
      const next = () => argv[++i];
      switch (a) {
        case "-i":
        case "--input":
          args.input = next()!;
          break;
        case "-o":
        case "--output":
          args.output = next()!;
          break;
        case "--model":
          args.model = next()!;
          break;
        case "--height":
          args.height = parseInt(next()!);
          break;
        case "--fps":
          args.fps = parseInt(next()!);
          break;
        case "--smooth":
          args.smooth = parseInt(next()!);
          break;
        case "--max-move":
          args.maxMove = parseInt(next()!);
          break;
        case "--max-accel":
          args.maxAccel = parseInt(next()!);
          break;
        case "--margin":
          args.margin = parseInt(next()!);
          break;
        case "--deadband":
          args.deadband = parseInt(next()!);
          break;
        case "--conf":
          args.conf = parseFloat(next()!);
          break;
        case "--device":
          args.device = next()!;
          break;
        case "--tracker":
          args.tracker = next()!;
          break;
        case "--backend":
          args.backend = next()!;
          break;
        case "--roi":
        case "--roi-size":
          args.roiSize = parseInt(next()!);
          break;
        case "--bootstrap-frames":
          args.bootstrapFrames = parseInt(next()!);
          break;
        case "--target-class":
          args.targetClass = next();
          break;
        case "--no-appearance":
          args.noAppearance = true;
          break;
        case "--profile":
          args.profile = true;
          break;
        case "--no-tta-recovery":
          args.noTtaRecovery = true;
          break;
        case "--fast":
          args.fast = true;
          break;
        case "--quality":
          args.quality = true;
          break;
        case "--imgsz":
          args.imgsz = parseInt(next()!);
          break;
        case "--use-ffmpeg":
          args.useFfmpeg = true;
          break;
        case "--det-interval":
          args.detInterval = parseInt(next()!);
          break;
        case "--jerk":
          args.jerk = parseInt(next()!);
          break;
        case "--strict-center":
          args.strictCenter = true;
          break;
        case "--sticky-window":
          args.stickyWindow = true;
          break;
        case "--center-bound":
          args.centerBound = parseInt(next()!);
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
          args.learnStride = parseInt(next()!);
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
}

class PythonRunner {
  private pythonPath: string;
  private scriptPath: string;

  constructor() {
    const venvPython = path.resolve(".venv/bin/python");
    const resolvedPython = fs.existsSync(venvPython) ? venvPython : "python3";
    this.pythonPath = process.env.VENV_PY || resolvedPython;
    this.scriptPath = path.resolve(__dirname, "../scripts/reframe.py");
  }

  run(rawArgs: ProcessingArgs) {
    const inputPath = path.resolve(String(rawArgs.input));
    const outputPath = path.resolve(String(rawArgs.output));
    if (!fs.existsSync(inputPath)) {
      console.error(`Input not found: ${inputPath}`);
      process.exit(1);
    }

    // Apply presets
    const args = { ...rawArgs };
    if (args.fast && args.quality) {
      console.warn("Both --fast and --quality provided. Using --quality.");
      args.fast = false;
    }

    if (args.fast) {
      args.model = args.model || "yolov8n.pt";
      args.backend = args.backend || "yolo-bytetrack";
      args.smooth = args.smooth ?? 9;
      args.maxMove = args.maxMove ?? 120;
      args.maxAccel = args.maxAccel ?? 60;
      args.deadband = args.deadband ?? 4;
      args.conf = args.conf ?? 0.25;
      args.roiSize = args.roiSize ?? 360;
      args.bootstrapFrames = args.bootstrapFrames ?? 32;
      args.noTtaRecovery = args.noTtaRecovery ?? true;
      args.detInterval = args.detInterval ?? 1;
      args.useFfmpeg = args.useFfmpeg ?? true;
    } else if (args.quality) {
      args.model = args.model || "yolov8x.pt";
      args.backend = args.backend || "yolo-bytetrack";
      args.smooth = args.smooth ?? 17;
      args.maxMove = args.maxMove ?? 60;
      args.maxAccel = args.maxAccel ?? 30;
      args.deadband = args.deadband ?? 2;
      args.conf = args.conf ?? 0.15;
      args.imgsz = args.imgsz ?? 960;
      args.roiSize = args.roiSize ?? 512;
      args.bootstrapFrames = args.bootstrapFrames ?? 64;
      args.profile = args.profile ?? true;
      args.detInterval = args.detInterval ?? 1;
      args.useFfmpeg = args.useFfmpeg ?? true;
    }

    const pyArgs = this.buildPythonCommand(args, {
      input: inputPath,
      output: outputPath,
    });

    console.log("Running:", this.pythonPath, pyArgs.join(" "));
    const proc = spawn(this.pythonPath, pyArgs, { stdio: "inherit" });
    proc.on("exit", (code) => process.exit(code ?? 0));
  }

  private buildPythonCommand(
    args: ProcessingArgs,
    paths: { input: string; output: string }
  ): string[] {
    const pyArgs = [this.scriptPath, "-i", paths.input, "-o", paths.output];

    // Add optional arguments
    if (args.model) pyArgs.push("--model", args.model);
    if (args.device) pyArgs.push("--device", args.device);
    if (args.height !== undefined) pyArgs.push("--height", String(args.height));
    if (args.fps !== undefined) pyArgs.push("--fps", String(args.fps));
    if (args.margin !== undefined) pyArgs.push("--margin", String(args.margin));
    if (args.smooth !== undefined) pyArgs.push("--smooth", String(args.smooth));
    if (args.maxMove !== undefined)
      pyArgs.push("--max-move", String(args.maxMove));
    if (args.maxAccel !== undefined)
      pyArgs.push("--max-accel", String(args.maxAccel));
    if (args.deadband !== undefined)
      pyArgs.push("--deadband", String(args.deadband));
    if (args.conf !== undefined) pyArgs.push("--conf", String(args.conf));
    if (args.bootstrapFrames !== undefined)
      pyArgs.push("--bootstrap-frames", String(args.bootstrapFrames));
    if (args.roiSize !== undefined) pyArgs.push("--roi", String(args.roiSize));
    if (args.tracker) pyArgs.push("--tracker", args.tracker);
    if (args.backend) pyArgs.push("--backend", args.backend);
    if (args.targetClass) pyArgs.push("--target-class", args.targetClass);
    if (args.noAppearance) pyArgs.push("--no-appearance");
    if (args.profile) pyArgs.push("--profile");
    if (args.noTtaRecovery) pyArgs.push("--no-tta-recovery");
    if (args.imgsz !== undefined) pyArgs.push("--imgsz", String(args.imgsz));
    if (args.useFfmpeg) pyArgs.push("--use-ffmpeg");
    if (args.detInterval !== undefined)
      pyArgs.push("--det-interval", String(args.detInterval));
    if (args.jerk !== undefined) pyArgs.push("--jerk", String(args.jerk));
    if (args.strictCenter) pyArgs.push("--strict-center");
    if (args.stickyWindow) pyArgs.push("--sticky-window");
    if (args.centerBound !== undefined)
      pyArgs.push("--center-bound", String(args.centerBound));
    if (args.detectEveryFrame) pyArgs.push("--detect-every-frame");
    if (args.fullDetect) pyArgs.push("--full-detect");
    if (args.threePass) pyArgs.push("--three-pass");
    if (args.learnStride !== undefined)
      pyArgs.push("--learn-stride", String(args.learnStride));

    return pyArgs;
  }
}

(function main() {
  const args = ArgsParser.parse(process.argv);
  if (args.help || !args.input || !args.output) {
    ArgsParser.printHelp();
    process.exit(args.help ? 0 : 1);
  }
  new PythonRunner().run(args);
})();
