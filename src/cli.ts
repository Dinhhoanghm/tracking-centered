#!/usr/bin/env node
import { spawn } from "child_process";
import * as path from "path";
import * as fs from "fs";

class ArgsParser {
  static printHelp() {
    console.log(
      `\nUsage: ball-reframe -i <input.mp4> -o <output.mp4> [options]\n\nOptions:\n  -i, --input <path>        Input video path (required)\n  -o, --output <path>       Output video path (required)\n  --model <path|name>       YOLO model (default: yolov8s.pt)\n  --height <int>            Output height (default: keep input)\n  --fps <number>            Override output FPS (default: keep input)\n  --smooth <int>            Savitzky-Golay window size (odd int, default: 11)\n  --max-move <int>          Max px shift per frame (default: 80)\n  --max-accel <int>         Max change in shift per frame (acceleration clamp, default: 40)\n  --margin <int>            Horizontal margin around ball in px (default: 60)\n  --conf <float>            Detection confidence threshold (default: 0.2)\n  --device <cpu|mps|cuda>   Inference device (default: auto)\n  --det-interval <int>      Run YOLO every N frames; track between (default: 1)\n  --auto-det                Adapt detection interval when tracking is stable\n  --imgsz <int>             YOLO input size (e.g. 640/960)\n  --tracker <auto|mosse|kcf|csrt>  Choose OpenCV tracker (default: auto)\n  --roi <int>               ROI width in px for YOLO detection around predicted center (default: 480)\n  --kf-q <float>            Kalman filter process noise (default: 0.05)\n  --kf-r <float>            Kalman filter measurement noise (default: 4.0)\n  --backend <yolo|yolo-bytetrack|yolo-botsort>  Tracking backend (default: yolo-bytetrack)\n  --bootstrap-frames <int>  Scan N frames at start to lock ball (default: 24)\n  --bootstrap-imgsz <int>   YOLO imgsz during bootstrap (default: 960)\n  --bootstrap-conf <float>  YOLO conf during bootstrap (default: 0.15)\n  --no-flow                 Disable optical-flow assist\n  --flow-gate <int>         Gate in px to prefer flow vs detection (default: 120)\n  -h, --help                Show help\n`
    );
  }

  static parse(argv: string[]) {
    const args: Record<string, string | boolean> = {};
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
          args.height = next()!;
          break;
        case "--fps":
          args.fps = next()!;
          break;
        case "--smooth":
          args.smooth = next()!;
          break;
        case "--max-move":
          args.maxMove = next()!;
          break;
        case "--max-accel":
          args.maxAccel = next()!;
          break;
        case "--margin":
          args.margin = next()!;
          break;
        case "--conf":
          args.conf = next()!;
          break;
        case "--device":
          args.device = next()!;
          break;
        case "--det-interval":
          args.detInterval = next()!;
          break;
        case "--auto-det":
          args.autoDet = true;
          break;
        case "--imgsz":
          args.imgsz = next()!;
          break;
        case "--tracker":
          args.tracker = next()!;
          break;
        case "--roi":
          args.roi = next()!;
          break;
        case "--kf-q":
          args.kfQ = next()!;
          break;
        case "--kf-r":
          args.kfR = next()!;
          break;
        case "--backend":
          args.backend = next()!;
          break;
        case "--bootstrap-frames":
          args.bootstrapFrames = next()!;
          break;
        case "--bootstrap-imgsz":
          args.bootstrapImgSz = next()!;
          break;
        case "--bootstrap-conf":
          args.bootstrapConf = next()!;
          break;
        case "--no-flow":
          args.noFlow = true;
          break;
        case "--flow-gate":
          args.flowGate = next()!;
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

  run(rawArgs: Record<string, string | boolean>) {
    const inputPath = path.resolve(String(rawArgs.input));
    const outputPath = path.resolve(String(rawArgs.output));
    if (!fs.existsSync(inputPath)) {
      console.error(`Input not found: ${inputPath}`);
      process.exit(1);
    }

    const pyArgs = [this.scriptPath, "-i", inputPath, "-o", outputPath];
    const push = (k: string, v?: string | boolean) => {
      if (v !== undefined && v !== false) pyArgs.push(k, String(v));
    };

    push("--model", rawArgs.model as string);
    push("--height", rawArgs.height as string);
    push("--fps", rawArgs.fps as string);
    push("--smooth", rawArgs.smooth as string);
    push("--max-move", rawArgs.maxMove as string);
    push("--max-accel", rawArgs.maxAccel as string);
    push("--margin", rawArgs.margin as string);
    push("--conf", rawArgs.conf as string);
    push("--device", rawArgs.device as string);
    push("--det-interval", rawArgs.detInterval as string);
    if (rawArgs.autoDet) pyArgs.push("--auto-det");
    push("--imgsz", rawArgs.imgsz as string);
    push("--tracker", rawArgs.tracker as string);
    push("--roi", rawArgs.roi as string);
    push("--kf-q", rawArgs.kfQ as string);
    push("--kf-r", rawArgs.kfR as string);
    push("--backend", rawArgs.backend as string);
    push("--bootstrap-frames", rawArgs.bootstrapFrames as string);
    push("--bootstrap-imgsz", rawArgs.bootstrapImgSz as string);
    push("--bootstrap-conf", rawArgs.bootstrapConf as string);
    if (rawArgs.noFlow) pyArgs.push("--no-flow");
    push("--flow-gate", rawArgs.flowGate as string);

    console.log("Running:", this.pythonPath, pyArgs.join(" "));
    const proc = spawn(this.pythonPath, pyArgs, { stdio: "inherit" });
    proc.on("exit", (code) => process.exit(code ?? 0));
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
