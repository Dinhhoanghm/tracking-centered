#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from filterpy.kalman import KalmanFilter  # type: ignore
except Exception:
    KalmanFilter = None  # type: ignore


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    num_frames: int


class YoloBallDetector:
    def __init__(self, model_name: str, device: Optional[str], conf: float, imgsz: Optional[int]) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Please run: pip install ultralytics")
        self.model = YOLO(model_name)
        self.device = self._select_device(device)
        self.conf = float(conf)
        self.imgsz = int(imgsz) if imgsz else None
        self.ball_class_ids = self._resolve_ball_classes()

    def _select_device(self, device: Optional[str]) -> str:
        if device:
            return device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    def _resolve_ball_classes(self) -> List[int]:
        names = getattr(self.model, "names", None)
        if not names:
            return [32]
        ball_ids: List[int] = []
        for cid, name in names.items():
            if "ball" in str(name).lower():
                ball_ids.append(int(cid))
        return ball_ids or [32]

    def _predict_on(self, frame_bgr: np.ndarray, *, conf: Optional[float] = None, imgsz: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        h, _ = frame_bgr.shape[:2]
        imgsz_eff = imgsz if imgsz else (self.imgsz if self.imgsz else max(320, (h // 32) * 32))
        results = self.model.predict(
            source=frame_bgr,
            conf=(conf if conf is not None else self.conf),
            verbose=False,
            device=self.device,
            classes=self.ball_class_ids,
            iou=0.5,
            imgsz=imgsz_eff,
        )
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        boxes = r.boxes
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        xys = boxes.xyxy.cpu().numpy()
        return xys, confs, clss

    def detect_best_bbox_xyxy(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        out = self._predict_on(frame_bgr)
        if out is None:
            return None
        xys, confs, clss = out
        best_bbox = None
        best_c = -1.0
        for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
            if int(cls_id) not in self.ball_class_ids:
                continue
            if float(c) > best_c:
                best_c = float(c)
                best_bbox = (float(x1), float(y1), float(x2), float(y2))
        return best_bbox

    def detect_best_bbox_xyxy_in_roi(
        self,
        frame_bgr: np.ndarray,
        roi_left: int,
        roi_width: int,
        pref_center_x: Optional[float] = None,
        conf_override: Optional[float] = None,
        imgsz_override: Optional[int] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        H, W = frame_bgr.shape[:2]
        roi_left = int(max(0, min(roi_left, W - 2)))
        roi_width = int(max(2, min(roi_width, W - roi_left)))
        roi = frame_bgr[:, roi_left : roi_left + roi_width]
        out = self._predict_on(roi, conf=conf_override, imgsz=imgsz_override)
        if out is None:
            return None
        xys, confs, clss = out
        best_score = -1.0
        best_bbox = None
        for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
            if int(cls_id) not in self.ball_class_ids:
                continue
            cx = (x1 + x2) / 2.0 + roi_left
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            conf_score = float(c)
            dist_penalty = 1.0
            if pref_center_x is not None:
                dist = abs(cx - float(pref_center_x))
                norm = max(1.0, roi_width / 2.0)
                dist_penalty = 1.0 / (1.0 + (dist / norm))
            area_penalty = 1.0 / (1.0 + area / (W * H * 0.02))
            score = conf_score * 0.7 + conf_score * dist_penalty * 0.2 + conf_score * area_penalty * 0.1
            if score > best_score:
                best_score = score
                best_bbox = (float(x1 + roi_left), float(y1), float(x2 + roi_left), float(y2))
        return best_bbox


class BoxTracker:
    def __init__(self, tracker_pref: Optional[str] = None) -> None:
        self.tracker = None
        self.active = False
        self.tracker_pref = (tracker_pref or "auto").lower()

    @staticmethod
    def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
        xi = int(round(x1))
        yi = int(round(y1))
        wi = max(1, int(round(x2 - x1)))
        hi = max(1, int(round(y2 - y1)))
        return xi, yi, wi, hi

    @staticmethod
    def _center_x_from_xywh(x: float, _y: float, w: float, _h: float) -> float:
        return x + w / 2.0

    def _candidate_paths(self) -> List[str]:
        order_map = {
            "mosse": ["legacy.TrackerMOSSE_create", "TrackerMOSSE_create"],
            "kcf": ["legacy.TrackerKCF_create", "TrackerKCF_create"],
            "csrt": ["legacy.TrackerCSRT_create", "TrackerCSRT_create"],
        }
        if self.tracker_pref in ("mosse", "kcf", "csrt"):
            ordered = order_map[self.tracker_pref]
        else:
            ordered = order_map["mosse"] + order_map["kcf"] + order_map["csrt"]
        return ordered

    def _create_tracker(self):
        tracker = None
        for path in self._candidate_paths():
            mod = cv2
            try:
                for attr in path.split("."):
                    mod = getattr(mod, attr)
                tracker = mod()
                if tracker is not None:
                    return tracker
            except Exception:
                continue
        raise RuntimeError("No supported OpenCV tracker available (MOSSE/KCF/CSRT)")

    def init_with_bbox(self, frame: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> bool:
        tracker = self._create_tracker()
        x1, y1, x2, y2 = bbox_xyxy
        x, y, w, h = self._xyxy_to_xywh(x1, y1, x2, y2)
        ok = tracker.init(frame, (int(x), int(y), int(w), int(h)))
        self.tracker = tracker
        self.active = bool(ok)
        return self.active

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[float]]:
        if not self.active or self.tracker is None:
            return False, None
        ok, box = self.tracker.update(frame)
        if not ok:
            self.active = False
            return False, None
        x, y, w, h = box
        return True, self._center_x_from_xywh(float(x), float(y), float(w), float(h))


class Kalman1D:
    def __init__(self, q: float = 0.05, r: float = 4.0) -> None:
        if KalmanFilter is None:
            self.x = None
            self.v = 0.0
            self.q = float(q)
            self.r = float(r)
            self.use_kf = False
        else:
            self.kf = KalmanFilter(dim_x=2, dim_z=1)
            self.kf.F = np.array([[1, 1], [0, 1]], dtype=float)
            self.kf.H = np.array([[1, 0]], dtype=float)
            self.kf.P = np.diag([1000.0, 1000.0])
            self.kf.R = np.array([[float(r)]], dtype=float)
            self.kf.Q = np.array([[0.25 * q, 0.5 * q], [0.5 * q, q]], dtype=float)
            self.initialized = False
            self.use_kf = True

    def predict(self) -> Optional[float]:
        if self.use_kf:
            if not getattr(self, "initialized", False):
                return None
            self.kf.predict()
            return float(self.kf.x[0, 0])
        else:
            if self.x is None:
                return None
            self.x = float(self.x + self.v)
            return self.x

    def update(self, measurement: Optional[float]) -> Optional[float]:
        if measurement is None:
            return self.predict()
        if self.use_kf:
            if not getattr(self, "initialized", False):
                self.kf.x = np.array([[measurement], [0.0]], dtype=float)
                self.initialized = True
                return float(self.kf.x[0, 0])
            self.kf.update(np.array([[measurement]], dtype=float))
            return float(self.kf.x[0, 0])
        else:
            if self.x is None:
                self.x = float(measurement)
                self.v = 0.0
            else:
                new_v = float(measurement - self.x)
                self.x = float(measurement)
                self.v = 0.8 * self.v + 0.2 * new_v
            return self.x


class TrajectorySmoother:
    def __init__(self, window_size: int = 15) -> None:
        self.window_size = int(window_size)

    @staticmethod
    def _interpolate_nans(values: np.ndarray) -> np.ndarray:
        x = np.arange(len(values))
        good = np.isfinite(values)
        if good.sum() == 0:
            return np.zeros_like(values)
        first_good = np.argmax(good)
        last_good = len(values) - 1 - np.argmax(good[::-1])
        values[:first_good] = values[first_good]
        values[last_good + 1 :] = values[last_good]
        good = np.isfinite(values)
        if good.all():
            return values
        values[~good] = np.interp(x=x[~good], xp=x[good], fp=values[good])
        return values

    def smooth(self, series: List[Optional[float]]) -> np.ndarray:
        xs = np.array([np.nan if v is None else float(v) for v in series], dtype=float)
        xs_interp = self._interpolate_nans(xs)
        n = len(xs_interp)
        win = max(5, min(self.window_size if self.window_size % 2 == 1 else self.window_size - 1, n if n % 2 == 1 else n - 1))
        if win < 5 or win > n:
            kernel = max(3, min(9, n))
            k2 = kernel // 2
            out = np.empty_like(xs_interp)
            for i in range(n):
                lo = max(0, i - k2)
                hi = min(n, i + k2 + 1)
                out[i] = xs_interp[lo:hi].mean()
            return out
        try:
            return savgol_filter(xs_interp, window_length=win, polyorder=2, mode="interp")
        except Exception:
            return xs_interp


class CropPlanner:
    def __init__(self, frame_width: int, output_width: int, max_move_per_frame: int, margin: int = 0, max_accel_per_frame: int = 0) -> None:
        self.frame_width = int(frame_width)
        self.output_width = int(output_width)
        self.max_move = int(max_move_per_frame)
        self.margin = max(0, int(margin))
        self.max_accel = max(0, int(max_accel_per_frame))
        half = self.output_width / 2
        self.min_center = half + self.margin
        self.max_center = self.frame_width - half - self.margin

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def plan_centers(self, smoothed_xs: np.ndarray) -> np.ndarray:
        centers = np.empty_like(smoothed_xs)
        prev_center: Optional[float] = None
        prev_delta: Optional[float] = None
        for i, cx in enumerate(smoothed_xs):
            cx = self._clamp(float(cx), self.min_center, self.max_center)
            if prev_center is None:
                centers[i] = cx
                prev_center = cx
                prev_delta = 0.0
            else:
                desired_delta = cx - prev_center
                if prev_delta is None:
                    prev_delta = 0.0
                if self.max_accel > 0:
                    accel = desired_delta - prev_delta
                    accel = self._clamp(accel, -self.max_accel, self.max_accel)
                    desired_delta = prev_delta + accel
                if abs(desired_delta) > self.max_move:
                    desired_delta = math.copysign(self.max_move, desired_delta)
                new_center = prev_center + desired_delta
                new_center = self._clamp(new_center, self.min_center, self.max_center)
                centers[i] = new_center
                prev_delta = new_center - prev_center
                prev_center = new_center
        return centers


class OpticalFlowHelper:
    def __init__(self, min_corners: int = 10) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # shape (N,1,2)
        self.min_corners = int(min_corners)

    @staticmethod
    def _bbox_to_rect(b: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = b
        x = int(round(x1))
        y = int(round(y1))
        w = max(1, int(round(x2 - x1)))
        h = max(1, int(round(y2 - y1)))
        return x, y, w, h

    def reset(self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self._bbox_to_rect(bbox_xyxy)
        x0 = max(0, x)
        y0 = max(0, y)
        roi = gray[y0 : y0 + h, x0 : x0 + w]
        if roi.size == 0:
            self.prev_pts = None
            self.prev_gray = gray
            return
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01, minDistance=3)
        if pts is not None:
            pts[:, 0, 0] += x0
            pts[:, 0, 1] += y0
        self.prev_pts = pts
        self.prev_gray = gray

    def track(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        if self.prev_pts is None or self.prev_gray is None or len(self.prev_pts) < self.min_corners:
            self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            self.prev_pts = None
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        if next_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts = None
            return None
        good_new = next_pts[status.flatten() == 1]
        good_old = self.prev_pts[status.flatten() == 1]
        if len(good_new) < self.min_corners:
            self.prev_gray = gray
            self.prev_pts = None
            return None
        disp = (good_new - good_old).reshape(-1, 2)
        dx = float(np.median(disp[:, 0]))
        dy = float(np.median(disp[:, 1]))
        # update state for next call
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        return dx, dy


class ReframerPipeline:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_name: str,
        device: Optional[str],
        conf: float,
        smooth_window: int,
        max_move: int,
        out_height: Optional[int],
        out_fps: Optional[float],
        det_interval: int,
        imgsz: Optional[int],
        margin: int,
        max_accel: int,
        auto_det: bool,
        tracker: str,
        roi_width: int,
        kf_q: float,
        kf_r: float,
        backend: str,
        bootstrap_frames: int,
        bootstrap_imgsz: Optional[int],
        bootstrap_conf: Optional[float],
        speed_mid: int,
        speed_high: int,
        roi_min: int,
        roi_max: int,
        roi_gain: float,
        outlier_thresh: int,
        outlier_frames: int,
        flow_enabled: bool = True,
        flow_gate: int = 120,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.detector = YoloBallDetector(model_name=model_name, device=device, conf=conf, imgsz=imgsz)
        self.smoother = TrajectorySmoother(window_size=smooth_window)
        self.max_move = max_move
        self.out_height = out_height
        self.out_fps = out_fps
        self.det_interval = max(1, int(det_interval))
        self.margin = max(0, int(margin))
        self.max_accel = max(0, int(max_accel))
        self.auto_det = bool(auto_det)
        self.tracker_pref = tracker
        self.roi_width = int(max(64, roi_width))
        self.kf = Kalman1D(q=float(kf_q), r=float(kf_r))
        self.backend = backend
        self.bootstrap_frames = max(1, int(bootstrap_frames))
        self.bootstrap_imgsz = int(bootstrap_imgsz) if bootstrap_imgsz else None
        self.bootstrap_conf = float(bootstrap_conf) if bootstrap_conf is not None else None
        self.speed_mid = int(speed_mid)
        self.speed_high = int(speed_high)
        self.roi_min = int(roi_min)
        self.roi_max = int(roi_max)
        self.roi_gain = float(roi_gain)
        self.outlier_thresh = int(outlier_thresh)
        self.outlier_frames = max(1, int(outlier_frames))
        self.flow_enabled = bool(flow_enabled)
        self.flow_gate = int(flow_gate)
        self.flow = OpticalFlowHelper(min_corners=10) if self.flow_enabled else None

    @staticmethod
    def _read_meta(path: str) -> VideoMeta:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit("Failed to open input video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return VideoMeta(width=width, height=height, fps=fps, num_frames=num_frames)

    @staticmethod
    def _compute_crop_width(input_height: int, input_width: int) -> int:
        crop_w = int(round(input_height * 9 / 16))
        crop_w -= crop_w % 2
        return max(2, min(crop_w, input_width - (input_width % 2)))

    @staticmethod
    def _compute_output_size(out_h: Optional[int], in_h: int) -> Tuple[int, int]:
        out_h_final = int(out_h) if out_h is not None else in_h
        out_w_final = int(round(out_h_final * 9 / 16))
        out_w_final -= out_w_final % 2
        return out_w_final, out_h_final

    def _bootstrap_initial_center(self) -> Optional[Tuple[int, Tuple[float, float, float, float], float]]:
        meta = self._read_meta(self.input_path)
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return None
        best_score = -1.0
        best = None
        frames_to_scan = min(self.bootstrap_frames, meta.num_frames or self.bootstrap_frames)
        for idx in range(frames_to_scan):
            ok, frame = cap.read()
            if not ok:
                break
            out = self.detector._predict_on(frame, conf=self.bootstrap_conf, imgsz=self.bootstrap_imgsz)
            if out is None:
                continue
            xys, confs, clss = out
            H, W = frame.shape[:2]
            for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                if int(cls_id) not in self.detector.ball_class_ids:
                    continue
                cx = (float(x1) + float(x2)) / 2.0
                area = max(1.0, float((x2 - x1) * (y2 - y1)))
                conf_score = float(c)
                area_penalty = 1.0 / (1.0 + area / (W * H * 0.02))
                score = conf_score * 0.85 + conf_score * area_penalty * 0.15
                if score > best_score:
                    best_score = score
                    best = (idx, (float(x1), float(y1), float(x2), float(y2)), cx)
        cap.release()
        return best

    def _first_pass_detect_track_ultra(self, tracker_yaml: str) -> List[Optional[float]]:
        xs: List[Optional[float]] = []
        boot = self._bootstrap_initial_center()
        if boot is not None:
            _, _, cx = boot
            self.kf.update(cx)
        try:
            gen = self.detector.model.track(
                source=self.input_path,
                conf=self.detector.conf,
                device=self.detector.device,
                imgsz=self.detector.imgsz if self.detector.imgsz else None,
                classes=self.detector.ball_class_ids,
                tracker=tracker_yaml,
                verbose=False,
                stream=True,
                persist=True,
            )
            for r in gen:
                cx_val: Optional[float] = None
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    xys = boxes.xyxy.cpu().numpy()
                    best_c = -1.0
                    for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                        if int(cls_id) not in self.detector.ball_class_ids:
                            continue
                        if float(c) > best_c:
                            best_c = float(c)
                            cx_val = (float(x1) + float(x2)) / 2.0
                if cx_val is None:
                    est = self.kf.update(None)
                    xs.append(float(est) if est is not None else None)
                else:
                    est = self.kf.update(cx_val)
                    xs.append(float(est) if est is not None else float(cx_val))
            return xs
        except Exception:
            return self._first_pass_detect_track()

    def _adaptive_params(self, base_interval: int, base_roi: int, prev_est: Optional[float], est: Optional[float]) -> Tuple[int, int]:
        if prev_est is None or est is None:
            return base_interval, max(self.roi_min, min(base_roi, self.roi_max))
        speed = abs(float(est) - float(prev_est))  # px/frame
        if speed >= self.speed_high:
            interval = 1
        elif speed >= self.speed_mid:
            interval = max(1, base_interval)
        else:
            interval = min(6, base_interval * 2)
        roi = int(round(base_roi * (1.0 + self.roi_gain * (speed / max(1.0, self.speed_high)))))
        roi = max(self.roi_min, min(roi, self.roi_max))
        return interval, roi

    def _first_pass_detect_track(self) -> List[Optional[float]]:
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to open input video")
        xs: List[Optional[float]] = []
        tracker = BoxTracker(tracker_pref=self.tracker_pref)

        frame_idx = 0
        frames_since_det = 1_000_000
        base_interval = self.det_interval
        cur_interval = base_interval
        stable_count = 0

        meta = self._read_meta(self.input_path)
        frame_w = meta.width

        boot = self._bootstrap_initial_center()
        if boot is not None:
            _, bbox0, cx0 = boot
            self.kf.update(cx0)
        else:
            self.kf.update(frame_w / 2.0)

        prev_est: Optional[float] = None
        outlier_streak = 0
        locked = False
        last_bbox: Optional[Tuple[float, float, float, float]] = None

        prev_frame: Optional[np.ndarray] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cx_pred = self.kf.predict()
            if cx_pred is None:
                cx_pred = frame_w / 2.0

            cur_interval, roi_w = self._adaptive_params(base_interval, self.roi_width, prev_est, cx_pred)
            roi_half = min(frame_w // 2, roi_w // 2)
            roi_left = int(round(float(cx_pred) - roi_half))
            roi_left = max(0, min(roi_left, frame_w - (roi_half * 2)))

            use_detect = (frames_since_det >= cur_interval) or (not tracker.active) or (not locked)
            cx_meas: Optional[float] = None
            used_flow = False

            if use_detect:
                detect_conf = self.bootstrap_conf if not locked else None
                detect_imgsz = self.bootstrap_imgsz if not locked else None
                if not locked:
                    bbox = self.detector.detect_best_bbox_xyxy_in_roi(
                        frame,
                        roi_left=0,
                        roi_width=frame_w,
                        pref_center_x=None,
                        conf_override=detect_conf,
                        imgsz_override=detect_imgsz,
                    )
                else:
                    bbox = self.detector.detect_best_bbox_xyxy_in_roi(
                        frame,
                        roi_left=roi_left,
                        roi_width=roi_w,
                        pref_center_x=cx_pred,
                        conf_override=None,
                        imgsz_override=None,
                    )
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cx_det = (x1 + x2) / 2.0
                    if locked and abs(cx_det - cx_pred) > self.outlier_thresh:
                        outlier_streak += 1
                        if outlier_streak >= self.outlier_frames:
                            tracker.init_with_bbox(frame, bbox)
                            cx_meas = cx_det
                            outlier_streak = 0
                            last_bbox = bbox
                            if self.flow is not None:
                                self.flow.reset(frame, bbox)
                        else:
                            ok_t, cx_t = tracker.update(frame)
                            if ok_t and cx_t is not None:
                                cx_meas = cx_t
                            frames_since_det += 1
                    else:
                        tracker.init_with_bbox(frame, bbox)
                        cx_meas = cx_det
                        outlier_streak = 0
                        frames_since_det = 0
                        stable_count = 0 if not tracker.active else stable_count
                        locked = True
                        last_bbox = bbox
                        if self.flow is not None:
                            self.flow.reset(frame, bbox)
                else:
                    ok_t, cx_t = tracker.update(frame)
                    if ok_t and cx_t is not None:
                        cx_meas = cx_t
                    frames_since_det += 1
            else:
                ok_t, cx_t = tracker.update(frame)
                if ok_t and cx_t is not None:
                    cx_meas = cx_t
                    frames_since_det += 1
                else:
                    tracker.active = False
                    frames_since_det = cur_interval
                    stable_count = 0

            # Optical flow assist when detection/tracker is weak or missing
            if self.flow is not None and prev_frame is not None:
                flow_disp = self.flow.track(frame)
                if flow_disp is not None and prev_est is not None:
                    dx, _dy = flow_disp
                    cx_flow = float(prev_est + dx)
                    if cx_meas is None:
                        cx_meas = cx_flow
                        used_flow = True
                    elif abs(cx_meas - cx_flow) > self.flow_gate:
                        # If detection/tracker is far from flow, prefer the one closer to prediction
                        if abs(cx_flow - cx_pred) < abs(cx_meas - cx_pred):
                            cx_meas = cx_flow
                            used_flow = True

            if cx_meas is None:
                est = self.kf.update(None)
            else:
                est = self.kf.update(cx_meas)

            xs.append(float(est) if est is not None else (float(cx_pred) if cx_pred is not None else None))
            prev_est = float(est) if est is not None else prev_est
            prev_frame = frame
            frame_idx += 1

        cap.release()
        return xs

    def _second_pass_write(self, centers: np.ndarray, crop_width: int, out_size: Tuple[int, int], fps_out: float) -> None:
        out_w, out_h = out_size
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to reopen input video for writing")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(self.output_path, fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            raise SystemExit("Failed to open VideoWriter. Try a different output path or codec.")

        half_crop = crop_width // 2
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cx = float(centers[idx])
            left = int(round(cx - half_crop))
            left = max(0, min(left, in_w - crop_width))
            crop = frame[:, left : left + crop_width]
            if (crop.shape[1], crop.shape[0]) != (out_w, out_h):
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            writer.write(crop)
            idx += 1

        cap.release()
        writer.release()

    def run(self) -> None:
        meta = self._read_meta(self.input_path)
        crop_w = self._compute_crop_width(meta.height, meta.width)
        out_w, out_h = self._compute_output_size(self.out_height, meta.height)
        fps_out = float(self.out_fps) if self.out_fps is not None else meta.fps

        if self.backend == "yolo-bytetrack":
            xs_raw = self._first_pass_detect_track_ultra("bytetrack.yaml")
        elif self.backend == "yolo-botsort":
            xs_raw = self._first_pass_detect_track_ultra("botsort.yaml")
        else:
            xs_raw = self._first_pass_detect_track()

        if len(xs_raw) == 0:
            raise SystemExit("No frames read from input video")

        xs_smooth = self.smoother.smooth(xs_raw)
        planner = CropPlanner(
            frame_width=meta.width,
            output_width=crop_w,
            max_move_per_frame=self.max_move,
            margin=self.margin,
            max_accel_per_frame=self.max_accel,
        )
        centers = planner.plan_centers(xs_smooth)

        self._second_pass_write(centers=centers, crop_width=crop_w, out_size=(out_w, out_h), fps_out=fps_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reframe 16:9 video to 9:16 centered on a sports ball")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model name or path")
    parser.add_argument("--height", type=int, default=None, help="Output height; default keep input height")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS; default keep input FPS")
    parser.add_argument("--smooth", type=int, default=11, help="Savitzky-Golay window size (odd); auto-adjust if too large")
    parser.add_argument("--max-move", type=int, default=80, help="Max allowed crop-center movement per frame in px")
    parser.add_argument("--max-accel", type=int, default=40, help="Max allowed change in movement per frame in px (acceleration clamp)")
    parser.add_argument("--margin", type=int, default=60, help="Horizontal margin to keep the ball away from crop edges in px")
    parser.add_argument("--conf", type=float, default=0.2, help="YOLO confidence threshold")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--det-interval", type=int, default=1, help="Run YOLO every N frames; track between detections")
    parser.add_argument("--auto-det", action="store_true", help="Adapt detection interval based on tracker stability")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO input size (e.g. 640, 960)")
    parser.add_argument("--tracker", default="auto", choices=["auto", "mosse", "kcf", "csrt"], help="OpenCV tracker to use for frames between detections")
    parser.add_argument("--roi", type=int, default=480, help="ROI width in pixels for YOLO detection around predicted center")
    parser.add_argument("--kf-q", type=float, default=0.05, help="Kalman filter process noise")
    parser.add_argument("--kf-r", type=float, default=4.0, help="Kalman filter measurement noise")
    parser.add_argument("--backend", choices=["yolo", "yolo-bytetrack", "yolo-botsort"], default="yolo-bytetrack", help="Tracking backend to use")
    parser.add_argument("--bootstrap-frames", type=int, default=24, help="Scan first N frames at higher quality to lock ball from start")
    parser.add_argument("--bootstrap-imgsz", type=int, default=960, help="YOLO imgsz for bootstrap stage")
    parser.add_argument("--bootstrap-conf", type=float, default=0.15, help="YOLO confidence for bootstrap stage")
    parser.add_argument("--speed-mid", type=int, default=16, help="Speed (px/frame) above which detection interval tightens")
    parser.add_argument("--speed-high", type=int, default=32, help="Speed (px/frame) above which detect every frame")
    parser.add_argument("--roi-min", type=int, default=480, help="Minimum ROI width")
    parser.add_argument("--roi-max", type=int, default=1280, help="Maximum ROI width")
    parser.add_argument("--roi-gain", type=float, default=1.0, help="How much ROI expands with speed (0..~2)")
    parser.add_argument("--outlier", type=int, default=160, help="Reject single detections farther than this px from prediction")
    parser.add_argument("--outlier-frames", type=int, default=2, help="Accept outlier if it persists this many frames")
    parser.add_argument("--no-flow", action="store_true", help="Disable optical-flow assist")
    parser.add_argument("--flow-gate", type=int, default=120, help="Gate in px to prefer optical flow vs detection when they disagree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ReframerPipeline(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        conf=args.conf,
        smooth_window=args.smooth,
        max_move=args.max_move,
        out_height=args.height,
        out_fps=args.fps,
        det_interval=args.det_interval,
        imgsz=args.imgsz,
        margin=args.margin,
        max_accel=args.max_accel,
        auto_det=bool(args.auto_det),
        tracker=args.tracker,
        roi_width=args.roi,
        kf_q=args.kf_q,
        kf_r=args.kf_r,
        backend=args.backend,
        bootstrap_frames=args.bootstrap_frames,
        bootstrap_imgsz=args.bootstrap_imgsz,
        bootstrap_conf=args.bootstrap_conf,
        speed_mid=args.speed_mid,
        speed_high=args.speed_high,
        roi_min=args.roi_min,
        roi_max=args.roi_max,
        roi_gain=args.roi_gain,
        outlier_thresh=args.outlier,
        outlier_frames=args.outlier_frames,
        flow_enabled=not bool(args.no_flow),
        flow_gate=args.flow_gate,
    )
    pipeline.run()
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main() 