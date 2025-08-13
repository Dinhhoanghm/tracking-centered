#!/usr/bin/env python3
import argparse
import math
import os
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter
import platform
import subprocess
from scipy.signal import butter, filtfilt, medfilt

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


class OptimizedYoloBallDetector:
    def __init__(self, model_name: str, device: Optional[str], conf: float, imgsz: Optional[int]) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Please run: pip install ultralytics")
        self.model = YOLO(model_name)
        self.device = self._select_device(device)
        self.conf = float(conf)
        # Prefer moderate imgsz on MPS for speed by default
        if imgsz:
            self.imgsz = int(imgsz)
        else:
            self.imgsz = 640 if self.device == "mps" else None
        self.ball_class_ids = self._resolve_ball_classes()
        
        # Control flags (may be overridden by pipeline)
        self.allow_tta_recovery: bool = True
        
        # Move model to target device (e.g., MPS on macOS)
        try:
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception:
            pass
        
        # Target appearance (will be set during bootstrap)
        self.selected_class_id: Optional[int] = None
        self.ref_hist: Optional[np.ndarray] = None
        
        # Pre-compile model for faster inference
        self._warmup_model()

    def set_target_appearance(self, class_id: Optional[int], ref_hist: Optional[np.ndarray]) -> None:
        self.selected_class_id = int(class_id) if class_id is not None else None
        self.ref_hist = ref_hist.copy() if ref_hist is not None else None

    @staticmethod
    def _compute_hs_hist(patch_bgr: np.ndarray) -> Optional[np.ndarray]:
        if patch_bgr is None or patch_bgr.size == 0:
            return None
        h, w = patch_bgr.shape[:2]
        if h < 2 or w < 2:
            return None
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    @staticmethod
    def _hist_similarity(hist_a: Optional[np.ndarray], hist_b: Optional[np.ndarray]) -> Optional[float]:
        if hist_a is None or hist_b is None:
            return None
        # Bhattacharyya distance in [0,1], lower is better
        d = float(cv2.compareHist(hist_a.astype('float32'), hist_b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA))
        sim = max(0.0, min(1.0, 1.0 - d))
        return sim

    def _warmup_model(self):
        """Warm up the model with a dummy frame for faster first inference"""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Try to fuse model layers for speed if supported
        try:
            if hasattr(self.model, "fuse"):
                self.model.fuse()
        except Exception:
            pass
        try:
            self.model.predict(source=dummy_frame, verbose=False, device=self.device)
            # Ensure MPS ops are kicked off and synchronized
            if self.device == "mps":
                try:
                    import torch  # type: ignore
                    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                except Exception:
                    pass
        except:
            pass

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
            return [32]  # sports ball in COCO
        # Normalize mapping to id->name
        if isinstance(names, dict):
            id_to_name = {int(k): str(v).lower() for k, v in names.items()}
        else:
            id_to_name = {i: str(n).lower() for i, n in enumerate(list(names))}
        # Accepted labels
        allowed_exact = {"sports ball", "basketball", "soccer ball", "tennis ball", "volleyball", "football", "handball", "rugby ball"}
        disallowed_substrings = {"bat", "glove", "racket", "racquet", "helmet"}
        ball_ids: List[int] = []
        for cid, name in id_to_name.items():
            if name in allowed_exact:
                ball_ids.append(int(cid))
                continue
            # Allow generic 'ball' but avoid items like 'baseball bat', 'baseball glove'
            if "ball" in name:
                if any(bad in name for bad in disallowed_substrings):
                    continue
                ball_ids.append(int(cid))
        return ball_ids or [32]

    def _predict_on_optimized(self, frame_bgr: np.ndarray, *, conf: Optional[float] = None, imgsz: Optional[int] = None, use_tta: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Optimized prediction with better preprocessing"""
        h, w = frame_bgr.shape[:2]
        
        # Dynamic image sizing for speed vs accuracy balance
        if imgsz is None:
            if h <= 480:
                imgsz_eff = 320  # Very fast for low res
            elif h <= 720:
                imgsz_eff = 480  # Balanced
            else:
                imgsz_eff = 480  # Faster default for HD
        else:
            imgsz_eff = imgsz
            
        # Use half precision for speed if available
        # Avoid half on MPS where it can be slower or unstable
        half = (self.device == "cuda")
        
        classes_param = [self.selected_class_id] if (self.selected_class_id is not None) else self.ball_class_ids
        
        # Light pre-processing for motion blur: mild sharpening + CLAHE when high-accuracy mode
        pre_frame = frame_bgr
        try:
            if use_tta or (imgsz_eff is not None and imgsz_eff >= 960):
                # Unsharp mask
                blur = cv2.GaussianBlur(pre_frame, (0, 0), sigmaX=1.0)
                sharp = cv2.addWeighted(pre_frame, 1.5, blur, -0.5, 0)
                # CLAHE on L channel
                lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l2 = clahe.apply(l)
                lab2 = cv2.merge((l2, a, b))
                pre_frame = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        except Exception:
            pre_frame = frame_bgr
        
        results = self.model.predict(
            source=pre_frame,
            conf=(conf if conf is not None else self.conf),
            verbose=False,
            device=self.device,
            classes=classes_param,
            iou=0.3,  # Lower IOU to keep small ball detections
            imgsz=imgsz_eff,
            half=half,
            augment=bool(use_tta),
            agnostic_nms=True,  # Faster NMS
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

    def detect_best_bbox_xyxy_in_roi_optimized(
        self,
        frame_bgr: np.ndarray,
        roi_left: int,
        roi_width: int,
        pref_center_x: Optional[float] = None,
        conf_override: Optional[float] = None,
        imgsz_override: Optional[int] = None,
        use_tta: bool = False,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Optimized ROI detection with better scoring"""
        H, W = frame_bgr.shape[:2]
        roi_left = int(max(0, min(roi_left, W - 2)))
        roi_width = int(max(2, min(roi_width, W - roi_left)))
        
        # Extract ROI efficiently
        roi = frame_bgr[:, roi_left : roi_left + roi_width]
        out = self._predict_on_optimized(roi, conf=conf_override, imgsz=imgsz_override, use_tta=use_tta)
        
        # Multi-scale ROI detect for tiny balls if needed
        if out is None and self.allow_tta_recovery:
            for scale in (1.5,):  # limit upscale attempts for speed
                h, w = roi.shape[:2]
                if h == 0 or w == 0:
                    break
                roi_up = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                out_up = self._predict_on_optimized(roi_up, conf=max(0.10, (conf_override or self.conf) * 0.7), imgsz=imgsz_override, use_tta=False)
                if out_up is not None:
                    xys, confs, clss = out_up
                    # Map back to original scale
                    xys = xys / scale
                    out = (xys, confs, clss)
                    break
        
        out_from_fallback = False
        if out is None:
            # Fallback: try full-frame with larger resolution and lower conf
            out = self._predict_on_optimized(
                frame_bgr,
                conf=max(0.06, (conf_override or self.conf) * 0.5),
                imgsz=(640 if self.device == 'mps' else max(960, imgsz_override or (self.imgsz or 640))),
                use_tta=use_tta,
            )
            out_from_fallback = out is not None
        # Heavy fallback: tiled scanning across ROI/full-frame for tiny or motion-blurred balls
        if out is None and self.allow_tta_recovery:
            # Respect throttling controls if present on pipeline
            heavy_ok = True
            try:
                heavy_ok = (getattr(self, 'heavy_every_frames', 6) <= 1) or ((getattr(self, '_heavy_frame_counter', 0) % max(1, getattr(self, 'heavy_every_frames', 6))) == 0)
            except Exception:
                heavy_ok = True
            if heavy_ok:
                t_start = time.perf_counter()
                tiles = min((3 if roi_width > 720 else 2), int(getattr(self, 'heavy_max_tiles', 2)))
                overlap = 0.2
                step = max(1, int(roi_width / tiles * (1 - overlap)))
                candidates = []
                scanned = 0
                for t in range(0, roi_width, step):
                    # Time budget check
                    if (time.perf_counter() - t_start) * 1000.0 > float(getattr(self, 'heavy_time_budget_ms', 12)):
                        break
                    t_left = int(max(0, min(t, roi_width - 2)))
                    t_right = int(min(roi_width, t_left + int(roi_width / tiles * (1 + overlap))))
                    if t_right - t_left < 4:
                        continue
                    tile = roi[:, t_left:t_right]
                    out_t = self._predict_on_optimized(
                        tile,
                        conf=max(0.05, (conf_override or self.conf) * 0.6),
                        imgsz=max(640, imgsz_override or (self.imgsz or 640)),
                        use_tta=True,
                    )
                    scanned += 1
                    if out_t is None:
                        continue
                    xys_t, confs_t, clss_t = out_t
                    for c, cls_id, (x1, y1, x2, y2) in zip(confs_t, clss_t, xys_t):
                        if int(cls_id) not in self.ball_class_ids:
                            continue
                        cx = (x1 + x2) / 2.0 + roi_left + t_left
                        cy = (y1 + y2) / 2.0
                        area = max(1.0, float((x2 - x1) * (y2 - y1)))
                        conf_score = float(c)
                        # Score similar to main path with preference to pref_center_x
                        dist_penalty = 1.0
                        if pref_center_x is not None:
                            dist = abs(cx - float(pref_center_x))
                            norm = max(1.0, (t_right - t_left) / 2.5)
                            dist_penalty = 1.0 / (1.0 + (dist / norm) ** 2)
                        area_ratio = area / (W * H)
                        if area_ratio < 0.00003:
                            area_penalty = 0.35
                        elif area_ratio > 0.06:
                            area_penalty = 0.6
                        else:
                            area_penalty = 1.0
                        score = conf_score * (0.6 + 0.2 * dist_penalty) * area_penalty
                        candidates.append((score, float(x1 + roi_left + t_left), float(y1), float(x2 + roi_left + t_left), float(y2)))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    _, bx1, by1, bx2, by2 = candidates[0]
                    return (bx1, by1, bx2, by2)
            try:
                self._heavy_frame_counter = getattr(self, '_heavy_frame_counter', 0) + 1
            except Exception:
                pass
        if out is None:
            return None
        xys, confs, clss = out
        if out_from_fallback:
            roi_left_effective = 0
            roi_width_effective = W
        else:
            roi_left_effective = roi_left
            roi_width_effective = roi_width
        
        best_score = -1.0
        best_bbox = None
        
        for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
            if int(cls_id) not in self.ball_class_ids:
                continue
            
            # Calculate center and metrics
            cx = (x1 + x2) / 2.0 + roi_left_effective
            cy = (y1 + y2) / 2.0
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            conf_score = float(c)
            
            # Enhanced scoring with multiple factors
            dist_penalty = 1.0
            if pref_center_x is not None:
                dist = abs(cx - float(pref_center_x))
                norm = max(1.0, roi_width_effective / 3.0)  # Tighter preference
                dist_penalty = 1.0 / (1.0 + (dist / norm) ** 2)
            
            # Prefer reasonable ball sizes (not too small/large)
            area_ratio = area / (W * H)
            if area_ratio < 0.00005:  # Very small
                area_penalty = 0.4
            elif area_ratio < 0.0003:  # Small
                area_penalty = 0.8
            elif area_ratio > 0.05:  # Too large
                area_penalty = 0.7
            else:
                area_penalty = 1.0
            
            # Aspect ratio penalty for non-circular objects
            aspect_ratio = (x2 - x1) / max(1, y2 - y1)
            aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0))
            
            # Appearance similarity if available
            patch = frame_bgr[int(max(0, y1)):int(min(H, y2)), int(max(0, x1 + roi_left_effective)):int(min(W, x2 + roi_left_effective))] if getattr(self, 'ref_hist', None) is not None else None
            cand_hist = self._compute_hs_hist(patch) if patch is not None else None
            hist_sim = self._hist_similarity(self.ref_hist, cand_hist) if cand_hist is not None else None
            hist_term = (hist_sim if (hist_sim is not None and getattr(self, 'ref_hist', None) is not None) else 0.0)
            
            # Combined score with better weighting
            score = (
                conf_score * 0.45 +
                conf_score * dist_penalty * 0.25 +
                conf_score * area_penalty * 0.1 +
                conf_score * aspect_penalty * 0.1 +
                (hist_term * 0.1 if getattr(self, 'ref_hist', None) is not None else 0.0)
            )
            
            if score > best_score:
                best_score = score
                best_bbox = (float(x1 + roi_left_effective), float(y1), float(x2 + roi_left_effective), float(y2))
        
        return best_bbox


class EnhancedKalman1D:
    """Enhanced Kalman filter with velocity and acceleration modeling"""
    def __init__(self, q: float = 0.05, r: float = 4.0) -> None:
        if KalmanFilter is None:
            # Fallback implementation with acceleration
            self.x = None
            self.v = 0.0
            self.a = 0.0
            self.q = float(q)
            self.r = float(r)
            self.use_kf = False
        else:
            # 3-state Kalman: position, velocity, acceleration
            self.kf = KalmanFilter(dim_x=3, dim_z=1)
            dt = 1.0  # Frame interval
            
            # State transition matrix [pos, vel, acc]
            self.kf.F = np.array([
                [1, dt, 0.5*dt*dt],
                [0, 1,  dt],
                [0, 0,  0.95]  # Acceleration decay
            ], dtype=float)
            
            self.kf.H = np.array([[1, 0, 0]], dtype=float)  # Measure position only
            self.kf.P = np.diag([1000.0, 100.0, 10.0])  # Initial uncertainty
            self.kf.R = np.array([[float(r)]], dtype=float)  # Measurement noise
            
            # Process noise with proper correlation
            q_val = float(q)
            self.kf.Q = np.array([
                [0.25*q_val*dt**4, 0.5*q_val*dt**3, 0.5*q_val*dt**2],
                [0.5*q_val*dt**3,  q_val*dt**2,     q_val*dt],
                [0.5*q_val*dt**2,  q_val*dt,        q_val]
            ], dtype=float)
            
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
            # Simple physics model
            self.v += self.a
            self.x += self.v
            self.a *= 0.9  # Damping
            return self.x

    def update(self, measurement: Optional[float]) -> Optional[float]:
        if measurement is None:
            return self.predict()
            
        if self.use_kf:
            if not getattr(self, "initialized", False):
                self.kf.x = np.array([[measurement], [0.0], [0.0]], dtype=float)
                self.initialized = True
                return float(self.kf.x[0, 0])
            self.kf.update(np.array([[measurement]], dtype=float))
            return float(self.kf.x[0, 0])
        else:
            if self.x is None:
                self.x = float(measurement)
                self.v = 0.0
                self.a = 0.0
            else:
                new_v = float(measurement - self.x)
                new_a = new_v - self.v
                self.x = float(measurement)
                self.v = 0.7 * self.v + 0.3 * new_v  # Smooth velocity
                self.a = 0.5 * self.a + 0.5 * new_a  # Smooth acceleration
            return self.x

    def get_velocity(self) -> float:
        """Get current velocity estimate"""
        if self.use_kf and getattr(self, "initialized", False):
            return float(self.kf.x[1, 0])
        return self.v if self.v is not None else 0.0

    def set_measurement_variance(self, variance: float) -> None:
        """Adjust measurement noise variance dynamically (smaller = more trust)."""
        if self.use_kf:
            self.kf.R = np.array([[float(variance)]], dtype=float)
        else:
            self.r = float(variance)

    def update_with_variance(self, measurement: Optional[float], variance: Optional[float]) -> Optional[float]:
        """Update with optional per-measurement variance.
        When variance is None, reuse the last set value.
        """
        if variance is not None:
            self.set_measurement_variance(float(variance))
        return self.update(measurement)


class ParallelBoxTracker:
    """Multi-tracker with parallel processing"""
    def __init__(self, tracker_pref: Optional[str] = None) -> None:
        self.trackers = []
        self.active_tracker_idx = 0
        self.tracker_pref = (tracker_pref or "auto").lower()
        self.confidence_scores = []
        
        # Initialize multiple trackers for redundancy
        self._init_multiple_trackers()

    def _init_multiple_trackers(self):
        """Initialize multiple tracker types for robustness"""
        tracker_types = ["mosse", "kcf", "csrt"]
        for t_type in tracker_types:
            try:
                tracker = self._create_tracker(t_type)
                if tracker is not None:
                    self.trackers.append({
                        'tracker': tracker,
                        'type': t_type,
                        'active': False,
                        'confidence': 0.0
                    })
            except:
                continue

    def _create_tracker(self, tracker_type: str):
        """Create specific tracker type"""
        try:
            if tracker_type == "mosse":
                return cv2.legacy.TrackerMOSSE_create()
            elif tracker_type == "kcf":
                return cv2.legacy.TrackerKCF_create()
            elif tracker_type == "csrt":
                return cv2.legacy.TrackerCSRT_create()
        except:
            try:
                if tracker_type == "mosse":
                    return cv2.TrackerMOSSE_create()
                elif tracker_type == "kcf":
                    return cv2.TrackerKCF_create()
                elif tracker_type == "csrt":
                    return cv2.TrackerCSRT_create()
            except:
                pass
        return None

    def init_with_bbox(self, frame: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> bool:
        """Initialize all trackers with the same bbox"""
        x1, y1, x2, y2 = bbox_xyxy
        x = int(round(x1))
        y = int(round(y1))
        w = max(1, int(round(x2 - x1)))
        h = max(1, int(round(y2 - y1)))
        
        success_count = 0
        for tracker_info in self.trackers:
            try:
                ok = tracker_info['tracker'].init(frame, (x, y, w, h))
                tracker_info['active'] = ok
                if ok:
                    success_count += 1
                    tracker_info['confidence'] = 1.0
            except:
                tracker_info['active'] = False
                
        return success_count > 0

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[float]]:
        """Update all trackers and return best result"""
        if not self.trackers:
            return False, None
            
        results = []
        active_count = 0
        
        for i, tracker_info in enumerate(self.trackers):
            if not tracker_info['active']:
                continue
                
            try:
                ok, box = tracker_info['tracker'].update(frame)
                if ok:
                    x, y, w, h = box
                    cx = float(x + w / 2.0)
                    # Simple confidence based on box stability
                    confidence = 0.8 if tracker_info['type'] == 'csrt' else 0.9
                    results.append((cx, confidence, i))
                    active_count += 1
                else:
                    tracker_info['active'] = False
            except:
                tracker_info['active'] = False
        
        if not results:
            return False, None
            
        # Return result from most confident tracker
        results.sort(key=lambda x: x[1], reverse=True)
        best_cx, best_conf, best_idx = results[0]
        self.active_tracker_idx = best_idx
        
        return True, best_cx


class OpticalFlowAssist:
    """Lucas–Kanade optical flow fallback to bridge detector gaps for long shots."""
    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_center_x: Optional[float] = None
        self.feature_box: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.prev_center_x = None
        self.feature_box = None

    def init_from_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        x, y, w, h = int(x1), int(y1), int(max(2, x2 - x1)), int(max(2, y2 - y1))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            self.reset()
            return
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.01, minDistance=5)
        if pts is not None:
            pts[:, 0, 0] += x
            pts[:, 0, 1] += y
        self.prev_gray = gray
        self.prev_pts = pts
        self.prev_center_x = float((x1 + x2) / 2.0)
        self.feature_box = (x, y, w, h)

    def update(self, frame_bgr: np.ndarray) -> Optional[float]:
        if self.prev_gray is None or self.prev_pts is None or self.prev_pts.size == 0:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            self.reset()
            return None
        good_new = next_pts[status.flatten() == 1]
        good_old = self.prev_pts[status.flatten() == 1]
        if good_new.size == 0 or good_old.size == 0:
            self.reset()
            return None
        # Compute median x shift to be robust to outliers
        dxs = (good_new[:, 0] - good_old[:, 0])
        dx_med = float(np.median(dxs))
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        if self.prev_center_x is None:
            return None
        return self.prev_center_x + dx_med


class OptimizedReframerPipeline:
    def __init__(self, **kwargs):
        # Copy all existing parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Enhanced components
        self.detector = OptimizedYoloBallDetector(
            model_name=kwargs.get('model_name', 'yolov8s.pt'),
            device=kwargs.get('device'),
            conf=kwargs.get('conf', 0.2),
            imgsz=kwargs.get('imgsz')
        )
        
        # If user provided a specific class name, map it
        target_class_name = getattr(self, 'target_class_name', None)
        if target_class_name:
            names = getattr(self.detector.model, 'names', None)
            id_to_name = {}
            if isinstance(names, dict):
                id_to_name = {int(k): str(v).lower() for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                id_to_name = {i: str(n).lower() for i, n in enumerate(list(names))}
            wanted = str(target_class_name).strip().lower()
            match_id = None
            for cid, cname in id_to_name.items():
                if cname == wanted:
                    match_id = int(cid)
                    break
            if match_id is not None:
                self.detector.ball_class_ids = [match_id]
                self.detector.selected_class_id = match_id
        
        self.kf = EnhancedKalman1D(
            q=kwargs.get('kf_q', 0.05),
            r=kwargs.get('kf_r', 4.0)
        )
        
        # Frame buffer for lookahead processing
        self.frame_buffer = queue.Queue(maxsize=10)
        self.processing_pool = ThreadPoolExecutor(max_workers=2)
        
        # Appearance toggle
        self.use_appearance = bool(kwargs.get('use_appearance', True))
        # Recovery toggle for heavy TTA searches
        self.allow_tta_recovery = bool(kwargs.get('allow_tta_recovery', True))
        # Sync flag down to detector to avoid attribute errors
        try:
            self.detector.allow_tta_recovery = self.allow_tta_recovery
        except Exception:
            pass
        # Profiling toggle
        self.profile = bool(kwargs.get('profile', False))
        self._perf = {
            'roi_detects': 0,
            'ff_detects': 0,
            'tta_detects': 0,
            'track_updates': 0,
            'misses': 0,
            'detect_time_s': 0.0,
            'frames': 0,
            't_start': time.perf_counter(),
        }
        # Baseline detection interval and ffmpeg flag
        self.base_det_interval = max(1, int(kwargs.get('base_det_interval', 2)))
        # Heavy recovery cooldown (frames) to avoid repeated global searches
        self.recovery_cooldown_frames = int(kwargs.get('recovery_cooldown_frames', 90))
        self._last_recovery_frame = -10_000
        self.use_ffmpeg = bool(kwargs.get('use_ffmpeg', False))
        # Three-pass learning settings
        self.three_pass = bool(kwargs.get('three_pass', False))
        self.learn_stride = max(1, int(kwargs.get('learn_stride', 2)))
        
        # Heavy fallback throttling controls
        self.heavy_every_frames = max(1, int(kwargs.get('heavy_every_frames', 6)))
        self.heavy_max_tiles = max(1, int(kwargs.get('heavy_max_tiles', 2)))
        self.heavy_time_budget_ms = max(0, int(kwargs.get('heavy_time_budget_ms', 12)))

    def _first_pass_detect_fullframe(self) -> List[Optional[float]]:
        """Per-frame full-frame YOLO detection for highest accuracy, seeded for early centering and with fallback on misses"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to open input video")
        meta = self._read_meta(self.input_path)
        xs: List[Optional[float]] = []
        prev_cx: Optional[float] = None
        # If bootstrap found appearance/class, use it implicitly via detector and seed initial center
        boot = self._bootstrap_initial_center_enhanced()
        if boot is not None:
            _, _, cx0 = boot
            prev_cx = float(cx0)
        # Choose imgsz: prefer provided, else device-optimized
        imgsz_eff = getattr(self.detector, 'imgsz', None) or (640 if self.detector.device == 'mps' else 960)
        misses = 0
        prev_prev_cx: Optional[float] = None
        flow = OpticalFlowAssist()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            # Estimate velocity from previous centers to size ROI
            vel = abs((prev_cx - prev_prev_cx)) if (prev_cx is not None and prev_prev_cx is not None) else 0.0
            base_roi = int(getattr(self, 'base_roi_width', 400))
            if vel > 60:
                roi_w = min(W, max(base_roi, base_roi + vel * 4.0))
            elif vel > 30:
                roi_w = min(W, max(base_roi, base_roi + vel * 2.2))
            elif vel > 15:
                roi_w = min(W, max(base_roi, base_roi + vel * 1.4))
            else:
                roi_w = base_roi
            roi_w = int(max(160, min(W, roi_w)))

            cx_choice: Optional[float] = None
            # 1) ROI-first detect around previous center (faster, accurate)
            if prev_cx is not None:
                roi_half = roi_w // 2
                roi_left = int(max(0, min(int(round(prev_cx)) - roi_half, W - roi_w)))
                bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                    frame,
                    roi_left=roi_left,
                    roi_width=roi_w,
                    pref_center_x=prev_cx,
                    conf_override=max(0.08, self.detector.conf * 0.8),
                    imgsz_override=(640 if self.detector.device == 'mps' else 960),
                    use_tta=(vel > 60 and self.allow_tta_recovery),
                )
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cx_choice = float((x1 + x2) / 2.0)
                    flow.init_from_bbox(frame, bbox)

            # 2) Full-frame robust fallback if ROI missed
            if cx_choice is None:
                # Adapt conf/imgsz with miss count and velocity
                use_tta = self.allow_tta_recovery and ((misses >= 4) or (vel > 60))
                big_imgsz = (960 if self.detector.device == 'mps' else (1536 if misses >= 8 or vel > 60 else 960))
                bbox_ff = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                    frame,
                    roi_left=0,
                    roi_width=W,
                    pref_center_x=prev_cx,
                    conf_override=max(0.06, self.detector.conf * (0.6 if misses >= 6 else 0.8)),
                    imgsz_override=big_imgsz,
                    use_tta=use_tta,
                )
                if bbox_ff is not None:
                    x1, y1, x2, y2 = bbox_ff
                    cx_choice = float((x1 + x2) / 2.0)
                    flow.init_from_bbox(frame, bbox_ff)

            # 3) If still nothing, try optical flow to bridge gap, else hold last center
            if cx_choice is None:
                cx_flow = flow.update(frame)
                if cx_flow is not None:
                    cx_choice = float(cx_flow)
                else:
                    cx_choice = prev_cx
                misses += 1
            else:
                misses = 0

            xs.append(cx_choice)
            prev_prev_cx = prev_cx
            prev_cx = cx_choice if cx_choice is not None else prev_cx
            self._perf['frames'] += 1
        cap.release()
        return xs

    def _bootstrap_initial_center_enhanced(self) -> Optional[Tuple[int, Tuple[float, float, float, float], float]]:
        """Enhanced bootstrap with better scoring"""
        meta = self._read_meta(self.input_path)
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return None
        
        best_score = -1.0
        best = None
        frames_to_scan = min(getattr(self, 'bootstrap_frames', 48), meta.num_frames or 48)
        
        # Scan more frames with different strategies
        step = max(1, frames_to_scan // 20)  # Sample frames throughout
        selected_class_local: Optional[int] = None
        ref_hist_local: Optional[np.ndarray] = None
        
        for idx in range(0, frames_to_scan, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            
            out = self.detector._predict_on_optimized(
                frame, 
                conf=0.1,  # Very low confidence for bootstrap
                imgsz=960
            )
            
            if out is None:
                continue
            
            xys, confs, clss = out
            H, W = frame.shape[:2]
            
            for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                if int(cls_id) not in self.detector.ball_class_ids:
                    continue
                
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                area = max(1.0, float((x2 - x1) * (y2 - y1)))
                conf_score = float(c)
                
                # Prefer center-screen positions
                center_dist = abs(cx - W/2) / (W/2)
                center_penalty = 1.0 / (1.0 + center_dist)
                
                # Prefer reasonable sizes
                area_ratio = area / (W * H)
                if 0.0005 < area_ratio < 0.02:
                    area_penalty = 1.0
                else:
                    area_penalty = 0.5
                
                # Prefer circular objects
                aspect_ratio = (x2 - x1) / max(1, y2 - y1)
                aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0))
                
                # Appearance model bootstrap
                patch = frame[int(max(0, y1)):int(min(H, y2)), int(max(0, x1)):int(min(W, x2))]
                cand_hist = self.detector._compute_hs_hist(patch) if self.use_appearance else None
                if self.use_appearance and ref_hist_local is None and cand_hist is not None and conf_score > 0.25:
                    ref_hist_local = cand_hist
                    selected_class_local = int(cls_id)
                hist_sim = (self.detector._hist_similarity(ref_hist_local, cand_hist) if self.use_appearance else None) or 0.5
                
                score = (
                    conf_score * 0.35 +
                    conf_score * center_penalty * 0.25 +
                    conf_score * area_penalty * 0.15 +
                    conf_score * aspect_penalty * 0.1 +
                    (hist_sim * 0.15 if self.use_appearance else 0.0)
                )
                
                if score > best_score:
                    best_score = score
                    best = (idx, (float(x1), float(y1), float(x2), float(y2)), cx)
        
        cap.release()
        
        # Commit selected class and histogram if found
        if selected_class_local is not None:
            self.detector.set_target_appearance(selected_class_local, ref_hist_local)
        
        return best

    def _first_pass_detect_track_optimized(self) -> List[Optional[float]]:
        """Optimized detection and tracking with parallel processing"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to open input video")
            
        xs: List[Optional[float]] = []
        tracker = ParallelBoxTracker(tracker_pref=getattr(self, 'tracker_pref', 'auto'))
        flow = OpticalFlowAssist()
        
        frame_idx = 0
        frames_since_det = 1_000_000
        consecutive_misses = 0
        stable_count = 0
        locked = False
        
        meta = self._read_meta(self.input_path)
        frame_w = meta.width
        
        # Enhanced bootstrap
        boot = self._bootstrap_initial_center_enhanced()
        if boot is not None:
            _, bbox0, cx0 = boot
            self.kf.update(cx0)
            locked = True
            flow.init_from_bbox(cap.read()[1] if cap.set(cv2.CAP_PROP_POS_FRAMES, boot[0]) else None, bbox0)  # best-effort
        else:
            self.kf.update(frame_w / 2.0)
        
        prev_est = None
        velocity_buffer = []
        base_roi = int(getattr(self, 'base_roi_width', 400))
        max_roi = min(frame_w, max(base_roi * 3, 800))
        base_interval = max(1, int(getattr(self, 'base_det_interval', 2)))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Predict next position with velocity consideration
            cx_pred = self.kf.predict()
            if cx_pred is None:
                cx_pred = frame_w / 2.0
            
            velocity = self.kf.get_velocity()
            velocity_buffer.append(abs(velocity))
            if len(velocity_buffer) > 5:
                velocity_buffer.pop(0)
            avg_velocity = np.mean(velocity_buffer)
            
            # Adaptive parameters based on velocity and user ROI
            if avg_velocity > 60:  # Extremely fast
                cur_interval = 1
                roi_w = int(min(max_roi, max(base_roi, base_roi + avg_velocity * 4.5)))
            elif avg_velocity > 30:  # Fast
                cur_interval = 1
                roi_w = int(min(max_roi, max(base_roi, base_roi + avg_velocity * 2.4)))
            elif avg_velocity > 15:  # Medium speed
                cur_interval = max(2, base_interval + 1)
                roi_w = int(min(max_roi, max(base_roi, base_roi + avg_velocity * 1.6)))
            else:  # Slow/stable
                cur_interval = max(4, base_interval + 3)
                roi_w = int(base_roi)
            roi_w = int(max(160, min(frame_w, roi_w)))
            
            # ROI calculation with velocity prediction
            roi_half = min(frame_w // 2, roi_w // 2)
            predicted_next_cx = cx_pred + velocity * 2.5  # Look ahead more for fast moves
            roi_left = int(round(predicted_next_cx - roi_half))
            roi_left = max(0, min(roi_left, frame_w - (roi_half * 2)))
            
            use_detect = (frames_since_det >= cur_interval) or (not tracker.trackers) or (not locked)
            if bool(getattr(self, 'detect_every_frame', False)):
                cur_interval = 1
                use_detect = True
            cx_meas = None
            meas_var: Optional[float] = None
            
            if use_detect:
                # Enhanced detection with velocity-based confidence
                conf_adjust = max(0.08, 0.28 - avg_velocity * 0.004)  # Lower conf for fast balls
                
                if locked:
                    t0 = time.perf_counter()
                    bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                        frame,
                        roi_left=roi_left,
                        roi_width=roi_w,
                        pref_center_x=predicted_next_cx,
                        conf_override=conf_adjust,
                        imgsz_override=(640 if (self.detector.device == 'mps') else (960 if avg_velocity > 30 else None)),
                        use_tta=False,
                    )
                    self._perf['detect_time_s'] += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                        frame,
                        roi_left=0,
                        roi_width=frame_w,
                        pref_center_x=None,
                        conf_override=0.1,
                        imgsz_override=(640 if self.detector.device == 'mps' else 720),
                        use_tta=False,
                    )
                    self._perf['detect_time_s'] += time.perf_counter() - t0
                
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cx_det = (x1 + x2) / 2.0
                    self._perf['roi_detects'] += 1 if locked else 0
                    self._perf['ff_detects'] += 0 if locked else 1
                    
                    # Enhanced outlier detection
                    if locked and abs(cx_det - cx_pred) > (120 + avg_velocity * 2.5):
                        # Still use tracker result for now
                        ok_t, cx_t = tracker.update(frame)
                        if ok_t and cx_t is not None:
                            cx_meas = cx_t
                            meas_var = 9.0  # tracker measurement variance (less trusted than detector)
                        frames_since_det += 1
                        consecutive_misses += 1
                        self._perf['misses'] += 1
                    else:
                        # Good detection, reinitialize tracker
                        tracker.init_with_bbox(frame, bbox)
                        cx_meas = cx_det
                        meas_var = 2.5  # detector measurement variance (more trusted)
                        frames_since_det = 0
                        consecutive_misses = 0
                        stable_count += 1
                        locked = True
                else:
                    # No detection, attempt a one-shot full-frame fallback before tracker
                    t0 = time.perf_counter()
                    bbox_ff = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                        frame,
                        roi_left=0,
                        roi_width=frame_w,
                        pref_center_x=None,
                        conf_override=0.08,
                        imgsz_override=720,
                        use_tta=False,
                    )
                    self._perf['detect_time_s'] += time.perf_counter() - t0
                    if bbox_ff is not None:
                        x1, y1, x2, y2 = bbox_ff
                        cx_meas = (x1 + x2) / 2.0
                        meas_var = 3.0
                        tracker.init_with_bbox(frame, bbox_ff)
                        frames_since_det = 0
                        consecutive_misses = 0
                        locked = True
                        self._perf['ff_detects'] += 1
                        if self.allow_tta_recovery:
                            self._perf['tta_detects'] += 1
                    else:
                        # Fall back to tracker if available, else optical flow
                        ok_t, cx_t = tracker.update(frame)
                        if ok_t and cx_t is not None:
                            cx_meas = cx_t
                            meas_var = 9.0
                        else:
                            cx_flow = flow.update(frame)
                            if cx_flow is not None:
                                cx_meas = cx_flow
                                meas_var = 16.0
                        frames_since_det += 1
                        consecutive_misses += 1
                        self._perf['misses'] += 1
            else:
                # Use tracker only (with optical flow assist)
                ok_t, cx_t = tracker.update(frame)
                if ok_t and cx_t is not None:
                    cx_meas = cx_t
                    meas_var = 9.0
                    frames_since_det += 1
                    consecutive_misses = 0
                    self._perf['track_updates'] += 1
                else:
                    cx_flow = flow.update(frame)
                    if cx_flow is not None:
                        cx_meas = cx_flow
                        meas_var = 16.0
                    else:
                        frames_since_det = cur_interval  # Force detection next frame
                        consecutive_misses += 1
                        stable_count = 0
                        self._perf['misses'] += 1
            
            # If we've missed too long, do a global high-res TTA search to re-acquire
            if cx_meas is None and consecutive_misses >= 8 and (frame_idx - self._last_recovery_frame) >= self.recovery_cooldown_frames:
                t0 = time.perf_counter()
                bbox_global = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                    frame,
                    roi_left=0,
                    roi_width=frame_w,
                    pref_center_x=None,
                    conf_override=0.06,
                    imgsz_override=(640 if self.detector.device == 'mps' else 960),
                    use_tta=self.allow_tta_recovery,
                )
                self._perf['detect_time_s'] += time.perf_counter() - t0
                if bbox_global is not None:
                    x1, y1, x2, y2 = bbox_global
                    cx_meas = (x1 + x2) / 2.0
                    meas_var = 3.0
                    tracker.init_with_bbox(frame, bbox_global)
                    frames_since_det = 0
                    consecutive_misses = 0
                    locked = True
                    self._perf['ff_detects'] += 1
                    if self.allow_tta_recovery:
                        self._perf['tta_detects'] += 1
                # mark last heavy recovery attempt
                self._last_recovery_frame = frame_idx
            
            # Update Kalman filter (adaptive measurement noise)
            if cx_meas is None:
                est = self.kf.update_with_variance(None, None)
            else:
                est = self.kf.update_with_variance(cx_meas, meas_var)
            
            xs.append(float(est) if est is not None else (float(cx_pred) if cx_pred is not None else None))
            self._perf['frames'] += 1
            if self.profile and (self._perf['frames'] % 60 == 0):
                elapsed = time.perf_counter() - self._perf['t_start']
                fps = self._perf['frames'] / max(1e-6, elapsed)
                print(f"Profile@{self._perf['frames']}: fps={fps:.1f} roi={self._perf['roi_detects']} ff={self._perf['ff_detects']} tta={self._perf['tta_detects']} misses={self._perf['misses']} detect_time={self._perf['detect_time_s']:.2f}s")
            prev_est = float(est) if est is not None else prev_est
            frame_idx += 1
            
        cap.release()
        return xs

    def _first_pass_detect_track_bytetrack(self) -> List[Optional[float]]:
        """Use Ultralytics ByteTrack to robustly follow the ball across frames"""
        meta = self._read_meta(self.input_path)
        frame_count_expected = meta.num_frames
        
        # Choose imgsz like in detector pred
        imgsz_eff = getattr(self.detector, 'imgsz', None) or (480 if meta.height > 720 else (416 if meta.height > 480 else 320))
        
        xs: List[Optional[float]] = []
        tracked_id: Optional[int] = None
        misses = 0
        
        # Initialize Kalman around center to fuse with tracker stream
        if not getattr(self.kf, 'initialized', False):
            self.kf.update(meta.width / 2.0)
        
        results_stream = self.detector.model.track(
            source=self.input_path,
            device=self.detector.device if self.detector.device in ["cpu", "cuda", "mps"] else None,
            persist=True,
            stream=True,
            conf=max(0.08, self.detector.conf * 0.9),
            iou=0.3,
            imgsz=imgsz_eff,
            classes=self.detector.ball_class_ids,
            save=False,
            verbose=False,
        )
        
        ref_hist = self.detector.ref_hist if getattr(self, 'use_appearance', True) else None
        prev_cx: Optional[float] = None
        
        for r in results_stream:
            cx_choice: Optional[float] = None
            meas_var: Optional[float] = None
            if r is None or r.boxes is None or len(r.boxes) == 0:
                # Fallback single-frame detect when tracker misses for too long
                misses += 1
                if hasattr(r, 'orig_img') and r.orig_img is not None and misses >= 5:
                    fallback = self.detector._predict_on_optimized(
                        r.orig_img,
                        conf=0.06,
                        imgsz=max(640, min(960, imgsz_eff)),
                        use_tta=self.allow_tta_recovery,
                    )
                    if fallback is not None:
                        xys, confs, clss = fallback
                        # Pick best center by confidence
                        best_i = int(np.argmax(confs))
                        x1, y1, x2, y2 = xys[best_i]
                        cx_choice = float((x1 + x2) / 2.0)
                        # Confidence → measurement variance mapping (higher conf → smaller variance)
                        conf_best = float(confs[best_i])
                        meas_var = float(np.interp(conf_best, [0.05, 0.9], [9.0, 2.0]))
                        misses = 0
                        tracked_id = None
                # Update KF even on miss to keep prediction flowing
                est = self.kf.update_with_variance(cx_choice, meas_var)
                xs.append(float(est) if est is not None else cx_choice)
                continue
            
            misses = 0
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, 'id', None) is not None else None
            
            # Filter to ball classes only (already set by classes param)
            candidates = []
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                if ids is not None:
                    tid = int(ids[i])
                else:
                    tid = -1
                conf = float(confs[i])
                
                # Compute center
                cx = (float(x1) + float(x2)) / 2.0
                
                # Appearance score if available
                hist_score = 0.0
                if ref_hist is not None and r.orig_img is not None:
                    H, W = r.orig_shape if hasattr(r, 'orig_shape') else r.orig_img.shape[:2]
                    x1i = int(max(0, x1)); y1i = int(max(0, y1)); x2i = int(min(W, x2)); y2i = int(min(H, y2))
                    patch = r.orig_img[y1i:y2i, x1i:x2i] if (y2i>y1i and x2i>x1i) else None
                    if patch is not None:
                        cand_hist = self.detector._compute_hs_hist(patch)
                        sim = self.detector._hist_similarity(ref_hist, cand_hist)
                        hist_score = float(sim) if sim is not None else 0.0
                
                # Distance bias to previous center to avoid jumps
                dist_bias = 0.0
                if prev_cx is not None:
                    dist = abs(cx - prev_cx)
                    # Penalize > 15% screen width shifts strongly
                    dist_bias = -min(1.0, dist / max(1.0, meta.width * 0.15)) * 0.3
                # Prioritize tracked id, otherwise highest (conf + hist + proximity)
                score = conf * 0.8 + hist_score * 0.2 + dist_bias
                candidates.append((tid, cx, score, conf))
            
            if tracked_id is not None and any(tid == tracked_id for tid, _, _, _ in candidates):
                for tid, cx, _, conf in candidates:
                    if tid == tracked_id:
                        cx_choice = cx
                        meas_var = float(np.interp(conf, [0.05, 0.9], [9.0, 2.0]))
                        break
            else:
                # Pick best candidate and set tracked id if available
                candidates.sort(key=lambda x: x[2], reverse=True)
                tid_best, cx_best, _, conf_best = candidates[0]
                cx_choice = cx_best
                meas_var = float(np.interp(conf_best, [0.05, 0.9], [9.0, 2.0]))
                if tid_best != -1:
                    tracked_id = tid_best
            
            est = self.kf.update_with_variance(cx_choice, meas_var)
            xs.append(float(est) if est is not None else cx_choice)
            prev_cx = cx_choice if cx_choice is not None else prev_cx
            self._perf['frames'] += 1
            if self.profile and (self._perf['frames'] % 60 == 0):
                elapsed = time.perf_counter() - self._perf['t_start']
                fps = self._perf['frames'] / max(1e-6, elapsed)
                print(f"Profile@{self._perf['frames']}: fps={fps:.1f} (bytetrack) misses={misses}")
         
        # Ensure length matches expected frames
        if frame_count_expected and len(xs) < frame_count_expected:
            xs.extend([xs[-1] if xs else None] * (frame_count_expected - len(xs)))
        return xs

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

    def run(self) -> None:
        """Run optimized pipeline"""
        print("Starting optimized ball tracking pipeline...")
        start_time = time.time()
        
        meta = self._read_meta(self.input_path)
        # Log device info to confirm Mac GPU usage
        try:
            import torch  # type: ignore
            mps_ok = getattr(torch.backends, "mps", None)
            mps_avail = bool(mps_ok.is_available()) if mps_ok else False
            print(f"Device: {self.detector.device} (MPS available={mps_avail})")
        except Exception:
            print(f"Device: {self.detector.device}")
        crop_w = self._compute_crop_width(meta.height, meta.width)
        out_w, out_h = self._compute_output_size(
            getattr(self, 'out_height', None), 
            meta.height
        )
        fps_out = getattr(self, 'out_fps', None) or meta.fps
        
        if self.three_pass:
            print("Phase 0: Learning ball appearance across video...")
            self._pass1_learn_ball_appearance()
        
        # Use optimized tracking
        print("Phase 1: Ball detection and tracking...")
        backend = getattr(self, 'backend', 'yolo')
        full_detect = bool(getattr(self, 'full_detect', False) or self.three_pass)
        if full_detect:
            xs_raw = self._first_pass_detect_fullframe()
        elif backend == 'yolo-bytetrack':
            try:
                xs_raw = self._first_pass_detect_track_bytetrack()
            except Exception as e:
                print(f"ByteTrack backend failed ({e}); falling back to YOLO+tracker")
                xs_raw = self._first_pass_detect_track_optimized()
        else:
            xs_raw = self._first_pass_detect_track_optimized()

        if self.profile:
            elapsed = time.perf_counter() - self._perf['t_start']
            fps = self._perf['frames'] / max(1e-6, elapsed)
            print(f"Profile: frames={self._perf['frames']} fps={fps:.1f} roi={self._perf['roi_detects']} ff={self._perf['ff_detects']} tta={self._perf['tta_detects']} misses={self._perf['misses']} detect_time={self._perf['detect_time_s']:.2f}s")
        
        if len(xs_raw) == 0:
            raise SystemExit("No frames read from input video")
            
        print(f"Phase 2: Trajectory smoothing ({len(xs_raw)} frames)...")
        smoother = TrajectorySmoother(window_size=getattr(self, 'smooth_window', 15))
        xs_smooth = smoother.smooth(xs_raw, fps=meta.fps)
        
        print("Phase 3: Crop planning...")
        if getattr(self, 'strict_center', False):
            # Center on ball but keep it inside the crop with a minimum margin near edges
            centers = self._plan_centers_strict_with_margin(
                xs_smooth, crop_w, meta.width, int(getattr(self, 'margin', 60))
            )
        elif getattr(self, 'sticky_window', False):
            centers = self._plan_centers_sticky(xs_smooth, crop_w, meta.width)
        else:
            planner = CropPlanner(
                frame_width=meta.width,
                output_width=crop_w,
                max_move_per_frame=getattr(self, 'max_move', 80),
                margin=getattr(self, 'margin', 60),
                max_accel_per_frame=getattr(self, 'max_accel', 40),
                deadband_px=getattr(self, 'deadband_px', 4),
                jerk_px=getattr(self, 'jerk_px', 0),
            )
            centers = planner.plan_centers(xs_smooth)
        
        print("Phase 4: Video rendering...")
        self._second_pass_write(centers=centers, crop_width=crop_w, out_size=(out_w, out_h), fps_out=fps_out)
        
        total_time = time.time() - start_time
        print(f"✅ Complete! Total time: {total_time:.1f}s ({len(xs_raw)/total_time:.1f} FPS)")
    
    def _plan_centers_sticky(self, smoothed_xs: np.ndarray, crop_width: int, frame_width: int) -> np.ndarray:
        """Keep previous crop center until the ball leaves a center bound; then recenter.
        Bound size is controlled by self.center_bound_px.
        """
        centers = np.empty_like(smoothed_xs)
        half = crop_width / 2.0
        min_center = half + float(getattr(self, 'margin', 60))
        max_center = frame_width - half - float(getattr(self, 'margin', 60))
        bound = float(getattr(self, 'center_bound_px', 40))
        prev_center: Optional[float] = None
        for i, cx in enumerate(smoothed_xs):
            cx = float(cx)
            if prev_center is None:
                new_c = max(min_center, min(max_center, cx))
            else:
                # Keep if within [prev_center - bound, prev_center + bound]
                if abs(cx - prev_center) <= bound:
                    new_c = prev_center
                else:
                    new_c = max(min_center, min(max_center, cx))
            centers[i] = new_c
            prev_center = new_c
        return centers

    def _plan_centers_strict_with_margin(self, smoothed_xs: np.ndarray, crop_width: int, frame_width: int, edge_margin: int) -> np.ndarray:
        centers = np.empty_like(smoothed_xs)
        half = crop_width / 2.0
        # Keep a small safety margin near frame borders so the ball is included even if not perfectly centered
        min_center = half + max(0, float(edge_margin) / 2.0)
        max_center = frame_width - half - max(0, float(edge_margin) / 2.0)
        for i, cx in enumerate(smoothed_xs):
            centers[i] = max(min_center, min(max_center, float(cx)))
        return centers

    # Keep all existing helper methods
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

    def _second_pass_write(self, centers: np.ndarray, crop_width: int, out_size: Tuple[int, int], fps_out: float) -> None:
        out_w, out_h = out_size
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to reopen input video for writing")

        # Use ffmpeg pipe if requested (faster encoders)
        if self.use_ffmpeg:
            # Choose encoder based on platform and GPU availability
            sys = platform.system().lower()
            encoder = 'libx264'
            if sys == 'darwin':
                encoder = 'h264_videotoolbox'
            elif sys == 'linux':
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        encoder = 'h264_nvenc'
                except Exception:
                    encoder = 'libx264'
            elif sys == 'windows':
                encoder = 'h264_nvenc'

            ff_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{out_w}x{out_h}', '-r', str(fps_out),
                '-i', '-',
                '-c:v', encoder,
                '-preset', 'veryfast',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                self.output_path,
            ]
            proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)
            pipe = proc.stdin
        else:
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

            if self.use_ffmpeg:
                pipe.write(crop.tobytes())  # type: ignore
            else:
                writer.write(crop)
            idx += 1

        cap.release()
        if self.use_ffmpeg:
            assert pipe is not None
            pipe.close()  # type: ignore
            proc.wait()
        else:
            writer.release()

    def _pass1_learn_ball_appearance(self) -> None:
        """Scan the whole video (with stride) to learn dominant ball class and appearance histogram."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return
        meta = self._read_meta(self.input_path)
        sum_hist_by_class: dict[int, np.ndarray] = {}
        weight_by_class: dict[int, float] = {}
        frames = int(meta.num_frames or 0)
        imgsz_eff = getattr(self.detector, 'imgsz', None) or (640 if self.detector.device == 'mps' else 480)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx % self.learn_stride) != 0:
                frame_idx += 1
                continue
            out = self.detector._predict_on_optimized(
                frame,
                conf=max(0.08, self.detector.conf * 0.7),
                imgsz=imgsz_eff,
                use_tta=False,
            )
            if out is not None:
                xys, confs, clss = out
                H, W = frame.shape[:2]
                for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                    if int(cls_id) not in self.detector.ball_class_ids:
                        continue
                    # Compute histogram in candidate box
                    x1i = int(max(0, x1)); y1i = int(max(0, y1)); x2i = int(min(W, x2)); y2i = int(min(H, y2))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    patch = frame[y1i:y2i, x1i:x2i]
                    cand_hist = self.detector._compute_hs_hist(patch)
                    if cand_hist is None:
                        continue
                    weight = float(c)
                    cid = int(cls_id)
                    if cid not in sum_hist_by_class:
                        sum_hist_by_class[cid] = cand_hist * weight
                        weight_by_class[cid] = weight
                    else:
                        sum_hist_by_class[cid] += cand_hist * weight
                        weight_by_class[cid] += weight
            frame_idx += 1
        cap.release()
        if not weight_by_class:
            return
        # Pick best class by total weight and compute normalized avg histogram
        best_c = max(weight_by_class.items(), key=lambda kv: kv[1])[0]
        avg_hist = sum_hist_by_class[best_c] / max(1e-6, weight_by_class[best_c])
        avg_hist = cv2.normalize(avg_hist, avg_hist).flatten()
        self.detector.set_target_appearance(best_c, avg_hist)


# Import remaining classes from original code
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

    def smooth(self, series: List[Optional[float]], fps: float = 30.0) -> np.ndarray:
        xs = np.array([np.nan if v is None else float(v) for v in series], dtype=float)
        xs_interp = self._interpolate_nans(xs)
        n = len(xs_interp)

        # Velocity estimate to adapt smoothing aggressiveness
        try:
            vel = np.abs(np.diff(xs_interp, prepend=xs_interp[0]))
            med_vel = float(np.median(vel))
        except Exception:
            med_vel = 0.0

        # 1) Median filter to remove spikes
        try:
            k_med = 7 if n >= 7 else (5 if n >= 5 else 3)
            xs_med = medfilt(xs_interp, kernel_size=k_med)
        except Exception:
            xs_med = xs_interp

        # 2) Zero-phase low-pass (Butterworth) with adaptive cutoff to avoid lag on fast motion
        try:
            # Increase cutoff if motion is fast to reduce lag
            base_cut = 1.2
            if med_vel > 24:
                cutoff_hz = 3.0
            elif med_vel > 12:
                cutoff_hz = 2.2
            else:
                cutoff_hz = base_cut
            nyq = 0.5 * max(1.0, float(fps))
            wn = min(0.99, max(0.01, cutoff_hz / nyq))
            b, a = butter(N=2, Wn=wn, btype='low')
            xs_lp = filtfilt(b, a, xs_med, method="gust")
        except Exception:
            xs_lp = xs_med

        # 3) Savitzky-Golay for gentle smoothing and preserving shape (adaptive window on speed)
        desired_win = self.window_size if self.window_size % 2 == 1 else self.window_size - 1
        if med_vel > 24:
            desired_win = min(desired_win, 7)
        elif med_vel > 12:
            desired_win = min(desired_win, 9)
        win = max(5, min(desired_win, n if n % 2 == 1 else n - 1))
        if win >= 5 and win <= n:
            try:
                xs_sg = savgol_filter(xs_lp, window_length=win, polyorder=2, mode="interp")
            except Exception:
                xs_sg = xs_lp
        else:
            xs_sg = xs_lp

        return xs_sg


class CropPlanner:
    def __init__(self, frame_width: int, output_width: int, max_move_per_frame: int, margin: int = 0, max_accel_per_frame: int = 0, deadband_px: int = 0, jerk_px: int = 0) -> None:
        self.frame_width = int(frame_width)
        self.output_width = int(output_width)
        self.max_move = int(max_move_per_frame)
        self.margin = max(0, int(margin))
        self.max_accel = max(0, int(max_accel_per_frame))
        self.deadband = max(0, int(deadband_px))
        self.jerk = max(0, int(jerk_px))
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
        prev_accel: float = 0.0
        for i, cx in enumerate(smoothed_xs):
            cx = self._clamp(float(cx), self.min_center, self.max_center)
            if prev_center is None:
                centers[i] = cx
                prev_center = cx
                prev_delta = 0.0
            else:
                desired_delta = cx - prev_center
                # Apply deadband to reduce micro jitter
                if abs(desired_delta) <= self.deadband:
                    desired_delta = 0.0
                if prev_delta is None:
                    prev_delta = 0.0
                if self.max_accel > 0:
                    accel = desired_delta - prev_delta
                    # Apply jerk limit first (limit change in acceleration)
                    if self.jerk > 0:
                        jerk = accel - prev_accel
                        jerk = self._clamp(jerk, -self.jerk, self.jerk)
                        accel = prev_accel + jerk
                    # Then acceleration clamp
                    accel = self._clamp(accel, -self.max_accel, self.max_accel)
                    desired_delta = prev_delta + accel
                    prev_accel = accel - prev_delta
                if abs(desired_delta) > self.max_move:
                    desired_delta = math.copysign(self.max_move, desired_delta)
                new_center = prev_center + desired_delta
                new_center = self._clamp(new_center, self.min_center, self.max_center)
                centers[i] = new_center
                prev_delta = new_center - prev_center
                prev_center = new_center
        return centers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized reframe 16:9 video to 9:16 centered on a sports ball")
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
    parser.add_argument("--deadband", type=int, default=4, help="Do not move crop for shifts smaller than this many pixels (jitter reduction)")
    # Compatibility with CLI options
    parser.add_argument("--bootstrap-frames", type=int, default=48, help="Initial frames to scan for bootstrapping the ball lock")
    parser.add_argument("--roi", type=int, default=400, help="Base ROI width (in px) for detection around predicted center")
    # New selection/appearance flags
    parser.add_argument("--target-class", type=str, default=None, help="Force a specific class name (e.g., 'tennis ball', 'basketball')")
    parser.add_argument("--no-appearance", action="store_true", help="Disable color histogram appearance model weighting")
    # Backend selection
    parser.add_argument("--backend", type=str, choices=["yolo", "yolo-bytetrack"], default="yolo", help="Tracking backend")
    parser.add_argument("--profile", action="store_true", help="Print periodic profiling info (FPS, misses, detects)")
    parser.add_argument("--no-tta-recovery", action="store_true", help="Disable heavy TTA full-frame recoveries for speed")
    # Speed/quality controls
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO input size (e.g., 480, 640, 960)")
    parser.add_argument("--det-interval", type=int, default=2, help="Baseline detection interval for YOLO backend (1=every frame)")
    parser.add_argument("--use-ffmpeg", action="store_true", help="Use ffmpeg pipe for faster video encoding")
    parser.add_argument("--jerk", type=int, default=0, help="Max allowed change in crop-center movement per frame in px (jerk clamp)")
    parser.add_argument("--recovery-cooldown", type=int, default=90, help="Frames to wait between heavy global recoveries")
    parser.add_argument("--detect-every-frame", action="store_true", help="Force YOLO detection on every frame (bypasses tracker interval)")
    parser.add_argument("--full-detect", action="store_true", help="Run per-frame full-frame detection pass for maximum accuracy (slower)")
    # New: strict centering mode
    parser.add_argument("--strict-center", action="store_true", help="Always center crop on smoothed ball x; bypass motion constraints.")
    # New: sticky window mode
    parser.add_argument("--sticky-window", action="store_true", help="Keep crop center until ball leaves a center bound; then recenter.")
    parser.add_argument("--center-bound", type=int, default=40, help="Half-width of the stickiness bound in pixels (default: 40)")
    parser.add_argument("--three-pass", action="store_true", help="Three-pass pipeline: learn appearance, detect every frame, then crop")
    parser.add_argument("--learn-stride", type=int, default=2, help="Frame stride for appearance learning pass (default: 2)")
    # Heavy fallback controls
    parser.add_argument("--heavy-every-frames", type=int, default=6, help="Only run heavy fallbacks every N frames when missing (default: 6)")
    parser.add_argument("--heavy-max-tiles", type=int, default=2, help="Max tiles to scan per heavy fallback (default: 2)")
    parser.add_argument("--heavy-time-budget-ms", type=int, default=12, help="Per-frame time budget in ms for heavy fallbacks (default: 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    pipeline = OptimizedReframerPipeline(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        conf=args.conf,
        smooth_window=args.smooth,
        max_move=args.max_move,
        out_height=args.height,
        out_fps=args.fps,
        margin=args.margin,
        max_accel=args.max_accel,
        deadband_px=args.deadband,
        kf_q=0.05,
        kf_r=4.0,
        bootstrap_frames=args.bootstrap_frames,
        base_roi_width=args.roi,
        target_class_name=(args.target_class or None),
        use_appearance=(not args.no_appearance),
        backend=args.backend,
        profile=bool(args.profile),
        allow_tta_recovery=(not args.no_tta_recovery),
        base_det_interval=max(1, int(args.det_interval)),
        use_ffmpeg=bool(args.use_ffmpeg),
        jerk_px=args.jerk,
        recovery_cooldown_frames=int(args.recovery_cooldown),
        detect_every_frame=bool(args.detect_every_frame),
        full_detect=bool(args.full_detect),
        three_pass=bool(args.three_pass),
        learn_stride=int(args.learn_stride),
        # New flag passed to pipeline
        strict_center=bool(args.strict_center),
        sticky_window=bool(args.sticky_window),
        center_bound_px=int(args.center_bound),
        heavy_every_frames=int(args.heavy_every_frames),
        heavy_max_tiles=int(args.heavy_max_tiles),
        heavy_time_budget_ms=int(args.heavy_time_budget_ms),
    )
    
    pipeline.run()
    print(f"✅ Optimized reframing complete: {args.output}")


if __name__ == "__main__":
    main()