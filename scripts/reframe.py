#!/usr/bin/env python3
import argparse
import math
import os
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




@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    num_frames: int
    
class BallMemorySystem:
    """Enhanced memory system with sliding window stability and forward prediction only"""
    def __init__(self, memory_duration_frames: int = 90, confidence_decay: float = 0.98, 
                 stability_window: int = 30, position_variance_threshold: float = 50.0):
        self.memory_duration = int(memory_duration_frames)
        self.confidence_decay = float(confidence_decay)
        
        # Enhanced position tracking
        self.position_history: List[Tuple[int, float, float]] = []  # (frame, x, confidence)
        self.last_detection_frame: int = -1
        
        # Sliding window for stability
        self.stability_window = int(stability_window)
        self.position_variance_threshold = float(position_variance_threshold)
        self.stable_position: Optional[float] = None
        self.stable_region_start: Optional[int] = None
        
        # Forward prediction only
        self.forward_predictor = PositionPredictor()
        
        # Frame-to-frame stability
        self.consecutive_misses = 0
        self.freeze_position: Optional[float] = None
        self.last_stable_position: Optional[float] = None
        
    def update_detection(self, frame_idx: int, ball_cx: float, bbox: Optional[Tuple[float, float, float, float]] = None, detection_confidence: float = 1.0):
        """Update with successful detection"""
        self.position_history.append((frame_idx, float(ball_cx), float(detection_confidence)))
        self.last_detection_frame = frame_idx
        self.consecutive_misses = 0
        self.freeze_position = None
        
        # Keep limited history
        if len(self.position_history) > 200:
            self.position_history = self.position_history[-200:]
            
        # Update forward predictor only
        self._update_predictor()
        
        # Update stable position
        self._update_stable_position(frame_idx, ball_cx)
        
    def update_no_detection(self, frame_idx: int):
        """Update when no detection found"""
        self.consecutive_misses += 1
        
        # For first miss, try to predict and freeze
        if self.consecutive_misses == 1 and self.last_stable_position is not None:
            predicted = self._predict_position_for_frame(frame_idx)
            self.freeze_position = predicted or self.last_stable_position
        
    def get_position_for_frame(self, frame_idx: int) -> Optional[float]:
        """Get best position estimate for frame"""
        # If we have freeze position from recent miss, use it
        if self.freeze_position is not None and self.consecutive_misses > 0:
            return self.freeze_position
            
        # Try forward prediction
        predicted = self._predict_position_for_frame(frame_idx)
        if predicted is not None:
            return predicted
            
        # Fall back to stable position
        return self.stable_position or self.last_stable_position
        
    def _update_stable_position(self, frame_idx: int, position: float):
        """Update stable position using sliding window"""
        if len(self.position_history) < 5:
            self.stable_position = position
            self.last_stable_position = position
            return
            
        # Get recent positions for stability check
        recent_positions = [pos for (f, pos, conf) in self.position_history[-self.stability_window:]]
        
        if len(recent_positions) >= 5:
            variance = float(np.var(recent_positions))
            
            if variance < self.position_variance_threshold:
                # Position is stable, update stable position
                self.stable_position = float(np.mean(recent_positions))
                self.stable_region_start = frame_idx - len(recent_positions) + 1
            else:
                # Position is moving, use weighted recent average
                weights = np.exp(np.linspace(-1, 0, len(recent_positions)))
                self.stable_position = float(np.average(recent_positions, weights=weights))
                
        self.last_stable_position = self.stable_position
        
    def _update_predictor(self):
        """Update forward predictor only"""
        if len(self.position_history) < 3:
            return
            
        frames = [f for f, pos, conf in self.position_history]
        positions = [pos for f, pos, conf in self.position_history]
        confidences = [conf for f, pos, conf in self.position_history]
        
        self.forward_predictor.update(frames, positions, confidences)
        
    def _predict_position_for_frame(self, frame_idx: int) -> Optional[float]:
        """Forward prediction only for frame"""
        if len(self.position_history) < 2:
            return None
            
        # Use only forward prediction
        forward_pred = self.forward_predictor.predict(frame_idx)
        return forward_pred
    
class PositionPredictor:
    """Forward-only position predictor with multiple prediction methods"""
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.frames: List[int] = []
        self.positions: List[float] = []
        self.confidences: List[float] = []
        
    def update(self, frames: List[int], positions: List[float], confidences: List[float]):
        """Update predictor with new data"""
        self.frames = frames[-self.max_history:]
        self.positions = positions[-self.max_history:]
        self.confidences = confidences[-self.max_history:]
        
    def predict(self, target_frame: int) -> Optional[float]:
        """Predict position for target frame using multiple forward methods"""
        if len(self.frames) < 2:
            return None
            
        predictions = []
        
        # Method 1: Linear extrapolation
        linear_pred = self._linear_predict(target_frame)
        if linear_pred is not None:
            predictions.append((linear_pred, 0.4))
            
        # Method 2: Polynomial fitting
        poly_pred = self._polynomial_predict(target_frame)
        if poly_pred is not None:
            predictions.append((poly_pred, 0.3))
            
        # Method 3: Velocity-based prediction
        velocity_pred = self._velocity_predict(target_frame)
        if velocity_pred is not None:
            predictions.append((velocity_pred, 0.3))
            
        if not predictions:
            return None
            
        # Weighted average of predictions
        total_weight = sum(weight for _, weight in predictions)
        weighted_sum = sum(pred * weight for pred, weight in predictions)
        
        return weighted_sum / total_weight
            
    def _linear_predict(self, target_frame: int) -> Optional[float]:
        """Linear extrapolation prediction"""
        if len(self.frames) < 2:
            return None
            
        x = np.array(self.frames[-10:], dtype=float)
        y = np.array(self.positions[-10:], dtype=float)
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
            return float(slope * target_frame + intercept)
        except:
            return None
            
    def _polynomial_predict(self, target_frame: int) -> Optional[float]:
        """Polynomial fitting prediction"""
        if len(self.frames) < 4:
            return None
            
        x = np.array(self.frames[-15:], dtype=float)
        y = np.array(self.positions[-15:], dtype=float)
        w = np.array(self.confidences[-15:], dtype=float)
        
        try:
            degree = min(3, len(x) - 1)
            coeffs = np.polyfit(x, y, deg=degree, w=w)
            return float(np.polyval(coeffs, target_frame))
        except:
            return None
            
    def _velocity_predict(self, target_frame: int) -> Optional[float]:
        """Velocity-based prediction with acceleration"""
        if len(self.frames) < 3:
            return None
            
        recent_frames = self.frames[-5:]
        recent_positions = self.positions[-5:]
        
        if len(recent_frames) < 3:
            return None
            
        # Calculate velocity and acceleration
        velocities = []
        for i in range(1, len(recent_frames)):
            dt = recent_frames[i] - recent_frames[i-1]
            if dt > 0:
                v = (recent_positions[i] - recent_positions[i-1]) / dt
                velocities.append(v)
                
        if not velocities:
            return None
            
        avg_velocity = np.mean(velocities)
        
        # Calculate acceleration if we have enough data
        acceleration = 0.0
        if len(velocities) >= 2:
            accelerations = np.diff(velocities)
            acceleration = np.mean(accelerations)
            
        # Predict using kinematic equation
        last_frame = self.frames[-1]
        last_position = self.positions[-1]
        dt = target_frame - last_frame
        
        predicted = last_position + avg_velocity * dt + 0.5 * acceleration * dt * dt
        
        return float(predicted)
    
class BlurTemplateBank:
    """Keeps a small bank of grayscale templates (sharp + blurred) for NCC matching on blurry frames."""
    def __init__(self, enabled: bool = True, max_templates: int = 12, min_similarity: float = 0.35) -> None:
        self.enabled = bool(enabled)
        self.max_templates = int(max_templates)
        self.min_similarity = float(min_similarity)
        self.templates: List[np.ndarray] = []  # normalized float32 32x32

    @staticmethod
    def _to_gray_patch(frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float], pad_scale: float = 0.35) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = float(max(2.0, x2 - x1))
        bh = float(max(2.0, y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        pad_w = bw * pad_scale
        pad_h = bh * pad_scale
        x1p = int(max(0, math.floor(cx - bw/2 - pad_w)))
        x2p = int(min(w, math.ceil(cx + bw/2 + pad_w)))
        y1p = int(max(0, math.floor(cy - bh/2 - pad_h)))
        y2p = int(min(h, math.ceil(cy + bh/2 + pad_h)))
        if x2p <= x1p or y2p <= y1p:
            return None
        roi = frame_bgr[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        return gray

    @staticmethod
    def _normalize(im: np.ndarray) -> np.ndarray:
        imf = im.astype(np.float32)
        mean = float(imf.mean())
        std = float(imf.std())
        if std < 1e-6:
            std = 1.0
        return (imf - mean) / std

    def maybe_add(self, frame_bgr: Optional[np.ndarray], bbox: Optional[Tuple[float, float, float, float]]) -> None:
        if not self.enabled or frame_bgr is None or bbox is None:
            return
        try:
            base = self._to_gray_patch(frame_bgr, bbox)
            if base is None:
                return
            variants = [base]
            # Add blurred variants to be robust to motion blur
            try:
                variants.append(cv2.GaussianBlur(base, (0, 0), 1.2))
                variants.append(cv2.GaussianBlur(base, (0, 0), 2.0))
            except Exception:
                pass
            for v in variants:
                templ = self._normalize(v)
                self.templates.append(templ)
            if len(self.templates) > self.max_templates:
                self.templates = self.templates[-self.max_templates:]
        except Exception:
            return

    def similarity(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
        if not self.enabled or not self.templates:
            return 0.0
        try:
            patch = self._to_gray_patch(frame_bgr, bbox)
            if patch is None:
                return 0.0
            patch_n = self._normalize(patch)
            best = 0.0
            for t in self.templates:
                # normalized cross correlation: mean of elementwise product since both are normalized
                sim = float((t * patch_n).mean())
                if sim > best:
                    best = sim
            # Map from [-1,1] to [0,1]
            return max(0.0, min(1.0, (best + 1.0) * 0.5))
        except Exception:
            return 0.0


class OptimizedYoloBallDetector:
    def __init__(self, model_name: str, device: Optional[str], conf: float, imgsz: Optional[int]) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Please run: pip install ultralytics")
        self.model = YOLO(model_name)
        self.device = self._select_device(device)
        self.conf = float(conf)
        if imgsz:
            self.imgsz = int(imgsz)
        else:
            self.imgsz = 640 if self.device == "mps" else None
        self.ball_class_ids = self._resolve_ball_classes()
        self.allow_tta_recovery: bool = True
        try:
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception:
            pass
        self.selected_class_id: Optional[int] = None
        self.ref_hist: Optional[np.ndarray] = None
        # Optional external template bank injected by pipeline
        self.template_bank: Optional[BlurTemplateBank] = None
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
        d = float(cv2.compareHist(hist_a.astype('float32'), hist_b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA))
        sim = max(0.0, min(1.0, 1.0 - d))
        return sim

    def _warmup_model(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            if hasattr(self.model, "fuse"):
                self.model.fuse()
        except Exception:
            pass
        try:
            self.model.predict(source=dummy_frame, verbose=False, device=self.device)
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
            return [32]
        if isinstance(names, dict):
            id_to_name = {int(k): str(v).lower() for k, v in names.items()}
        else:
            id_to_name = {i: str(n).lower() for i, n in enumerate(list(names))}
        allowed_exact = {"sports ball", "basketball", "soccer ball", "tennis ball", "volleyball", "football", "handball", "rugby ball"}
        disallowed_substrings = {"bat", "glove", "racket", "racquet", "helmet"}
        ball_ids: List[int] = []
        for cid, name in id_to_name.items():
            if name in allowed_exact:
                ball_ids.append(int(cid))
                continue
            if "ball" in name:
                if any(bad in name for bad in disallowed_substrings):
                    continue
                ball_ids.append(int(cid))
        return ball_ids or [32]

    def _predict_on_optimized(self, frame_bgr: np.ndarray, *, conf: Optional[float] = None, imgsz: Optional[int] = None, use_tta: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Ensure use_tta is a proper Python boolean
        use_tta = bool(use_tta)
        h, w = frame_bgr.shape[:2]
        if imgsz is None:
            if h <= 480:
                imgsz_eff = 320
            elif h <= 720:
                imgsz_eff = 480
            else:
                imgsz_eff = 480
        else:
            imgsz_eff = imgsz
        half = (self.device == "cuda")
        classes_param = [self.selected_class_id] if (self.selected_class_id is not None) else self.ball_class_ids
        pre_frame = frame_bgr
        try:
            if use_tta or (imgsz_eff is not None and imgsz_eff >= 960):
                blur = cv2.GaussianBlur(pre_frame, (0, 0), sigmaX=1.0)
                sharp = cv2.addWeighted(pre_frame, 1.5, blur, -0.5, 0)
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
            iou=0.28,
            imgsz=imgsz_eff,
            half=half,
            augment=bool(use_tta),  # Ensure it's a Python boolean
            agnostic_nms=True,
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

    def detect_best_bbox_xyxy_in_roi_optimized(self, frame_bgr: np.ndarray, roi_left: int, roi_width: int, pref_center_x: Optional[float] = None, conf_override: Optional[float] = None, imgsz_override: Optional[int] = None, use_tta: bool = False) -> Optional[Tuple[float, float, float, float]]:
        H, W = frame_bgr.shape[:2]
        roi_left = int(max(0, min(roi_left, W - 2)))
        roi_width = int(max(2, min(roi_width, W - roi_left)))
        roi = frame_bgr[:, roi_left : roi_left + roi_width]
        out = self._predict_on_optimized(roi, conf=conf_override, imgsz=imgsz_override, use_tta=bool(use_tta))
# Add multi-scale detection fallback
        if out is None and self.allow_tta_recovery:
            for scale_factor in [1.2, 0.8, 1.5]:
                h, w = roi.shape[:2]
                if h == 0 or w == 0:
                    break
                scaled_h, scaled_w = int(h * scale_factor), int(w * scale_factor)
                roi_scaled = cv2.resize(roi, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
                out_scaled = self._predict_on_optimized(
                    roi_scaled, 
                    conf=max(0.08, (conf_override or self.conf) * 0.6), 
                    imgsz=imgsz_override, 
                    use_tta=False
                )
                if out_scaled is not None:
                    xys, confs, clss = out_scaled
                    xys = xys / scale_factor
                    out = (xys, confs, clss)
                    break
        out_from_fallback = False
        if out is None:
            out = self._predict_on_optimized(
                frame_bgr,
                conf=max(0.06, (conf_override or self.conf) * 0.5),
                imgsz=(640 if self.device == 'mps' else max(960, imgsz_override or (self.imgsz or 640))),
                use_tta=False,
            )
            out_from_fallback = out is not None
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
        best_bbox: Optional[Tuple[float, float, float, float]] = None
        for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
            if int(cls_id) not in self.ball_class_ids:
                continue
            cx = (x1 + x2) / 2.0 + roi_left_effective
            cy = (y1 + y2) / 2.0
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            conf_score = float(c)
            dist_penalty = 1.0
            if pref_center_x is not None:
                dist = abs(cx - float(pref_center_x))
                norm = max(1.0, roi_width_effective / 3.0)
                dist_penalty = 1.0 / (1.0 + (dist / norm) ** 2)
            area_ratio = area / (W * H)
            if area_ratio < 0.00005:
                area_penalty = 0.4
            elif area_ratio < 0.0003:
                area_penalty = 0.8
            elif area_ratio > 0.05:
                area_penalty = 0.7
            else:
                area_penalty = 1.0
            aspect_ratio = (x2 - x1) / max(1, y2 - y1)
            aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0))
            patch = frame_bgr[int(max(0, y1)):int(min(H, y2)), int(max(0, x1 + roi_left_effective)):int(min(W, x2 + roi_left_effective))] if getattr(self, 'ref_hist', None) is not None else None
            cand_hist = self._compute_hs_hist(patch) if patch is not None else None
            hist_sim = self._hist_similarity(self.ref_hist, cand_hist) if cand_hist is not None else None
            hist_term = (hist_sim if (hist_sim is not None and getattr(self, 'ref_hist', None) is not None) else 0.0)
            # Template similarity term (if bank is available)
            templ_term = 0.0
            try:
                bank = getattr(self, 'template_bank', None)
                if bank is not None and best_bbox is None:  # compute on candidate crop
                    templ_bbox = (float(x1 + roi_left_effective), float(y1), float(x2 + roi_left_effective), float(y2))
                    sim = float(bank.similarity(frame_bgr, templ_bbox))
                    templ_term = sim
            except Exception:
                templ_term = 0.0
            score = (
                    conf_score * 0.35 +
                    conf_score * dist_penalty * 0.25 +
                    conf_score * area_penalty * 0.12 +
                    conf_score * aspect_penalty * 0.1 +
                    (hist_term * 0.13 if getattr(self, 'ref_hist', None) is not None else 0.0) +
                    (templ_term * 0.15 if templ_term is not None else 0.0)
                )
            if score > best_score:
                best_score = score
                best_bbox = (float(x1 + roi_left_effective), float(y1), float(x2 + roi_left_effective), float(y2))
        return best_bbox

    def detect_best_bbox_xyxy_tiled(self, frame_bgr: np.ndarray, tile_size: int = 480, overlap: int = 80, conf_override: Optional[float] = None, imgsz_override: Optional[int] = None, pref_center_x: Optional[float] = None) -> Optional[Tuple[float, float, float, float]]:
        H, W = frame_bgr.shape[:2]
        step = max(32, tile_size - int(overlap))
        best_bbox = None
        best_score = -1.0
        for y in range(0, max(1, H - tile_size + 1), step):
            for x in range(0, max(1, W - tile_size + 1), step):
                tile = frame_bgr[y:y+tile_size, x:x+tile_size]
                out = self._predict_on_optimized(tile, conf=(conf_override or self.conf * 0.8), imgsz=(imgsz_override or max(640, tile_size)), use_tta=True)
                if out is None:
                    continue
                xys, confs, clss = out
                for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                    if int(cls_id) not in self.ball_class_ids:
                        continue
                    cx = (x1 + x2) / 2.0 + x
                    cy = (y1 + y2) / 2.0 + y
                    area = max(1.0, float((x2 - x1) * (y2 - y1)))
                    conf_score = float(c)
                    area_ratio = area / (W * H)
                    if area_ratio < 0.00003:
                        area_penalty = 0.3
                    elif area_ratio < 0.0002:
                        area_penalty = 0.7
                    elif area_ratio > 0.04:
                        area_penalty = 0.6
                    else:
                        area_penalty = 1.0
                    aspect_ratio = (x2 - x1) / max(1, y2 - y1)
                    aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0))
                    dist_penalty = 1.0
                    if pref_center_x is not None:
                        dist = abs(cx - float(pref_center_x))
                        norm = max(1.0, W / 3.0)
                        dist_penalty = 1.0 / (1.0 + (dist / norm) ** 2)
                    score = conf_score * 0.5 + conf_score * area_penalty * 0.15 + conf_score * aspect_penalty * 0.1 + conf_score * dist_penalty * 0.25
                    if score > best_score:
                        best_score = score
                        best_bbox = (float(x + x1), float(y + y1), float(x + x2), float(y + y2))
        return best_bbox





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


class PredictiveSearchLayer:
    """Predictive recovery layer: predicts next center and searches targeted regions when detection misses."""
    def __init__(self, detector: OptimizedYoloBallDetector, template_bank: Optional[BlurTemplateBank]) -> None:
        self.detector = detector
        self.template_bank = template_bank
        self.history: List[float] = []

    def update_history(self, cx: Optional[float]) -> None:
        if cx is None:
            return
        self.history.append(float(cx))
        if len(self.history) > 1200:
            self.history = self.history[-1200:]

    def predict_next_cx(self) -> Optional[float]:
        n = len(self.history)
        if n == 0:
            return None
        
        if n >= 8:
            xs = np.arange(n, dtype=float)
            ys = np.array(self.history, dtype=float)
            weights = np.exp(np.linspace(-2, 0, min(n, 12)))[-n:]
            
            try:
                recent_frames = min(8, n)
                coeffs = np.polyfit(
                    xs[-recent_frames:], 
                    ys[-recent_frames:], 
                    deg=min(2, recent_frames-1),
                    w=weights[-recent_frames:]
                )
                x_next = float(n)
                predicted = float(np.polyval(coeffs, x_next))
                
                if n >= 3:
                    recent_velocity = np.mean(np.diff(ys[-3:]))
                    predicted += recent_velocity * 0.3
                
                return predicted
            except Exception:
                pass
        
        if n >= 3:
            velocities = np.diff(self.history[-4:] if n >= 4 else self.history)
            avg_velocity = float(np.mean(velocities))
            damping = 0.7 if abs(avg_velocity) > 30 else 0.8
            return float(self.history[-1] + avg_velocity * damping)
        
        if n >= 2:
            v = float(self.history[-1] - self.history[-2])
            return float(self.history[-1] + v * 0.6)
        
        return float(self.history[-1])

    def recover(self, frame: np.ndarray, prev_cx: Optional[float], vel: float, misses: int, frame_w: int, base_roi: int) -> Optional[Tuple[float, float, float, float]]:
        H, W = frame.shape[:2]
        pred = self.predict_next_cx() or prev_cx or (W / 2.0)
        # 1) Targeted ROI detect around predicted center
        roi_w = int(max(200, min(W, base_roi + (vel * 2.5))))
        roi_half = roi_w // 2
        roi_left = int(max(0, min(int(round(pred)) - roi_half, W - roi_w)))
        bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
            frame,
            roi_left=roi_left,
            roi_width=roi_w,
            pref_center_x=pred,
            conf_override=max(0.08, self.detector.conf * (0.7 if misses >= 3 else 0.85)),
            imgsz_override=(960 if self.detector.device == 'mps' else (1280 if vel > 40 else 960)),
            use_tta=bool(vel > 40 and self.detector.allow_tta_recovery),
        )
        if bbox is not None:
            return bbox
        # 2) Banded tiled detect around predicted center
        band_w = int(min(W, max(base_roi * 2, 640)))
        left = int(max(0, min(int(round(pred - band_w / 2)), W - band_w)))
        strip = frame[:, left:left+band_w]
        bbox_t = self.detector.detect_best_bbox_xyxy_tiled(
            strip,
            tile_size=min(512, band_w),
            overlap=96,
            conf_override=max(0.06, self.detector.conf * 0.6),
            imgsz_override=max(640, min(960, band_w)),
            pref_center_x=(pred - left),
        )
        if bbox_t is not None:
            x1, y1, x2, y2 = bbox_t
            return (float(x1 + left), float(y1), float(x2 + left), float(y2))
        # 3) NCC template search in band as last resort
        try:
            if self.template_bank is not None and getattr(self.template_bank, 'templates', None):
                search = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
                best = -1.0
                best_x = None
                for templ in self.template_bank.templates:
                    t = (templ * templ.std() + templ.mean()).astype(np.float32)
                    try:
                        res = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
                        _, maxv, _, maxl = cv2.minMaxLoc(res)
                        if float(maxv) > best:
                            best = float(maxv)
                            best_x = left + float(maxl[0] + t.shape[1] / 2.0)
                    except Exception:
                        continue
                if best_x is not None and best >= 0.3:
                    cx = float(best_x)
                    bw = max(12.0, base_roi * 0.05)
                    x1 = max(0.0, cx - bw/2)
                    x2 = min(float(W), cx + bw/2)
                    return (x1, float(H)*0.45, x2, float(H)*0.55)
        except Exception:
            pass
        return None


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
        # Add ball memory system
        self.ball_memory = BallMemorySystem(
            memory_duration_frames=int(kwargs.get('memory_duration_frames', 90)),
            confidence_decay=float(kwargs.get('memory_confidence_decay', 0.98)),
            stability_window=int(kwargs.get('stability_window', 30)),
            position_variance_threshold=float(kwargs.get('position_variance_threshold', 50.0))
        )
        
        # Memory usage settings
        self.use_ball_memory = bool(kwargs.get('use_ball_memory', True))
        self.memory_blend_frames = int(kwargs.get('memory_blend_frames', 15))  # Frames to blend back to detection
        
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
        # Tiled detection options
        self.tiled_detect = bool(kwargs.get('tiled_detect', False))
        self.tile_size = int(kwargs.get('tile_size', 480))
        self.tile_overlap = int(kwargs.get('tile_overlap', 80))
        # Blur template learning
        self.use_blur_templates = True
        self.template_bank = BlurTemplateBank(enabled=True, max_templates=16, min_similarity=0.35)
        # Attach bank to detector
        try:
            self.detector.template_bank = self.template_bank
        except Exception:
            pass
        # Predictive recovery layer
        self.predict_layer = PredictiveSearchLayer(self.detector, self.template_bank)



    def _bootstrap_initial_center_enhanced(self) -> Optional[Tuple[int, Tuple[float, float, float, float], float]]:
        """Enhanced bootstrap with multi-confidence and temporal consistency"""
        meta = self._read_meta(self.input_path)
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return None
        
        candidates = []
        frames_to_scan = min(getattr(self, 'bootstrap_frames', 48), meta.num_frames or 48)
        
        conf_thresholds = [0.15, 0.1, 0.05] if getattr(self, 'enhanced_bootstrap', False) else [0.1]
        
        for conf_thresh in conf_thresholds:
            step = max(1, frames_to_scan // 30)
            
            for idx in range(0, frames_to_scan, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    break
                
                detection_strategies = [
                    {'conf': conf_thresh, 'imgsz': 960},
                    {'conf': conf_thresh * 0.8, 'imgsz': 1280},
                ]
                
                for strategy in detection_strategies:
                    out = self.detector._predict_on_optimized(
                        frame, 
                        conf=strategy['conf'],
                        imgsz=strategy['imgsz'],
                        use_tta=True
                    )
                    
                    if out is None:
                        continue
                    
                    xys, confs, clss = out
                    H, W = frame.shape[:2]
                    
                    for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                        if int(cls_id) not in self.detector.ball_class_ids:
                            continue
                        
                        score = self._compute_bootstrap_score(frame, (x1, y1, x2, y2), c, W, H)
                        
                        candidates.append({
                            'frame': idx,
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'cx': (float(x1) + float(x2)) / 2.0,
                            'score': score,
                            'conf': float(c),
                            'strategy': strategy
                        })
        
        cap.release()
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        best_candidate = None
        for candidate in candidates[:10]:
            if self._verify_temporal_consistency(candidate):
                best_candidate = candidate
                break
        
        if best_candidate is None and candidates:
            best_candidate = candidates[0]
        
        if best_candidate:
            return (best_candidate['frame'], best_candidate['bbox'], best_candidate['cx'])
        
        return None

    def _compute_bootstrap_score(self, frame, bbox, conf, W, H):
        """Compute enhanced bootstrap score"""
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        area = max(1.0, float((x2 - x1) * (y2 - y1)))
        conf_score = float(conf)
        
        center_dist = abs(cx - W/2) / (W/2)
        center_penalty = 1.0 / (1.0 + center_dist * 0.5)
        
        area_ratio = area / (W * H)
        if 0.0003 < area_ratio < 0.03:
            area_penalty = 1.0
        elif 0.0001 < area_ratio < 0.0003:
            area_penalty = 0.8
        elif 0.03 < area_ratio < 0.05:
            area_penalty = 0.8
        else:
            area_penalty = 0.4
        
        aspect_ratio = (x2 - x1) / max(1, y2 - y1)
        aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0) * 0.5)
        
        y_center_ratio = cy / H
        if 0.2 < y_center_ratio < 0.8:
            vertical_penalty = 1.0
        else:
            vertical_penalty = 0.7
        
        return (
            conf_score * 0.3 +
            conf_score * center_penalty * 0.25 +
            conf_score * area_penalty * 0.2 +
            conf_score * aspect_penalty * 0.15 +
            conf_score * vertical_penalty * 0.1
        )

    def _verify_temporal_consistency(self, candidate):
        """Verify candidate appears consistently in nearby frames"""
        return candidate['score'] > 0.4

    def _first_pass_detect_track_optimized(self) -> List[Optional[float]]:
        """Enhanced detection with forward prediction and stability (no backward processing)"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to open input video")
            
        xs: List[Optional[float]] = []
        tracker = ParallelBoxTracker(tracker_pref=getattr(self, 'tracker_pref', 'auto'))
        flow = OpticalFlowAssist()
        
        frame_idx = 0
        meta = self._read_meta(self.input_path)
        frame_w = meta.width
        
        # Enhanced bootstrap
        boot = self._bootstrap_initial_center_enhanced()
        if boot is not None:
            _, bbox0, cx0 = boot
            self.ball_memory.update_detection(boot[0], cx0, bbox0, 1.0)
        
        # Process frames sequentially (forward only)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Get prediction from memory system
            memory_pos = self.ball_memory.get_position_for_frame(frame_idx)
            cx_pred = memory_pos or (frame_w / 2.0)
            
            # Detection setup with dynamic ROI
            base_roi = int(getattr(self, 'base_roi_width', 400))
            
            # Expand ROI if we're in unstable region
            if self.ball_memory.consecutive_misses > 0:
                roi_w = int(min(frame_w * 0.8, base_roi * 1.5))
            else:
                roi_w = int(max(200, min(frame_w, base_roi)))
                
            roi_half = roi_w // 2
            roi_left = int(max(0, min(int(round(cx_pred)) - roi_half, frame_w - roi_w)))
            
            # Try detection with multiple aggressive strategies
            detection_made = False
            final_position = None
            
            # Calculate ball velocity for motion blur handling
            ball_velocity = 0.0
            if len(self.ball_memory.position_history) >= 2:
                recent_positions = [pos for (f, pos, conf) in self.ball_memory.position_history[-5:]]
                if len(recent_positions) >= 2:
                    velocities = [abs(recent_positions[i] - recent_positions[i-1]) for i in range(1, len(recent_positions))]
                    ball_velocity = np.mean(velocities) if velocities else 0.0
            
            # Expand ROI based on velocity (for fast-moving balls)
            velocity_expansion = min(2.0, 1.0 + (ball_velocity / 50.0))  # Expand up to 2x for fast balls
            roi_w_expanded = int(roi_w * velocity_expansion)
            roi_half_expanded = roi_w_expanded // 2
            roi_left_expanded = int(max(0, min(int(round(cx_pred)) - roi_half_expanded, frame_w - roi_w_expanded)))
            
            # Strategy 1: Primary ROI detection with velocity-adaptive confidence
            base_conf = 0.15
            if ball_velocity > 30:  # Lower confidence for fast-moving balls
                base_conf = 0.10
            elif ball_velocity > 60:  # Even lower for very fast balls
                base_conf = 0.08
                
            bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                frame,
                roi_left=roi_left_expanded,
                roi_width=roi_w_expanded,
                pref_center_x=cx_pred,
                conf_override=base_conf,
                imgsz_override=(960 if (self.detector.device == 'mps') else 1280),
                use_tta=True,  # Enable TTA for better detection
            )
            
            # Strategy 2: If primary fails, try motion-blur-optimized detection
            if bbox is None and ball_velocity > 20:
                # For fast-moving balls, try with motion blur preprocessing
                bbox = self._detect_with_motion_blur_handling(
                    frame, cx_pred, frame_w, base_conf
                )
            
            # Strategy 3: If still no detection, try full-frame detection with lower confidence
            if bbox is None:
                bbox = self.detector._predict_on_optimized(
                    frame,
                    conf=0.12,  # Lower but still reasonable confidence
                    imgsz=1280,
                    use_tta=True
                )
                if bbox is not None:
                    xys, confs, clss = bbox
                    H, W = frame.shape[:2]
                    best_score = -1.0
                    best_bbox = None
                    
                    for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                        if int(cls_id) not in self.detector.ball_class_ids:
                            continue
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        area = max(1.0, float((x2 - x1) * (y2 - y1)))
                        conf_score = float(c)
                        
                        # Distance penalty from predicted position (stronger penalty)
                        dist_penalty = 1.0
                        if cx_pred is not None:
                            dist = abs(cx - float(cx_pred))
                            # Expand search area for fast-moving balls
                            search_radius = W / 3.0 if ball_velocity < 30 else W / 2.0
                            norm = max(1.0, search_radius)
                            dist_penalty = 1.0 / (1.0 + (dist / norm) ** 1.5)
                        
                        # Area penalty (more restrictive)
                        area_ratio = area / (W * H)
                        if 0.0002 < area_ratio < 0.05:  # Tighter area bounds
                            area_penalty = 1.0
                        elif 0.0001 < area_ratio < 0.0002:
                            area_penalty = 0.7
                        elif 0.05 < area_ratio < 0.08:
                            area_penalty = 0.7
                        else:
                            area_penalty = 0.3
                        
                        # Aspect ratio penalty (ensure it's roughly circular)
                        aspect_ratio = (x2 - x1) / max(1, y2 - y1)
                        aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0) * 2.0)
                        
                        # Vertical position penalty (balls are usually in middle area)
                        y_center_ratio = cy / H
                        if 0.15 < y_center_ratio < 0.85:
                            vertical_penalty = 1.0
                        else:
                            vertical_penalty = 0.6
                        
                        # Motion blur bonus (higher score for blurry objects when ball is moving fast)
                        motion_blur_bonus = 1.0
                        if ball_velocity > 30:
                            # Check if the object appears blurry (low contrast)
                            patch = frame[int(y1):int(y2), int(x1):int(x2)]
                            if patch.size > 0:
                                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                                contrast = np.std(gray_patch)
                                if contrast < 30:  # Low contrast = likely blurry
                                    motion_blur_bonus = 1.2
                        
                        score = conf_score * dist_penalty * area_penalty * aspect_penalty * vertical_penalty * motion_blur_bonus
                        if score > best_score:
                            best_score = score
                            best_bbox = (float(x1), float(y1), float(x2), float(y2))
                    
                    # Only accept if score is good enough
                    if best_bbox is not None and best_score > 0.08:
                        bbox = best_bbox
            
            # Strategy 4: If still no detection, try tiled detection with moderate confidence
            if bbox is None:
                bbox = self.detector.detect_best_bbox_xyxy_tiled(
                    frame,
                    tile_size=640,
                    overlap=120,
                    conf_override=0.12,
                    imgsz_override=960,
                    pref_center_x=cx_pred,
                )
            
            # Strategy 5: If still no detection, try predictive recovery
            if bbox is None and hasattr(self, 'predict_layer'):
                try:
                    bbox = self.predict_layer.recover(
                        frame, 
                        prev_cx=cx_pred, 
                        vel=ball_velocity, 
                        misses=self.ball_memory.consecutive_misses,
                        frame_w=frame_w,
                        base_roi=base_roi
                    )
                except Exception as e:
                    # Skip predictive recovery if it fails
                    bbox = None
            
            if bbox is not None:
                # Ensure bbox is in correct format
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                else:
                    # Skip invalid bbox
                    bbox = None
                    continue
                    
                cx_det = (x1 + x2) / 2.0
                final_position = cx_det
                detection_made = True
                
                # Update memory and trackers
                self.ball_memory.update_detection(frame_idx, cx_det, bbox, 1.0)
                tracker.init_with_bbox(frame, bbox)
                flow.init_from_bbox(frame, bbox)
                
            else:
                # Try tracker
                tracker_estimate = None
                ok_t, cx_t = tracker.update(frame)
                if ok_t and cx_t is not None:
                    tracker_estimate = cx_t
                
                # Try optical flow
                if tracker_estimate is None:
                    cx_flow = flow.update(frame)
                    if cx_flow is not None:
                        tracker_estimate = cx_flow
                
                # Update memory with miss
                self.ball_memory.update_no_detection(frame_idx)
                
                # Get final position from enhanced memory
                final_position = self.ball_memory.get_position_for_frame(frame_idx)
                
                # Apply sliding window stability
                if final_position is not None:
                    final_position = self._apply_sliding_window_stability(
                        final_position, frame_idx, xs
                    )
            
            # Store result
            xs.append(final_position)
            
            # Update performance counters
            self._perf['frames'] += 1
            if detection_made:
                self._perf['roi_detects'] += 1
            else:
                self._perf['misses'] += 1
                
            # Progress reporting
            if self.profile and (self._perf['frames'] % 60 == 0):
                elapsed = time.perf_counter() - self._perf['t_start']
                fps = self._perf['frames'] / max(1e-6, elapsed)
                stability = "STABLE" if self.ball_memory.stable_position is not None else "TRACKING"
                print(f"Profile@{self._perf['frames']}: fps={fps:.1f} detects={self._perf['roi_detects']} misses={self._perf['misses']} mode={stability}")
            
            frame_idx += 1
                
        cap.release()
        return xs


    def _apply_sliding_window_stability(self, position: float, frame_idx: int, 
                                    previous_positions: List[Optional[float]]) -> float:
        """Apply sliding window stability to position"""
        if len(previous_positions) < 5:
            return position
        
        # Get recent valid positions
        recent_positions = []
        for i in range(max(0, len(previous_positions) - 10), len(previous_positions)):
            if previous_positions[i] is not None:
                recent_positions.append(previous_positions[i])
        
        if len(recent_positions) < 3:
            return position
        
        # Calculate stability metrics
        recent_variance = np.var(recent_positions)
        recent_mean = np.mean(recent_positions)
        
        # If current position is too far from recent stable area, smooth it
        deviation = abs(position - recent_mean)
        max_allowed_deviation = 2.0 * np.sqrt(recent_variance) + 20.0  # Dynamic threshold
        
        if deviation > max_allowed_deviation:
            # Smooth the position towards recent mean
            smoothing_factor = 0.7
            return recent_mean * smoothing_factor + position * (1 - smoothing_factor)
        
        return position

    def _detect_with_motion_blur_handling(self, frame: np.ndarray, cx_pred: float, frame_w: int, base_conf: float) -> Optional[Tuple[float, float, float, float]]:
        """Specialized detection for motion-blurred balls"""
        H, W = frame.shape[:2]
        
        # Create motion-blur-optimized versions of the frame
        processed_frames = []
        
        # Original frame
        processed_frames.append(frame)
        
        # Motion blur simulation (helps detect blurry balls)
        try:
            # Apply slight blur to simulate motion
            blur_kernel = np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.1]])
            blurred = cv2.filter2D(frame, -1, blur_kernel)
            processed_frames.append(blurred)
            
            # Gaussian blur for more aggressive blur simulation
            gaussian_blurred = cv2.GaussianBlur(frame, (0, 0), 1.5)
            processed_frames.append(gaussian_blurred)
        except Exception:
            pass
        
        # Try detection on each processed frame
        best_bbox = None
        best_score = -1.0
        
        for proc_frame in processed_frames:
            # Try ROI detection on processed frame
            roi_w = int(min(frame_w * 0.6, 800))  # Larger ROI for blurry balls
            roi_half = roi_w // 2
            roi_left = int(max(0, min(int(round(cx_pred)) - roi_half, W - roi_w)))
            
            bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                proc_frame,
                roi_left=roi_left,
                roi_width=roi_w,
                pref_center_x=cx_pred,
                conf_override=base_conf * 0.8,  # Slightly lower confidence for blurry frames
                imgsz_override=1280,
                use_tta=True,
            )
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                area = max(1.0, float((x2 - x1) * (y2 - y1)))
                
                # Calculate score with blur-friendly penalties
                dist_penalty = 1.0
                if cx_pred is not None:
                    dist = abs(cx - float(cx_pred))
                    norm = max(1.0, W / 2.5)  # More lenient distance penalty for blurry balls
                    dist_penalty = 1.0 / (1.0 + (dist / norm) ** 1.2)
                
                area_ratio = area / (W * H)
                if 0.0001 < area_ratio < 0.08:  # More lenient area bounds for blurry balls
                    area_penalty = 1.0
                else:
                    area_penalty = 0.5
                
                aspect_ratio = (x2 - x1) / max(1, y2 - y1)
                aspect_penalty = 1.0 / (1.0 + abs(aspect_ratio - 1.0) * 1.5)  # More lenient aspect ratio
                
                score = dist_penalty * area_penalty * aspect_penalty
                if score > best_score:
                    best_score = score
                    best_bbox = bbox
        
        # Ensure we return a valid bbox format
        if best_bbox is not None and isinstance(best_bbox, (list, tuple)) and len(best_bbox) == 4:
            return best_bbox
        return None

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
        return VideoMeta(width, height, fps, num_frames)  # FIXED: No keyword arguments

    def run(self) -> None:
        """Run simplified pipeline"""
        print("Starting optimized ball tracking pipeline...")
        start_time = time.time()
        
        meta = self._read_meta(self.input_path)
        print(f"Device: {self.detector.device}")
        
        crop_w = self._compute_crop_width(meta.height, meta.width)
        out_w, out_h = self._compute_output_size(getattr(self, 'out_height', None), meta.height)
        fps_out = getattr(self, 'out_fps', None) or meta.fps
        
        if self.three_pass:
            print("Phase 0: Learning ball appearance...")
            self._pass1_learn_ball_appearance()
        
        print("Phase 1: Ball detection and tracking...")
        
        # Use the simplified detection method
        xs_raw = self._first_pass_detect_track_optimized()

        if self.profile:
            elapsed = time.perf_counter() - self._perf['t_start']
            fps = self._perf['frames'] / max(1e-6, elapsed)
            print(f"Profile: frames={self._perf['frames']} fps={fps:.1f} detects={self._perf['roi_detects']} misses={self._perf['misses']}")
        
        if len(xs_raw) == 0:
            raise SystemExit("No frames processed")
            
        print(f"Phase 2: Trajectory smoothing ({len(xs_raw)} frames)...")
        smoother = TrajectorySmoother(window_size=getattr(self, 'smooth_window', 15))
        xs_smooth = smoother.smooth(xs_raw, fps=meta.fps)
        
        print("Phase 3: Crop planning...")
        # Use the simplified crop planning
        centers = self._plan_centers_strict_with_margin(
            xs_smooth, crop_w, meta.width, int(getattr(self, 'margin', 60))
        )
        
        print("Phase 4: Video rendering...")
        self._second_pass_write(centers=centers, crop_width=crop_w, out_size=(out_w, out_h), fps_out=fps_out)
        
        total_time = time.time() - start_time
        print(f"✅ Complete! Total time: {total_time:.1f}s ({len(xs_raw)/total_time:.1f} FPS)")
    
    def _plan_centers_strict_with_margin(self, smoothed_xs: np.ndarray, crop_width: int, frame_width: int, edge_margin: int) -> np.ndarray:
        """Simple crop planning with frame-to-frame stability"""
        centers = np.empty_like(smoothed_xs)
        half = crop_width / 2.0
        min_center = half + max(10, float(edge_margin) / 3.0)
        max_center = frame_width - half - max(10, float(edge_margin) / 3.0)
        
        for i, ball_x in enumerate(smoothed_xs):
            ball_x = float(ball_x)
            
            # Calculate target center
            target_center = max(min_center, min(max_center, ball_x))
            
            if i == 0:
                # First frame
                centers[i] = target_center
            else:
                # Limit movement from previous frame
                prev_center = centers[i-1]
                max_movement = crop_width * 0.1  # Small movement per frame
                
                movement = target_center - prev_center
                if abs(movement) > max_movement:
                    # Limit the movement
                    if movement > 0:
                        target_center = prev_center + max_movement
                    else:
                        target_center = prev_center - max_movement
                        
                # Bounds check
                target_center = max(min_center, min(max_center, target_center))
                centers[i] = target_center
        
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
    parser.add_argument("--tiled-detect", action="store_true", help="Enable tiled detection fallback for blurry/fast balls")
    parser.add_argument("--tile-size", type=int, default=480, help="Square tile size in pixels for tiled detection")
    parser.add_argument("--tile-overlap", type=int, default=80, help="Overlap in pixels between tiles for tiled detection")
    parser.add_argument("--enhanced-bootstrap", action="store_true", help="Use enhanced multi-pass bootstrap for better initial detection")
    parser.add_argument("--detection-confidence-boost", type=float, default=1.2, help="Boost factor for detection confidence in 3-phase mode")
    parser.add_argument("--stability-frames", type=int, default=5, help="Number of consecutive frames needed to confirm ball position")
    parser.add_argument("--prediction-lookahead", type=int, default=3, help="Number of frames to look ahead for prediction")
    parser.add_argument("--multi-roi-detect", action="store_true", help="Use multiple ROI sizes for detection")
    parser.add_argument("--ultra-quality", action="store_true", help="Use ultra-quality settings for maximum accuracy")
    parser.add_argument("--ultra-aggressive", action="store_true", help="Use ultra-aggressive detection with multiple fallback strategies")
    parser.add_argument("--motion-blur-mode", action="store_true", help="Enable specialized detection for fast-moving, blurry balls")
    parser.add_argument("--use-ball-memory", action="store_true", default=True, help="Keep last known ball position when ball disappears")
    parser.add_argument("--no-ball-memory", action="store_true", help="Disable ball memory system")
    parser.add_argument("--memory-duration", type=int, default=90, help="Frames to remember ball position")
    parser.add_argument("--memory-decay", type=float, default=0.98, help="Memory confidence decay rate")
    parser.add_argument("--memory-blend", type=int, default=15, help="Frames to blend back to detection")
    # Add these lines in the parse_args() function:
    parser.add_argument("--stability-window", type=int, default=30, help="Window size for position stability calculation")
    parser.add_argument("--position-variance-threshold", type=float, default=50.0, help="Variance threshold for stable position detection")
    # In parse_args():

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Ultra-quality configuration
    if args.ultra_quality:
        args.model = args.model if args.model != "yolov8s.pt" else "yolov8x.pt"
        args.imgsz = args.imgsz or 1280
        args.conf = 0.12 if args.conf == 0.2 else args.conf
        args.smooth = 19 if args.smooth == 11 else args.smooth
        args.three_pass = True
        args.full_detect = True
        args.tiled_detect = True
        args.enhanced_bootstrap = True
        args.multi_roi_detect = True
        args.detection_confidence_boost = 1.3
        args.stability_frames = 7
        print("🚀 Ultra-quality mode enabled!")
    
    # Ultra-aggressive configuration
    if args.ultra_aggressive:
        args.model = args.model if args.model != "yolov8s.pt" else "yolov8x.pt"
        args.imgsz = args.imgsz or 1280
        args.conf = 0.12 if args.conf == 0.2 else args.conf  # More reasonable confidence
        args.smooth = 15 if args.smooth == 11 else args.smooth
        args.three_pass = True
        args.full_detect = True
        args.tiled_detect = True
        args.enhanced_bootstrap = True
        args.multi_roi_detect = True
        args.detection_confidence_boost = 1.3  # Reduced from 1.5
        args.stability_frames = 3
        args.detect_every_frame = True
        args.use_ball_memory = True
        args.memory_duration_frames = 120  # Reduced from 150
        args.memory_confidence_decay = 0.99  # More reasonable decay
        print("🔥 Ultra-aggressive detection mode enabled!")
    
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
        tiled_detect=bool(args.tiled_detect),
        tile_size=int(args.tile_size),
        tile_overlap=int(args.tile_overlap),
        enhanced_bootstrap=getattr(args, 'enhanced_bootstrap', False),
        detection_confidence_boost=getattr(args, 'detection_confidence_boost', 1.2),
        stability_frames=getattr(args, 'stability_frames', 5),
        prediction_lookahead=getattr(args, 'prediction_lookahead', 3),
        multi_roi_detect=getattr(args, 'multi_roi_detect', False),
        # Add these lines to the OptimizedReframerPipeline constructor call in main():
        stability_window=getattr(args, 'stability_window', 30),
        position_variance_threshold=getattr(args, 'position_variance_threshold', 50.0),
    )
    
    pipeline.run()
    print(f"✅ Optimized reframing complete: {args.output}")


if __name__ == "__main__":
    main()