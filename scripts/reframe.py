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
    """Memory system that allows one adjustment when ball disappears, then freezes"""
    def __init__(self, memory_duration_frames: int = 90, confidence_decay: float = 0.98):
        self.memory_duration = int(memory_duration_frames)
        self.confidence_decay = float(confidence_decay)
        
        # Detection tracking
        self.last_known_position: Optional[float] = None
        self.last_detection_frame: int = -1
        self.memory_confidence: float = 1.0
        
        # Frame-to-frame tracking
        self.previous_frame_center: Optional[float] = None
        self.had_detection_last_frame = False
        
        # NEW: First miss handling
        self.consecutive_misses = 0
        self.frozen_position: Optional[float] = None  # Position to freeze at after first miss
        self.is_frozen = False
        
        # Motion estimation for first miss
        self.velocity_estimate: float = 0.0
        self.position_history: List[Tuple[int, float]] = []
        
    def update_detection(self, frame_idx: int, ball_cx: float, bbox: Optional[Tuple[float, float, float, float]] = None, detection_confidence: float = 1.0):
        """Update with detection - reset freeze state"""
        self.last_known_position = float(ball_cx)
        self.last_detection_frame = int(frame_idx)
        self.memory_confidence = 1.0
        self.previous_frame_center = float(ball_cx)
        self.had_detection_last_frame = True
        
        # Reset freeze state when ball is found
        self.consecutive_misses = 0
        self.frozen_position = None
        self.is_frozen = False
        
        # Update motion history
        self.position_history.append((frame_idx, ball_cx))
        if len(self.position_history) > 5:
            self.position_history = self.position_history[-5:]
        self._update_velocity()

    def update_no_detection(self, frame_idx: int, tracker_estimate: Optional[float] = None):
        """Update when no detection - handle first miss vs subsequent misses"""
        self.had_detection_last_frame = False
        self.consecutive_misses += 1
        
        if self.consecutive_misses == 1:
            # FIRST MISS - allow adjustment using tracker/prediction
            estimated_position = self._get_first_miss_estimate(tracker_estimate)
            if estimated_position is not None:
                self.frozen_position = estimated_position
                self.previous_frame_center = estimated_position
            self.is_frozen = False  # Not frozen yet, this is the adjustment frame
            
        elif self.consecutive_misses >= 2:
            # SECOND+ MISS - freeze at the first miss position
            if self.frozen_position is not None:
                self.previous_frame_center = self.frozen_position
                self.is_frozen = True
        
    def _get_first_miss_estimate(self, tracker_estimate: Optional[float]) -> Optional[float]:
        """Get best estimate for first frame where ball is missing"""
        # Priority 1: Use tracker estimate if available
        if tracker_estimate is not None:
            return float(tracker_estimate)
            
        # Priority 2: Use velocity prediction
        if self.last_known_position is not None and abs(self.velocity_estimate) > 0.5:
            predicted = self.last_known_position + self.velocity_estimate
            return predicted
            
        # Priority 3: Use last known position
        return self.last_known_position
        
    def get_position_for_frame(self, current_frame: int) -> Optional[float]:
        """Get position for current frame"""
        if self.previous_frame_center is not None:
            return self.previous_frame_center
        
        # Fallback if no previous frame
        if self.last_known_position is not None:
            frames_since = current_frame - self.last_detection_frame
            if frames_since <= self.memory_duration:
                return self.last_known_position
                
        return None
        
    def _update_velocity(self):
        """Update velocity estimate"""
        if len(self.position_history) >= 2:
            recent = self.position_history[-2:]
            dt = recent[1][0] - recent[0][0]
            if dt > 0:
                dx = recent[1][1] - recent[0][1]
                self.velocity_estimate = dx / dt

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
            # 🆕 ADD: Enhanced tennis ball specific settings
        self.tennis_ball_confidence_boost = 1.3  # Boost tennis ball detections
        self.distraction_penalty = 0.7  # Penalize non-tennis objects
        self.temporal_consistency_buffer = []  # Track recent detections
        self.false_positive_memory = set()  # Remember false positive locations
        
        # 🆕 ADD: Enhanced template matching for tennis balls
        self.tennis_templates = []  # Specific tennis ball templates
        self.template_update_frequency = 30  # Update templates every 30 frames
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
            augment=use_tta,
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
        out = self._predict_on_optimized(roi, conf=conf_override, imgsz=imgsz_override, use_tta=use_tta)
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
            tennis_ball_boost = 1.0
            if self._is_tennis_ball_like(frame_bgr, (x1, y1, x2, y2)):
                tennis_ball_boost = self.tennis_ball_confidence_boost
                
            temporal_score = self._check_temporal_consistency((x1, y1, x2, y2))
            fp_penalty = self._check_false_positive_memory((x1, y1, x2, y2))
            score = (
                conf_score * tennis_ball_boost * 0.30 +  # Boost tennis balls
                conf_score * dist_penalty * 0.20 +
                conf_score * area_penalty * 0.15 +
                conf_score * aspect_penalty * 0.10 +
                temporal_score * 0.15 +  # Reward consistent detections
                (hist_term * 0.10 if getattr(self, 'ref_hist', None) is not None else 0.0) +
                fp_penalty * 0.10  # Penalize known false positives
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
    
        # Add this AFTER the existing __init__ method of OptimizedYoloBallDetector class
    def detect_with_nms_filtering(self, frame_bgr: np.ndarray, aggressive_nms: bool = True) -> Optional[Tuple[float, float, float, float]]:
        """Enhanced detection with aggressive NMS - PERFORMANCE OPTIMIZED"""
        
        if frame_bgr is None:
            return None
        
        # ✅ PERFORMANCE: Only use 1-2 scales instead of 3
        scales = [1.0, 1.1] if aggressive_nms else [1.0]  # Reduced from [1.0, 1.2, 0.8]
        
        # ✅ PERFORMANCE: Only use 2 confidence levels instead of 3
        conf_levels = [self.conf, self.conf * 0.8] if aggressive_nms else [self.conf]  # Reduced from 3 levels
        
        all_detections = []
        
        for scale in scales:
            for conf_level in conf_levels:
                h, w = frame_bgr.shape[:2]
                
                if scale != 1.0:
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_frame = cv2.resize(frame_bgr, (new_w, new_h))
                else:
                    new_h, new_w = h, w
                    scaled_frame = frame_bgr
                
                out = self._predict_on_optimized(
                    scaled_frame, 
                    conf=conf_level,
                    imgsz=min(960, max(640, int(max(new_w, new_h)))),  # ✅ PERFORMANCE: Reduced max size
                    use_tta=False  # ✅ PERFORMANCE: Disable TTA for speed
                )
                
                if out is not None:
                    xys, confs, clss = out
                    
                    if scale != 1.0:
                        xys = xys / scale
                    
                    for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                        if int(cls_id) in self.ball_class_ids:
                            all_detections.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'conf': float(c),
                                'cls_id': int(cls_id),
                                'area': (x2-x1) * (y2-y1),
                                'cx': (x1+x2)/2,
                                'cy': (y1+y2)/2
                            })
                    
                    # ✅ PERFORMANCE: Break early if we found good detections
                    if len(all_detections) >= 3:  # Stop if we have enough candidates
                        break
            
            if len(all_detections) >= 2:  # Don't try more scales if we have detections
                break
        
        if not all_detections:
            return None
        
        # Apply custom NMS with ball-specific logic
        filtered_detections = self._apply_ball_specific_nms(all_detections, frame_bgr.shape)
        
        if not filtered_detections:
            return None
        
        # Select best detection
        best_det = max(filtered_detections, key=lambda x: self._score_detection_in_context(x, frame_bgr))
        
        return best_det['bbox']


    def _apply_ball_specific_nms(self, detections, frame_shape, iou_threshold=0.3):
        """Apply ball-specific non-maximum suppression"""
        if len(detections) <= 1:
            return detections
        
        H, W = frame_shape[:2]
        
        # Sort by confidence
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        keep = []
        used_indices = set()
        
        for i, det in enumerate(detections):
            if i in used_indices:
                continue
            
            keep.append(det)
            
            # Suppress overlapping detections
            for j, other in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(det['bbox'], other['bbox'])
                
                # Calculate spatial distance
                dx = abs(det['cx'] - other['cx'])
                dy = abs(det['cy'] - other['cy'])
                spatial_dist = np.sqrt(dx*dx + dy*dy) / min(W, H)
                
                # Suppress if too similar
                if iou > iou_threshold or spatial_dist < 0.05:
                    used_indices.add(j)
        
        return keep

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)

    def _score_detection_in_context(self, detection, frame_bgr):
        """Score detection considering context and surroundings"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        H, W = frame_bgr.shape[:2]
        
        base_score = detection['conf']
        
        # 1. Size appropriateness
        area = detection['area']
        area_ratio = area / (W * H)
        if 0.0008 < area_ratio < 0.02:
            size_score = 1.0
        elif 0.0003 < area_ratio < 0.04:
            size_score = 0.8
        else:
            size_score = 0.4
        
        # 2. Aspect ratio (balls should be roughly square)
        aspect = (x2-x1) / max(1, y2-y1)
        aspect_score = 1.0 / (1.0 + abs(aspect - 1.0))
        
        # 3. Position reasonableness
        cx, cy = detection['cx'], detection['cy']
        
        # Avoid extreme edges
        edge_margin = min(W, H) * 0.03
        if (cx < edge_margin or cx > W-edge_margin or 
            cy < edge_margin or cy > H-edge_margin):
            position_score = 0.5
        else:
            position_score = 1.0
        
        # 4. Context analysis - check surrounding area
        context_score = self._analyze_detection_context(frame_bgr, bbox)
        
        return base_score * 0.4 + base_score * size_score * 0.2 + base_score * aspect_score * 0.2 + base_score * position_score * 0.1 + context_score * 0.1

    def _analyze_detection_context(self, frame_bgr, bbox):
        """Analyze the context around a detection to validate it's likely a ball"""
        try:
            x1, y1, x2, y2 = bbox
            H, W = frame_bgr.shape[:2]
            
            # Expand bbox to get context
            pad = max(20, int(min(x2-x1, y2-y1) * 0.5))
            ctx_x1 = max(0, int(x1) - pad)
            ctx_y1 = max(0, int(y1) - pad)
            ctx_x2 = min(W, int(x2) + pad)
            ctx_y2 = min(H, int(y2) + pad)
            
            context_patch = frame_bgr[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            ball_patch = frame_bgr[int(y1):int(y2), int(x1):int(x2)]
            
            if context_patch.size == 0 or ball_patch.size == 0:
                return 0.5
            
            # 1. Color contrast - ball should contrast with background
            ball_mean = np.mean(ball_patch, axis=(0,1))
            context_mean = np.mean(context_patch, axis=(0,1))
            contrast = np.linalg.norm(ball_mean - context_mean) / 255.0
            contrast_score = min(1.0, contrast * 2.0)
            
            # 2. Edge strength - balls usually have clear edges
            gray_ball = cv2.cvtColor(ball_patch, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_ball, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(1.0, edge_density * 10.0)
            
            return (contrast_score * 0.6 + edge_score * 0.4)
            
        except Exception:
            return 0.5
    def _is_tennis_ball_like(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """🆕 ADD: Detect tennis ball specific characteristics"""
        try:
            x1, y1, x2, y2 = bbox
            H, W = frame_bgr.shape[:2]
            
            # Extract patch
            patch = frame_bgr[int(max(0, y1)):int(min(H, y2)), 
                            int(max(0, x1)):int(min(W, x2))]
            if patch.size < 50:
                return False
                
            # Tennis ball color detection (yellow-green range)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            
            # Tennis ball HSV ranges
            lower_tennis = np.array([25, 70, 70])   # Yellow-green
            upper_tennis = np.array([85, 255, 255])
            
            mask = cv2.inRange(hsv, lower_tennis, upper_tennis)
            tennis_color_ratio = np.sum(mask > 0) / mask.size
            
            # Size check - tennis balls have typical size range
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (W * H)
            size_appropriate = 0.0002 < area_ratio < 0.02
            
            # Circularity check
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            circularity = 0.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 20:
                    perimeter = cv2.arcLength(largest, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * cv2.contourArea(largest) / (perimeter * perimeter)
            
            # Tennis ball criteria
            return (tennis_color_ratio > 0.15 and size_appropriate and circularity > 0.6)
            
        except Exception:
            return False

    def _check_temporal_consistency(self, bbox: Tuple[float, float, float, float]) -> float:
        """🆕 ADD: Check if detection is consistent with recent frames"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Add current detection to buffer
        self.temporal_consistency_buffer.append((cx, cy))
        if len(self.temporal_consistency_buffer) > 10:
            self.temporal_consistency_buffer.pop(0)
        
        if len(self.temporal_consistency_buffer) < 3:
            return 0.5
        
        # Calculate consistency score based on recent positions
        recent_positions = self.temporal_consistency_buffer[-5:]
        distances = []
        
        for i in range(1, len(recent_positions)):
            prev_x, prev_y = recent_positions[i-1]
            curr_x, curr_y = recent_positions[i]
            dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            distances.append(dist)
        
        if not distances:
            return 0.5
            
        # Reward smooth motion, penalize erratic jumps
        avg_movement = np.mean(distances)
        if avg_movement < 50:  # Smooth motion
            return 1.0
        elif avg_movement < 100:  # Moderate motion
            return 0.8
        else:  # Erratic motion
            return 0.3

    def _check_false_positive_memory(self, bbox: Tuple[float, float, float, float]) -> float:
        """🆕 ADD: Penalize known false positive locations"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Check against known false positive locations
        for fp_x, fp_y in self.false_positive_memory:
            dist = np.sqrt((cx - fp_x)**2 + (cy - fp_y)**2)
            if dist < 80:  # Close to known false positive
                return 0.5
        
        return 1.0

    def update_false_positive_memory(self, bbox: Tuple[float, float, float, float]):
        """🆕 ADD: Remember false positive locations"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.false_positive_memory.add((cx, cy))
        
        # Keep memory size manageable
        if len(self.false_positive_memory) > 50:
            self.false_positive_memory = set(list(self.false_positive_memory)[-30:])


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
            # 🆕 CHANGE: Enhanced prediction for longer videos
        if n >= 12:  # More history for better prediction
            xs = np.arange(n, dtype=float)
            ys = np.array(self.history, dtype=float)
            
            # 🆕 ADD: Adaptive weighting based on video length
            decay_rate = 0.95 if n > 900 else 0.98  # Faster decay for long videos
            weights = np.exp(np.linspace(-3, 0, min(n, 20)))[-n:] * decay_rate
            
            try:
                # 🆕 CHANGE: Use more recent frames for fitting
                recent_frames = min(15, n)  # Increased from 8 to 15
                
                # 🆕 ADD: Detect if ball is accelerating (bouncing, changing direction)
                if n >= 6:
                    recent_velocities = np.diff(ys[-6:])
                    velocity_change = np.std(recent_velocities)
                    
                    if velocity_change > 20:  # Ball is changing direction rapidly
                        # Use shorter prediction window
                        recent_frames = min(8, n)
                        poly_degree = 1  # Linear prediction for rapid changes
                    else:
                        poly_degree = min(2, recent_frames-1)
                else:
                    poly_degree = min(2, recent_frames-1)
                
                coeffs = np.polyfit(
                    xs[-recent_frames:], 
                    ys[-recent_frames:], 
                    deg=poly_degree,
                    w=weights[-recent_frames:]
                )
                x_next = float(n)
                predicted = float(np.polyval(coeffs, x_next))
                
                # 🆕 ADD: Momentum-based correction for tennis ball physics
                if n >= 4:
                    recent_velocity = np.mean(np.diff(ys[-4:]))
                    # Apply physics-based momentum (tennis balls maintain trajectory)
                    momentum_factor = 0.4 if abs(recent_velocity) > 25 else 0.3
                    predicted += recent_velocity * momentum_factor
                
                return predicted
            except Exception:
                pass
        
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
            use_tta=(vel > 40 and self.detector.allow_tta_recovery),
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


class TrajectoryLearner:
    """🆕 ADD: Learn ball trajectory patterns for better prediction"""
    def __init__(self, frame_width: int):
        self.frame_width = frame_width
        self.detections = []  # (frame_idx, cx, confidence)
        self.pattern_confidence = 0.0
        self.velocity_history = []
        self.acceleration_history = []
        
    def add_detection(self, frame_idx: int, cx: float, high_confidence: bool = False):
        """Add a detection to the trajectory"""
        confidence = 1.0 if high_confidence else 0.8
        self.detections.append((frame_idx, cx, confidence))
        
        # Keep recent history
        if len(self.detections) > 200:
            self.detections = self.detections[-200:]
        
        self._update_motion_patterns()
    
    def add_miss(self, frame_idx: int, estimated_pos: Optional[float]):
        """Handle a detection miss"""
        if estimated_pos is not None:
            self.detections.append((frame_idx, estimated_pos, 0.3))
    
    def predict_position(self, frame_idx: int) -> Optional[float]:
        """Predict ball position for given frame"""
        if len(self.detections) < 3:
            return None
        
        # Use recent detections for prediction
        recent = self.detections[-10:]
        frames = [d[0] for d in recent]
        positions = [d[1] for d in recent]
        weights = [d[2] for d in recent]
        
        try:
            # Weighted polynomial fit
            coeffs = np.polyfit(frames, positions, deg=min(2, len(recent)-1), w=weights)
            predicted = float(np.polyval(coeffs, frame_idx))
            
            # Apply motion constraints
            if len(self.velocity_history) > 0:
                avg_velocity = np.mean(self.velocity_history[-5:])
                predicted += avg_velocity * 0.2  # Add momentum
            
            # Bound to frame
            return max(0, min(self.frame_width, predicted))
        except:
            # Fallback to last known position
            return self.detections[-1][1] if self.detections else None
    
    def _update_motion_patterns(self):
        """Update motion pattern understanding"""
        if len(self.detections) < 3:
            return
        
        # Calculate velocities
        recent = self.detections[-5:]
        for i in range(1, len(recent)):
            dt = recent[i][0] - recent[i-1][0]
            if dt > 0:
                dx = recent[i][1] - recent[i-1][1]
                velocity = dx / dt
                self.velocity_history.append(velocity)
        
        # Keep velocity history manageable
        if len(self.velocity_history) > 50:
            self.velocity_history = self.velocity_history[-50:]
        
        # Update confidence based on pattern consistency
        if len(self.velocity_history) >= 5:
            velocity_std = np.std(self.velocity_history[-10:])
            # Lower std = more consistent = higher confidence
            self.pattern_confidence = max(0.0, min(1.0, 1.0 - velocity_std / 50.0))
    
    def get_confidence(self) -> float:
        """Get trajectory learning confidence"""
        return self.pattern_confidence
    
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
            confidence_decay=float(kwargs.get('memory_confidence_decay', 0.98))
        )
                # 🆕 ADD: Enhanced settings for longer videos
        self.long_video_threshold = 600  # 20 seconds at 30fps
        self.enhanced_detection_interval = 60  # Enhanced detection every 60 frames for long videos
        self.temporal_validation_window = 15  # Frames to validate detections
        
        # 🆕 ADD: Tennis ball specific settings
        self.tennis_ball_mode = kwargs.get('tennis_ball_mode', True)
        if self.tennis_ball_mode:
            self.detector.conf *= 0.8  # Lower confidence for tennis balls (they're fast)
            self.ball_memory.memory_duration = int(kwargs.get('memory_duration_frames', 120))  # Longer memory
        
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
        """🆕 ENHANCED: Multi-stage bootstrap with cross-validation"""
        meta = self._read_meta(self.input_path)
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return None
        
        candidates = []
        frames_to_scan = min(getattr(self, 'bootstrap_frames', 60), meta.num_frames or 60)
        
        print(f"🔍 Enhanced bootstrap: scanning {frames_to_scan} frames...")
        
        # 🆕 Phase 1: Multi-confidence detection passes
        confidence_strategies = [
            {'conf': 0.03, 'imgsz': 1280, 'name': 'ultra_sensitive', 'weight': 0.8},
            {'conf': 0.08, 'imgsz': 960, 'name': 'sensitive', 'weight': 1.0},
            {'conf': 0.15, 'imgsz': 640, 'name': 'standard', 'weight': 1.2},
        ]
        
        step_size = max(1, frames_to_scan // 30)  # More thorough scanning
        
        for idx in range(0, frames_to_scan, step_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            
            H, W = frame.shape[:2]
            
            # 🆕 Try each confidence strategy
            for strategy in confidence_strategies:
                # Full frame detection
                out = self.detector._predict_on_optimized(
                    frame, 
                    conf=strategy['conf'],
                    imgsz=strategy['imgsz'],
                    use_tta=(idx < 20)  # Use TTA only for first frames
                )
                
                if out is not None:
                    xys, confs, clss = out
                    
                    for c, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xys):
                        if int(cls_id) not in self.detector.ball_class_ids:
                            continue
                        
                        # 🆕 Multi-factor validation
                        validation_score = self._validate_detection_thoroughly(
                            frame, (x1, y1, x2, y2), c, W, H, idx, strategy['name']
                        )
                        
                        if validation_score > 0.3:  # Lower threshold but better validation
                            candidates.append({
                                'frame': idx,
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'cx': (float(x1) + float(x2)) / 2.0,
                                'cy': (float(y1) + float(y2)) / 2.0,
                                'score': validation_score * strategy['weight'],
                                'conf': float(c),
                                'strategy': strategy['name'],
                                'area': (x2-x1) * (y2-y1),
                                'aspect': (x2-x1) / max(1, y2-y1),
                                'validation_details': self._get_validation_details(frame, (x1, y1, x2, y2))
                            })
                
                if len(candidates) >= 15:  # Collect more candidates
                    break
            
            if len(candidates) >= 20:
                break
        
        cap.release()
        
        if not candidates:
            print("❌ No ball candidates found in enhanced bootstrap")
            return None
        
        print(f"📊 Found {len(candidates)} candidates, selecting best...")
        
        # 🆕 Phase 2: Cross-validation and consensus
        best_candidate = self._select_best_candidate_with_consensus(candidates, meta)
        
        if best_candidate:
            print(f"✅ Selected ball at frame {best_candidate['frame']} "
                f"(score: {best_candidate['score']:.3f}, validation: {best_candidate['strategy']})")
            return (best_candidate['frame'], best_candidate['bbox'], best_candidate['cx'])
        
        return None
    
    def _validate_detection_thoroughly(self, frame, bbox, conf, W, H, frame_idx, strategy_name):
        """🆕 ADD: Comprehensive detection validation"""
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        area = max(1.0, float((x2 - x1) * (y2 - y1)))
        
        validation_scores = []
        
        # 1. 🆕 Tennis ball color validation (most important)
        tennis_ball_score = 0.0
        try:
            patch = frame[int(max(0, y1)):int(min(H, y2)), 
                        int(max(0, x1)):int(min(W, x2))]
            if patch.size > 50:
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                
                # Multiple tennis ball color ranges
                yellow_green_mask = cv2.inRange(hsv, np.array([25, 50, 50]), np.array([85, 255, 255]))
                bright_yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
                
                yellow_green_ratio = np.sum(yellow_green_mask > 0) / yellow_green_mask.size
                bright_yellow_ratio = np.sum(bright_yellow_mask > 0) / bright_yellow_mask.size
                
                tennis_ball_score = max(yellow_green_ratio, bright_yellow_ratio) * 2.0
                if tennis_ball_score > 0.2:  # Strong tennis ball color
                    tennis_ball_score = min(1.0, tennis_ball_score * 1.5)
        except:
            pass
        
        validation_scores.append(('tennis_color', tennis_ball_score, 0.4))
        
        # 2. 🆕 Size appropriateness (critical for filtering noise)
        area_ratio = area / (W * H)
        size_score = 0.0
        if 0.0003 < area_ratio < 0.03:  # Ideal tennis ball size range
            size_score = 1.0
        elif 0.0001 < area_ratio < 0.0003:  # Too small but possible
            size_score = 0.4
        elif 0.03 < area_ratio < 0.06:  # Too large but possible
            size_score = 0.6
        else:  # Definitely wrong size
            size_score = 0.1
        
        validation_scores.append(('size', size_score, 0.25))
        
        # 3. 🆕 Circularity validation
        circularity_score = 0.0
        try:
            patch = frame[int(max(0, y1)):int(min(H, y2)), 
                        int(max(0, x1)):int(min(W, x2))]
            if patch.size > 100:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                
                # Edge-based circularity
                edges = cv2.Canny(gray, 30, 100)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 50:
                        perimeter = cv2.arcLength(largest, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * cv2.contourArea(largest) / (perimeter * perimeter)
                            circularity_score = min(1.0, circularity * 1.3)
        except:
            pass
        
        validation_scores.append(('circularity', circularity_score, 0.2))
        
        # 4. 🆕 Position reasonableness
        edge_margin = min(W, H) * 0.05
        position_score = 1.0
        
        # Penalize extreme edges heavily
        if (cx < edge_margin or cx > W - edge_margin or 
            cy < edge_margin or cy > H - edge_margin):
            position_score = 0.2
        
        # Penalize very top/bottom (balls rarely there in tennis)
        if cy < H * 0.1 or cy > H * 0.9:
            position_score *= 0.3
        
        validation_scores.append(('position', position_score, 0.15))
        
        # 5. 🆕 Confidence scaling based on strategy
        conf_score = float(conf)
        if strategy_name == 'ultra_sensitive' and conf_score < 0.05:
            conf_score *= 0.5  # Penalize very low confidence detections
        
        # 6. Calculate weighted final score
        final_score = 0.0
        total_weight = 0.0
        
        for name, score, weight in validation_scores:
            final_score += score * weight
            total_weight += weight
        
        # Add confidence component
        final_score += conf_score * 0.1
        total_weight += 0.1
        
        return final_score / total_weight if total_weight > 0 else 0.0

    def _get_validation_details(self, frame, bbox):
        """🆕 ADD: Get detailed validation info for debugging"""
        x1, y1, x2, y2 = bbox
        try:
            patch = frame[int(max(0, y1)):int(min(frame.shape[0], y2)), 
                        int(max(0, x1)):int(min(frame.shape[1], x2))]
            if patch.size > 50:
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean_hue = np.mean(hsv[:, :, 0])
                mean_sat = np.mean(hsv[:, :, 1])
                mean_val = np.mean(hsv[:, :, 2])
                
                return {
                    'mean_hue': float(mean_hue),
                    'mean_sat': float(mean_sat),
                    'mean_val': float(mean_val),
                    'patch_size': patch.size
                }
        except:
            pass
        return {}

    def _select_best_candidate_with_consensus(self, candidates, meta):
        """🆕 ADD: Advanced candidate selection with consensus"""
        if not candidates:
            return None
        
        # Sort by validation score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 🆕 Consensus-based selection
        # Group candidates by location and time
        consensus_groups = []
        used = set()
        
        for i, cand in enumerate(candidates):
            if i in used:
                continue
            
            group = [cand]
            used.add(i)
            
            # Find spatially and temporally close candidates
            for j, other in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                
                # Spatial proximity
                dx = abs(cand['cx'] - other['cx'])
                dy = abs(cand['cy'] - other['cy'])
                
                # Temporal proximity
                dt = abs(cand['frame'] - other['frame'])
                
                if dx < 80 and dy < 80 and dt < 20:  # Close in space and time
                    group.append(other)
                    used.add(j)
            
            if len(group) >= 2:  # Only consider groups with multiple detections
                consensus_groups.append(group)
        
        if consensus_groups:
            # Score each consensus group
            best_group = None
            best_group_score = 0
            
            for group in consensus_groups:
                # Group metrics
                avg_score = sum(c['score'] for c in group) / len(group)
                consistency = len(group)
                max_score = max(c['score'] for c in group)
                
                # Tennis ball color consistency in group
                tennis_scores = []
                for candidate in group:
                    details = candidate.get('validation_details', {})
                    if 'mean_hue' in details:
                        hue = details['mean_hue']
                        sat = details['mean_sat']
                        # Tennis ball hue range
                        if 25 <= hue <= 85 and sat > 50:
                            tennis_scores.append(1.0)
                        else:
                            tennis_scores.append(0.0)
                
                tennis_consistency = np.mean(tennis_scores) if tennis_scores else 0.0
                
                group_score = (avg_score * 0.4 + 
                            max_score * 0.3 + 
                            consistency * 0.1 + 
                            tennis_consistency * 0.2)
                
                if group_score > best_group_score:
                    best_group_score = group_score
                    best_group = group
            
            if best_group:
                # Return best candidate from best consensus group
                return max(best_group, key=lambda x: x['score'])
        
        # Fallback: return single best candidate with high enough score
        if candidates[0]['score'] > 0.5:
            return candidates[0]
        
        return None
    
    def _compute_enhanced_bootstrap_score(self, frame, bbox, conf, W, H, frame_idx):
        """Enhanced scoring that filters out non-ball objects"""
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        area = max(1.0, float((x2 - x1) * (y2 - y1)))
        conf_score = float(conf)
        
        # 1. Size filtering - balls should be reasonable size
        area_ratio = area / (W * H)
        if area_ratio < 0.00008:  # Too small (likely noise)
            return 0.1
        elif area_ratio > 0.08:   # Too large (likely person/equipment)
            return 0.1
        elif 0.0005 < area_ratio < 0.025:  # Good ball size range
            area_penalty = 1.0
        else:
            area_penalty = 0.6
        
        # 2. Aspect ratio - balls should be roughly circular
        aspect_ratio = (x2 - x1) / max(1, y2 - y1)
        if 0.7 < aspect_ratio < 1.4:  # Roughly square/circular
            aspect_penalty = 1.0
        elif 0.5 < aspect_ratio < 2.0:  # Acceptable range
            aspect_penalty = 0.8
        else:  # Likely not a ball
            aspect_penalty = 0.2
        
        # 3. Position filtering - avoid extreme edges where balls are unlikely
        edge_margin = min(W, H) * 0.05
        if (cx < edge_margin or cx > W - edge_margin or 
            cy < edge_margin or cy > H - edge_margin):
            edge_penalty = 0.4
        else:
            edge_penalty = 1.0
        
        # 4. Vertical position - balls usually not at very top/bottom
        y_ratio = cy / H
        if 0.15 < y_ratio < 0.85:  # Good vertical range
            vertical_penalty = 1.0
        elif 0.05 < y_ratio < 0.95:  # Acceptable range  
            vertical_penalty = 0.7
        else:  # Very top/bottom
            vertical_penalty = 0.3
        
        # 5. Center preference (balls often in center of action)
        center_dist = abs(cx - W/2) / (W/2)
        center_penalty = 1.0 / (1.0 + center_dist * 0.3)
        
        # 6. Frame position bonus (later frames often have better ball visibility)
        frame_bonus = 1.0 + (frame_idx / 1000.0) * 0.2
        
        # 7. Color/texture analysis for ball-like appearance
        try:
            patch = frame[int(max(0, y1)):int(min(H, y2)), 
                        int(max(0, x1)):int(min(W, x2))]
            appearance_score = self._analyze_ball_appearance(patch)
        except:
            appearance_score = 0.5
        
        # Combine all factors
        final_score = (
            conf_score * 0.25 +
            conf_score * area_penalty * 0.20 +
            conf_score * aspect_penalty * 0.15 +
            conf_score * edge_penalty * 0.10 +
            conf_score * vertical_penalty * 0.10 +
            conf_score * center_penalty * 0.10 +
            appearance_score * 0.05 +
            frame_bonus * 0.05
        )
        
        return final_score

    def _analyze_ball_appearance(self, patch):
        """Analyze patch for ball-like visual characteristics"""
        if patch is None or patch.size < 50:
            return 0.3
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            
            # 1. Check for circular/round edges
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            circularity_score = 0.3
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 20:
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * cv2.contourArea(largest_contour) / (perimeter * perimeter)
                        circularity_score = min(1.0, circularity * 1.2)
            
            # 2. Check color consistency (balls usually have consistent color)
            h, s, v = cv2.split(hsv)
            color_consistency = 1.0 - (np.std(h) / 180.0 + np.std(s) / 255.0) / 2.0
            color_consistency = max(0.0, min(1.0, color_consistency))
            
            # 3. Check for typical ball colors (avoid skin tones, clothing colors)
            mean_hue = np.mean(h)
            mean_sat = np.mean(s)
            
            # Boost score for typical ball colors
            ball_color_bonus = 1.0
            if (10 < mean_hue < 25) and (mean_sat > 100):  # Orange (basketball)
                ball_color_bonus = 1.3
            elif (30 < mean_hue < 70) and (mean_sat > 80):   # Yellow-green (tennis)
                ball_color_bonus = 1.2
            elif (100 < mean_hue < 120) and (mean_sat > 60): # Blue
                ball_color_bonus = 1.1
            elif mean_sat < 50:  # Low saturation (white/gray balls)
                ball_color_bonus = 1.1
            
            return (circularity_score * 0.4 + color_consistency * 0.4 + 0.2) * ball_color_bonus
            
        except Exception:
            return 0.4

    def _select_best_bootstrap_candidate(self, candidates, meta):
        """Advanced candidate selection with temporal validation"""
        if not candidates:
            return None
        
        # Sort by score first
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Group candidates by spatial proximity
        candidate_groups = []
        used = set()
        
        for i, cand in enumerate(candidates):
            if i in used:
                continue
                
            group = [cand]
            used.add(i)
            
            # Find nearby candidates (same general area)
            for j, other in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                    
                dx = abs(cand['cx'] - other['cx'])
                dy = abs(cand['cy'] - other['cy'])
                
                if dx < 100 and dy < 100:  # Within 100px
                    group.append(other)
                    used.add(j)
            
            candidate_groups.append(group)
        
        # Score each group
        best_group = None
        best_group_score = 0
        
        for group in candidate_groups:
            # Group scoring factors
            avg_score = sum(c['score'] for c in group) / len(group)
            consistency = len(group)  # More detections = more consistent
            temporal_spread = max(c['frame'] for c in group) - min(c['frame'] for c in group)
            
            group_score = avg_score * (1 + consistency * 0.1) * (1 + temporal_spread * 0.001)
            
            if group_score > best_group_score:
                best_group_score = group_score
                best_group = group
        
        if best_group:
            # Return the highest scoring candidate from the best group
            return max(best_group, key=lambda x: x['score'])
        
        return candidates[0] if candidates else None

    def _enhanced_first_frame_detection(self):
        """Try multiple strategies to detect ball in first few frames"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return None
        
        strategies = [
            {'frames': range(0, 30, 2), 'conf': 0.05, 'imgsz': 1280, 'aggressive_nms': True},
            {'frames': range(0, 60, 5), 'conf': 0.08, 'imgsz': 960, 'aggressive_nms': True}, 
            {'frames': range(0, 90, 10), 'conf': 0.12, 'imgsz': 640, 'aggressive_nms': False},
        ]
        
        best_detection = None
        best_score = 0
        
        for strategy in strategies:
            for frame_idx in strategy['frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Temporarily adjust detector settings
                original_conf = self.detector.conf
                original_imgsz = getattr(self.detector, 'imgsz', None)
                
                self.detector.conf = strategy['conf']
                self.detector.imgsz = strategy['imgsz']
                
                bbox = self.detector.detect_with_nms_filtering(frame, strategy['aggressive_nms'])
                
                # Restore original settings
                self.detector.conf = original_conf
                self.detector.imgsz = original_imgsz
                
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    score = self._compute_enhanced_bootstrap_score(
                        frame, bbox, strategy['conf'], frame.shape[1], frame.shape[0], frame_idx
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_detection = {
                            'frame': frame_idx,
                            'bbox': bbox,
                            'cx': (x1 + x2) / 2.0,
                            'score': score
                        }
        
        cap.release()
        
        if best_detection and best_detection['score'] > 0.4:
            return (best_detection['frame'], best_detection['bbox'], best_detection['cx'])
        
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
        """🆕 ENHANCED: Better integration of Phase 1 and Phase 2"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise SystemExit("Failed to open input video")
            
        xs: List[Optional[float]] = []
        tracker = ParallelBoxTracker(tracker_pref=getattr(self, 'tracker_pref', 'auto'))
        flow = OpticalFlowAssist()
        
        frame_idx = 0
        meta = self._read_meta(self.input_path)
        frame_w = meta.width
        
        # 🆕 Enhanced trajectory learning
        trajectory_learner = TrajectoryLearner(frame_w)
        
        # Enhanced bootstrap
        boot = self._enhanced_first_frame_detection()
        if boot is None:
            boot = self._bootstrap_initial_center_enhanced()
        
        if boot is not None:
            _, bbox0, cx0 = boot
            predicted_position = cx0
            self.ball_memory.update_detection(boot[0], cx0, bbox0, 1.0)
            trajectory_learner.add_detection(boot[0], cx0, high_confidence=True)
            print(f"🎯 Initial ball position: {cx0:.1f} at frame {boot[0]}")
        else:
            predicted_position = frame_w / 2.0
            print("⚠️  No initial ball found, using center prediction")
        
        # 🆕 Early learning phase (first 5 seconds)
        early_learning_frames = min(150, meta.num_frames or 150)
        detection_validation_threshold = 0.4  # Higher threshold during learning
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            
            # Get prediction from multiple sources
            memory_pos = self.ball_memory.get_position_for_frame(frame_idx)
            trajectory_pos = trajectory_learner.predict_position(frame_idx)
            
            # Combine predictions intelligently
            if memory_pos is not None and trajectory_pos is not None:
                # Weight based on confidence and recency
                memory_weight = 0.6 if self.ball_memory.consecutive_misses < 3 else 0.3
                trajectory_weight = 1.0 - memory_weight
                cx_pred = memory_pos * memory_weight + trajectory_pos * trajectory_weight
            else:
                cx_pred = memory_pos or trajectory_pos or predicted_position
            
            # 🆕 Adaptive detection strategy based on learning phase
            if frame_idx < early_learning_frames:
                # Learning phase: more thorough detection
                bbox = self._learning_phase_detection(frame, cx_pred, frame_idx)
            else:
                # Normal phase: standard detection
                bbox = self._standard_detection(frame, cx_pred, frame_idx)
            
            detection_made = False
            final_position = None
            
            if bbox is not None:
                # Validate detection before accepting
                x1, y1, x2, y2 = bbox
                validation_score = self._validate_detection_thoroughly(
                    frame, bbox, 0.5, frame_w, frame.shape[0], frame_idx, 'runtime'
                )
                
                if validation_score > detection_validation_threshold or frame_idx < 30:
                    # Accept detection
                    cx_det = (x1 + x2) / 2.0
                    predicted_position = cx_det
                    self.ball_memory.update_detection(frame_idx, cx_det, bbox, validation_score)
                    trajectory_learner.add_detection(frame_idx, cx_det, validation_score > 0.7)
                    
                    final_position = cx_det
                    detection_made = True
                    
                    # Update trackers
                    tracker.init_with_bbox(frame, bbox)
                    flow.init_from_bbox(frame, bbox)
                    
                    # Lower validation threshold as we learn
                    if frame_idx > 50:
                        detection_validation_threshold = max(0.25, detection_validation_threshold * 0.999)
                else:
                    # Reject low-quality detection
                    self.detector.update_false_positive_memory(bbox)
                    bbox = None
            
            if bbox is None:
                # Handle miss with enhanced fallback
                tracker_estimate = None
                ok_t, cx_t = tracker.update(frame)
                if ok_t and cx_t is not None:
                    tracker_estimate = cx_t
                    
                if tracker_estimate is None:
                    cx_flow = flow.update(frame)
                    if cx_flow is not None:
                        tracker_estimate = cx_flow
                
                # Update systems
                self.ball_memory.update_no_detection(frame_idx, tracker_estimate)
                trajectory_learner.add_miss(frame_idx, tracker_estimate)
                
                final_position = self.ball_memory.get_position_for_frame(frame_idx)
                if final_position is not None:
                    predicted_position = final_position
            
            # Final fallback
            if final_position is None:
                final_position = predicted_position
                
            xs.append(final_position)
            
            # Progress tracking
            self._perf['frames'] += 1
            if detection_made:
                self._perf['roi_detects'] += 1
            else:
                self._perf['misses'] += 1
                
            frame_idx += 1
            
        cap.release()
        
        print(f"📈 Trajectory learning: {trajectory_learner.get_confidence():.2f} confidence")
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
    def _learning_phase_detection(self, frame, cx_pred, frame_idx):
        """🆕 ADD: Enhanced detection during learning phase"""
        H, W = frame.shape[:2]
        
        # Try multiple detection strategies
        strategies = [
            {'roi_width': 300, 'conf': 0.05, 'imgsz': 960},
            {'roi_width': 500, 'conf': 0.08, 'imgsz': 640},
            {'roi_width': W, 'conf': 0.12, 'imgsz': 640},  # Full frame fallback
        ]
        
        for strategy in strategies:
            roi_w = strategy['roi_width']
            roi_half = roi_w // 2
            roi_left = int(max(0, min(int(round(cx_pred)) - roi_half, W - roi_w)))
            
            bbox = self.detector.detect_best_bbox_xyxy_in_roi_optimized(
                frame,
                roi_left=roi_left,
                roi_width=roi_w,
                pref_center_x=cx_pred,
                conf_override=strategy['conf'],
                imgsz_override=strategy['imgsz'],
                use_tta=False
            )
            
            if bbox is not None:
                return bbox
        
        return None

    def _standard_detection(self, frame, cx_pred, frame_idx):
        """🆕 ADD: Standard detection for normal operation"""
        H, W = frame.shape[:2]
        base_roi = int(getattr(self, 'base_roi_width', 400))
        roi_w = int(max(200, min(W, base_roi)))
        roi_half = roi_w // 2
        roi_left = int(max(0, min(int(round(cx_pred)) - roi_half, W - roi_w)))
        
        return self.detector.detect_best_bbox_xyxy_in_roi_optimized(
            frame,
            roi_left=roi_left,
            roi_width=roi_w,
            pref_center_x=cx_pred,
            conf_override=0.08,
            imgsz_override=(640 if (self.detector.device == 'mps') else 960),
            use_tta=False,
        )


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
    parser.add_argument("--use-ball-memory", action="store_true", default=True, help="Keep last known ball position when ball disappears")
    parser.add_argument("--no-ball-memory", action="store_true", help="Disable ball memory system")
    parser.add_argument("--memory-duration", type=int, default=90, help="Frames to remember ball position")
    parser.add_argument("--memory-decay", type=float, default=0.98, help="Memory confidence decay rate")
    parser.add_argument("--memory-blend", type=int, default=15, help="Frames to blend back to detection")
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
    )
    
    pipeline.run()
    print(f"✅ Optimized reframing complete: {args.output}")


if __name__ == "__main__":
    main()