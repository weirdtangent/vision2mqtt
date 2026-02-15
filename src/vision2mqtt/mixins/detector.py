# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import TYPE_CHECKING, Any

from vision2mqtt.models.events import DetectedObject, MotionEvent, VisionResult

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt


class DetectorMixin:
    _detector_model: Any = None

    async def init_detector(self: Vision2Mqtt) -> None:
        backend = self.vision_config["backend"]
        model_path = self.vision_config["model"]

        if backend == "ultralytics":
            self.logger.info(f"initializing ultralytics backend with model: {model_path}")
            self._detector_model = await asyncio.to_thread(self._load_ultralytics_model, model_path)
        elif backend == "axcl":
            self.logger.info(f"initializing axcl backend with model: {model_path}")
            self._detector_model = await asyncio.to_thread(self._load_axcl_model, model_path)

        self.logger.info(f"detector ready (backend={backend})")

    def _load_ultralytics_model(self: Vision2Mqtt, model_path: str) -> Any:
        from ultralytics import YOLO

        model = YOLO(model_path)
        return model

    @staticmethod
    def _resolve_input_layout(input_shape: list[Any]) -> tuple[int, bool]:
        """Derive the square spatial input size and layout from a model input shape.

        Handles NHWC [1, H, W, 3] and NCHW [1, 3, H, W] layouts.
        Returns (spatial_size, is_nchw).
        Raises ValueError if the spatial size cannot be determined.
        """
        if len(input_shape) == 4:
            _, d1, d2, d3 = input_shape
            # NHWC: last dim is channels (1 or 3)
            if isinstance(d3, int) and d3 in (1, 3) and isinstance(d1, int) and isinstance(d2, int):
                h, w, nchw = d1, d2, False
            # NCHW: second dim is channels (1 or 3)
            elif isinstance(d1, int) and d1 in (1, 3) and isinstance(d2, int) and isinstance(d3, int):
                h, w, nchw = d2, d3, True
            else:
                raise ValueError(f"Unable to determine spatial dimensions from shape {input_shape!r}")
            if h != w:
                raise ValueError(f"Expected square spatial dimensions but got H={h}, W={w} from shape {input_shape!r}")
            return h, nchw
        raise ValueError(f"Expected 4D input shape but got {input_shape!r}")

    def _load_axcl_model(self: Vision2Mqtt, model_path: str) -> Any:
        import axengine

        session = axengine.InferenceSession(model_path)

        # cache input metadata to avoid per-frame overhead
        input_info = session.get_inputs()[0]
        self._axcl_input_name: str = input_info.name
        self._axcl_input_size: int
        self._axcl_nchw: bool
        self._axcl_input_size, self._axcl_nchw = DetectorMixin._resolve_input_layout(list(input_info.shape))

        return session

    async def detect_objects(self: Vision2Mqtt, event: MotionEvent) -> VisionResult:
        backend = self.vision_config["backend"]
        start = time.monotonic()

        if backend == "ultralytics":
            objects = await asyncio.to_thread(self._detect_ultralytics, event.image_b64)
        elif backend == "axcl":
            objects = await asyncio.to_thread(self._detect_axcl, event.image_b64)
        else:
            objects = []

        elapsed_ms = (time.monotonic() - start) * 1000

        # filter by confidence first
        min_conf = self.vision_config["min_confidence"]
        confident = [obj for obj in objects if obj.confidence >= min_conf]

        # save all confident detections before label filtering (for composite logic)
        all_detections = [DetectedObject(label=obj.raw_label, raw_label=obj.raw_label, confidence=round(obj.confidence, 3), bbox=obj.bbox) for obj in confident]

        # filter by configured labels
        filtered = []
        for obj in confident:
            simplified = self.get_simplified_label(obj.raw_label)
            if simplified:
                filtered.append(
                    DetectedObject(
                        label=simplified,
                        raw_label=obj.raw_label,
                        confidence=round(obj.confidence, 3),
                        bbox=obj.bbox,
                    )
                )

        return VisionResult(objects=filtered, all_detections=all_detections, processing_time_ms=round(elapsed_ms, 1))

    def _detect_ultralytics(self: Vision2Mqtt, image_b64: str) -> list[DetectedObject]:
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))

        results = self._detector_model(image, verbose=False)
        objects = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            img_h, img_w = result.orig_shape
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                raw_label = result.names[cls_id]

                # normalize bbox to 0-1
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = [
                    round(x1 / img_w, 4),
                    round(y1 / img_h, 4),
                    round(x2 / img_w, 4),
                    round(y2 / img_h, 4),
                ]

                objects.append(
                    DetectedObject(
                        label=raw_label,
                        raw_label=raw_label,
                        confidence=conf,
                        bbox=bbox,
                    )
                )
        return objects

    # COCO class names (80 classes) used by YOLO models
    _COCO_NAMES: list[str] = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # Pre-NMS confidence threshold to reduce box count before expensive NMS loop
    _PRE_NMS_CONF: float = 0.25

    @staticmethod
    def _postprocess_yolo_raw(
        outputs: list[Any],
        input_size: int,
        num_classes: int = 80,
        dfl_bins: int = 16,
        nms_iou: float = 0.45,
        pre_nms_conf: float = _PRE_NMS_CONF,
    ) -> Any:
        """Decode raw YOLO feature maps into [x1, y1, x2, y2, conf, class_id] detections.

        Handles multi-head output (e.g. 3 scales: 80x80, 40x40, 20x20) with either:
        - DFL bounding box regression (64 channels = 4 * 16 bins) + class scores (YOLO11: 144ch)
        - Direct bounding box regression (4 channels) + class scores (YOLO26: 84ch)

        The format is auto-detected from the channel count of each head.
        """
        import numpy as np

        # Accepted channel counts: DFL (4*16 + 80 = 144) or direct (4 + 80 = 84)
        dfl_channels = 4 * dfl_bins  # 64
        accepted_channels = {dfl_channels + num_classes, 4 + num_classes}  # {144, 84}

        all_boxes = []
        all_scores = []

        skipped_heads = 0
        for head in outputs:
            # head shape: (1, H, W, C) or (1, C, H, W)
            if head.ndim == 4:
                head = head[0]  # remove batch dim -> (H, W, C) or (C, H, W)

            # detect and handle CHW layout
            if head.shape[0] in accepted_channels and head.shape[0] != head.shape[-1]:
                head = np.transpose(head, (1, 2, 0))  # CHW -> HWC

            grid_h, grid_w, channels = head.shape
            if channels not in accepted_channels:
                skipped_heads += 1
                continue

            # derive actual bbox channels and DFL bin count from this head
            head_bbox_channels = channels - num_classes  # 64 (DFL) or 4 (direct)
            head_dfl_bins = head_bbox_channels // 4  # 16 or 1

            stride = input_size / grid_h

            # split bbox regs and class scores
            bbox_raw = head[..., :head_bbox_channels]
            cls_raw = head[..., head_bbox_channels:]  # (H, W, 80)

            # numerically stable sigmoid for class scores
            cls_scores = np.where(cls_raw >= 0, 1.0 / (1.0 + np.exp(-cls_raw)), np.exp(cls_raw) / (1.0 + np.exp(cls_raw)))

            if head_dfl_bins > 1:
                # DFL decode (YOLO11): reshape to (H, W, 4, N), softmax over bins, weighted sum
                bbox_dfl = bbox_raw.reshape(grid_h, grid_w, 4, head_dfl_bins)
                bbox_dfl_exp = np.exp(bbox_dfl - np.max(bbox_dfl, axis=-1, keepdims=True))
                bbox_dfl_softmax = bbox_dfl_exp / (np.sum(bbox_dfl_exp, axis=-1, keepdims=True) + 1e-9)
                dfl_weights = np.arange(head_dfl_bins, dtype=np.float32)
                bbox_decoded = np.sum(bbox_dfl_softmax * dfl_weights, axis=-1)  # (H, W, 4)
            else:
                # Direct regression (YOLO26): 4 channels are raw distance values
                bbox_decoded = bbox_raw.reshape(grid_h, grid_w, 4)

            # build grid offsets
            gy, gx = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")

            # convert dist-from-center to absolute coords in input space
            x1 = (gx + 0.5 - bbox_decoded[..., 0]) * stride
            y1 = (gy + 0.5 - bbox_decoded[..., 1]) * stride
            x2 = (gx + 0.5 + bbox_decoded[..., 2]) * stride
            y2 = (gy + 0.5 + bbox_decoded[..., 3]) * stride

            boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)  # (H*W, 4)
            scores = cls_scores.reshape(-1, num_classes)  # (H*W, 80)

            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            if skipped_heads == len(outputs) and len(outputs) > 0:
                raise ValueError(f"All {len(outputs)} output heads skipped: channel mismatch (expected one of {sorted(accepted_channels)})")
            return np.empty((0, 6), dtype=np.float32)

        all_boxes_arr = np.concatenate(all_boxes, axis=0)  # (N, 4)
        all_scores_arr = np.concatenate(all_scores, axis=0)  # (N, 80)

        # per-cell best class
        cls_ids = np.argmax(all_scores_arr, axis=1)  # (N,)
        confs = all_scores_arr[np.arange(len(cls_ids)), cls_ids]  # (N,)

        # pre-NMS confidence filter to reduce box count
        conf_mask = confs >= pre_nms_conf
        if not np.any(conf_mask):
            return np.empty((0, 6), dtype=np.float32)
        all_boxes_arr = all_boxes_arr[conf_mask]
        cls_ids = cls_ids[conf_mask]
        confs = confs[conf_mask]

        # greedy NMS (class-agnostic)
        order = np.argsort(-confs)
        keep: list[int] = []
        suppressed = np.zeros(len(order), dtype=bool)
        for i in range(len(order)):
            idx = order[i]
            if suppressed[i]:
                continue
            keep.append(idx)
            # compute IoU with remaining
            xx1 = np.maximum(all_boxes_arr[idx, 0], all_boxes_arr[order[i + 1 :], 0])
            yy1 = np.maximum(all_boxes_arr[idx, 1], all_boxes_arr[order[i + 1 :], 1])
            xx2 = np.minimum(all_boxes_arr[idx, 2], all_boxes_arr[order[i + 1 :], 2])
            yy2 = np.minimum(all_boxes_arr[idx, 3], all_boxes_arr[order[i + 1 :], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_a = (all_boxes_arr[idx, 2] - all_boxes_arr[idx, 0]) * (all_boxes_arr[idx, 3] - all_boxes_arr[idx, 1])
            area_b = (all_boxes_arr[order[i + 1 :], 2] - all_boxes_arr[order[i + 1 :], 0]) * (
                all_boxes_arr[order[i + 1 :], 3] - all_boxes_arr[order[i + 1 :], 1]
            )
            iou = inter / (area_a + area_b - inter + 1e-6)
            suppressed[i + 1 :] |= iou > nms_iou

        if not keep:
            return np.empty((0, 6), dtype=np.float32)

        keep_arr = np.array(keep)
        return np.column_stack([all_boxes_arr[keep_arr], confs[keep_arr], cls_ids[keep_arr].astype(np.float32)])  # (K, 6)

    def _detect_axcl(self: Vision2Mqtt, image_b64: str) -> list[DetectedObject]:
        import numpy as np
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size

        input_size = self._axcl_input_size

        # resize with letterbox
        scale = min(input_size / img_w, input_size / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        padded = Image.new("RGB", (input_size, input_size), (114, 114, 114))
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))

        input_data = np.array(padded, dtype=np.uint8)[np.newaxis, ...]  # NHWC (1, H, W, 3)
        if self._axcl_nchw:
            input_data = np.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW

        # run inference
        outputs = self._detector_model.run(None, {self._axcl_input_name: input_data})

        # postprocess raw YOLO feature maps into detections
        detections = DetectorMixin._postprocess_yolo_raw(outputs, input_size)

        coco_names = DetectorMixin._COCO_NAMES
        objects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            conf = float(conf)
            cls_id = int(cls_id)

            if cls_id < 0 or cls_id >= len(coco_names):
                continue

            raw_label = coco_names[cls_id]

            # undo letterbox: map back to original image coords, then normalize
            x1 = (float(x1) - pad_x) / scale / img_w
            y1 = (float(y1) - pad_y) / scale / img_h
            x2 = (float(x2) - pad_x) / scale / img_w
            y2 = (float(y2) - pad_y) / scale / img_h

            bbox = [round(max(0, x1), 4), round(max(0, y1), 4), round(min(1, x2), 4), round(min(1, y2), 4)]

            objects.append(
                DetectedObject(
                    label=raw_label,
                    raw_label=raw_label,
                    confidence=conf,
                    bbox=bbox,
                )
            )

        return objects
