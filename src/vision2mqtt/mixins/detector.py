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

        # filter by configured labels and confidence
        min_conf = self.vision_config["min_confidence"]
        filtered = []
        for obj in objects:
            if obj.confidence < min_conf:
                continue
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

        return VisionResult(objects=filtered, processing_time_ms=round(elapsed_ms, 1))

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

        # parse YOLO output (standard format: [batch, num_detections, 6] or similar)
        # The exact output format depends on the .axmodel export
        # Common: each detection = [x1, y1, x2, y2, confidence, class_id]
        objects = []
        if outputs and len(outputs) > 0:
            detections = outputs[0]
            if hasattr(detections, "shape") and len(detections.shape) == 3:
                detections = detections[0]  # remove batch dim

            # COCO class names (80 classes)
            coco_names = [
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

            for det in detections:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls_id = det[:6]
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
