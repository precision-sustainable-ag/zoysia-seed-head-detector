from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Protocol, runtime_checkable, Union

from omegaconf import DictConfig
import yaml

from cvat_sdk.core.proxies.tasks import Task as CvatTask

log = logging.getLogger(__name__)


# =====================================================================
# ========================== CVAT Formatters ==========================
# =====================================================================


@runtime_checkable
class AnnotationFormatter(Protocol):
    """
    Strategy interface for formatting a CVAT task into a target dataset format.
    The exporter will:
      1) call `begin(...)` once with task metadata & labels,
      2) call `add_image(...)` per frame,
      3) call `add_rectangle(...)` per rectangle shape,
      4) finally call `write(out_dir)` to persist files and return the main artifact path.
    """

    def begin(
        self,
        task: CvatTask,
        frames_meta: List[Dict[str, Any]],
        labels: List[Dict[str, Any]],
    ) -> None: ...

    def add_image(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
    ) -> None: ...

    def add_rectangle(
        self,
        image_id: int,
        label_id: int,
        points: List[float],                # [x1,y1,x2,y2]
        attributes: Optional[List[Dict]] = None,
    ) -> None: ...

    def write(self, out_dir: Path) -> Path: ...



# ========================== COCO Formatter =========================

@dataclass
class CocoFormatter(AnnotationFormatter):
    """
    COCO Detection v1.0 formatter.
    - Forces a background category with id=0.
    - Maps CVAT label_id -> COCO category_id (others start at 1).
    - Exports rectangles as COCO bboxes, no segmentation.

    Parameters
    ----------
    background_names : names treated as background (case-insensitive).
    always_include_background : include synthetic background if none exists on server.
    """

    name: str = "coco"
    background_names: Tuple[str, ...] = ("background",)
    always_include_background: bool = True

    def __post_init__(self) -> None:
        self._bg_name_set = {norm_name(n) for n in self.background_names}
        self._coco: Dict[str, Any] = self._new_coco()
        self._labelid_to_catid: Dict[int, int] = {}
        self._next_ann_id: int = 1

    @staticmethod
    def _new_coco() -> Dict[str, Any]:
        return {
            "info": {"description": "COCO detection export from CVAT", "version": "1.0", "year": 2025},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

    # ---- lifecycle ----

    def begin(self, task: CvatTask, frames_meta: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> None:
        # deterministic ordering for stable category ids
        labels = list(labels)
        labels.sort(key=lambda d: (norm_name(d.get("name")), int(d.get("id") or 10**9)))
        # split background (first match wins)
        background_label = None
        others: List[Dict[str, Any]] = []
        for lab in labels:
            if background_label is None and norm_name(lab.get("name")) in self._bg_name_set:
                background_label = lab
            else:
                others.append(lab)

        # id=0 background
        if background_label is not None or self.always_include_background:
            bg_name = (background_label.get("name") if background_label else None) or "background"
            bg_super = (background_label.get("parent_id") if background_label else "") or ""
            self._coco["categories"].append({"id": 0, "name": bg_name, "supercategory": str(bg_super)})
            if background_label and isinstance(background_label.get("id"), int):
                self._labelid_to_catid[int(background_label["id"])] = 0

        # others -> 1..N
        next_id = 1
        for lab in others:
            name = lab.get("name") or f"class_{next_id}"
            supercat = lab.get("parent") or lab.get("parent_id") or ""
            self._coco["categories"].append({"id": next_id, "name": name, "supercategory": str(supercat)})
            if isinstance(lab.get("id"), int):
                self._labelid_to_catid[int(lab["id"])] = next_id
            next_id += 1

    def add_image(self, image_id: int, file_name: str, width: int, height: int) -> None:
        self._coco["images"].append(
            {"id": image_id, "file_name": file_name, "width": int(width or 0), "height": int(height or 0)}
        )

    def add_rectangle(
        self,
        image_id: int,
        label_id: int,
        points: List[float],
        attributes: Optional[List[Dict]] = None,
    ) -> None:
        if len(points) != 4:
            return
        cat_id = self._labelid_to_catid.get(label_id)
        if cat_id is None:
            # unknown label id -> skip
            return
        x, y, w, h = _rectangle_points_to_xywh(points)
        ann = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [x, y, w, h],
            "area": float(w * h),
            "iscrowd": 0,
            "segmentation": [],
        }
        if attributes:
            ann["attributes"] = {a.get("name"): a.get("value") for a in attributes if "name" in a}
        self._coco["annotations"].append(ann)
        self._next_ann_id += 1

    def write(self, out_dir: Path) -> Path:
        ensure_dir(out_dir)
        out_json = out_dir / "annotations.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(self._coco, f, ensure_ascii=False, indent=2, separators=(",", ":"))
        return out_json

# ========================== YOLO Formatter =========================

@dataclass
class YoloDet10Formatter(AnnotationFormatter):
    """
    Ultralytics YOLO Detection v1.0 formatter.

    - Background is *excluded* from class names and annotations.
    - Class ids are 0..(K-1) over non-background labels in deterministic order.
    - One .txt per image under labels/, lines: "<cls> <xc> <yc> <w> <h>" (all normalized).
    - Writes dataset.yaml with paths pointing to the 'images' directory by default.

    Parameters
    ----------
    background_names : Tuple[str, ...]
        Names treated as background (case-insensitive).
    dataset_name : str
        Name to put in dataset.yaml's "name".
    put_paths_relative_to_outdir : bool
        If True, dataset.yaml uses relative paths "images" and "labels".
    """
    name: str = "yolo_det"
    background_names: Tuple[str, ...] = ("background",)
    dataset_name: str = "cvat_yolo_det10_export"
    put_paths_relative_to_outdir: bool = True

    # internal state
    _bg_name_set: set = field(init=False, repr=False)
    _labelid_to_classid: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _class_names: List[str] = field(default_factory=list, init=False, repr=False)
    _image_sizes: Dict[int, Tuple[int, int]] = field(default_factory=dict, init=False, repr=False)  # image_id -> (W,H)
    _image_files: Dict[int, str] = field(default_factory=dict, init=False, repr=False)             # image_id -> file_name
    _per_image_anns: Dict[int, List[Tuple[int, float, float, float, float]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._bg_name_set = {norm_name(n) for n in self.background_names}

    # ---- lifecycle ----

    def begin(self, task, frames_meta: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> None:
        # deterministic order for stable class ids
        labels = list(labels)
        labels.sort(key=lambda d: (norm_name(d.get("name")), int(d.get("id") or 10**9)))

        # build class list (exclude background names)
        class_names: List[str] = []
        labelid_to_classid: Dict[int, int] = {}
        for lab in labels:
            nm = lab.get("name")
            lid = lab.get("id")
            if norm_name(nm) in self._bg_name_set:
                continue
            if isinstance(lid, int):
                labelid_to_classid[lid] = len(class_names)
            class_names.append(nm or f"class_{len(class_names)}")

        self._class_names = class_names
        self._labelid_to_classid = labelid_to_classid
        self._per_image_anns.clear()
        self._image_sizes.clear()
        self._image_files.clear()

    def add_image(self, image_id: int, file_name: str, width: int, height: int) -> None:
        self._image_sizes[image_id] = (int(width or 0), int(height or 0))
        self._image_files[image_id] = file_name
        self._per_image_anns.setdefault(image_id, [])

    def add_rectangle(
        self,
        image_id: int,
        label_id: int,
        points: List[float],
        attributes: Optional[List[Dict]] = None,
    ) -> None:
        if len(points) != 4:
            return
        # map CVAT label -> YOLO class; skip unknowns (e.g., background)
        cls = self._labelid_to_classid.get(label_id)
        if cls is None:
            return

        # CVAT rect points: [x1,y1,x2,y2]
        x1, y1, x2, y2 = points
        x = min(x1, x2); y = min(y1, y2)
        w = abs(x2 - x1); h = abs(y2 - y1)

        W, H = self._image_sizes.get(image_id, (0, 0))
        if W <= 0 or H <= 0:
            # cannot normalize; skip to avoid invalid rows
            return

        xc = (x + w / 2.0) / W
        yc = (y + h / 2.0) / H
        wn = w / W
        hn = h / H

        self._per_image_anns.setdefault(image_id, []).append((cls, xc, yc, wn, hn))

    def write(self, out_dir: Path) -> Path:
        images_dir = out_dir / "images"   # exporter is responsible for image downloads (if any)
        labels_dir = out_dir / "labels"
        ensure_dir(labels_dir)

        # write label files
        for image_id, rows in self._per_image_anns.items():
            stem = Path(self._image_files.get(image_id, f"frame_{image_id:06d}.jpg")).stem
            txt_path = labels_dir / f"{stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for cls, xc, yc, wn, hn in rows:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        # dataset.yaml (minimal, single bucket)
        # You can later adjust to real train/val/test splits.
        yaml_dict = {
            "path": ".",
            "train": "images" if self.put_paths_relative_to_outdir else str(images_dir),
            "val":   "images" if self.put_paths_relative_to_outdir else str(images_dir),
            "names": {i: n for i, n in enumerate(self._class_names)},
            "task": "detect",
            "nc": len(self._class_names),
        }
        yaml_path = out_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            # write simple YAML without importing PyYAML
            # (Ultralytics accepts JSON superset; but we’ll hand-roll a tiny YAML)
            f.write(f"name: {self.dataset_name}\n")
            f.write(f"path: {yaml_dict['path']}\n")
            f.write(f"train: {yaml_dict['train']}\n")
            f.write(f"val: {yaml_dict['val']}\n")
            f.write("task: detect\n")
            f.write(f"nc: {yaml_dict['nc']}\n")
            f.write("names:\n")
            for i, n in enumerate(self._class_names):
                # quote names defensively
                f.write(f"  {i}: \"{n}\"\n")

        # Primary artifact for YOLO is dataset.yaml
        return yaml_path



# ------ Yolo2Coco converter ------------------------

class YOLO2COCOConverter:
    """
    Convert a folder of Ultralytics YOLO .txt labels to a COCO JSON ready for CVAT.

    Assumptions:
      - One .txt per image, same filename stem.
      - YOLO format lines: <class> <cx> <cy> <w> <h>   (all normalized to [0,1])
      - Detection (bounding boxes) task (no polygons).

    label_map may be provided in one of three forms:
      1) dict[int, str] : yolo_class_index -> category_name
      2) dict[str, int] : category_name -> desired_category_id (COCO ids)
      3) list[str]      : index position is yolo_class_index, value is category_name

    Example:
        converter = YOLO2COCOConverter("/data/zoysia/yolo_export", {0: "seed_head"})
        converter.convert("zoysia_seed_heads_coco.json")
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root_dir = Path(cfg.paths.workdir) / cfg.cvat.detect.importer.root_dir
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"
        self.label_map = cfg.cvat.detect.importer.label_map
        self.image_exts = (".jpg", ".jpeg")

    def convert(self, out_json: Union[str, Path]) -> Path:
        root = Path(self.root_dir)
        assert root.exists(), f"Root not found: {root}"

        # Build categories mapping: yolo_idx -> (coco_id, name)
        idx_to_catid, catid_to_name = self._normalize_label_map(self.label_map)
        categories = [
            {"id": cid, "name": catid_to_name[cid], "supercategory": ""}
            for cid in sorted(catid_to_name.keys())
        ]

        images, annotations = [], []
        ann_id = 1

        # Collect image files (use stems to locate corresponding .txt)
        image_files = sorted([p for p in self.image_dir.rglob("*") if p.suffix.lower() in self.image_exts])
        log.info(f"Found {len(image_files)} images under {root}")

        seen_stems = set()
        for img_path in image_files:
            stem = img_path.stem
            if stem in seen_stems:
                # In case multiple images with same stem but different extensions exist, prefer first found.
                log.warning(f"Duplicate stem encountered, keeping first: {stem}")
                continue
            seen_stems.add(stem)

            try:
                width, height = self._read_image_size(img_path)
            except Exception as e:
                log.warning(f"Skip unreadable image {img_path}: {e}")
                continue

            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": str(img_path.name),
                "width": width,
                "height": height
            })

            label_dir = self.label_dir

            label_path = label_dir / f"{stem}.txt"
            if not label_path.exists():
                # Remove image record if we only want annotated images
                images.pop()
                continue

            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                images.pop()
                continue

            for li, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    log.warning(f"Bad label line ({label_path}:{li}): {line}")
                    continue
                try:
                    cls_raw = parts[0]
                    cx, cy, w, h = map(float, parts[1:5])
                    # class id comes as int or str; map to COCO id via idx_to_catid
                    yolo_idx = int(cls_raw)
                    if yolo_idx not in idx_to_catid:
                        log.warning(f"Unknown class index {yolo_idx} at {label_path}:{li}; skipping.")
                        continue
                    coco_cid = idx_to_catid[yolo_idx]

                    # Convert normalized center-wh to absolute xywh
                    x_abs, y_abs, w_abs, h_abs = self._yolo_to_coco_bbox(cx, cy, w, h, width, height)

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": coco_cid,
                        "bbox": [x_abs, y_abs, w_abs, h_abs],
                        "area": float(w_abs * h_abs),
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1
                except Exception as e:
                    log.warning(f"Failed to parse line ({label_path}:{li}): {line} ({e})")
                    continue

        coco = {
            "info": {
                "description": "Converted from Ultralytics YOLO labels",
                "version": "1.0",
            },
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        log.info(f"COCO saved: {out_path}  (images: {len(images)}, annotations: {len(annotations)})")
        return out_path

    # ---------- helpers ----------

    @staticmethod
    def _read_image_size(path: Path) -> Tuple[int, int]:
        with Image.open(path) as im:
            w, h = im.size
        return w, h

    @staticmethod
    def _yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float, W: int, H: int) -> List[float]:
        # YOLO normalized center (cx,cy), size (w,h) -> absolute top-left (x,y) and size (w,h)
        x = (cx - w / 2.0) * W
        y = (cy - h / 2.0) * H
        bw = w * W
        bh = h * H
        # Clamp to image bounds and ensure non-negative
        x = max(0.0, min(float(W), x))
        y = max(0.0, min(float(H), y))
        # Trim width/height if they overflow
        if x + bw > W:
            bw = max(0.0, float(W) - x)
        if y + bh > H:
            bh = max(0.0, float(H) - y)
        return [float(x), float(y), float(bw), float(bh)]

    @staticmethod
    def _normalize_label_map(label_map: Dict) -> Tuple[Dict[int, int], Dict[int, str]]:
        """
        Returns:
          idx_to_catid: maps YOLO class index -> COCO category id
          catid_to_name: maps COCO category id -> name
        """
        if isinstance(label_map, list):
            # list[str] => indices -> names; assign COCO ids starting at 1
            names = list(label_map)
            catid_to_name = {i + 1: n for i, n in enumerate(names)}
            idx_to_catid = {i: i + 1 for i in range(len(names))}
            return idx_to_catid, catid_to_name

        if all(isinstance(k, int) for k in label_map.keys()):
            # dict[int, str] => yolo_idx -> name; assign COCO ids starting at 1 by order of yolo_idx
            items = sorted(label_map.items(), key=lambda kv: kv[0])  # sort by yolo idx
            catid_to_name = {i + 1: name for i, (_, name) in enumerate(items)}
            # map each yolo idx to its new catid (1-based in order)
            yolo_to_catid = {yolo_idx: i + 1 for i, (yolo_idx, _) in enumerate(items)}
            return yolo_to_catid, catid_to_name

        if all(isinstance(k, str) for k in label_map.keys()):
            # dict[str, int] => name -> desired COCO id
            name_to_id = dict(label_map)
            catid_to_name = {cid: name for name, cid in name_to_id.items()}
            # build yolo_idx -> coco_id by ordering names alphabetically and mapping to provided ids
            # (caller likely also controls YOLO class order externally; we'll assume yolo_idx aligns to sorted names)
            sorted_names = sorted(name_to_id.keys())
            idx_to_catid = {i: name_to_id[name] for i, name in enumerate(sorted_names)}
            return idx_to_catid, catid_to_name

        raise ValueError("Unsupported label_map format. Use dict[int,str], dict[str,int], or list[str].")

# =====================================================================
# ========================== Utils/Helpers ============================
# =====================================================================

# ========================== Helpers =========================

def norm_name(s: Any) -> str:
    return str(s or "").strip().casefold()

def _rectangle_points_to_xywh(points: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = points
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return float(x), float(y), float(w), float(h)

def get_formatter(format_name: str) -> AnnotationFormatter:
    """
    Factory for selecting a formatter by name. Add new formats here.
    Accepts friendly names like "coco 1.0", "Coco1.0", etc.
    """
    key = norm_name(format_name).replace(" ", "")
    if key in {"coco1.0", "coco", "cocodetection", "cocodet"}:
        return CocoFormatter(background_names=("background",), always_include_background=True)
    if key in {"yolo1.0", "yolodetection1.0", "ultralyticsyolo", "yolo", "yolo_det_1_0", "Ultralytics Yolo Detect 1.0"}:
        # background is ignored for YOLO
        return YoloDet10Formatter(background_names=("background",))
    raise ValueError(f"Unsupported detection format: {format_name!r}")

def chunk(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])

def parse_yolo_names(names: Any) -> List[str]:
    """
    Accepts dataset.yaml 'names' as list or {id:name} dict.
    """
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        # map 0..N-1 in order
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    raise ValueError("Unsupported 'names' type in dataset.yaml; expected list or dict.")


# ========================== Path/Dir Utils =========================

def read_yaml_from_dictconfig(cfg: DictConfig) -> Dict[str, Any]:
    path = Path(cfg.paths.key_path)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)    
    return loaded

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ========================== Utils =========================

# --- CVAT Operator Class ---

class CVATOperator:
    def __init__(self, cfg: DictConfig) -> None:
        keys = read_yaml_from_dictconfig(cfg)
        self.url = keys["CVAT"]["url"]
        self.email = keys["CVAT"]["email"]
        self.password = keys["CVAT"]["password"]
        self.org_id = keys["CVAT"]["org_id"]
        self.org_slug = keys["CVAT"]["org_slug"]
        
        self.project_id = cfg.cvat.project_id
        self.task_id = cfg.cvat.task_id

        # Label format
        self.label_format = cfg.cvat.label_format
        self.class_names = cfg.cvat.class_names
        self.image_quality = cfg.cvat.image_quality

