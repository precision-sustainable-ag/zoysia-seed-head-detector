#!/usr/bin/env python3
"""
Prepare a CVAT-ready ZIP for a given batch_id in a chosen dataset format.

Formats
-------
- images_only
    <zip>/
      images/train|val|test/<files>

- yolo_det_1_0 (Ultralytics YOLO Detection v1.0)
    <zip>/
      dataset.yaml
      images/train|val|test/<files>
      labels/train|val|test/<.txt>   # optional if you have YOLO labels

- camvid (semantic segmentation)
    <zip>/
      classes.txt
      images/train|val|test/<files>
      labels/train|val|test/<.png>   # optional if you have masks

Config (OmegaConf)
------------------
batch_id: "TX_2024-07-07"

paths:
  cropped_data_dir: "/abs/path/to/batch/crops"   # your current source
  cvat_prep_dir: "/abs/path/for/zips"            # where zip is written
  yolo_label_source_dir: null                    # optional
  camvid_label_source_dir: null                  # optional

cvat:
  type: "images_only"            # images_only | yolo_det_1_0 | camvid
  class_names: ["weed","crop"]   # used for yolo_det_1_0 and camvid
  splits: {train: 0.8, val: 0.2, test: 0.0}
  seed: 1337

zip:
  compression: "deflate"         # deflate | store
  overwrite: false

runtime:
  log_level: "INFO"
"""

from __future__ import annotations

import logging
import os
import random
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ------------------------------- Utils ----------------------------------------


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)

def make_unique_names(paths: List[Path], keep_rel: bool, root: Path) -> Dict[Path, Path]:
    assigned: Dict[str, int] = {}
    mapping: Dict[Path, Path] = {}
    for src in paths:
        if keep_rel:
            rel = relative_to_root(src, root)
            key = str(rel).replace(os.sep, "/")
        else:
            rel = Path(src.name)
            key = rel.name
        if key not in assigned:
            assigned[key] = 0
            mapping[src] = rel
            continue
        assigned[key] += 1
        stem, suffix = rel.stem, rel.suffix
        parent = rel.parent if keep_rel else Path(".")
        mapping[src] = parent / f"{stem}__dup{assigned[key]}{suffix}"
    return mapping

def choose_compression(name: str) -> int:
    return zipfile.ZIP_STORED if str(name).lower() == "store" else zipfile.ZIP_DEFLATED

def split_paths(paths: List[Path], splits: Dict[str, float], seed: int) -> Dict[str, List[Path]]:
    total = round((splits.get("train", 0) + splits.get("val", 0) + splits.get("test", 0)), 6)
    assert abs(total - 1.0) < 1e-6, f"splits must sum to 1.0, got {total}"
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * splits["train"]))
    return {
        "train": shuffled[:n_train]
    }

def write_text(fp: Path, text: str) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(text, encoding="utf-8")

# ------------------------------- Core Class -----------------------------------

class CvatImagesPackager:
    """
    Collect images from `paths.cropped_data_dir` and build a CVAT-ready ZIP archive
    in one of the supported dataset layouts.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.crop_dir = Path(cfg.paths.cropped_data_dir)
        self.outputs_dir = Path(cfg.paths.cvat_prep_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = self.cfg.cvat.class_names

    # -------- collect --------
    def collect(self) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        paths = [p for p in sorted(self.crop_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
        if not paths:
            raise FileNotFoundError(f"No images found under {self.crop_dir}")
        log.info("Found %d images in %s", len(paths), self.crop_dir)
        return paths

    # -------- builders for each dataset.type --------
    def _build_images_only_tree(self, build_dir: Path, images_by_split: Dict[str, List[Path]]) -> None:
        for split, files in images_by_split.items():
            out_dir = build_dir / "images" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                (out_dir / src.name).write_bytes(src.read_bytes())

    def _build_yolo_det_1_0_tree(self, build_dir: Path, images_by_split: Dict[str, List[Path]]) -> None:
        names = list(self.class_names or [])
        # dataset.yaml
        ds_yaml = [
            "path: .",
            "train: images/train",
            f"nc: {len(names)}",
            "names: [" + ", ".join(names) + "]" if names else "names: []",
            "",
        ]
        write_text(build_dir / "dataset.yaml", "\n".join(ds_yaml))
        
        for split, files in images_by_split.items():
            img_out = build_dir / "images" / split
            lbl_out = build_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for src in files:
                (img_out / src.name).write_bytes(src.read_bytes())
        log.info("No YOLO label dir provided; created empty labels/ tree (valid for images-first import).")
        
    def _build_camvid_tree(self, build_dir: Path, images_by_split: Dict[str, List[Path]]) -> None:
        """
        CamVid layout + <split>.txt lists.
        - classes.txt at repo root (one class per line)
        - images/<split>/*.*
        - labels/<split>/ (left empty unless masks are later added)
        - <split>.txt: "images/<split>/<file> labels/<split>/<file_stem>.png" per line
        """
        names = list(self.class_names or [])
        write_text(build_dir / "label_colors.txt", "\n".join(names) + ("\n" if names else ""))

        # You said the annot folder will be blank → no mask_root, but we still write <split>.txt
        for split, files in images_by_split.items():
            img_out = build_dir / "images" / split
            img_out.mkdir(parents=True, exist_ok=True)

            # build the <split>.txt content
            lines = []
            for src in files:
                # copy image
                (img_out / src.name).write_bytes(src.read_bytes())

                # expected mask path (not created now)
                img_rel = f"images/{split}/{src.name}"
                lines.append(f"{img_rel}")

            # write <split>.txt at the zip root (e.g., train.txt)
            list_path = build_dir / f"{split}.txt"
            write_text(list_path, "\n".join(lines) + ("\n" if lines else ""))

        log.info("CamVid structure built with empty labels/ and <split>.txt listing image→mask pairs.")

    # -------- zip --------
    def build_zip(self, batch_id: str, build_dir: Path) -> Path:
        out_name = f"{batch_id}_{self.cfg.cvat.type}_{timestamp()}.zip"
        out_path = self.outputs_dir / out_name
        if out_path.exists():
            raise FileExistsError(f"Archive already exists: {out_path} (set zip.overwrite=true)")

        with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for p in build_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(build_dir).as_posix())
        return out_path

    # -------- pipeline --------
    def run(self) -> Path:
        images = self.collect()
        # splits
        splits = dict(self.cfg.cvat.splits)
        images_by_split = split_paths(images, splits=splits, seed=int(self.cfg.cvat.seed))

        # build temp tree under outputs_dir/__build_*
        build_dir = self.outputs_dir / f"__build_{self.batch_id}_{timestamp()}"
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            dtype = str(self.cfg.cvat.type).lower()
            if dtype == "images_only":
                self._build_images_only_tree(build_dir, images_by_split)
            elif dtype == "yolo_det_1_0":
                self._build_yolo_det_1_0_tree(build_dir, images_by_split)
            elif dtype == "camvid":
                self._build_camvid_tree(build_dir, images_by_split)
            else:
                raise ValueError(f"Unknown dataset.type: {self.cfg.cvat.type}")

            out_zip = self.build_zip(self.batch_id, build_dir)
            log.info("Done. Ready for CVAT import: %s", out_zip)
            return out_zip
        finally:
            # Clean build dir
            for p in sorted(build_dir.rglob("*"), reverse=True):
                try:
                    p.unlink() if p.is_file() else p.rmdir()
                except Exception:
                    pass
            try:
                build_dir.rmdir()
            except Exception:
                pass

# ------------------------------- Hydra Entry ----------------------------------

def main(cfg: DictConfig) -> None:
    log.debug("Config:\n%s", OmegaConf.to_yaml(cfg))
    packager = CvatImagesPackager(cfg)
    try:
        packager.run()
    except Exception as e:
        log.exception("Failed to build CVAT ZIP: %s", e)
        raise
