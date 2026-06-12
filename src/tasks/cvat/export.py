from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
from pathlib import Path
from typing import Any, Dict,List, Optional
import logging

from omegaconf import DictConfig
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import Task as CvatTask
from cvat_sdk.api_client.models import DataMetaRead
from src.tasks.cvat.cvat_helpers import read_yaml_from_dictconfig, get_formatter, ensure_dir, norm_name, AnnotationFormatter

log = logging.getLogger(__name__)


# =========================== CVAT Exporter =========================

class CVATExporter:
    """
    Pulls frames, labels, and shapes from a CVAT task, then delegates to a formatter.

    You can extend by adding new formatter classes and mapping them in `get_formatter(...)`.
    """
    def __init__(self, cfg: DictConfig):
        keys = read_yaml_from_dictconfig(cfg)["CVAT"]
        self.cfg = cfg
        self.client_base_url = keys.get("url")
        self.email = keys.get("email")
        self.password = keys.get("password")
        self.org_slug = keys.get("org_slug")
        self.download_images = bool(getattr(cfg.cvat.detect.export, "save_images", False))
        self.fmt_name = str(cfg.cvat.detect.export.label_format)
        self.formatter: AnnotationFormatter = get_formatter(self.fmt_name)  # inject a concrete formatter

        

    # ----------------- Public API ----------------- #
    def export_task(self, task_id: int) -> Path:
        with self._connect() as client:
            log.info(f"Connected to CVAT at {self.client_base_url} as {self.email}")
            task = self._retrieve_task(client, task_id)
            log.info(f"Retrieved task: {task.name} (ID: {task.id})")
            self.task_name = task.name
            self._config_save_location()
            frames = self._get_frames_meta(client, task_id)
            log.info(f"Retrieved metadata for {len(frames)} frames")
            labels = self._get_labels(client, task_id)
            log.info(f"Retrieved {len(labels)} labels")

            self._init_formatter(task, frames, labels)

            image_id_for_frame = self._process_frames(task, frames, self.out_dir_images)
            self._process_shapes(task, image_id_for_frame)

            return self._write_artifacts(self.out_dir)
    
    def _config_save_location(self):
        
        deg_dir = None
        if "deg" in self.task_name:
            if "_20deg" in self.task_name:
                deg_dir = "20deg"
                batch_name = self.task_name.split("_20deg")[0]

            elif "_0deg" in self.task_name:
                deg_dir = "0deg"
                batch_name = self.task_name.split("_0deg")[0]
            else:
                raise ValueError(f"Degree information not found in task name: {self.task_name}")
            self.out_dir =  Path(self.cfg.paths.batches_dir) / batch_name / deg_dir / "detections" / self.fmt_name
        else:   
            self.out_dir =  Path(self.cfg.paths.workdir) / self.cfg.cvat.detect.export.out_dir / self.task_name / self.fmt_name
        ensure_dir(self.out_dir)
        self.out_dir_images = self.out_dir / "images"
        if self.download_images:
            ensure_dir(self.out_dir_images)
    
    # ----------------- Internals ------------------ #
    def _connect(self):
        client = make_client(self.client_base_url)
        client.login((self.email, self.password))
        if self.org_slug:
            client.organization_slug = self.org_slug
        return client

    @staticmethod
    def _retrieve_task(client, task_id: int) -> CvatTask:
        return client.tasks.retrieve(task_id)

    @staticmethod
    def _get_frames_meta(client, task_id: int) -> List[Dict[str, Any]]:
        meta: DataMetaRead = client.api_client.tasks_api.retrieve_data_meta(task_id)
        if isinstance(meta, (list, tuple)):
            meta = meta[0]

        frames = getattr(meta, "frames", None) or []
        deleted = set(getattr(meta, "deleted_frames", None) or [])
        start = int(getattr(meta, "start_frame", 0))
        stop  = int(getattr(meta, "stop_frame", -1))
        # Real frame indices that currently exist in the task (after GUI deletions)
        valid_indices = [i for i in range(start, stop + 1) if i not in deleted]

        from pprint import pprint
        # pprint(valid_indices, sort_dicts=False)
        # exit()
        # pprint(meta, sort_dicts=False)
        # `frames` are ordered to match remaining media; pair them with their real indices
        if len(frames) != len(valid_indices):
            # Not fatal, but helps diagnose odd states (joins/slices, etc.)
            log.warning("Frame meta count (%d) != valid index count (%d)", len(frames), len(valid_indices))
        
        merged = []
        for idx, f in enumerate(frames):
            if idx in valid_indices:
                fd = f.to_dict() if hasattr(f, "to_dict") else dict(f)
                fd["frame"] = idx  # <- keep the real frame index
                merged.append(fd)
        # print()
        
        return merged

    @staticmethod
    def _get_frames_meta(client, task_id: int) -> List[Dict[str, Any]]:
        meta: DataMetaRead = client.api_client.tasks_api.retrieve_data_meta(task_id)
        if isinstance(meta, (list, tuple)):
            meta = meta[0]

        frames = getattr(meta, "frames", None) or []
        deleted = set(getattr(meta, "deleted_frames", None) or [])

        start = int(getattr(meta, "start_frame", 0))
        stop = int(getattr(meta, "stop_frame", -1))

        valid_indices = [i for i in range(start, stop + 1) if i not in deleted]

        if len(frames) != len(valid_indices):
            log.warning(
                "Frame meta count (%d) != valid index count (%d)",
                len(frames),
                len(valid_indices),
            )

        merged = []

        # Pair each metadata row with its real CVAT frame index
        for real_frame_idx, f in zip(valid_indices, frames):
            fd = f.to_dict() if hasattr(f, "to_dict") else dict(f)
            fd["frame"] = real_frame_idx
            merged.append(fd)

        return merged   

    @staticmethod
    def _get_labels(client, task_id: int) -> List[Dict[str, Any]]:
        page, _ = client.api_client.labels_api.list(task_id=task_id)
        return page.to_dict().get("results", [])

    def _init_formatter(
        self,
        task: CvatTask,
        frames: List[Dict[str, Any]],
        labels: List[Dict[str, Any]],
    ) -> None:
        self.formatter.begin(task=task, frames_meta=frames, labels=labels)

    def _process_frames(
        self,
        task: CvatTask,
        frames: List[Dict[str, Any]],
        images_dir: Path,
    ) -> Dict[int, int]:
        """
        Add images to formatter and (optionally) download them.
        Returns a map: frame_idx -> image_id
        """
        image_id_for_frame: Dict[int, int] = {}
        
        for _, f in enumerate(frames):
            frame_idx = int(f.get("frame"))
            file_name = f.get("name") or f"frame_{frame_idx:06d}.jpg"
            width = int(f.get("width") or 0)
            height = int(f.get("height") or 0)

            image_id_for_frame[frame_idx] = frame_idx
            self.formatter.add_image(
                image_id=frame_idx, file_name=file_name, width=width, height=height
            )

            if self.download_images:
                self._download_image(task, frame_idx, images_dir / file_name)
        return image_id_for_frame

    def _process_shapes(
        self,
        task: CvatTask,
        image_id_for_frame: Dict[int, int],
    ) -> None:
        """
        Convert CVAT shapes (rectangles) into formatter annotations.
        """
        labeled = task.get_annotations()  # dict-like with "shapes"
        shapes = labeled.get("shapes") or []
        for shp in shapes:
            if norm_name(shp.get("type")) != "rectangle":
                continue
            points = shp.get("points") or []
            if len(points) != 4:
                continue

            frame_idx = int(shp.get("frame"))
            image_id = image_id_for_frame.get(frame_idx)
            if image_id is None:
                continue

            label_id = shp.get("label_id")
            attributes = shp.get("attributes") or []

            self.formatter.add_rectangle(
                image_id=image_id,
                label_id=label_id,
                points=points,
                attributes=attributes,
            )

    def _write_artifacts(self, out_dir: Path) -> Path:
        return self.formatter.write(out_dir)
        
    @staticmethod
    def _download_image(task: CvatTask, frame_idx: int, dst_path: Path) -> None:
        
        # CVAT SDK will stream the frame directly to the file
        frame_bytes = task.get_frame(frame_idx, quality='original')
        im = Image.open(frame_bytes)
        im.save(dst_path)

# ========================= Formatter Selection =====================



# ============================== Hydra ==============================

def main(cfg: DictConfig):
    """
    Expected cfg keys:
      cfg.cvat.task_id
      cfg.cvat.detect.train_project_name
      cfg.cvat.detection.format     # e.g., "Coco 1.0"
      cfg.download_images (optional)

    CVAT creds under:
      read_yaml_from_dictconfig(cfg)["CVAT"] -> url, email, password, org_slug
    """    
    task_id = int(cfg.cvat.detect.export.task_id)
    log.info(f"Starting CVAT export for task {task_id} with format {cfg.cvat.detect.export.label_format}")
    exporter = CVATExporter(cfg)
    out_path = exporter.export_task(task_id)
    log.info(f"Export complete: {out_path}")