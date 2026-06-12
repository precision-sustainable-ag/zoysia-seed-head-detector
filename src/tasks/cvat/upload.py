from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List

from omegaconf import DictConfig
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import Task as CvatTask
from cvat_sdk.core.proxies.tasks import ResourceType
from src.tasks.cvat.cvat_helpers import read_yaml_from_dictconfig, YOLO2COCOConverter

log = logging.getLogger(__name__)

class CVATTaskFromCOCO:
    """
    Create (or reuse) a CVAT Task **within a specific Project**, upload images,
    and import COCO annotations.
    """

    def __init__(self, cfg: DictConfig):
        keys = read_yaml_from_dictconfig(cfg)["CVAT"]
        self.url: str = keys.get("url")
        self.email: str = keys.get("email")
        self.password: str = keys.get("password")
        self.org_slug: Optional[str] = keys.get("org_slug", None)

    def create_task_and_import_coco(
        self,
        project_id: int,
        task_name: str,
        images_dir: Path,
        coco_json: Path,
        image_quality: int = 75,
        subset: Optional[str] = None,
    ) -> CvatTask:
        """
        Create/reuse a task inside a given CVAT project, upload images, and import COCO annotations.
        """
        images_dir = Path(images_dir)
        coco_json = Path(coco_json)
        assert images_dir.is_dir(), f"Images dir not found: {images_dir}"
        assert coco_json.is_file(), f"COCO json not found: {coco_json}"

        categories = self._read_coco_categories(coco_json)
        img_paths = self._scan_images(images_dir)
        if not img_paths:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        with self._connect() as client:
            # Validate project exists (and capture its name for logs)
            proj = client.projects.retrieve(project_id)
            log.info(f"Using project {proj.id} ({proj.name})")

            task = self._create_from_data(
                client,
                task_name=task_name,
                project_id=project_id,
                imgs=img_paths,
                annotation_path=str(coco_json),
                annotation_format="COCO 1.0",
                image_quality=image_quality,
            )

            # # Make task in project
            # task = self._get_or_create_task(client, project_id, task_name)
            # # Import images to task
            # self._upload_images(client, task, img_paths, image_quality=image_quality, subset=subset)

            # # Import COCO annotations
            # task.import_annotations(format_name="COCO 1.0", filename=str(coco_json))
            # log.info(f"Imported COCO annotations into task {task.id} ({task_name}) in project {project_id}")

            return task

    # ---------- internals ---------- #

    def _connect(self):
        client = make_client(self.url)
        client.login((self.email, self.password))
        if self.org_slug:
            client.organization_slug = self.org_slug
        return client

    @staticmethod
    def _scan_images(images_dir: Path) -> List[Path]:
        paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg"]])
        log.info(f"Found {len(paths)} images under {images_dir}")
        return paths

    @staticmethod
    def _read_coco_categories(coco_json: Path) -> List[Dict[str, Any]]:
        data = json.loads(coco_json.read_text(encoding="utf-8"))
        cats = data.get("categories") or []
        if not cats:
            raise ValueError("COCO file has no 'categories'.")
        labels = [{"name": str(c["name"])} for c in cats]
        # De-dupe by name, preserving order
        seen, uniq = set(), []
        for lab in labels:
            if lab["name"] not in seen:
                uniq.append(lab)
                seen.add(lab["name"])
        log.info(f"COCO categories -> {len(uniq)} labels: {[l['name'] for l in uniq]}")
        return uniq

    def _get_or_create_task(
        self,
        client,
        project_id: int,
        task_name: str,
    ) -> CvatTask:
        """
        Look for a task with this name inside the given project; create if missing.
        """
        # Fetch tasks and filter by project_id + name
        existing = [t for t in client.tasks.list() if getattr(t, "project_id", None) == project_id and t.name == task_name]
        if existing:
            t = existing[0]
            log.info(f"Reusing existing task: {t.id} ({t.name}) in project {project_id}")
            return t

        log.info(f"Creating new task '{task_name}' in project {project_id}")
        # Some SDK versions accept kwargs, others want spec={...}
        task = client.tasks.create(spec={"name": task_name, "project_id": project_id})
        log.info(f"Created task: {task.id} ({task.name}) in project {project_id}")
        return task

    @staticmethod
    def _upload_images(client, task: CvatTask, image_paths: Sequence[Path], image_quality: int, subset: Optional[str]):
        log.info(f"Uploading {len(image_paths)} images to task {task.id} (subset={subset})")
        if hasattr(task, "upload_data"):
            task.upload_data(
                resources=[str(p) for p in image_paths],
                resource_type=ResourceType.LOCAL,
                image_quality=int(image_quality),
                # data_params={"image_quality": int(image_quality)}
            )
            return
        raise RuntimeError("CVAT SDK does not expose an upload_data method on Task or client.tasks")
    

    @staticmethod
    def _create_from_data(
        client,
        task_name: str,
        project_id: int,
        imgs: List[Path],
        annotation_path: str,
        annotation_format: str = "COCO 1.0",
        image_quality: int = 100,
    ) -> CvatTask:
        """
        """
        # print(imgs)
        imgs = [str(p) for p in imgs]
        print(annotation_format)
        task = client.tasks.create_from_data(
            spec={"name": task_name, "project_id": project_id},
            resource_type=ResourceType.LOCAL,
            resources=imgs,
            annotation_path = annotation_path,
            annotation_format = annotation_format,
            data_params={"image_quality": int(image_quality)},   # <- preview quality
        )
        log.info(f"Created Task: {task.id} ({task.name}) in project {project_id}")
        return task
        

# ---------------- Example main (Hydra/DictConfig) ---------------- #

def main(cfg: DictConfig):
    """
    Expected cfg:

    CVAT:
      url: "https://cvat.example.org"
      email: "user@example.org"
      password: "secret"
      org_slug: "my-org"     # optional

    cvat:
      detect:
        importer:
          project_id: 123
          task_name: "Zoysia_Seediness_v1"
          images_dir: "/data/zoysia/images"
          coco_json: "/data/zoysia/annotations/instances_train.json"
          subset: "train"       # optional
          image_quality: 75     # optional
    """
    imp = cfg.cvat.detect.importer
    yolo2coco = YOLO2COCOConverter(cfg)
    coco_json = yolo2coco.convert(out_json=Path(yolo2coco.root_dir)  / "annotation.json")
    image_dir = Path(cfg.cvat.detect.importer.root_dir) / "images"
    runner = CVATTaskFromCOCO(cfg)
    task = runner.create_task_and_import_coco(
        project_id=int(imp.project_id),
        task_name=str(imp.task_name),
        images_dir=image_dir,
        coco_json=Path(coco_json),
        image_quality=int(getattr(imp, "image_quality", 100)),
        subset=getattr(imp, "subset", None),
    )
    log.info(f"Done. Task id: {task.id}  name: {task.name}  (project {int(imp.project_id)})")
