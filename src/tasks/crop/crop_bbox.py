import cv2
from pathlib import Path
from pprint import pprint
import omegaconf
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.utils import (
    find_lts_dir,
    get_jsons,
    read_json,
    get_developed_images,
)

log = logging.getLogger(__name__)


def process_json_file(
    json_file: Path,
    developed_image_lookup: dict[str, Path],
    selected_cam_angles: set[int],
    angle_save_dirs: dict[int, Path],
) -> dict:
    """
    Process one JSON metadata file:
    - validate camera angle
    - find matching developed image
    - crop valid bounding boxes
    - save crops
    - return processing counts
    """

    result = {
        "json_file": str(json_file),
        "saved_crops": [],
        "skipped_wrong_angle": 0,
        "skipped_small": 0,
        "skipped_missing_image": 0,
        "skipped_missing_image_id": 0,
        "skipped_mismatch": 0,
        "skipped_imread": 0,
        "skipped_empty_crop": 0,
        "skipped_left_side_20deg": 0,
        "skipped_category_28": 0,
        "skipped_missing_bbox": 0,
    }

    data = read_json(json_file)

    cam_angle = data.get("camera_info", {}).get("cam_angle", 0)
    cam_angle = int(cam_angle)

    if cam_angle not in selected_cam_angles:
        result["skipped_wrong_angle"] += 1
        return result

    save_dir = angle_save_dirs[cam_angle]

    image_id = data.get("image_id", None)
    if image_id is None:
        result["skipped_missing_image_id"] += 1
        return result

    developed_image = developed_image_lookup.get(image_id)

    if developed_image is None or not developed_image.exists():
        result["skipped_missing_image"] += 1
        return result

    if json_file.stem != image_id:
        result["skipped_mismatch"] += 1
        return result

    img = cv2.imread(str(developed_image))
    if img is None:
        result["skipped_imread"] += 1
        return result

    image_height, image_width = img.shape[:2]

    for annotation in data.get("annotations", []):
        cutout_id = annotation.get("cutout_id", None)
        bbox_xywh = annotation.get("bbox_xywh", None)
        category_class_id = annotation.get("category_class_id", None)

        if bbox_xywh is None:
            result["skipped_missing_bbox"] += 1
            continue

        if category_class_id == 28:
            result["skipped_category_28"] += 1
            continue

        x, y, w, h = bbox_xywh

        # Keep this unchanged so crop geometry stays identical.
        if h < 500 or w < 500:
            result["skipped_small"] += 1
            continue

        # For 20-degree images, only keep bboxes whose center is right of image center.
        if cam_angle == 20:
            bbox_center_x = x + w // 2
            image_center_x = image_width // 2

            if bbox_center_x <= image_center_x:
                result["skipped_left_side_20deg"] += 1
                continue

        cropped_img = img[y:y + h, x:x + w]

        if cropped_img.size == 0:
            result["skipped_empty_crop"] += 1
            continue

        cropped_name = f"{cutout_id}.jpg"
        output_path = save_dir / cropped_name

        cv2.imwrite(
            str(output_path),
            cropped_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

        result["saved_crops"].append(cropped_name)

    return result


def main(cfg) -> None:
    """Crop bounding boxes from developed images, optionally filtering by camera angle."""

    log.info("Cropping bounding boxes from input images...")
    pprint(omegaconf.OmegaConf.to_container(cfg, resolve=True))

    batch_id = cfg.batch_id
    nfs_locations = cfg.paths.lts_locations

    # Config option:
    # crop.cam_angles: [0, 20], [0], or [20]
    selected_cam_angles = cfg.get("crop", {}).get("cam_angles", [0, 20])
    selected_cam_angles = {int(angle) for angle in selected_cam_angles}

    valid_angles = {0, 20}
    invalid_angles = selected_cam_angles - valid_angles
    if invalid_angles:
        raise ValueError(
            f"Invalid crop.cam_angles values: {invalid_angles}. "
            f"Valid options are 0 and 20."
        )

    # Optional config option:
    # crop.num_workers: 8
    num_workers = int(cfg.get("crop", {}).get("num_workers", 8))

    lts_location = find_lts_dir(
        batch_id=batch_id,
        nfs_locations=nfs_locations,
        local=False,
        developed=True,
        jpgs=True,
    )

    data_type = "semifield-developed-images"
    lts_batch_dir = Path(lts_location) / data_type / batch_id

    local_cropped_data = Path(cfg.paths.cropped_data_dir)

    angle_save_dirs = {}
    for angle in selected_cam_angles:
        save_dir = local_cropped_data / f"{angle}deg"
        save_dir.mkdir(parents=True, exist_ok=True)
        angle_save_dirs[angle] = save_dir

    json_files = get_jsons(lts_batch_dir)
    developed_images = get_developed_images(lts_batch_dir)

    developed_image_lookup = {img.stem: img for img in developed_images}

    saved_crops = []
    skipped_wrong_angle = 0
    skipped_small = 0
    skipped_missing_image = 0
    skipped_missing_image_id = 0
    skipped_mismatch = 0
    skipped_imread = 0
    skipped_empty_crop = 0
    skipped_left_side_20deg = 0
    skipped_category_28 = 0
    skipped_missing_bbox = 0

    log.info(f"Processing {len(json_files)} JSON files with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_json_file,
                json_file,
                developed_image_lookup,
                selected_cam_angles,
                angle_save_dirs,
            )
            for json_file in json_files
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                log.exception("Worker failed while processing a JSON file")
                continue

            saved_crops.extend(result["saved_crops"])

            skipped_wrong_angle += result["skipped_wrong_angle"]
            skipped_small += result["skipped_small"]
            skipped_missing_image += result["skipped_missing_image"]
            skipped_missing_image_id += result["skipped_missing_image_id"]
            skipped_mismatch += result["skipped_mismatch"]
            skipped_imread += result["skipped_imread"]
            skipped_empty_crop += result["skipped_empty_crop"]
            skipped_left_side_20deg += result["skipped_left_side_20deg"]
            skipped_category_28 += result["skipped_category_28"]
            skipped_missing_bbox += result["skipped_missing_bbox"]

    log.info(f"Developed images found: {len(developed_images)}")
    log.info(f"JSON files processed: {len(json_files)}")
    log.info(f"Skipped JSONs due to cam_angle filter: {skipped_wrong_angle}")
    log.info(f"Skipped JSONs due to missing image_id: {skipped_missing_image_id}")
    log.info(f"Skipped missing developed images: {skipped_missing_image}")
    log.info(f"Skipped JSON/image_id mismatches: {skipped_mismatch}")
    log.info(f"Skipped images where cv2.imread returned None: {skipped_imread}")
    log.info(f"Skipped small crops: {skipped_small}")
    log.info(f"Skipped 20deg crops left of image center: {skipped_left_side_20deg}")
    log.info(f"Skipped category_class_id == 28: {skipped_category_28}")
    log.info(f"Skipped annotations missing bbox_xywh: {skipped_missing_bbox}")
    log.info(f"Skipped empty crops: {skipped_empty_crop}")
    log.info(f"Total cropped images saved: {len(saved_crops)}")