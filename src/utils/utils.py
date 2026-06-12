from pathlib import Path
import logging
import math
import shutil
from typing import Dict, Any, List
import subprocess
import json
import os
import time

log = logging.getLogger(__name__)

def replace_all_nan_with_null(obj: Any) -> Any:
    """
    Recursively replace all NaN values with None (JSON null).
    """
    if isinstance(obj, dict):
        return {k: replace_all_nan_with_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_all_nan_with_null(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

def save_json_file(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary to a JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        
def safe_save_json(data: Any, path: Path) -> None:
    cleaned = replace_all_nan_with_null(data)
    save_json_file(path, cleaned)


def get_jsons(batch: Path) -> List[Path]:
    """
    Get all JSON files in the given batch directory.
    """
    json_dir = batch / "metadata"
    return sorted(json_dir.glob("*.json"))

def read_json(file_path: Path) -> Dict[str, Any]:
    """
    Read a JSON file and return its content.
    """
    with open(file_path, "r") as file:
        data = json.load(file)#, parse_constant=strict_parse_constant)

    return data

def get_developed_images(batch: Path):
    """
    Get all developed images in the given batch directory.
    """
    image_dir = batch / "images"
    return sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.JPG"))

# Find the batch NFS location from a list of possible parent directories
def find_lts_dir(batch_id: str, nfs_locations: list[str], local: bool = False,
                 developed: bool = False, dngs: bool = False, jpgs: bool = False) -> Path | None:
    """
    Searches for the specified batch directory within the given NFS locations and checks for the presence and completeness of RAW files.
    Args:
        batch_id (str): The identifier of the batch to search for.
        nfs_locations (list): A list of NFS locations (directories) to search within.
        local (bool): true - searches for batch data in local directory.
        developed (bool): true - searches for jpg in semifield-developed-images, false - searches for raws in semifield-upload.
    Returns:
        Path: The NFS location where the batch was found with complete RAW
        files, or None if the batch is not found or the files are incomplete.
    Logs:
        - Info: Logs the NFS location and the number of RAW files found if the batch is found and the files are complete.
        - Error: Logs an error message if the batch is not found, if no RAW files are found, or if the RAW files are not completely uploaded.
    """
    dir_found, files_found, upload_complete = False, False, False
    batch_location = None
    for nfs_location in nfs_locations:
        nfs_location = Path(nfs_location)
        if local:
            if developed:
                batch_location = Path(
                    "data") / nfs_location.name / "semifield-developed-images" / batch_id
            else:
                batch_location = Path(
                    "data") / nfs_location.name / "semifield-upload" / batch_id
        else:
            if developed:
                batch_location = nfs_location / "semifield-developed-images" / batch_id
            else:
                batch_location = nfs_location / "semifield-upload" / batch_id
        # Check if the batch directory exists
        if batch_location.exists():
            dir_found = True
            if developed:
                if dngs:
                    dng_location = batch_location / "dngs"
                    if dng_location.exists():
                        return nfs_location
                elif jpgs:
                    files = list(Path(batch_location, "images").glob("*.jpg")) + list(
                        Path(batch_location, "images").glob("*.JPG"))
                else:
                    files = list(Path(batch_location, "pngs").glob("*.png")) + list(
                        Path(batch_location, "pngs").glob("*.PNG"))
            else:
                files = list(batch_location.glob("*.RAW")) + list(
                    batch_location.glob("*.raw"))
            # Check if any RAW files are present
            if files:
                # todo: md5 checksum for data verification?
                log.info(
                    f"Batch {batch_id} found in {batch_location} with {len(files)} {'RAW' if not developed else ('JPG' if jpgs else 'PNG')} files")
                return nfs_location
    if not dir_found:
        log.error(
            f"Batch {batch_id} not found in NFS locations: {nfs_locations}")
    elif not files_found:
        log.error(
            f"Batch {batch_id} found in {batch_location} but no RAW files found")
    elif not upload_complete:
        log.error(
            f"Batch {batch_id} found in {batch_location} but RAW files are not completely uploaded")
    return None


def save_yaml_to_lts(cfg, lts_dev_dir):
    try:
        yaml_path = Path(cfg.paths.artifact_path)
        batch_id = cfg.batch_id
        yaml_dst_dir = Path(lts_dev_dir) / batch_id / "inspection"
        yaml_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(yaml_path, yaml_dst_dir)
    except Exception as e:
        log.error(f"Failed to save YAML file: {e}")

def create_issue(cfg, issue_type, tsk: str = None, error_msg: str = None):
    batch_id = cfg.batch_id
    user_id = cfg.gh_reviewer
    lts_dev_dir = cfg.paths.lts_developed_directory
    if not lts_dev_dir:
        lts_dev_dir = Path(find_lts_dir(batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)) / "semifield-developed-images"
    lts_dev_dir_name = Path(lts_dev_dir).parent.name

    save_yaml_to_lts(cfg, lts_dev_dir)

    if issue_type == "report":
        trigger_payload = {
                "event_type": "report-generated",
                "client_payload": {
                    "batch_id": batch_id,
                    "assignee": user_id,  # from cfg.report.reviewers
                    "lts_developed": lts_dev_dir_name
                }
            }
        
    elif issue_type == "failure":
        trigger_payload = {
                "event_type": "failure-reported",
                "client_payload": {
                    "batch_id": batch_id,
                    "assignee": user_id,
                    "task_name": tsk,
                    "error_msg": error_msg,
                    "lts_developed": lts_dev_dir_name
                }
            }
    subprocess.run([
                "curl", "-X", "POST", "https://api.github.com/repos/precision-sustainable-ag/SemiF-Preprocessing/dispatches",
                "-H", f"Authorization: token {os.environ['GITHUB_PAT']}",
                "-H", "Accept: application/vnd.github.v3+json",
                "-d", json.dumps(trigger_payload)
            ], check=True)


def retry_nfs_access(path: Path, 
                     mode: str = "read", 
                     retries: int = 5, 
                     delay: float = 2.0,
                     backoff: float = 1.5) -> bool:
    """
    Retry access to a Path (NFS) multiple times if PermissionError or OSError occurs.

    Args:
        path (Path): Path object pointing to a file or directory.
        mode (str): "read" (check existence/readability) or "write" (try writing a temp file).
        retries (int): Max number of retries.
        delay (float): Initial delay between retries in seconds.
        backoff (float): Backoff multiplier to increase delay.

    Returns:
        bool: True if access eventually succeeds, False otherwise.
    """
    assert mode in ["read", "write"], "mode must be 'read' or 'write'"

    for attempt in range(retries):
        try:
            if mode == "read":
                if not path.exists():
                    raise FileNotFoundError(f"{path} does not exist")
                if path.is_dir():
                    _ = list(path.iterdir())  # trigger PermissionError if any
                else:
                    _ = path.read_bytes()[:1]  # just try to read a byte

            elif mode == "write":
                test_file = path / ".nfs_test"
                test_file.write_text("test")
                test_file.unlink()

            log.info(f"NFS access succeeded on attempt {attempt+1}: {path}")
            return True

        except (PermissionError, OSError) as e:
            log.warning(f"Attempt {attempt+1} failed to access {path}: {e}")
            time.sleep(delay)
            delay *= backoff

    log.error(f"NFS access failed after {retries} attempts: {path}")
    return False
