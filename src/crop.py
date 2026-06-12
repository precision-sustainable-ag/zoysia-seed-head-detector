import logging
import hydra
from omegaconf import DictConfig

from src.tasks.crop.crop_bbox import main as crop_bbox

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "crop_bbox": crop_bbox,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting image cropping tasks...")

    task_dict = cfg.tasks.crop

    for task, enabled in task_dict.items():

        if task in TASK_REGISTRY:
            if enabled:
                log.info(f"Running task {task}")
                try:
                    TASK_REGISTRY[task](cfg)
                except Exception as e:
                    log.error(f"Error running task {task}: {e}")
                    raise
            else:
                log.info(f"Skipping task {task} as it is disabled in the config")

        else:
            log.error(f"Task {task} not found in cropping task registry")
            raise ValueError(f"Task {task} not found in cropping task registry")

    log.info("Image cropping complete.")

    return

if __name__ == "__main__":
    main()