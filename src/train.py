import logging
import hydra
from omegaconf import DictConfig

from src.tasks.train.train_detector import main as train_detector
log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "train_detector": train_detector,
    # Add more tasks here as needed
}
TASK_NAME = "train"

# @hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting {TASK_NAME} tasks...")

    task_dict = cfg.tasks.get(TASK_NAME, {})

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
            log.error(f"Task {task} not found in {TASK_NAME} task registry")
            raise ValueError(f"Task {task} not found in {TASK_NAME} task registry")

    log.info(f"{TASK_NAME} complete.")

    return
