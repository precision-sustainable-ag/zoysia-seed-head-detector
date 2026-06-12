import logging
import hydra
from omegaconf import DictConfig

from tasks.detect.train_detector import main as train_detector

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "train_detector": train_detector,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting training tasks...")

    task_dict = cfg.tasks.train

    for task, enabled in task_dict.items():

        if task in TASK_REGISTRY and enabled:
            log.info(f"Running task {task}")
            try:
                TASK_REGISTRY[task](cfg)
            except Exception as e:
                log.error(f"Error running task {task}: {e}")
                raise

        else:
            log.error(f"Task {task} not found in training task registry")
            raise ValueError(f"Task {task} not found in training task registry")

    log.info("Training complete.")

    return

if __name__ == "__main__":
    main()