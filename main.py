import logging
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf  # Do not confuse with dataclass.MISSING
# Import the task functions
from src.crop import main as crop
from src.cvat import main as cvat
from src.segment import main as segment
from src.train import main as train
from src.evaluate import main as evaluate


log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "crop": crop,
    "cvat": cvat,
    "segment": segment,
    "train": train,
    "evaluate": evaluate,

    # Add more tasks here as needed
}


log = logging.getLogger(__name__)
# Get the logger for the Azure SDK

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.tasks)}")
    
    mode = cfg.mode

    assert mode in TASK_REGISTRY, f"Unknown mode: {mode}"

    try:
        TASK_REGISTRY[mode](cfg)

    except Exception as e:
        log.exception("Failed")
        return


if __name__ == "__main__":
    main()