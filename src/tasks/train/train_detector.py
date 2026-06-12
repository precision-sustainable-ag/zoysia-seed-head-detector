from ultralytics import YOLO


import cv2
from pathlib import Path
from pprint import pprint
import omegaconf
from src.utils.utils import find_lts_dir, get_jsons, read_json, get_developed_images

def main(cfg) -> None:
    # Load a model
    
    model = YOLO(cfg.train.model)  # load a pretrained model (recommended for training)


    # Train the model with the two most idle GPUs
    results = model.train(
        project=cfg.train.project,
        name=cfg.train.name,
        exist_ok=cfg.train.exist_ok,
        data=cfg.train.train_yaml,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        rect=cfg.train.rect,
        cache=cfg.train.cache,
        workers=cfg.train.workers,
        pretrained=cfg.train.pretrained,
        optimizer=cfg.train.optimizer,
        weight_decay=cfg.train.weight_decay,
        momentum=cfg.train.momentum,
        cos_lr=cfg.train.cos_lr,
        freeze=cfg.train.freeze,
        classes=cfg.train.classes,
        fraction=cfg.train.fraction,
        close_mosaic=cfg.train.close_mosaic,
        val=cfg.train.val,
        plots=cfg.train.plots,
        save=cfg.train.save,
        amp=cfg.train.amp,
        box=cfg.train.box,
        cls=cfg.train.cls,
        dfl=cfg.train.dfl,
        deterministic=cfg.train.deterministic,
        seed=cfg.train.seed,
        save_period=cfg.train.save_period,
        resume=cfg.train.resume,
        warmup_epochs=cfg.train.warmup_epochs,
        warmup_momentum=cfg.train.warmup_momentum,
        warmup_bias_lr=cfg.train.warmup_bias_lr,
        imgsz=cfg.train.imgsz,
        device=cfg.train.device,
        batch=cfg.train.batch,
        single_cls=cfg.train.single_cls,
        multi_scale=cfg.train.multi_scale,
        lr0=cfg.train.lr0,
        lrf=cfg.train.lrf,
        hsv_h=cfg.train.augments.hsv_h,
        hsv_s=cfg.train.augments.hsv_s,
        hsv_v=cfg.train.augments.hsv_v,
        degrees=cfg.train.augments.degrees,
        translate=cfg.train.augments.translate,
        scale=cfg.train.augments.scale,
        shear=cfg.train.augments.shear,
        perspective=cfg.train.augments.perspective,
        flipud=cfg.train.augments.flipud,
        fliplr=cfg.train.augments.fliplr,
        bgr=cfg.train.augments.bgr,
        mosaic=cfg.train.augments.mosaic,
        mixup=cfg.train.augments.mixup,
        cutmix=cfg.train.augments.cutmix,
        copy_paste=cfg.train.augments.copy_paste,
        auto_augment=cfg.train.augments.auto_augment,
        erasing=cfg.train.augments.erasing,
        max_det=cfg.train.max_det,
        
    )