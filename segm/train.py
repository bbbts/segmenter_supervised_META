#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml
import torch
import click
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params
import segm.engine as engine
from segm.engine import evaluate
from contextlib import suppress
from timm.utils import NativeScaler

IGNORE_LABEL = 255  # consistent ignore



@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int)
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
def main(
    log_dir, dataset, im_size, crop_size, window_size, window_stride,
    backbone, decoder, optimizer, scheduler, weight_decay,
    dropout, drop_path, batch_size, epochs, learning_rate,
    normalization, eval_freq, amp, resume
):
    # ---- Distributed setup ----
    ptu.set_gpu_mode(True)
    distributed.init_process()

    # ---- Load config ----
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    decoder_cfg = cfg["decoder"]["mask_transformer"] if "mask_transformer" in decoder else cfg["decoder"][decoder]

    # ---- Model config ----
    im_size = im_size or dataset_cfg["im_size"]
    crop_size = crop_size or dataset_cfg.get("crop_size", im_size)
    window_size = window_size or dataset_cfg.get("window_size", im_size)
    window_stride = window_stride or dataset_cfg.get("window_stride", im_size)

    model_cfg.update({
        "image_size": (crop_size, crop_size),
        "backbone": backbone,
        "dropout": dropout,
        "drop_path_rate": drop_path,
    })
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # ---- Dataset config ----
    world_batch_size = batch_size or dataset_cfg["batch_size"]
    num_epochs = epochs or dataset_cfg["epochs"]
    lr = learning_rate or dataset_cfg["learning_rate"]
    eval_freq = eval_freq or dataset_cfg.get("eval_freq", 1)
    if normalization:
        model_cfg["normalization"] = normalization

    batch_size = max(1, world_batch_size // max(1, ptu.world_size))

    dataset_kwargs = dict(
        dataset=dataset,
        image_size=im_size,
        crop_size=crop_size,
        batch_size=batch_size,
        normalization=model_cfg.get("normalization", "vit"),
        split="train",
        num_workers=10,
        root=dataset_cfg.get("data_root", dataset_cfg.get("root", None)),
    )

    # ---- Create datasets ----
    print(f"Creating training dataset for {dataset}...")
    train_dataset = create_dataset(dataset_kwargs)

    # ---- Detect validation split folder name (validation or val)
    val_split = None
    for split_name in ["validation", "val"]:
        try:
            val_kwargs = dataset_kwargs.copy()
            val_kwargs["split"] = split_name
            val_kwargs["batch_size"] = 1
            val_dataset = create_dataset(val_kwargs)
            val_split = split_name
            print(f"Detected validation split: '{val_split}'")
            break
        except RuntimeError as e:
            if "No images or masks found" in str(e) or "No images" in str(e):
                continue
            raise

    if val_split is None:
        raise RuntimeError("No validation split found. Expected folder 'validation' or 'val'.")

    # ---- Wrap in DataLoaders ----
    def make_loader(dataset_obj, batch_size, shuffle=True):
        sampler = DistributedSampler(dataset_obj, shuffle=shuffle) if ptu.distributed else None
        loader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=(sampler is None) and shuffle,
            sampler=sampler,
            num_workers=10,
            pin_memory=True,
        )
        loader.sampler_obj = sampler
        return loader

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Validation ('{val_split}') dataset length: {len(val_loader.dataset)}")

    n_cls = train_dataset.n_cls

    # ---- Model ----
    model_cfg["n_cls"] = n_cls
    model = create_segmenter(model_cfg)
    model.to(ptu.device)

    # ---- Optimizer & scheduler ----
    iter_max = len(train_loader) * num_epochs
    iter_warmup = 0.0

    optimizer_kwargs = dict(
        opt=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
        clip_grad=None,
        sched=scheduler,
        epochs=num_epochs,
        min_lr=1e-5,
        poly_power=0.9,
        poly_step_size=1,
        iter_max=iter_max,
        iter_warmup=iter_warmup,
    )
    opt_args = type('', (), {})()
    for k, v in optimizer_kwargs.items():
        setattr(opt_args, k, v)

    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)

    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # ---- Resume ----
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pth"
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # ---- Save config ----
    variant = dict(
        world_batch_size=world_batch_size,
        dataset_kwargs=dataset_kwargs,
        net_kwargs=model_cfg,
        optimizer_kwargs=optimizer_kwargs,
        amp=amp,
        log_dir=str(log_dir),
        inference_kwargs=dict(im_size=im_size, window_size=window_size, window_stride=window_stride),
    )
    with open(log_dir / "variant.yml", "w") as f:
        yaml.dump(variant, f)

    print(f"Encoder parameters: {num_params(model.encoder)}")
    print(f"Decoder parameters: {num_params(model.decoder)}")

    # ---- Train loop ----
    for epoch in range(num_epochs):
        # set epoch on sampler if present (distributed)
        if hasattr(train_loader, "sampler_obj") and train_loader.sampler_obj is not None:
            if hasattr(train_loader.sampler_obj, "set_epoch"):
                train_loader.sampler_obj.set_epoch(epoch)

        # Train one epoch; pass val_loader so validation loss is computed & plotted
        train_logger = engine.train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            log_dir=str(log_dir),
            val_loader=val_loader,
        )

        # Print a compact one-liner showing the losses (including validation)
        print(f"[Epoch {epoch+1}/{num_epochs}] Losses: {train_logger}", flush=True)

        # ---- Save checkpoint ----
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict(),
                epoch=epoch,
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            torch.save(snapshot, checkpoint_path)

        # ---- Evaluation metrics (full) ----
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            # Build val_seg_gt mapping: keys = filename without extension
            val_seg_gt = {}
            for idx in range(len(val_loader.dataset)):
                item = val_loader.dataset[idx]
                mask = item.get("segmentation", item.get("mask"))
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask
                file_id = item.get("id", f"img_{idx}")
                file_id = os.path.splitext(file_id)[0]
                val_seg_gt[file_id] = mask_np

            eval_logger = evaluate(
                model, val_loader, val_seg_gt, window_size, window_stride, amp_autocast, log_dir=str(log_dir), epoch=epoch
            )

            # Print evaluation metrics (Python dict)
            print(f"Evaluation Metrics [Epoch {epoch}]: {eval_logger}", flush=True)
            print("")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(0)


if __name__ == "__main__":
    main()
