import argparse
import os
import sys
from typing import Dict

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.utils.logger import CaptionImageLogger, SetupCallback
from src.utils.util_modules import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        default=[],
        help="Paths to base configuration files.",
    )
    parser.add_argument(
        "-r",
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Resume training from a checkpoint.",
    )
    parser.add_argument(
        "--loss_with_epoch",
        type=bool,
        default=True,
        help="If True, calculate and log loss for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="If set, enables logging with available experiment trackers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for saving the trained model.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="If specified, training will be performed on the CPU.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Specify the mixed precision mode during training.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Define the maximum number of epochs for training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="1",
        help="Specify checkpointing frequency.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logging",
        help="Project directory for logging.",
    )
    parser.add_argument(
        "--mode", type=str, default="feat", help="Mode of model: feature or entropy."
    )
    parser.add_argument(
        "--batch_frequency", type=int, default=5000, help="Log images every n batches."
    )
    parser.add_argument(
        "--max_images", type=int, default=16, help="Maximum number of images to log."
    )
    return parser


def training_function(config: Dict, args: argparse.Namespace):
    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.with_tracking else None,
        project_dir=args.project_dir if args.with_tracking else None,
    )

    if args.with_tracking:
        run = os.path.splitext(os.path.basename(__file__))[0]
        config_dict = OmegaConf.to_container(config, resolve=True)
        accelerator.init_trackers(run, config_dict)

    data = instantiate_from_config(config.data)
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    model = instantiate_from_config(config.model).to(accelerator.device)
    lr = config.scheduler.base_learning_rate

    optimizer_ae = optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quantize.parameters())
        + list(model.quant_conv.parameters())
        + list(model.post_quant_conv.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
        weight_decay=1e-5,  # Added weight decay for regularization
    )
    optimizer_disc = optim.Adam(
        model.loss.discriminator.parameters(),
        lr=lr,
        betas=(0.5, 0.9),
        weight_decay=1e-5,
    )

    steps_per_epoch = len(train_dataloader)
    warmup_steps = int(steps_per_epoch * config.scheduler.warmup_epochs_ratio)
    min_learning_rate = config.model.get("min_learning_rate", 0.0)

    # Combine scheduler initialization into a single function
    def create_scheduler(optimizer, scheduler_type, warmup_steps, min_lr):
        if scheduler_type == "linear-warmup_cosine-decay":
            return CosineAnnealingLR(optimizer, warmup_steps, min_lr)
        return ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    scheduler_ae = create_scheduler(
        optimizer_ae,
        config["scheduler"]["scheduler_type"],
        warmup_steps,
        min_learning_rate,
    )
    scheduler_disc = create_scheduler(
        optimizer_disc,
        config["scheduler"]["scheduler_type"],
        warmup_steps,
        min_learning_rate,
    )

    (
        model,
        optimizer_ae,
        optimizer_disc,
        train_dataloader,
        val_dataloader,
        scheduler_ae,
        scheduler_disc,
    ) = accelerator.prepare(
        model,
        optimizer_ae,
        optimizer_disc,
        train_dataloader,
        val_dataloader,
        scheduler_ae,
        scheduler_disc,
    )

    starting_epoch, overall_step = 0, 0
    best_val_loss = float("inf")

    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        checkpoint_name = os.path.basename(args.resume_from_checkpoint)
        starting_epoch = (
            int(checkpoint_name.split("_")[1]) + 1 if "epoch" in checkpoint_name else 0
        )

    setup_callback = SetupCallback(
        resume=args.resume_from_checkpoint,
        now="",
        logdir=accelerator.project_dir,
        ckptdir=os.path.join(accelerator.project_dir, "checkpoints"),
        cfgdir=os.path.join(accelerator.project_dir, "configs"),
        config=config,
        argv_content=sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
    )
    image_logger = CaptionImageLogger(
        batch_frequency=args.batch_frequency, max_images=args.max_images
    )

    setup_callback.on_training_start()

    for epoch in tqdm(range(starting_epoch, args.max_epochs), desc="Epoch"):
        model.train()

        if args.with_tracking:
            total_loss = 0

        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):
            with accelerator.autocast():
                x = model.module.get_input(batch, model.module.image_key).to(
                    accelerator.device
                )
                if args.mode == "feat":
                    xrec, qloss, indices, gate = model.module(x)
                else:
                    xrec, qloss, indices, gate, x_entropy = model.module(x)

                ratio = indices.sum() / (indices.numel())
                aeloss, log_dict_ae = model.module.calculate_loss(
                    x, xrec, qloss, epoch, optimizer_idx=0, gate=gate
                )

                accelerator.backward(aeloss)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Gradient clipping
                optimizer_ae.step()
                optimizer_ae.zero_grad()
                (
                    scheduler_ae.step()
                    if isinstance(scheduler_ae, CosineAnnealingLR)
                    else scheduler_ae.step(aeloss)
                )

                discloss, log_dict_disc = model.module.calculate_loss(
                    x, xrec, qloss, epoch, optimizer_idx=1, gate=gate
                )

                accelerator.backward(discloss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_disc.step()
                optimizer_disc.zero_grad()
                (
                    scheduler_disc.step()
                    if isinstance(scheduler_disc, CosineAnnealingLR)
                    else scheduler_disc.step(discloss)
                )

                # Use a single function to handle logging
                def log_training_metrics(loss_ae, loss_disc, step):
                    total_loss = loss_ae + loss_disc
                    accelerator.log(
                        {
                            "train_aeloss": aeloss.item(),
                            "train_discloss": discloss.item(),
                            "train_fine_ratio": ratio.item(),
                            "train_loss": total_loss.item(),
                            **{
                                k: v.item() if v.numel() == 1 else v.tolist()
                                for k, v in log_dict_ae.items()
                            },
                            **{
                                k: v.item() if v.numel() == 1 else v.tolist()
                                for k, v in log_dict_disc.items()
                            },
                        },
                        step=step,
                    )

                # Update logging in training loop
                if args.with_tracking:
                    log_training_metrics(aeloss, discloss, overall_step)

                if batch_idx % args.batch_frequency == 0:
                    image_logger.log_img(
                        model,
                        batch,
                        batch_idx,
                        split="train",
                        mode=args.mode,
                        step=overall_step,
                        accelerator=accelerator,
                    )

                overall_step += 1

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_dataloader, desc="Validation", leave=False)
            ):
                with accelerator.autocast():
                    x = model.module.get_input(batch, model.module.image_key).to(
                        accelerator.device
                    )
                    if args.mode == "feat":
                        xrec, qloss, indices, gate = model.module(x)
                    else:
                        xrec, qloss, indices, gate, x_entropy = model.module(x)

                    aeloss, log_dict_ae = model.module.calculate_loss(
                        x, xrec, qloss, epoch, optimizer_idx=0, gate=gate
                    )
                    discloss, log_dict_disc = model.module.calculate_loss(
                        x, xrec, qloss, epoch, optimizer_idx=1, gate=gate
                    )

                    if args.with_tracking:
                        total_val_loss = aeloss + discloss
                        accelerator.log(
                            {
                                "val_aeloss": aeloss.item(),
                                "val_discloss": discloss.item(),
                                "val_fine_ratio": indices.sum() / indices.numel(),
                                "val_loss": total_val_loss.item(),
                                **{
                                    f"val_{k}": (
                                        v.item() if v.numel() == 1 else v.tolist()
                                    )
                                    for k, v in log_dict_ae.items()
                                },
                                **{
                                    f"val_{k}": (
                                        v.item() if v.numel() == 1 else v.tolist()
                                    )
                                    for k, v in log_dict_disc.items()
                                },
                            },
                            step=overall_step,
                        )

                    total_val_loss += aeloss.item() + discloss.item()

                    if batch_idx % args.batch_frequency == 0:
                        image_logger.log_img(
                            model,
                            batch,
                            batch_idx,
                            split="val",
                            mode=args.mode,
                            epoch=epoch,
                            step=overall_step,
                            accelerator=accelerator,
                        )

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                print(
                    f"\n New best validation loss: {best_val_loss:.4f}, saving checkpoint..."
                )
                accelerator.save_state(
                    os.path.join(args.output_dir, "best_checkpoint.ckpt"),
                    safe_serialization=False,
                )

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    training_function(config=config, args=args)
