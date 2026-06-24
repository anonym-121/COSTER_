"""
Training script for VectorMapCVAE.

Usage:
    python -m cvae.train --config cvae/config.py \
        --train /path/to/scenarionet/training \
        --test /path/to/scenarionet/validation \
        --ckpt /path/to/checkpoint_dir \
        [--seed 1] [--device cuda:0]
"""
import os
import sys
import time
import gc
import math
import importlib.util
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from .model import VectorMapCVAE
    from .data import ScenarioNetCVAEDataset
except ImportError:  # Allow `python cvae/train.py` during development.
    from model import VectorMapCVAE
    from data import ScenarioNetCVAEDataset


def seed_all(seed_val):
    import random
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state(device):
    import random
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device) if torch.cuda.is_available() and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
    )


def set_rng_state(state, device):
    import random
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])


def ADE_FDE(y_pred, y_gt):
    """
    y_pred: (n_samples, horizon, N, 2)  or (horizon, N, 2)
    y_gt:   (horizon, N, 2)
    """
    err = (y_pred - y_gt).norm(dim=-1)
    if err.dim() == 2:
        return err.mean(-1), err[-1]
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
        return ade, fde


def main():
    parser = argparse.ArgumentParser(description="Train VectorMapCVAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config.py")
    parser.add_argument("--train", type=str, nargs="+", default=None, help="Training data directories")
    parser.add_argument("--test", type=str, nargs="+", default=None, help="Test data directories")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    args = parser.parse_args()

    # ── Load config ──
    spec = importlib.util.spec_from_file_location(
        "config", args.config,
        submodule_search_locations=[os.path.dirname(args.config)],
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_all(args.seed)
    init_rng_state = get_rng_state(device)

    # ── Datasets ──
    train_data_loader, test_data_loader = None, None

    if args.train:
        fraction = args.fraction if args.fraction else getattr(config, "fraction", None)
        train_dataset = ScenarioNetCVAEDataset(
            data_dir=args.train,
            **config.train_dataset,
            fraction=fraction,
            seed=args.seed,
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=ScenarioNetCVAEDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        batches = len(train_data_loader)
        print(f"Training: {len(train_dataset)} scenarios, {batches} batches/epoch")

    test_dirs = args.test if args.test else args.train
    if test_dirs:
        test_dataset = ScenarioNetCVAEDataset(
            data_dir=test_dirs,
            **config.test_dataset,
            fraction=args.fraction,
            seed=args.seed,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=ScenarioNetCVAEDataset.collate_fn,
            drop_last=False,
            pin_memory=True,
        )
        print(f"Testing: {len(test_dataset)} scenarios")

    # ── Model ──
    model = VectorMapCVAE(**config.model)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        weight_decay=getattr(config, "weight_decay", 1e-4),
    )
    start_epoch = 0
    ade_best = 1e6
    fde_best = 1e6

    # ── Resume from checkpoint ──
    if args.ckpt:
        os.makedirs(args.ckpt, exist_ok=True)
        ckpt_last = os.path.join(args.ckpt, "ckpt-last")
        ckpt_best = os.path.join(args.ckpt, "ckpt-best")

        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=device)
            ade_best = state_dict.get("ade", 1e6)
            fde_best = state_dict.get("fde", 1e6)

        if os.path.exists(ckpt_last) and args.train:
            print(f"Resuming from: {ckpt_last}")
            state_dict = torch.load(ckpt_last, map_location=device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
            if "rng_state" in state_dict:
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
                set_rng_state(rng_state, device)
            start_epoch = state_dict.get("epoch", 0)
            print(f"  Epoch: {start_epoch}, ADE: {state_dict.get('ade', '?'):.4f}, FDE: {state_dict.get('fde', '?'):.4f}")
        elif not args.train and os.path.exists(ckpt_best):
            print(f"Loading best checkpoint: {ckpt_best}")
            state_dict = torch.load(ckpt_best, map_location=device)
            model.load_state_dict(state_dict["model"])

    end_epoch = start_epoch + 1 if train_data_loader is None else config.epochs

    # ── LR Scheduler (Cosine Annealing with Warmup) ──
    lr_scheduler = None
    lr_warmup_epochs = getattr(config, "lr_warmup_epochs", 5)
    use_cosine_lr = getattr(config, "use_cosine_lr", True)
    if use_cosine_lr and train_data_loader is not None:
        def lr_lambda(epoch):
            if epoch < lr_warmup_epochs:
                return epoch / max(1, lr_warmup_epochs)
            progress = (epoch - lr_warmup_epochs) / max(1, config.epochs - lr_warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if start_epoch > 0:
            for _ in range(start_epoch):
                lr_scheduler.step()
        print(f"LR Scheduler: Cosine Annealing (warmup={lr_warmup_epochs} epochs)")

    # ── Logger ──
    logger = None
    if args.train and args.ckpt:
        logger = SummaryWriter(log_dir=args.ckpt)

    rng_state = init_rng_state
    kl_warmup = getattr(config, "kl_warmup_epochs", 30)
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)
    ss_start_epoch = getattr(config, "ss_start_epoch", 50)
    ss_max_ratio = getattr(config, "ss_max_ratio", 0.5)
    ss_ramp_epochs = getattr(config, "ss_ramp_epochs", 100)
    time_weight_exp = getattr(config, "time_weight_exp", 0.0)

    use_amp = getattr(config, "use_amp", True) and "cuda" in str(device)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    if use_amp:
        print("AMP (Mixed Precision) enabled — dtype=bfloat16")

    # ── Training Loop ──
    for epoch in range(start_epoch + 1, end_epoch + 1):
        beta = min(1.0, epoch / max(1, kl_warmup))

        if epoch < ss_start_epoch:
            ss_ratio = 0.0
        else:
            ss_ratio = min(ss_max_ratio,
                           ss_max_ratio * (epoch - ss_start_epoch) / max(1, ss_ramp_epochs))

        # ========== Train ==========
        losses = None
        if train_data_loader is not None and epoch <= config.epochs:
            ss_str = f", ss={ss_ratio:.3f}" if ss_ratio > 0 else ""
            print(f"\nEpoch {epoch}/{config.epochs}  (β={beta:.3f}{ss_str})")
            tic = time.time()
            set_rng_state(rng_state, device)
            losses = {}
            model.train()

            for batch_idx, batch in enumerate(train_data_loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                neighbor = batch["neighbor"].to(device)
                map_feature = batch["map_feature"].to(device)
                map_mask = batch["map_mask"].to(device)
                map_position = batch["map_position"].to(device)
                map_heading = batch["map_heading"].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    err, kl = model(
                        x=x, y=y, neighbor=neighbor,
                        map_feature=map_feature, map_mask=map_mask,
                        map_position=map_position, map_heading=map_heading,
                        scheduled_sampling_ratio=ss_ratio,
                    )
                    loss_dict = model.loss(
                        err, kl, beta=beta,
                        time_weight_exp=time_weight_exp,
                    )

                scaler.scale(loss_dict["loss"]).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                for k, v in loss_dict.items():
                    if k not in losses:
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k] * batch_idx + v.item()) / (batch_idx + 1)

                if (batch_idx + 1) % max(1, batches // 20) == 0 or batch_idx == batches - 1:
                    elapsed = int(time.time() - tic)
                    loss_str = " - ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                    sys.stdout.write(
                        f"\r\033[K  {batch_idx + 1}/{batches} -- {elapsed}s - {loss_str}"
                    )

            rng_state = get_rng_state(device)
            print()

        # Step LR scheduler
        if lr_scheduler is not None and losses is not None:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            if logger is not None:
                logger.add_scalar("train/lr", current_lr, epoch)

        gc.collect()
        torch.cuda.empty_cache()

        # ========== Test ==========
        ade, fde, ade_d, fde_d = 1e4, 1e4, 1e4, 1e4
        perform_test = (train_data_loader is None or getattr(config, "test_since", 0) <= epoch) \
                       and test_data_loader is not None

        if perform_test:
            print("  Evaluating...", end="")
            model.eval()
            ADE_list, FDE_list = [], []
            ADE_d_list, FDE_d_list = [], []
            set_rng_state(init_rng_state, device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                for batch in test_data_loader:
                    try:
                        x = batch["x"].to(device)
                        y = batch["y"].to(device)
                        neighbor = batch["neighbor"].to(device)
                        map_feature = batch["map_feature"].to(device)
                        map_mask = batch["map_mask"].to(device)
                        map_position = batch["map_position"].to(device)
                        map_heading = batch["map_heading"].to(device)

                        # Stochastic prediction (minADE/minFDE)
                        y_pred = model(
                            x=x, neighbor=neighbor,
                            map_feature=map_feature, map_mask=map_mask,
                            map_position=map_position, map_heading=map_heading,
                            n_predictions=config.pred_samples,
                        )
                        a, f = ADE_FDE(y_pred, y)
                        a = torch.min(a, dim=0)[0]
                        f = torch.min(f, dim=0)[0]
                        ADE_list.append(a)
                        FDE_list.append(f)

                        # Deterministic prediction
                        y_pred_d = model(
                            x=x, neighbor=neighbor,
                            map_feature=map_feature, map_mask=map_mask,
                            map_position=map_position, map_heading=map_heading,
                            n_predictions=0,
                        )
                        a_d, f_d = ADE_FDE(y_pred_d, y)
                        ADE_d_list.append(a_d)
                        FDE_d_list.append(f_d)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print("\n  [WARN] OOM in eval batch — skipped", end="")
                        continue

            ade = torch.cat(ADE_list).mean().item()
            fde = torch.cat(FDE_list).mean().item()
            ade_d = torch.cat(ADE_d_list).mean().item()
            fde_d = torch.cat(FDE_d_list).mean().item()
            print(f"\r\033[K  ADE: {ade:.4f}/{ade_d:.4f}; FDE: {fde:.4f}/{fde_d:.4f}")

        # ========== Save ==========
        if losses is not None and args.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar(f"train/{k}", v, epoch)
                logger.add_scalar("train/beta", beta, epoch)
                if perform_test:
                    logger.add_scalars("eval", dict(
                        ADE_min=ade, FDE_min=fde,
                        ADE_deter=ade_d, FDE_deter=fde_d,
                    ), epoch)

            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade, fde=fde, ade_d=ade_d, fde_d=fde_d,
                epoch=epoch,
                rng_state=rng_state,
            )
            torch.save(state, os.path.join(args.ckpt, "ckpt-last"))

            if ade < ade_best:
                ade_best = ade
                fde_best = fde
                best_state = dict(
                    model=state["model"],
                    ade=ade, fde=fde, ade_d=ade_d, fde_d=fde_d,
                    epoch=epoch, rng_state=rng_state,
                )
                torch.save(best_state, os.path.join(args.ckpt, "ckpt-best"))
                print(f"  ★ New best: ADE={ade:.4f}, FDE={fde:.4f}")

    if logger is not None:
        logger.close()
    print("Done.")


if __name__ == "__main__":
    main()
