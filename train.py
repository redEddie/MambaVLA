"""
MambaVLA training entry point for LIBERO datasets.

Usage:
    python train.py                          # libero_spatial, default settings
    python train.py --suite libero_goal      # different suite
    python train.py --epochs 1000 --bs 128   # custom hyperparameters
    python train.py --model transformer      # use transformer backbone
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MambaVLA import train_policy
from MambaVLA.dataset import LiberoDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaVLA on LIBERO")

    # Data
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_goal", "libero_object", "libero_10", "libero_90"],
        help="LIBERO benchmark suite",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./libero/datasets",
        help="Root directory containing benchmark suite folders",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="mamba",
        choices=["mamba", "transformer"],
        help="Backbone type",
    )
    parser.add_argument("--latent_dim",   type=int, default=256)
    parser.add_argument("--embed_dim",    type=int, default=256)
    parser.add_argument("--n_layer",      type=int, default=5)
    parser.add_argument("--d_intermediate", type=int, default=256)
    parser.add_argument("--action_seq_len", type=int, default=10)
    parser.add_argument("--action_dim",   type=int, default=7)
    parser.add_argument("--lang_emb_dim", type=int, default=512)
    parser.add_argument("--sampling_steps", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--bs",     type=int, default=64,  dest="batch_size")
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_ema",  action="store_true")
    parser.add_argument("--scaler",  type=str, default="minmax", choices=["minmax", "action"])

    # Checkpointing
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: ./outputs/<suite>)",
    )
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None,
                        help="재개할 체크포인트 경로 (예: outputs/libero_spatial/epoch_00100.pt)")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="MambaVLA")
    parser.add_argument("--wandb_name",    type=str, default=None)
    parser.add_argument("--no_wandb",      action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Dataset ---
    dataset_dir = os.path.join(args.data_root, args.suite)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    log.info(f"Loading dataset from {dataset_dir}")
    dataset = LiberoDataset(
        dataset_dir=dataset_dir,
        action_seq_len=args.action_seq_len,
    )

    # --- Save directory ---
    save_dir = args.save_dir or os.path.join("outputs", args.suite)
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {save_dir}")

    # --- Transformer config (only used when --model transformer) ---
    transformer_cfg = None
    if args.model == "transformer":
        transformer_cfg = {
            "n_heads": 8,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "mlp_pdrop": 0.0,
            "bias": False,
        }

    # --- Train ---
    log.info(f"Starting training: {args.suite} | backbone={args.model} | "
             f"epochs={args.epochs} | bs={args.batch_size} | lr={args.lr}")

    model, trainer = train_policy(
        dataloader=dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        n_layer=args.n_layer,
        d_intermediate=args.d_intermediate,
        action_seq_len=args.action_seq_len,
        action_dim=args.action_dim,
        lang_emb_dim=args.lang_emb_dim,
        sampling_steps=args.sampling_steps,
        save_dir=save_dir,
        save_freq=args.save_freq,
        enable_ema=not args.no_ema,
        enable_data_scaling=True,
        data_scaler_type=args.scaler,
        dataloader_workers=args.workers,
        model_type=args.model,
        transformer_cfg=transformer_cfg,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_name=args.wandb_name or f"{args.suite}_{args.model}",
        resume=args.resume,
    )

    log.info("Done.")
    return model, trainer


if __name__ == "__main__":
    main()
