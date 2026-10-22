"""
Entry point for running MYTHRI SLM experiments.

Usage:
    python run_experiment.py --experiment 0
    python run_experiment.py --experiment 0 --quick  # smaller dataset for fast testing
"""

import os
# Auto-configure ROCm target architecture for AMD consumer GPUs (e.g., Radeon RX 6500 / Navi 24)
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Run MYTHRI SLM experiments")
    parser.add_argument(
        "--experiment", type=str, default="0",
        help="Experiment number to run (default: 0)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller dataset, fewer epochs for fast testing"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (e.g. rocm, rocm:0, cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and run evaluation directly on a trained checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file for eval-only mode (.safetensors or .pt)"
    )
    parser.add_argument(
        "--pretrained-backbone", type=str, default=None,
        help="Pretrained backbone to use (e.g. 'distilbert-base-uncased')"
    )
    parser.add_argument(
        "--pretrained-embeddings", type=str, default=None,
        help="Initialize embeddings from pretrained model (e.g. 'distilbert-base-uncased')"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size per training step (default: from config, recommend 8 for 4GB VRAM)"
    )
    parser.add_argument(
        "--grad-accumulation", type=int, default=None,
        help="Gradient accumulation steps (default: from config, recommend 8 when batch_size=8)"
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze pretrained backbone weights (train heads only, uses ~305 MB VRAM)"
    )
    parser.add_argument(
        "--unfreeze-backbone", action="store_true",
        help="Unfreeze pretrained backbone weights for full fine-tuning (uses ~1.2 GB VRAM)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs to train (overrides config)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max training samples to use with stratified minority preservation (e.g. 50000, 100000)"
    )
    parser.add_argument(
        "--full-dataset", action="store_true",
        help="Train on 100%% of the prepared dataset (462k+ examples)"
    )
    args = parser.parse_args()
    
    if args.experiment in ["0", "1", "1c"]:
        from experiments import experiment_0
        if args.pretrained_backbone:
            experiment_0.CONFIG["model"]["pretrained_backbone"] = args.pretrained_backbone
            experiment_0.CONFIG["model"]["hidden_dim"] = 768
            if "training" in experiment_0.CONFIG:
                experiment_0.CONFIG["training"]["backbone_learning_rate"] = 2.0e-5
            print(f"Using pretrained backbone: {args.pretrained_backbone} (hidden_dim=768, backbone_lr=2e-5)")
        if args.pretrained_embeddings:
            experiment_0.CONFIG["model"]["pretrained_embeddings"] = args.pretrained_embeddings
            print(f"Using pretrained embeddings: {args.pretrained_embeddings}")

        if args.epochs is not None:
            experiment_0.CONFIG["training"]["epochs"] = args.epochs
            print(f"Overriding training epochs: {args.epochs}")

        if args.max_samples is not None:
            experiment_0.CONFIG["data"]["max_train_samples"] = args.max_samples
            print(f"Setting stratified max_train_samples: {args.max_samples:,}")
        elif args.full_dataset:
            experiment_0.CONFIG["data"]["max_train_samples"] = None
            print("Full dataset mode enabled (100% of available data)")

        if args.quick:
            # Override config for quick testing
            experiment_0.CONFIG["data"]["max_train_samples"] = 2000
            experiment_0.CONFIG["data"]["max_eval_samples"] = 500
            experiment_0.CONFIG["training"]["epochs"] = 3
            experiment_0.CONFIG["training"]["batch_size"] = 16
            experiment_0.CONFIG["training"]["grad_accumulation"] = 2
            print("Running in QUICK mode (small dataset, 3 epochs, batch_size=16)")

        if args.batch_size is not None:
            experiment_0.CONFIG["training"]["batch_size"] = args.batch_size
            print(f"Overriding batch_size: {args.batch_size}")
        if args.grad_accumulation is not None:
            experiment_0.CONFIG["training"]["grad_accumulation"] = args.grad_accumulation
        if args.freeze_backbone:
            experiment_0.CONFIG["model"]["freeze_backbone"] = True
            print("Freezing pretrained backbone (training heads only, minimal VRAM)")
        elif args.unfreeze_backbone:
            experiment_0.CONFIG["model"]["freeze_backbone"] = False
            print("Unfreezing pretrained backbone (full fine-tuning)")
        
        from experiments.experiment_0 import main as run_exp_0
        target_device = args.device
        if target_device and (target_device.startswith("rocm") or target_device.startswith("hip")):
            target_device = target_device.replace("rocm", "cuda").replace("hip", "cuda")
        run_exp_0(device_name=target_device, eval_only=args.eval_only, checkpoint_path=args.checkpoint)

    else:
        print(f"Experiment {args.experiment} not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()