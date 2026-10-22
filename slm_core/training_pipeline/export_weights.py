"""
Utility script to export PyTorch checkpoints (.pt) to safetensors (.safetensors).

Usage:
    python model_training/export_weights.py --checkpoint checkpoints/exp_0/best_model.pt
    python model_training/export_weights.py --checkpoint checkpoints/exp_0/best_model.pt --output checkpoints/exp_0/cognitive_slm.safetensors
"""

import argparse
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def export_checkpoint(checkpoint_path: str, output_path: str = None):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        root_relative = ROOT_DIR / checkpoint_path
        if root_relative.exists():
            ckpt_path = root_relative
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    if output_path is None:
        out_path = ckpt_path.with_suffix(".safetensors")
    else:
        out_path = Path(output_path)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Extracted 'model_state_dict' ({len(state_dict)} tensors)")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        print(f"Using root dictionary as state_dict ({len(state_dict)} tensors)")
    else:
        raise ValueError("Unsupported checkpoint structure.")

    # Save to safetensors format
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(out_path))

    pt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    st_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved safetensors to: {out_path}")
    print(f"Original .pt size: {pt_size_mb:.2f} MB -> Safetensors size: {st_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export .pt checkpoints to .safetensors")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/exp_0/best_model.pt",
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path for .safetensors file",
    )
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
