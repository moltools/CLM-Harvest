#!/usr/bin/env python3

"""Plot loss for trained CLM models."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from harvest.loader import ModelType, load_clm


def cli() -> argparse.Namespace:
    """
    Command line parser.

    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--work-dir", type=str, required=True, help="Path to the working directory.")
    parser.add_argument("--model-type", type=str, required=True, choices=[t.name.lower() for t in ModelType], help="Type of CLM to sample.")
    parser.add_argument("--enum-number", type=int, required=True, help="Number of SMILES enumerations used during training.")
    return parser.parse_args()


def main() -> None:
    """
    Main function.
    """
    args = cli()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_clm(
        args.model_dir,
        model_type=args.model_type,
        enum_number=args.enum_number,
        device=torch.device("cpu"),
    )

    num_folds = len(model_config.folds)

    # Based on number of folds, create 3xN grid of loss plots
    num_cols = 3
    num_rows = math.ceil(num_folds / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    for fold_idx, fold in enumerate(model_config.folds):
        ax = axs[fold_idx]

        loss_path = fold.loss_path
        loss_df = pd.read_csv(loss_path, sep=",")

        train_df = loss_df[loss_df["outcome"] == "training loss"].sort_values("minibatch")
        val_df = loss_df[loss_df["outcome"] == "validation loss"].sort_values("minibatch")

        ax.plot(train_df["minibatch"], train_df["value"], label="Training loss")
        ax.plot(val_df["minibatch"], val_df["value"], label="Validation loss")

        ax.set_xlabel("Minibatch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Fold {fold_idx + 1}")
        ax.legend(loc="upper right")

    # Hide unused axes
    for ax in axs[num_folds:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(work_dir / "loss.png", dpi=300)
    plt.close()



if __name__ == "__main__":
    main()
