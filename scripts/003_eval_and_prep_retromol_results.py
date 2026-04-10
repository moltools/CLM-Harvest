#!/usr/bin/env python3

"""Parse RetroMol results and create a dataset for training a conditional CLM."""

import argparse
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

from harvest.retromol import iter_jsonl
from retromol.model.result import Result


log = logging.getLogger(__name__)


def cli() -> argparse.Namespace:
    """
    Command line argument parsing.

    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True, help="Working directory.")
    parser.add_argument("--jsonl", type=str, required=True, help="JSONL file with RetroMol results.")
    parser.add_argument("--model-type", type=str, required=True, help="Model type.")
    return parser.parse_args()


def main() -> None:
    """
    Main function.
    """
    args = cli()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    num_rows = 1
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    # Parse JSON and gather statistics
    coverages: list[float] = []
    num_found_monomers: list[int] = []
    num_unidentified_monomers: list[int] = []
    found_monomer_names: list[str] = []

    for d in tqdm(iter_jsonl(args.jsonl)):
        r = Result.from_dict(d)

        coverage = r.calculate_coverage()
        coverages.append(coverage)

        ns = r.linear_readout.assembly_graph.monomer_nodes()
        num_found = 0
        num_unidentified = 0
        for n in ns:
            if n.identified and n.identity:
                num_found += 1
                monomer_name = n.identity.name
                found_monomer_names.append(monomer_name)
            else:
                num_unidentified += 1

        num_found_monomers.append(num_found)
        num_unidentified_monomers.append(num_unidentified)

    # Plot coverages
    ax = axs[0]
    x = np.linspace(0, 1, 100)
    coverages_kde = gaussian_kde(coverages)
    ax.plot(x, coverages_kde(x), label="Coverage")
    ax.set_title("Coverage")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.legend()

    # Plot num found monomers barplot; mean and std
    ax = axs[1]
    mean_num_found = np.mean(num_found_monomers)
    mean_num_unidentified = np.mean(num_unidentified_monomers)
    std_num_found = np.std(num_found_monomers)
    std_num_unidentified = np.std(num_unidentified_monomers)
    ax.bar(["Count identified", "Count unidentified"], [mean_num_found, mean_num_unidentified], yerr=[std_num_found, std_num_unidentified], capsize=10)
    ax.set_title("Found monomers per compound")
    ax.set_ylabel("Average count")

    # Plot Marimekko of top 10 most abundant identified monomers
    ax = axs[2]

    monomer_counts = Counter(found_monomer_names)
    total_found = sum(monomer_counts.values())

    top10 = monomer_counts.most_common(10)
    top10_total = sum(count for _, count in top10)
    other_count = total_found - top10_total

    labels = [name for name, _ in top10]
    widths = [count / total_found for _, count in top10]

    if other_count > 0:
        labels.append("Other")
        widths.append(other_count / total_found)

    x_left = 0.0
    for label, width in zip(labels, widths):
        ax.bar(x_left, 1.0, width=width, align="edge", edgecolor="black")
        x_center = x_left + width / 2
        if width > 0.03:
            ax.text(x_center, 0.5, f"{label} ({width:.1%})", ha="center", va="center", rotation=90, fontsize=8)
        x_left += width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Top identified monomers")
    ax.set_xlabel("Fraction of all identified monomers")
    ax.set_ylabel("Relative abundance")
    ax.set_yticks([])

    # Hide unused axes
    for ax in axs[3:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(work_dir / "retromol_results.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
