#!/usr/bin/env python3

import argparse


def cli() -> argparse.Namespace:
    """
    Command line argument parser.

    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True, help="Work directory.")
    parser.add_argument("--true-compounds", type=str, required=True, help="Path to CSV with true compounds.")
    parser.add_argument("--sampled-compounds", type=str, required=True, help="Path to CSV with sampled compounds.")
    parser.add_argument("--true-compounds-smiles-col", type=str, required=False, default="smiles", help="Column name for SMILES in true compounds CSV.")
    parser.add_argument("--sampled-compounds-smiles-col", type=str, required=False, default="smiles", help="Column name for SMILES in sampled compounds CSV.")
    return parser.parse_args()


def main() -> None:
    """
    Main function.
    """
    args = cli()

    # For true and sampled:
    # - show distributions molecular weights (density plot)
    # - show distributions generated atom counts (bar plot %; major ones + other)
    # - show for generated compound the distribution of counts regenerated/resampled SMILES with log10 x-axis
    # - select for all unique comounds with <0.8 Tc (r=3, b=2048) to any real compound
    # - select again but additionally make sure that not any pair in picked has >=0.8 Tc (r=3, b=2048)

    # Separate script:
    # - parse real and picked sampled lipopeptides with RetroMol
    # - visualize RetroMol results (monomer counts etc.)
    # - create training set based on picked with any coverage, and a fitlered set where every compound has coverage >0.7


if __name__ == "__main__":
    main()
