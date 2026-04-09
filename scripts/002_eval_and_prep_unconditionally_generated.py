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


if __name__ == "__main__":
    main()
