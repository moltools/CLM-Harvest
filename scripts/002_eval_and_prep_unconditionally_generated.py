#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect, TanimotoSimilarity


RDLogger.DisableLog("rdApp.*")


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


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol object.

    :param smiles: SMILES string.
    :return: RDKit Mol object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    return mol


def mol_to_inchikey_conn(mol: Chem.Mol) -> str:
    """
    Convert a RDKit Mol object to an InChI string.

    :param mol: RDKit Mol object.
    :return: InChI string.
    """
    # inchikey = Chem.MolToInchiKey(mol).split("-")[0]
    #
    # # Connectivity string is 14 characters long; we ignore stereochemistry and version information for uniqueness checks
    # assert len(inchikey) == 14, "Incorrect InChI string length!"
    #
    # return inchikey
    return Chem.MolToInchiKey(mol)


def mol_to_fingerprint(mol: Chem.Mol, radius: int, num_bits: int) -> ExplicitBitVect:
    """
    Convert a RDKit Mol object to a Morgan fingerprint.

    :param mol: RDKit Mol object.
    :param radius: Radius of the fingerprint.
    :param num_bits: Number of bits of the fingerprint.
    :return: Morgan fingerprint.
    """
    generator = GetMorganGenerator(radius=radius, fpSize=num_bits, includeChirality=False)
    fingerprint = generator.GetFingerprint(mol)

    return fingerprint


def calc_tc(fp1: ExplicitBitVect, fp2: ExplicitBitVect) -> float:
    """
    Calculate the Tanimoto similarity between two RDKit fingerprints.

    :param fp1: the first fingerprint
    :param fp2: the second fingerprint
    :return: the Tanimoto similarity score
    .. note:: perfect similarity returns 1.0, no similarity returns 0.0
    """
    similarity = TanimotoSimilarity(fp1, fp2)

    return similarity


def calc_weight(mol: Chem.Mol) -> float:
    """
    Calculate the exact molecular weight of a RDKit Mol object.

    :param mol: RDKit Mol object.
    """
    wt = ExactMolWt(mol)

    return wt



@dataclass(frozen=True)
class Compound:
    """
    Class representing a compound.

    :param smiles: SMILES string.
    .. note:: Compound objects are hashed on inchikey_conn property.
    """

    smiles: str = field(compare=False)

    mol: Chem.Mol = field(init=False, compare=False, repr=False)
    inchikey_conn: str = field(init=False)
    fingerprint: ExplicitBitVect = field(init=False, compare=False, repr=False)
    weight: float = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization method to compute the mol, inchikey_conn, and fingerprint attributes.
        """
        mol = smiles_to_mol(self.smiles)
        inchikey_conn = mol_to_inchikey_conn(mol)
        fingerprint = mol_to_fingerprint(mol, radius=3, num_bits=2048)
        weight = calc_weight(mol)

        object.__setattr__(self, "mol", mol)
        object.__setattr__(self, "inchikey_conn", inchikey_conn)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "weight", weight)


def parse_compounds(path: str, smiles_col: str) -> tuple[int, Counter, dict[str, Compound]]:
    """
    Parse compounds from a CSV file.

    :param path: Path to the CSV file.
    :param smiles_col: SMILES string column.
    :return: A tuple containing the total number of compounds, a Counter of compound counts, and a dictionary mapping InChI connectivity strings to Compound objects.
    """
    # Parse in true and sampled compounds
    total: int = 0
    compound_counts = Counter()
    compounds = {}

    with open(path, "r") as f:
        header = f.readline()
        smiles_col_idx = header.index(smiles_col)

        for line_idx, line in tqdm(enumerate(f.readlines())):
            total += 1

            line_items = line.strip().split(",")
            smiles = line_items[smiles_col_idx]
            try:
                compound = Compound(smiles)
                compound_counts[compound] += 1
                if compound not in compounds:
                    compounds[compound] = compound
            except Exception:
                continue

            # DM: only here for testing; remove later
            # if line_idx >= 10_000:
            #     break

    return total, compound_counts, compounds


def main() -> None:
    """
    Main function.
    """
    args = cli()

    # Parse in true and sampled compounds
    true_total, true_compound_counts, true_compounds = parse_compounds(args.true_compounds, args.true_compounds_smiles_col)
    sampled_total, sampled_compound_counts, sampled_compounds = parse_compounds(args.sampled_compounds, args.sampled_compounds_smiles_col)

    perc_valid = len(sampled_compounds) / sampled_total
    print(sampled_total, perc_valid)

    # For true and sampled:
    # - percentage of valid generated compounds
    # - select for all unique comounds with <0.8 Tc (r=3, b=2048) to any real compound
    # - show distributions molecular weights (density plot)
    # - show distributions generated atom counts (bar plot %; major ones + other)
    # - show for generated compound the distribution of counts regenerated/resampled SMILES with log10 x-axis
    # - select again but additionally make sure that not any pair in picked has >=0.8 Tc (r=3, b=2048)

    # Separate script:
    # - parse real and picked sampled lipopeptides with RetroMol
    # - visualize RetroMol results (monomer counts etc.)
    # - create training set based on picked with any coverage, and a filtered set where every compound has coverage >0.7

    # Additionally:
    # - retrain H001 on both datasets and evaluate
    # - encode graph approaches H002 and H003; draw out networks


if __name__ == "__main__":
    main()
