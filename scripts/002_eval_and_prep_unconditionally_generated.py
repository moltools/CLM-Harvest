#!/usr/bin/env python3

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
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
    parser.add_argument("--size-test", type=int, required=False, default=1000, help="Size of test set.")
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
    inchikey = Chem.MolToInchiKey(mol).split("-")[0]

    # Connectivity string is 14 characters long; we ignore stereochemistry and version information for uniqueness checks
    assert len(inchikey) == 14, "Incorrect InChI string length!"

    return inchikey


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
    :return: the exact molecular weight of a RDKit Mol object.
    """
    wt = ExactMolWt(mol)

    return wt


def count_atoms(mol: Chem.Mol) -> defaultdict[str, int]:
    """
    Count the number of atoms in a RDKit Mol object.

    :param mol: RDKit Mol object.
    :return: Dictionary of atom name and number of atoms.
    """
    counts: defaultdict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] += 1

    return counts


def count_macrocycles(mol: Chem.Mol, min_ring_size: int = 12) -> int:
    """
    Count macrocyclic rings in a molecule.

    :param mol: RDKit Mol object.
    :param min_ring_size: Minimum ring size to call a ring a macrocycle.
    :return: Number of macrocyclic rings.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    return sum(1 for ring in atom_rings if len(ring) >= min_ring_size)


PEPTIDE_BOND_SMARTS = "[NX3][CX3](=[OX1])"
PEPTIDE_BOND_QUERY = Chem.MolFromSmarts(PEPTIDE_BOND_SMARTS)


def count_peptide_bonds(mol: Chem.Mol) -> int:
    """
    Count peptide/amide bonds in a molecule.

    :param mol: RDKit Mol object.
    :return: Number of peptide bonds.
    """
    return len(mol.GetSubstructMatches(PEPTIDE_BOND_QUERY))


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

    count_c: int = field(init=False, compare=False, repr=False)
    count_n: int = field(init=False, compare=False, repr=False)
    count_o: int = field(init=False, compare=False, repr=False)
    count_s: int = field(init=False, compare=False, repr=False)
    count_p: int = field(init=False, compare=False, repr=False)
    count_halogens: int = field(init=False, compare=False, repr=False)

    count_macrocycles: int = field(init=False, compare=False, repr=False)
    count_peptide_bonds: int = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization method to compute the mol, inchikey_conn, and fingerprint attributes.
        """
        mol = smiles_to_mol(self.smiles)
        inchikey_conn = mol_to_inchikey_conn(mol)
        fingerprint = mol_to_fingerprint(mol, radius=3, num_bits=2048)
        weight = calc_weight(mol)
        atom_counts = count_atoms(mol)
        macrocycle_count = count_macrocycles(mol)
        peptide_bond_count = count_peptide_bonds(mol)

        object.__setattr__(self, "mol", mol)
        object.__setattr__(self, "inchikey_conn", inchikey_conn)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "weight", weight)

        object.__setattr__(self, "count_c", atom_counts["C"])
        object.__setattr__(self, "count_n", atom_counts["N"])
        object.__setattr__(self, "count_o", atom_counts["O"])
        object.__setattr__(self, "count_s", atom_counts["S"])
        object.__setattr__(self, "count_p", atom_counts["P"])
        object.__setattr__(self, "count_halogens", atom_counts["F"] + atom_counts["Cl"] + atom_counts["Br"] + atom_counts["I"])

        object.__setattr__(self, "count_macrocycles", macrocycle_count)
        object.__setattr__(self, "count_peptide_bonds", peptide_bond_count)


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
                    compounds[compound.inchikey_conn] = compound
            except Exception:
                continue

    return total, compound_counts, compounds


def main() -> None:
    """
    Main function.
    """
    args = cli()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Parse in true and sampled compounds
    true_total, true_compound_counts, true_compounds = parse_compounds(args.true_compounds, args.true_compounds_smiles_col)
    sampled_total, sampled_compound_counts, sampled_compounds = parse_compounds(args.sampled_compounds, args.sampled_compounds_smiles_col)

    # Pick a random test set from sampled
    random_keys: list[str] = random.sample(list(sampled_compounds.keys()), k=args.size_test)
    test_compounds: list[Compound] = [sampled_compounds[k] for k in random_keys]

    num_rows = 2
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    # Plot barchart: ratio valid vs. ratio invalid
    ratio_valid = len(sampled_compounds) / sampled_total

    dissimilar_compounds: list[Compound] = []
    for inchikey_conn, sampled_compound in tqdm(sampled_compounds.items()):
        if any(calc_tc(sampled_compound.fingerprint, ref_compound.fingerprint) >= 0.8 for ref_compound in list(true_compounds.values()) + test_compounds):
            continue
        dissimilar_compounds.append(sampled_compound)
    num_dissimilar = len(dissimilar_compounds)
    ratio_dissimilar = num_dissimilar / sampled_total

    # Write dissimilar and test compound SMILES to output files
    out_dissimilar = work_dir / "sampled_dissimilar.csv"
    with open(out_dissimilar, "w") as f:
        f.write("smiles\n")
        for compound in dissimilar_compounds:
            f.write(compound.smiles + "\n")

    out_test = work_dir / "sampled_test.csv"
    with open(out_test, "w") as f:
        f.write("smiles\n")
        for compound in test_compounds:
            f.write(compound.smiles + "\n")

    ax = axs[0]
    ax.bar(["Valid samples", "Dissimilar to true and test\n" + r"($T_c \leq 0.8$)"], [ratio_valid, ratio_dissimilar], color=["green", "blue"])
    ax.set_title("Validity and uniqueness generated SMILES")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1)

    # Plot smooth density plot of molecular weights for true and sampled compounds
    ax = axs[1]
    true_weights = np.array([compound.weight for compound in true_compounds.values()])
    dissimilar_weights = np.array([compound.weight for compound in dissimilar_compounds])
    test_weights = np.array([compound.weight for compound in test_compounds])

    all_weights = np.concatenate([true_weights, dissimilar_weights, test_weights])
    x = np.linspace(all_weights.min(), all_weights.max(), 500)

    if len(true_weights) > 1:
        true_kde = gaussian_kde(true_weights)
        ax.plot(x, true_kde(x), label="True compounds")

    if len(dissimilar_weights) > 1:
        sampled_kde = gaussian_kde(dissimilar_weights)
        ax.plot(x, sampled_kde(x), label="Train compounds (synthetic)")

    if len(test_weights) > 1:
        sampled_kde = gaussian_kde(test_weights)
        ax.plot(x, sampled_kde(x), label="Test compounds (synthetic)")

    ax.set_title("Molecular weight distribution")
    ax.set_xlabel("Exact molecular weight")
    ax.set_ylabel("Density")
    ax.legend()

    # Plot stacked bar chart of atom count composition
    ax = axs[2]

    groups = {
        "True compounds": list(true_compounds.values()),
        "Train compounds\n(synthetic)": dissimilar_compounds,
        "Test compounds\n(synthetic)": test_compounds,
    }

    atom_labels = ["C", "N", "O", "S", "P", "Halogens"]

    group_means: dict[str, dict[str, float]] = {}
    for group_name, compounds in groups.items():
        n = len(compounds)
        if n == 0:
            group_means[group_name] = {atom: 0.0 for atom in atom_labels}
            continue

        group_means[group_name] = {
            "C": sum(compound.count_c for compound in compounds) / n,
            "N": sum(compound.count_n for compound in compounds) / n,
            "O": sum(compound.count_o for compound in compounds) / n,
            "S": sum(compound.count_s for compound in compounds) / n,
            "P": sum(compound.count_p for compound in compounds) / n,
            "Halogens": sum(compound.count_halogens for compound in compounds) / n,
        }

    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))

    for atom in atom_labels:
        values = [group_means[group_name][atom] for group_name in groups]
        ax.bar(x, values, bottom=bottom, label=atom)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(list(groups.keys()))
    ax.set_title("Mean atom counts per compound")
    ax.set_ylabel("Mean atom count")
    ax.legend()

    # Plot average +/- std of macrocycles and peptide bonds
    ax = axs[3]

    group_names = list(groups.keys())

    macro_means = []
    macro_stds = []
    peptide_means = []
    peptide_stds = []

    for group_name in group_names:
        compounds = groups[group_name]

        macro_counts = np.array([compound.count_macrocycles for compound in compounds], dtype=float)
        peptide_counts = np.array([compound.count_peptide_bonds for compound in compounds], dtype=float)

        if len(macro_counts) == 0:
            macro_means.append(0.0)
            macro_stds.append(0.0)
        else:
            macro_means.append(macro_counts.mean())
            macro_stds.append(macro_counts.std())

        if len(peptide_counts) == 0:
            peptide_means.append(0.0)
            peptide_stds.append(0.0)
        else:
            peptide_means.append(peptide_counts.mean())
            peptide_stds.append(peptide_counts.std())

    x = np.arange(len(group_names))
    width = 0.35

    ax.bar(x - width / 2, macro_means, width, yerr=macro_stds, capsize=4, label="Macrocycles")
    ax.bar(x + width / 2, peptide_means, width, yerr=peptide_stds, capsize=4, label="Peptide bonds")

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_title("Macrocycles and peptide bonds")
    ax.set_ylabel("Mean count per compound")
    ax.legend()

    # Plot distribution of how often unique generated compounds were resampled
    ax = axs[4]

    generated_counts = np.array(list(sampled_compound_counts.values()), dtype=int)
    generated_counts = generated_counts[generated_counts > 0]

    count_distribution = Counter(generated_counts)

    x_vals = sorted(count_distribution.keys())
    y_vals = [count_distribution[x] for x in x_vals]

    ax.bar(x_vals, y_vals)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title("Resampling frequency of\ngenerated compounds")
    ax.set_xlabel("Count per unique generated compound")
    ax.set_ylabel("Number of unique compounds")

    # Hide unused axes
    for ax in axs[5:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(work_dir / "eval_unconditionally_generated.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
