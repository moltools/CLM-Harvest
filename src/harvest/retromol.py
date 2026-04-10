"""Module with functionalities to run RetroMol."""

import json
import logging
import yaml
from pathlib import Path
from typing import Any, Generator

from tqdm import tqdm

from retromol.model.rules import RuleSet, ReactionRule, MatchingRule
from retromol.io.streaming import run_retromol_stream, stream_table_rows


log = logging.getLogger(__name__)


BATCH_SIZE = 2000
POOL_CHUNKSIZE = 50
MAXTASKSPERCHILD = 2000


def load_ruleset(reaction_rules_path: Path, matching_rules_path: Path) -> RuleSet:
    """
    Loads reaction and matching rules from YAML files.

    :param reaction_rules_path: Path to reaction rules file.
    :param matching_rules_path: Path to matching rules file.
    :return: Reaction and matching rules.
    """
    with open(reaction_rules_path, "r") as f:
        reaction_rules_data = yaml.safe_load(f)
    reaction_rules: list[ReactionRule] = [ReactionRule.from_dict(d) for d in reaction_rules_data]

    with open(matching_rules_path, "r") as f:
        matching_rules_data = yaml.safe_load(f)
    matching_rules: list[MatchingRule] = [MatchingRule.from_dict(d) for d in matching_rules_data]

    return RuleSet(
        reaction_rules=reaction_rules,
        matching_rules=matching_rules,
        match_stereochemistry=False,
    )


def cmd_retromol(
    data_path: Path | str,
    reaction_rules_path: Path | str,
    matching_rules_path: Path | str,
    out_dir: Path | str,
    smiles_col: str,
    num_workers: int,
) -> None:
    """
    Runs RetroMol.

    :param data_path: Path to input SMILES strings.
    :param reaction_rules_path: Path to reaction rules file.
    :param matching_rules_path: Path to matching rules file.
    :param out_dir: Path to output directory.
    :param smiles_col: SMILES string column.
    :param num_workers: Number of processes to use.
    """
    out_dir: Path = Path(out_dir)

    ruleset = load_ruleset(Path(reaction_rules_path), Path(matching_rules_path))

    # Setup table streamer
    source_iter = stream_table_rows(str(data_path), sep=",", chunksize=20_000)

    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = Path(out_dir) / "retromol_results.jsonl"
    jsonl_fh = open(out_path, "a", buffering=1)

    pbar = tqdm()
    for evt in run_retromol_stream(
        ruleset=ruleset,
        row_iter=source_iter,
        smiles_col=smiles_col,
        workers=num_workers,
        batch_size=BATCH_SIZE,
        pool_chunksize=POOL_CHUNKSIZE,
        maxtasksperchild=MAXTASKSPERCHILD,
    ):
        pbar.update(1)

        if evt.result is not None:
            jsonl_fh.write(json.dumps(evt.result) + "\n")

    if jsonl_fh:
        jsonl_fh.close()


def iter_jsonl(path: str) -> Generator[dict[str, Any], None, None]:
    """
    Generator that yields JSON objects from a JSONL file.

    :param path: Path to JSONL file.
    :yield: JSON object from each line of the file.
    """
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)