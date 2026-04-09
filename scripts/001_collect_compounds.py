#!/usr/bin/env python3

import argparse
import json
import logging
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Set
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import aiohttp
import asyncio
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Download URLs
DOWNLOAD_URL_MIBIG = r"https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz"
DOWNLOAD_URL_NPATLAS = r"https://www.npatlas.org/static/downloads/NPAtlas_download.json"

# NPClassifier API call configuration
CHUNK_SIZE = 100
NP_CONCURRENCY = 8
NP_RATE = 10.0
NP_TIMEOUT = 10
NP_RETRIES = 3

# NPClassifier API settings
NP_API_BASES = [
    "https://npclassifier.ucsd.edu/classify",
    "https://npclassifier.gnps2.org/classify",
]

NP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "BioNexus/0.2 (NPClassifier bulk annotate)",
}


def cli() -> argparse.Namespace:
    """
    Command line parser.

    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True, help="Working directory in which all downloaded and created files are stored.")
    parser.add_argument("--compound-classes", nargs="+", required=True, help="List of compound classes to select for; compound is selected if it is classified as at least one of the given classes.")
    parser.add_argument("--extra-smiles", type=str, required=False, help="Path to .smi file with one SMILES per line.")
    parser.add_argument("--extra-smiles-has-header", action="store_true", default=False, help="Extra SMILES file has a header line.")
    return parser.parse_args()


async def _np_fetch_one(
    session: aiohttp.ClientSession,
    smiles: str,
    *,
    timeout_s: int,
    retries: int,
    delay_between: float | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """
    Fetch NPClassifier annotation for a single SMILES string.

    :param session: aiohttp client session.
    :param smiles: SMILES string to query.
    :param timeout_s: Request timeout in seconds.
    :param retries: Number of retries on failure.
    :param delay_between: Delay between requests in seconds.
    :param semaphore: asyncio semaphore for limiting concurrency.
    :return: NPClassifier annotation blob or None.
    """
    async with semaphore:
        if delay_between:
            await asyncio.sleep(delay_between)

        backoff = 0.5
        last_err: Exception | None = None

        for attempt in range(retries + 1):
            for base in NP_API_BASES:
                try:
                    async with session.get(
                        base,
                        params={"smiles": smiles},
                        timeout=timeout_s,
                        headers=NP_HEADERS,
                    ) as resp:
                        status = resp.status
                        text = await resp.text()
                        if status == 200:
                            # Header sometimes wrong; try parse anyway
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                pass

                        if status == 429:
                            ra = resp.headers.get("Retry-After")
                            wait = float(ra) if (ra and ra.isdigit()) else backoff * (1 + random.random())
                            await asyncio.sleep(wait)
                            continue

                        if 500 <= status < 600:
                            last_err = RuntimeError(f"{base} returned {status}")
                            continue

                        if 400 <= status < 500:
                            log.warning(f"NPClassifier 4xx ({status}) for SMILES={smiles}")
                            return None

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # Log and retry
                    last_err = e

            if attempt < retries:
                # Exponential backoff with jitter
                await asyncio.sleep(backoff * (1 + random.random()))
                backoff *= 2

        if last_err:
            log.error(f"NPClassifier giving up for SMILES={smiles} :: {last_err}")

        return None


async def np_fetch_many(
    smiles_list: list[str],
    *,
    concurrency: int,
    rate_per_sec: float | None,
    timeout_s: int,
    retries: int,
    pbar: tqdm | None = None,
) -> dict[str, dict[str, Any] | None]:
    """
    Fetch NPClassifier annotations for multiple SMILES strings concurrently.

    :param smiles_list: List of SMILES strings to query.
    :param concurrency: Maximum number of concurrent requests.
    :param rate_per_sec: Maximum request rate per second.
    :param timeout_s: Request timeout in seconds.
    :param retries: Number of retries on failure.
    :param pbar: Optional tqdm progress bar.
    :return: Dictionary mapping SMILES strings to their NPClassifier annotation blobs or None.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    delay_between = (1.0 / rate_per_sec) if rate_per_sec and rate_per_sec > 0 else None

    connector = aiohttp.TCPConnector(
        limit=max(4, concurrency),
        limit_per_host=max(2, concurrency // 2),
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=None)

    out: dict[str, dict[str, Any] | None] = {}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=NP_HEADERS) as session:
        async def _run(smi: str):
            out[smi] = await _np_fetch_one(
                session, smi,
                timeout_s=timeout_s,
                retries=retries,
                delay_between=delay_between,
                semaphore=semaphore,
            )
            if pbar is not None:
                pbar.update(1)

        tasks = [asyncio.create_task(_run(smi)) for smi in smiles_list]
        await asyncio.gather(*tasks)

    return out


def download(work_dir: Path, url: str) -> Path:
    """
    Download file from URL to work directory if it does not already exist.

    :param work_dir: Work directory to download file to.
    :param url: URL to download.
    :return: Path to the downloaded file.
    """
    filename = Path(urlparse(url).path).name
    out_path = work_dir / filename

    if not out_path.exists():
        log.info(f"Downloading {filename}...")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        log.info(f"Done downloading {filename}!")
    else:
        log.info(f"File {out_path} already exists!")

    assert out_path.exists(), f"{filename} not found at {out_path}!"

    return out_path


def iter_mibig_json_records(mibig_path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Iterate over JSON files inside a .tar.gz MIBiG archive.

    :param mibig_path: Path to MIBiG .tar.gz archive.
    :return: Yields (member_name, parsed_json_dict).
    """
    mibig_path = Path(mibig_path)

    with tarfile.open(mibig_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not member.name.endswith(".json"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            with f:
                yield member.name, json.load(f)


def smiles_to_mol(smiles: str) -> Mol:
    """
    Convert SMILES string to RDKit Mol object.

    :param smiles: SMILES string.
    :return: RDKit Mol object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return mol


def remove_stereochemistry(mol: Mol) -> Mol:
    """
    Remove stereochemistry from RDKit Mol object.

    :param mol: RDKit Mol object.
    :return: RDKit Mol object.
    """
    mol = Chem.Mol(mol)  # makes sure operation is not in-place
    Chem.RemoveStereochemistry(mol)
    return mol


def mol_to_inchikey(mol: Mol) -> str:
    """
    Convert RDKit Mol object to InChIKey.

    :param mol: RDKit Mol object.
    :return: InChIKey string.
    """
    return Chem.MolToInchiKey(mol)


@dataclass(frozen=True)
class Compound:
    """
    Data class representing a chemical compound with its SMILES, RDKit Mol object (without stereochemistry), non-stereo SMILES, and InChIKey.

    :param smiles: SMILES string.
    """

    smiles: str
    mol: Mol = field(init=False, repr=False, compare=False)
    smiles_no_stereo: str = field(init=False, compare=False)
    inchikey: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization processing to parse SMILES, remove stereochemistry, and compute InChIKey.
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {self.smiles}")

        mol_no_stereo = remove_stereochemistry(mol)
        smiles_no_stereo = Chem.MolToSmiles(mol_no_stereo, isomericSmiles=False)
        inchikey = mol_to_inchikey(mol_no_stereo)

        object.__setattr__(self, "mol", mol_no_stereo)
        object.__setattr__(self, "smiles_no_stereo", smiles_no_stereo)
        object.__setattr__(self, "inchikey", inchikey)


def main() -> None:
    """
    Main function.
    """
    args = cli()

    # Create work directory if it does not already exists
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    query_classes: Set[str] = set([cl.lower() for cl in args.compound_classes])

    collected_extra: Set[Compound] = set()
    collected: Set[Compound] = set()

    # Parse compounds from extra-smiles file
    if args.extra_smiles is not None:
        extra_smiles = Path(args.extra_smiles)
        with open(extra_smiles, "r") as f:
            if args.extra_smiles_has_header:
                f.readline()

            for line in f:
                smiles = line.strip()
                compound = Compound(smiles)
                collected_extra.add(compound)

    collected_extra_compounds = len(collected_extra)
    log.info(f"Collected {collected_extra_compounds} extra compounds!")

    # Download MIBiG and collect SMILES, remove stereochemistry, and do quick validity check
    mibig_path = download(work_dir, DOWNLOAD_URL_MIBIG)
    for _, record in tqdm(iter_mibig_json_records(mibig_path)):
        for compound in record.get("compounds", []):
            if smiles := compound.get("structure", None):
                try:
                    compound = Compound(smiles)
                    collected.add(compound)
                except ValueError:
                    pass

    collected_mibig_compounds = len(collected)
    log.info(f"Collected {collected_mibig_compounds} unique MIBiG compounds!")

    # Download NPAtlas and collect SMILES, remove stereochemistry, and do quick validity check
    npatlas_path = download(work_dir, DOWNLOAD_URL_NPATLAS)
    with open(npatlas_path, "r") as f:
        data = json.load(f)
        for record in tqdm(data):
            if smiles := record.get("smiles", None):
                compound = Compound(smiles)
                collected.add(compound)

    collected_npatlas_compounds = len(collected) - collected_mibig_compounds
    log.info(f"Collected {collected_npatlas_compounds} unique NPAtlas compounds!")

    # Create output file
    out_path = work_dir / "collected_compounds.txt"
    with open(out_path, "w") as f:

        # Write header
        f.write("smiles\n")

        # Extra compounds don't have to be classified; always included
        for compound in collected_extra:
            f.write(compound.smiles + "\n")

        # Loop over collected compounds in chunks and collect NPClassifier annotations
        for i in tqdm(range(0, len(collected), CHUNK_SIZE)):
            chunk = list(collected)[i : i + CHUNK_SIZE]
            smiles_list = [c.smiles_no_stereo for c in chunk]

            annotations = asyncio.run(np_fetch_many(
                smiles_list,
                concurrency=NP_CONCURRENCY,
                rate_per_sec=NP_RATE,
                timeout_s=NP_TIMEOUT,
                retries=NP_RETRIES,
            ))

            for smiles, annotations in annotations.items():
                if annotations:
                    class_results = annotations.get("class_results", [])
                    class_results: Set[str] = set([cl.lower() for cl in class_results])

                    # Check if class_results have any overlap with query_classes
                    has_overlap = bool(class_results & query_classes)
                    if has_overlap:
                        f.write(smiles + "\n")

            # Flush after every chunk
            f.flush()


if __name__ == "__main__":
    main()
