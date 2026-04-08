"""Module contains funtionalities for sampling a trained CLM."""

import logging
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm

from clm.models import RNN
from harvest.loader import ModelType, load_clm


log = logging.getLogger(__name__)


def sample_unconditional_clm(
    model: RNN,
    num_samples: int,
    *,
    batch_size: int = 1024,
    max_len: int = 250,
) -> Iterator[str]:
    """
    Sample a trained CLM.

    :param model: Model to be sampled.
    :param num_samples: Number of samples to generate.
    :param batch_size: Batch size to use.
    :param max_len: Maximum length of the sampled SMILES.
    :return: Iterator over sampled SMILES.
    """
    model.eval()

    with torch.inference_mode():
        remaining = num_samples

        while remaining > 0:
            this_batch = min(batch_size, remaining)

            batch_smiles = model.sample(
                n_sequences=this_batch,
                max_len=max_len,
                return_smiles=True,
                return_losses=False,
                descriptors=None,
            )

            for s in batch_smiles:
                yield s

            remaining -= this_batch


def cmd_sample_clm(
    model_dir: Path | str,
    model_type: ModelType,
    enum_number: int,
    out_dir: Path | str,
    device: torch.device | str,
    num_samples: int,
) -> None:
    """
    Sample a trained CLM.

    :param model_dir: Path to the output directory of a trained CLM.
    :param model_type: Type of model to be sampled.
    :param enum_number: Number of SMILES numerations used during training.
    :param out_dir: Path to directory to write sampled compounds and other statistics to.
    :param device: Device to run the model on.
    :param num_samples: Number of samples to generate.
    """
    model_config = load_clm(model_dir, model_type, enum_number, device)
    models = model_config.load_models()
    log.info(f"Loaded {len(models)} models!")

    num_models = len(models)
    samples_per_model = num_samples // num_models
    remainder = num_samples % num_models

    # We are going to stream generated SMILES to output file
    sampled = 0
    output_path = Path(out_dir) / "sampled.csv"
    with open(output_path, "w") as f:
        # Write header
        f.write("smiles\n")

        for model_idx, model in enumerate(tqdm(models)):
            # Determine number of samples to generate with model fold
            num_samples = samples_per_model + (remainder if model_idx == num_models - 1 else 0)

            # Generate samples and write out
            for s in tqdm(sample_unconditional_clm(model, num_samples)):
                f.write(s + "\n")
                sampled += 1

                # Flush every 1000 samples
                if sampled % 1000 == 0:
                    f.flush()
