"""Module contains functionalities for indexing and loading a trained CLM from disk."""

import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import torch

from clm.datasets import Vocabulary
from clm.models import RNN

CLM_INPUTS_VOCAB_PATTERN = r"^train_(.+)_SMILES_(\d+)\.vocabulary$"
CLM_MODELS_MODEL_PATTERN = r"^(.+)_SMILES_(\d+)_\d+_model\.pt$"
CLM_MODELS_LOSS_PATTERN = r"^(.+)_SMILES_(\d+)_\d+_loss\.csv\.gz$"


class ModelType(Enum):
    """
    Enum for the type of CLM.

    :cvar UNCONDITIONAL: Unconditional CLM.
    """

    UNCONDITIONAL = auto()


@dataclass(frozen=True)
class Fold:
    """
    Represents a fold of a trained CLM.

    :param fold_iter: Fold iteration number.
    :param vocab_path: Path to the fold's vocabulary file.
    :param model_path: Path to the fold's model file.
    :param loss_path: Path to the fold's loss file.
    """

    fold_iter: int
    vocab_path: Path
    model_path: Path
    loss_path: Path


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a trained CLM.

    :param model_type: Type of model loaded.
    :param enum_number: Number of SMILES enumerations used during training.
    :param dataset_name: Name of the dataset loaded.
    :param folds: List of found Folds.
    :param device: Device to load the CLM from.
    """

    model_type: ModelType
    enum_number: int
    dataset_name: str
    folds: list[Fold]
    device: torch.device

    def load_models(self) -> list[RNN]:
        """
        Loads the RNN models for each fold.

        :return: List of RNN models.
        """
        models: list[RNN] = []
        for fold in self.folds:
            vocab = Vocabulary(vocab_file=str(fold.vocab_path))
            model = RNN(
                vocabulary=vocab,
                rnn_type="LSTM",
                embedding_size=128,
                hidden_size=1024,
                n_layers=3,
                dropout=0,
            )

            state = torch.load(str(fold.model_path), map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()

            models.append(model)

        return models


def load_clm(
    model_dir: Path | str,
    model_type: ModelType,
    enum_number: int,
    device: torch.device | str,
) -> ModelConfig:
    """
    Loads a trained CLM from disk.

    :param model_dir: Path to the directory containing the trained CLM.
    :param model_type: Type of model to load.
    :param enum_number: Number of SMILES enumerations used during training.
    :param device: Device to load the CLM from.
    :return: ModelConfig object.
    """
    # Configure device; check if cuda available if requested
    device: torch.device = torch.device(device)
    cuda_available = torch.cuda.is_available()
    if not cuda_available and device.type == "cuda":
        raise ValueError("CUDA is not available but device is set to 'cuda'!")

    model_dir = Path(model_dir)

    inputs_path = model_dir / str(enum_number) / "prior" / "inputs"
    assert inputs_path.exists(), f"{inputs_path} does not exist!"

    models_path = model_dir / str(enum_number) / "prior" / "models"
    assert models_path.exists(), f"{models_path} does not exist!"

    # Find vocab file paths from inputs folder
    vocab_re = re.compile(CLM_INPUTS_VOCAB_PATTERN)
    vocab_paths = {}
    for path in inputs_path.iterdir():
        if not path.is_file():
            continue

        match = vocab_re.match(str(path.name))
        if match:
            dataset_name = match.group(1)
            fold_iter = int(match.group(2))

            if fold_iter in vocab_paths:
                raise KeyError(f"Fold {fold_iter} already exists in vocabs!")

            vocab_paths[fold_iter] = dict(
                vocab_path=str(path),
                dataset_name=dataset_name,
            )

    # Find model file paths from models folder
    model_re = re.compile(CLM_MODELS_MODEL_PATTERN)
    model_paths = {}
    for path in models_path.iterdir():
        if not path.is_file():
            continue

        match = model_re.match(str(path.name))
        if match:
            dataset_name = match.group(1)
            fold_iter = int(match.group(2))

            if fold_iter in model_paths:
                raise KeyError(f"Fold {fold_iter} already exists in models!")

            model_paths[fold_iter] = dict(
                model_path=str(path),
                dataset_name=dataset_name,
            )

    # Additionally, find the loss files
    loss_re = re.compile(CLM_MODELS_LOSS_PATTERN)
    loss_paths = {}
    for path in models_path.iterdir():
        if not path.is_file():
            continue

        match = loss_re.match(str(path.name))
        if match:
            dataset_name = match.group(1)
            fold_iter = int(match.group(2))

            if fold_iter in loss_paths:
                raise KeyError(f"Fold {fold_iter} already exists in losses!")

            loss_paths[fold_iter] = dict(
                loss_path=str(path),
                dataset_name=dataset_name,
            )

    # Check that we found at least one vocab and at least one model
    assert len(vocab_paths) > 0, "Need at least one vocab file to load a model!"
    assert len(model_paths) > 0, "Need at least one model file to load a model!"
    assert len(loss_paths) > 0, "Need at least one loss file to load a model!"

    # Check that vocab, model, and loss folds match; same number of files and same fold numbers
    assert set(vocab_paths.keys()) == set(model_paths.keys()), f"Vocab and model folds do not match!"
    assert set(model_paths.keys()) == set(loss_paths.keys()), f"Vocab and loss folds do not match!"

    # Check that all found vocab, model, and loss files are associated to the same dataset
    dataset_names = \
        set([v["dataset_name"] for v in vocab_paths.values()] \
            + [m["dataset_name"] for m in model_paths.values()] \
            + [l["dataset_name"] for l in loss_paths.values()])
    assert len(dataset_names) == 1, f"Found vocab, model, and loss files associated to multiple datasets: {dataset_names}!"
    dataset_name = list(dataset_names)[0]

    # Match vocab and model files
    folds: list[Fold] = []
    for fold_iter in vocab_paths.keys():
        vocab_path = vocab_paths[fold_iter]["vocab_path"]
        model_path = model_paths[fold_iter]["model_path"]
        loss_path = loss_paths[fold_iter]["loss_path"]
        folds.append(Fold(
            fold_iter=int(fold_iter) + 1,  # 0-based, so increment with one
            vocab_path=Path(vocab_path),
            model_path=Path(model_path),
            loss_path=Path(loss_path),
        ))

    # Sort folds based on fold_iter
    folds.sort(key=lambda x: x.fold_iter)

    return ModelConfig(
        model_type=model_type,
        enum_number=enum_number,
        dataset_name=dataset_name,
        folds=folds,
        device=device,
    )
