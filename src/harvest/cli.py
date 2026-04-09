#!/usr/bin/env python3

"""Harvest command line interface."""

import argparse
import os
import sys
import shlex
import subprocess
from pathlib import Path

from harvest.logging import setup_logging, add_file_handler
from harvest.loader import ModelType
from harvest.sampling import cmd_sample_clm


_SLURM_FLAGS_WITH_VALUE = {
    "--part",
    "--cpus",
    "--mem",
    "--time",
    "--gres",
    "--job-name",
}
_SLURM_FLAGS_BOOL = {"--slurm"}


def _strip_slurm_flags(argv: list[str]) -> list[str]:
    """
    Return copy of argv with slurm options removed.

    :param argv: List of command line arguments.
    :return: Cleaned list of command line arguments.
    .. note:: This is what is passed to the inner CLI when using slurm submission.
    """
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "--snakemake-args":
            # Preserve passthrough args verbatim (may include --slurm)
            cleaned.extend(argv[i:])
            break

        if arg in _SLURM_FLAGS_BOOL:
            i += 1
            continue

        if arg in _SLURM_FLAGS_WITH_VALUE:
            # Skip flag + its value
            i += 2
            continue

        cleaned.append(arg)
        i += 1

    return cleaned


def _submit_via_slurm(slurm_args: argparse.Namespace, cli_argv: list[str]) -> None:
    """
    Submit the current Harvest command to Slurm using sbatch.

    :param slurm_args: Parsed command line arguments (including slurm options).
    :param cli_argv: List of command line arguments to pass to the inner Harvest CLI.
    """
    python = sys.executable

    # Get output directory from CLI args (fallback to cwd for commands without --out-dir)
    output_dir = os.path.abspath(getattr(slurm_args, "out_dir", os.getcwd()))

    # Ensure log directory exists
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Slurm settings (CLI wins over env, env wins over defaults)
    partition = slurm_args.part or os.environ.get("PARTITION", "skinniderlab")
    cpus = slurm_args.cpus or int(os.environ.get("CPUS", "8"))
    mem = slurm_args.mem or os.environ.get("MEM", "16G")
    time = slurm_args.time or os.environ.get("TIME", "24:00:00")
    gres = slurm_args.gres or os.environ.get("GRES", "gpu:1")
    job_name = slurm_args.job_name or os.environ.get("JOB_NAME", "harvest_job")

    # The command that will run on the node: python -m harvest.cli <args...>
    inner_cmd = [python, "-m", "harvest.cli", *cli_argv]
    inner_cmd_str = shlex.join(inner_cmd)

    sbatch_cmd = [
        "sbatch",
        "-J", job_name,
        "-p", partition,
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        "-o", os.path.join(output_dir, "logs", "harvest_%x_%j.out"),
        "-e", os.path.join(output_dir, "logs", "harvest_%x_%j.err"),
    ]

    if gres:
        sbatch_cmd.append(f"--gres={gres}")

    # Wrap inner command so wet get some basic info + timing
    wrap_script = f"""set -euo pipefail
echo "Node: $(hostname)"
echo "Using Python: {python}"
echo "CPUs: ${{OMP_NUM_THREADS:-{cpus}}}; Mem limit: {mem}"
/usr/bin/time -v {inner_cmd_str}
"""

    sbatch_cmd.extend(["--export", f"ALL,OMP_NUM_THREADS={cpus},MKL_NUM_THREADS={cpus},PYTHONUNBUFFERED=1"])
    sbatch_cmd.extend(["--wrap", wrap_script])

    if slurm_args.dry_run:
        print("[DRY RUN] Would submit Harvest job to slurm with:")
        print(" sbatch", " ".join(shlex.quote(x) for x in sbatch_cmd[1:]))
        print("\n[DRY RUN] Full --wrap script:\n")
        print(wrap_script)
        return

    print("Submitting Harvest job to Slurm:")
    print(" sbatch", " ".join(shlex.quote(x) for x in sbatch_cmd[1:]))
    subprocess.run(sbatch_cmd, check=True)


def cli(argv: list[str] | None = None) -> None:
    """
    Command line parser.

    :param argv: List of command line arguments (defaults to sys.argv[1:]).
    :return: Parsed command line arguments.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # Global Slurm options (only meaningful when --slurm is set)
    parser.add_argument("--slurm", action="store_true", help="Submit this Harvest command as a slurm job instead of running it locally.")
    parser.add_argument("--part", default=None, help="Slurm partition (queue) name.")
    parser.add_argument("--cpus", type=int, default=None, help="n=Number of CPUs per task for slurm.")
    parser.add_argument("--mem", default=None, help="Memory request for slurm (e.g., 16G).")
    parser.add_argument("--time", type=str, default=None, help="Time limit for slurm (e.g., 24:00:00).")
    parser.add_argument("--gres", type=str, default=None, help="Slurm generic resources (e.g., gpu:1).")
    parser.add_argument("--job-name", type=str, default=None, help="Slurm job name.")
    parser.add_argument("--dry-run", action="store_true", help="If set, only print the slurm submission command without actually submitting.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Common arguments; Slurm expects an output directory for logs, so we require it for all commands to simplify the interface when using --slurm
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=str, required=True, help="Directory to save output results.")
    common.add_argument("--device", type=str, required=False, default="cpu", help="Device to run on.")
    common.add_argument("--log-level", type=str, required=False, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")

    # Subparser for sampling a CLM.
    psample = sub.add_parser("sample", parents=[common], help="Sample a CLM.")
    psample.add_argument("--model-dir", type=str, required=True, help="path to output dir of trained CLM.")
    psample.add_argument("--model-type", type=str, required=True, choices=[t.name.lower() for t in ModelType], help="Type of CLM to sample.")
    psample.add_argument("--enum-number", type=int, required=True, help="Number of SMILES enumerations used during training.")
    psample.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate (split equally across folds).")
    psample.set_defaults(func=lambda args: cmd_sample_clm(
        model_dir=args.model_dir,
        model_type=ModelType[args.model_type.upper()],
        enum_number=args.enum_number,
        out_dir=args.out_dir,
        device=args.device,
        num_samples=args.num_samples,
    ))

    args = parser.parse_args(argv)

    # Make sure out_dir exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger and add file handler
    setup_logging(level=args.log_level)
    log_file_path: Path = out_dir / "log.txt"
    add_file_handler(log_file_path, level=args.log_level)

    if getattr(args, "slurm", False):
        # Rebuild CLI argv without slurm-only flags
        cli_argv = _strip_slurm_flags(argv or [])
        _submit_via_slurm(args, cli_argv)
    else:
        args.func(args)


def main(argv: list[str] | None = None) -> None:
    """
    Main function.
    """
    cli(argv)


if __name__ == "__main__":
    main()
