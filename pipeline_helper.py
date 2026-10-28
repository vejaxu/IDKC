"""Internal data/plot helper called by main.m; does not start MATLAB or search."""

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
dependencies = Path(os.environ.get("IDKC_PLOT_DEPENDENCIES", PROJECT / ".local" / "plot_dependencies"))
if dependencies.is_dir():
    sys.path.insert(0, str(dependencies))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".local" / "plot_cache"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

from pipeline.data import BY_NAME, DatasetSpec, load_dataset
from pipeline.plotting import plot_best_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--key", choices=tuple(BY_NAME))
    source.add_argument("--input-file", type=Path)
    prepare.add_argument("--name")
    prepare.add_argument("--data-root", type=Path)
    prepare.add_argument("--x-key", default="data")
    prepare.add_argument("--y-key", default="class")
    prepare.add_argument("--pos-key")
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--skip-mat", action="store_true")
    plot = commands.add_parser("plot")
    plot.add_argument("--work-dir", type=Path, required=True)
    plot.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.key:
                if args.data_root is None:
                    raise ValueError("Configured input requires --data-root")
                spec = BY_NAME[args.key]
                source_file = args.data_root / spec.filename
            else:
                spec = DatasetSpec(args.name or args.input_file.stem, str(args.input_file),
                                   args.x_key, args.y_key, args.pos_key)
                source_file = args.input_file
            dataset = load_dataset(spec, source_file)
            args.work_dir.mkdir(parents=True, exist_ok=True)
            if not args.skip_mat:
                dataset.save_mat(args.work_dir / "input.mat")
            info = dict(dataset.info, source_input=str(dataset.source),
                        x_key=spec.x_key, y_key=spec.y_key, pos_key=spec.pos_key)
            (args.work_dir / "dataset.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            info = json.loads((args.work_dir / "dataset.json").read_text(encoding="utf-8"))
            spec = DatasetSpec(info["dataset"], info["source_input"],
                               info["x_key"], info["y_key"], info["pos_key"])
            dataset = load_dataset(spec, info["source_input"])
            if dataset.info != {key: info[key] for key in dataset.info}:
                raise ValueError("Dataset dimensions/labels changed after preparation")
            plot_best_run(dataset, args.work_dir / "result", seed=args.seed)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"IDKC helper failed: {error}\n")


if __name__ == "__main__":
    main()
