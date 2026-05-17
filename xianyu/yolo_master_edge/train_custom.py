#!/usr/bin/env python3
"""Train YOLO-Master on the local tools-detection dataset (finetune or scratch)."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_master_edge import (
    DetectionTrainConfig,
    YoloMasterDetectionPipeline,
    YoloMasterVariant,
)
from yolo_master_edge.plot_metrics import plot_run_metrics

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATA = _ROOT / "dataset" / "data.yaml"
_DEFAULT_REPO = _ROOT / "YOLO-Master"
_DEFAULT_PROJECT = _ROOT / "artifacts" / "yolo_master_runs"
_WEIGHTS_URL = (
    "https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/"
    "YOLO-Master-v0.1-N/YOLO-Master-v0.1-N.pt"
)
_DEFAULT_WEIGHTS = _ROOT / "yolo_master_n.pt"


def _ensure_pretrained_weights(path: Path = _DEFAULT_WEIGHTS) -> Path:
    if path.is_file():
        return path
    try:
        import urllib.request

        print(f"Downloading pretrained weights to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_WEIGHTS_URL, path)
    except Exception as e:
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_weights.sh or download from {_WEIGHTS_URL}"
        ) from e
    return path


def _wsl_data_path(data: Path) -> str:
    """Use WSL mount path when running under Linux/WSL."""
    s = str(data.resolve())
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO-Master on dataset/data.yaml")
    parser.add_argument(
        "--mode",
        choices=("finetune", "scratch"),
        default="finetune",
        help="finetune: load yolo_master_*.pt; scratch: random init from YAML",
    )
    parser.add_argument("--variant", default="n", choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--repo", type=Path, default=_DEFAULT_REPO)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", type=Path, default=_DEFAULT_PROJECT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-after", action="store_true", help="Run val() after training")
    parser.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX after val")
    parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics training plots")
    parser.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Only plot metrics for an existing run dir (skip training)",
    )
    args = parser.parse_args()

    if args.plot_only is not None:
        plot_run_metrics(args.plot_only)
        return

    variant_map = {
        "n": YoloMasterVariant.V01_N,
        "s": YoloMasterVariant.V01_S,
        "m": YoloMasterVariant.V01_M,
        "l": YoloMasterVariant.V01_L,
        "x": YoloMasterVariant.V01_X,
    }
    variant = variant_map[args.variant]
    data_path = _wsl_data_path(args.data)
    epochs = args.epochs if args.epochs is not None else (50 if args.mode == "finetune" else 100)
    run_name = args.name or f"{args.mode}_{args.variant}"

    pipe = YoloMasterDetectionPipeline(
        variant=variant,
        yolo_master_repo=args.repo,
    )
    if args.mode == "scratch":
        pipe.load_from_yaml()
    else:
        weights = _ensure_pretrained_weights()
        pipe.load(model_spec=weights)

    cfg = DetectionTrainConfig(
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=run_name,
        plots=not args.no_plots,
        extra_train_kwargs={"workers": args.workers, "exist_ok": True},
    )
    run_dir = args.project / run_name
    print(f"Training mode={args.mode} variant={args.variant} data={data_path} name={run_name}")
    pipe.train(data_path, train_config=cfg)

    if not args.no_plots and (run_dir / "results.csv").is_file():
        print(f"Plotting loss & metrics -> {run_dir}")
        plot_run_metrics(run_dir)

    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    if args.val_after or args.export_onnx:
        if best_pt.is_file():
            pipe.load(model_spec=best_pt)
        pipe.val(data_path)

    if args.export_onnx:
        pipe.export(format="onnx")


if __name__ == "__main__":
    main()
