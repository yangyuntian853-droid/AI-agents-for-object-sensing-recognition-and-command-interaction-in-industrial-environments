"""Plot training loss and detection metrics from Ultralytics results.csv."""

from __future__ import annotations

import csv
from pathlib import Path
def _read_results_csv(csv_path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows: list[dict[str, float]] = []
        for raw in reader:
            row: dict[str, float] = {}
            for k, v in raw.items():
                if k == "epoch":
                    row[k] = float(v)
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
        return fieldnames, rows


def plot_run_metrics(run_dir: str | Path, *, show: bool = False) -> list[Path]:
    """
    Generate loss / metric curve PNGs under `run_dir`.

    Writes:
      - loss_curves.png   (train & val box/cls/dfl/moe loss)
      - metrics_curves.png (precision, recall, mAP50, mAP50-95)
      - results.png       (Ultralytics bundled plot, if available)
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No results.csv in {run_dir}")

    try:
        from ultralytics.utils.plotting import plot_results

        plot_results(file=str(csv_path))
    except Exception as exc:
        print(f"[plot_metrics] ultralytics plot_results skipped: {exc}")

    import matplotlib.pyplot as plt

    fieldnames, rows = _read_results_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]

    train_loss_cols = [c for c in fieldnames if c.startswith("train/") and "loss" in c]
    val_loss_cols = [c for c in fieldnames if c.startswith("val/") and "loss" in c]
    metric_cols = [c for c in fieldnames if c.startswith("metrics/")]

    saved: list[Path] = []

    if train_loss_cols or val_loss_cols:
        fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Loss curves — {run_dir.name}", fontsize=13)

        for col in train_loss_cols:
            ys = [r.get(col, float("nan")) for r in rows]
            ax_train.plot(epochs, ys, "o-", linewidth=1.8, markersize=4, label=col.split("/", 1)[-1])
        ax_train.set_ylabel("train loss")
        ax_train.legend(loc="upper right", fontsize=8)
        ax_train.grid(True, alpha=0.3)

        for col in val_loss_cols:
            ys = [r.get(col, float("nan")) for r in rows]
            ax_val.plot(epochs, ys, "s-", linewidth=1.8, markersize=4, label=col.split("/", 1)[-1])
        ax_val.set_xlabel("epoch")
        ax_val.set_ylabel("val loss")
        ax_val.legend(loc="upper right", fontsize=8)
        ax_val.grid(True, alpha=0.3)

        fig.tight_layout()
        loss_path = run_dir / "loss_curves.png"
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        saved.append(loss_path)
        print(f"Saved {loss_path}")

    if metric_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"Detection metrics — {run_dir.name}", fontsize=13)
        for col in metric_cols:
            ys = [r.get(col, float("nan")) for r in rows]
            label = col.replace("metrics/", "").replace("(B)", "")
            ax.plot(epochs, ys, "o-", linewidth=1.8, markersize=4, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        metrics_path = run_dir / "metrics_curves.png"
        fig.savefig(metrics_path, dpi=150)
        plt.close(fig)
        saved.append(metrics_path)
        print(f"Saved {metrics_path}")

    results_png = run_dir / "results.png"
    if results_png.is_file():
        saved.append(results_png)

    if show:
        import matplotlib.pyplot as plt

        for p in saved:
            if p.suffix == ".png":
                plt.figure()
                plt.imshow(plt.imread(p))
                plt.axis("off")
                plt.title(p.name)
        plt.show()

    return saved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot YOLO-Master training curves from results.csv")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path("artifacts/yolo_master_runs/ft_smoke"),
        help="Training run directory containing results.csv",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_run_metrics(args.run_dir, show=args.show)


if __name__ == "__main__":
    main()
