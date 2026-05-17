# yolo_master_edge
**现在没有和机器人对齐，只是作为cv流程的overview，等和机器人对齐的时候应该要做大调整**


本目录是对 [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master) 的**薄封装**：在与你现有 `llm_edge` 相同的仓库里，提供统一入口的**目标检测**管线（加载 → 推理 → 训练 / 验证 → 导出），便于「检测 + 语言」组合落地。

上游论文与实现说明见官方仓库；本 README 描述**本包怎么用**，并记录在本机 **tools-detection 数据集（119 类）** 上完成的微调与配套脚本。

## 本次已完成工作（2026-05）

| 项 | 说明 |
|----|------|
| 数据集 | `dataset/`：2660 训练图 + 870 验证图，YOLO 标注，`names` 0–118 共 119 类 |
| 数据配置 | `dataset/data.yaml`：WSL 路径 `/mnt/d/.../dataset`，替代原 `newdata.yaml` 的 Kaggle 路径 |
| 环境 | WSL `conda` 环境 `yolo`；`YOLO-Master` 可编辑安装；`flash_attn`；PyTorch **cu128**（`scripts/setup_cuda_torch.sh`） |
| 预训练权重 | `yolo_master_n.pt`（YOLO-Master-v0.1-N，可 `scripts/download_weights.sh` 下载） |
| **微调训练** | `ft_n`：50 epoch，`batch=8`，GPU `device=0`，输出见 `artifacts/yolo_master_runs/ft_n/` |
| 从零训练 | `scratch_n` 曾启动后手动停止（未完成 100 epoch）；需时可再跑 |
| 曲线与指标图 | `plot_metrics.py` → `loss_curves.png`、`metrics_curves.png`；训练自带 PR/F1/混淆矩阵等 |
| 一键脚本 | `train_custom.py`、`scripts/run_train.sh`、`scripts/train_pipeline.sh`、`scripts/verify_env.sh` |

**微调最终结果（epoch 50，验证集）**：mAP50 ≈ **0.92**，mAP50-95 ≈ **0.82**，Precision ≈ **0.95**，Recall ≈ **0.86**。  
**部署权重**：`artifacts/yolo_master_runs/ft_n/weights/best.pt`（val / ONNX 导出见下文「验证与导出」）。

## 与上游的关系

- YOLO-Master 在工程上基于 **Ultralytics** API（`from ultralytics import YOLO`），但代码与权重与 PyPI 上的官方 `ultralytics` **不等价**。
- 你必须在克隆的 **YOLO-Master 仓库根目录**执行 `pip install -r requirements.txt` 与 **`pip install -e .`**，让当前环境里的 `ultralytics` 来自 Tencent 这份源码；**不要**单独 `pip install ultralytics` 指望能加载 YOLO-Master 的模型与 MoE 等扩展。

上游许可证为 **AGPL-3.0**，商用、闭源分发与网络服务请自行做合规评估。

## 环境

- Python **3.10+** 推荐（与上游及 PyTorch 生态更一致）。
- **NVIDIA GPU** 强烈推荐；CPU 仅适合小规模试跑。
- 将**父目录** `tiaozhanbei` 加入 `PYTHONPATH`，以便 `import yolo_master_edge`（本包目录名为 `yolo_master_edge`，不是 `src`）。

### WSL + conda（本机实测）

```bash
# 1. 上游可编辑安装（克隆目录已放在本包下的 YOLO-Master/）
cd /mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/YOLO-Master
pip install -r requirements.txt
pip install -e .

# 2. 环境变量（写入 ~/.bashrc 或每次训练前 export）
export YOLO_MASTER_ROOT=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/YOLO-Master
export PYTHONPATH=/mnt/d/dl/keyan/tiaozhanbei:$PYTHONPATH

# 3. 自检
bash /mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/scripts/verify_env.sh
```

若 `torch.cuda.is_available()` 为 False，多为 PyTorch CUDA 与驱动不匹配。本机使用：

```bash
bash scripts/setup_cuda_torch.sh   # 安装 torch 2.11+cu128
```

`flash_attn` 需与当前 torch/CUDA 版本匹配的 wheel；安装问题见上游文档或本仓库历史记录。

## 安装上游（一次性）

若尚未克隆，可在 `yolo_master_edge` 下：

```text
git clone https://github.com/Tencent/YOLO-Master YOLO-Master
cd YOLO-Master
pip install -r requirements.txt
pip install -e .
```

设置 **`YOLO_MASTER_ROOT`** 为上述克隆根目录的绝对路径，便于 **`load_from_yaml()`**（从仓库内 YAML 构建网络，适合从零训练）。

## 本包模块

| 模块 / 脚本 | 说明 |
|-------------|------|
| `config.py` | `YoloMasterVariant`（v0.1 N/S/M/L/X）、`DetectionTrainConfig`（含 `plots`、`extra_train_kwargs`） |
| `presets.py` | 默认权重名（如 `yolo_master_n.pt`）、YAML 相对路径、`default_yolo_master_root()` |
| `pipeline.py` | **`YoloMasterDetectionPipeline`**：主入口类 |
| `results.py` | `detections_from_result`：单张 `Results` → `list[dict]` |
| `plot_metrics.py` | `plot_run_metrics`：从 `results.csv` 绘制 loss / mAP 曲线 |
| `train_custom.py` | 本地数据集一键训练（`--mode finetune|scratch`） |
| `scripts/run_train.sh` | 激活 `yolo` 环境并调用 `train_custom.py` |
| `scripts/train_pipeline.sh` | 顺序：微调 → 从零 → val+ONNX（可按需修改） |
| `scripts/download_weights.sh` | 下载 `yolo_master_n.pt` |
| `scripts/verify_env.sh` | 检查 ultralytics 来源、CUDA、数据路径 |
| `scripts/setup_cuda_torch.sh` | 重装 PyTorch cu128 |

## 本地数据集（119 类 tools-detection）

```text
dataset/
  images/train/    # 2660
  images/val/      # 870
  labels/train/
  labels/val/
  data.yaml        # 训练用（WSL path 指向本目录）
  newdata.yaml     # 原始 Kaggle 路径，仅作参考
```

`data.yaml` 中 `path` 必须为 **WSL 路径**（如 `/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/dataset`），在 Windows 下直接训练易找不到图片。

### 微调（推荐，已跑通）

加载官方 `yolo_master_n.pt`，在自定义 119 类上训练；检测头会随 `nc` 自动适配。

```bash
# WSL
bash scripts/run_train.sh \
  --mode finetune --variant n --epochs 50 --batch 8 \
  --device 0 --name ft_n --workers 4
```

权重与日志默认目录：`artifacts/yolo_master_runs/<name>/`。

- `weights/best.pt`、`weights/last.pt`
- `results.csv`：每 epoch 的 loss 与 mAP
- 训练结束自动生成：`BoxPR_curve.png`、`confusion_matrix.png`、`val_batch*_pred.jpg` 等

首次运行若缺少 `yolo_master_n.pt`，`train_custom.py` 会从 Hugging Face 自动下载。

### 从零训练（随机初始化）

```bash
bash scripts/run_train.sh \
  --mode scratch --variant n --epochs 100 --batch 8 \
  --device 0 --name scratch_n --workers 4
```

收敛慢，建议 epoch ≥100；OOM 时将 `--batch` 改为 4 或 2。

### 绘制 loss / 检测指标曲线

训练完成后（或已有 `results.csv`）：

```bash
cd /mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge
python plot_metrics.py artifacts/yolo_master_runs/ft_n
```

生成：

- `loss_curves.png` — train/val 的 box、cls、dfl、moe loss
- `metrics_curves.png` — precision、recall、mAP50、mAP50-95
- `results.png` — Ultralytics 汇总图（若环境可用）

## 快速开始：推理

**本机 119 类微调权重**（推荐）：

```python
from pathlib import Path
from yolo_master_edge import YoloMasterDetectionPipeline, YoloMasterVariant

pipe = YoloMasterDetectionPipeline(variant=YoloMasterVariant.V01_N)
pipe.load(model_spec=Path("artifacts/yolo_master_runs/ft_n/weights/best.pt"))
dets = pipe.predict_detections(Path("dataset/images/val/0.jpg"))
```

**官方预训练权重**（COCO 80 类，非本数据集）：

```python
pipe = YoloMasterDetectionPipeline(variant=YoloMasterVariant.V01_N)
pipe.load()  # yolo_master_n.pt

# 返回每张图一组框：xyxy、confidence、class_id、class_name
dets = pipe.predict_detections(Path("path/to/image.jpg"))
print(dets[0])
```

`predict(...)` 与 `predict_detections(...)` 的额外关键字参数会**原样传给**上游 `model.predict`（例如 `conf=`、`device=`、`cluster=`、`sparse_sahi=` 等，以后者仓库文档为准）。

## 从零训练：按 YAML 构建模型

需本地已有 YOLO-Master 克隆，并设置 `YOLO_MASTER_ROOT` **或**在构造管线时传入 `yolo_master_repo=`。

```python
from pathlib import Path

from yolo_master_edge import YoloMasterDetectionPipeline, YoloMasterVariant, DetectionTrainConfig

pipe = YoloMasterDetectionPipeline(
    variant=YoloMasterVariant.V01_N,
    yolo_master_repo=Path(r"D:\repos\YOLO-Master"),
)
pipe.load_from_yaml()

# data 为 Ultralytics 风格的数据集 yaml
cfg = DetectionTrainConfig(epochs=100, imgsz=640, batch=16, project=Path("artifacts/yolo_runs"))
pipe.train("coco8.yaml", train_config=cfg)
```

训练中的 **MoE / LoRA** 等扩展参数，请直接通过 `DetectionTrainConfig.extra_train_kwargs` 或 `train(..., **kwargs)` 传入，命名与含义以 [YOLO-Master README](https://github.com/Tencent/YOLO-Master/blob/main/README.md) 为准。

## 验证与导出

对微调得到的 `best.pt`：

```python
from pathlib import Path
from yolo_master_edge import YoloMasterDetectionPipeline, YoloMasterVariant

DATA = "/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/dataset/data.yaml"
best = Path("artifacts/yolo_master_runs/ft_n/weights/best.pt")

pipe = YoloMasterDetectionPipeline(variant=YoloMasterVariant.V01_N)
pipe.load(model_spec=best)
metrics = pipe.val(DATA)
print(metrics)
onnx_path = pipe.export(format="onnx")
print(onnx_path)
```

或使用 CLI 等价命令（在设置好 `YOLO_MASTER_ROOT` 后）：

```bash
yolo detect val \
  model=artifacts/yolo_master_runs/ft_n/weights/best.pt \
  data=dataset/data.yaml
```

## 训练产出目录说明

```text
artifacts/yolo_master_runs/ft_n/
  weights/best.pt          # 验证集最优
  weights/last.pt          # 最后一轮
  results.csv              # 指标表
  loss_curves.png          # plot_metrics 生成
  metrics_curves.png
  confusion_matrix*.png
  Box*.png                 # P/R/F1/PR 曲线
  val_batch*_labels.jpg    # 验证集可视化
  val_batch*_pred.jpg
artifacts/train_logs/      # train_pipeline 的 tee 日志
```

## 常见问题

| 现象 | 处理 |
|------|------|
| 找不到图片 | `data.yaml` 的 `path` 用 WSL `/mnt/d/...`，勿用 `D:\` |
| `import ultralytics` 不是 YOLO-Master | 在 `YOLO-Master/` 下 `pip install -e .`，勿单独 `pip install ultralytics` |
| CUDA unavailable | 运行 `scripts/setup_cuda_torch.sh` 或升级 NVIDIA 驱动 |
| OOM | 减小 `--batch`、`--imgsz`，或 `extra_train_kwargs` 启用 LoRA |
| 类别数错误 | `names` 须含 0–118；标签 class id ≤ 118 |

## 延迟粗测

```python
stats = pipe.benchmark_predict("path/to/image.jpg", runs=20, warmup=2)
# stats: mean_s / min_s / max_s
```

## 自定义权重路径

构造时传入 `model_spec=`，或之后调用 `load(model_spec=...)`，支持本地 `.pt` 绝对/相对路径；裸文件名（如 `yolo_master_n.pt`）在未对应本地文件时交由上游解析（通常用于自动拉取）。

---

更上层的仓库总览（含 `llm_edge`）见上一级目录的 [README.md](../README.md)。
