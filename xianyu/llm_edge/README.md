# llm_edge
**😓当前NLP的文件只是一个空中楼阁，还不是能和机器人部分对齐的pipeline，仅作为overview**

本目录是仓库里的**语言侧薄封装**，与 [`yolo_master_edge`](../yolo_master_edge) 并列，面向工业场景：

| 能力 | 做什么 | 典型部署 |
|------|--------|----------|
| **指令解析（在线）** | 把口语指令变成结构化 JSON（意图 + 槽位） | 厂区工控机 CPU，BERT 或规则 |
| **Tinytron 微调（离线）** | 对 [Tinytron-Qwen-0.5B](https://huggingface.co/Agnuxo/Tinytron-Qwen-0.5B-Instruct_CODE_Python_Spanish_English_16bit) 做 SFT / DPO | 有 GPU 的开发机 |

一句话：**YOLO 看现场，llm_edge 听懂人话**；解析结果可交给调度 / 机械臂逻辑（需你方后续对接）。

---

## 目录

1. [核心概念（新手必读）](#核心概念新手必读)
2. [仓库结构](#仓库结构)
3. [环境安装](#环境安装)
4. [5 分钟上手：解析一条指令](#5-分钟上手解析一条指令)
5. [输出 JSON 长什么样](#输出-json-长什么样)
6. [数据怎么准备](#数据怎么准备)
7. [训练流程](#训练流程)
8. [Python API](#python-api)
9. [命令行一览](#命令行一览)
10. [与 yolo_master_edge 组合](#与-yolo_master_edge-组合)
11. [环境变量](#环境变量)
12. [常见问题](#常见问题)

---

## 核心概念（新手必读）

### 意图识别（Intent）

判断用户**想干什么**，例如：

| 用户说法 | 意图 `intent` |
|----------|----------------|
| 给我一把扳手 | `retrieve`（获取） |
| 帮我把滚柱放到料箱第三格 | `place`（放置） |
| 把零件移到左侧 | `move`（移动） |
| 抓取螺丝 | `grasp`（抓取） |
| 装配端盖 | `assemble`（装配） |

本包预定义意图见 [`schemas/intents.py`](schemas/intents.py)。

### 槽位填充（Slot Filling）

从句子里抽出**关键实体**，例如工具、物体、方位、位置：

- 输入：`帮我把滚柱放到料箱的第三个格子中`
- 槽位：`object=滚柱`，`target=料箱`，`position=第3格`，`action=place`

### 两条技术路线（不要混用场景）

```text
┌─────────────────────────────────────────────────────────────┐
│  在线推理（工控机 / 边缘端，要快）                              │
│  InstructionParser → 轻量中文 BERT（intent + slot NER）        │
│  无训练权重时 → 规则 fallback（仅 demo，精度有限）              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  离线训练（需 GPU，数据到位后再跑）                              │
│  TinytronFinetunePipeline → SFT → DPO                         │
│  用于把大模型对齐「工业指令 → JSON」风格（不参与默认在线解析）    │
└─────────────────────────────────────────────────────────────┘
```

### SFT 和 DPO 是什么？

- **SFT（Supervised Fine-Tuning）**：用「输入指令 + 标准答案 JSON」教模型模仿正确输出。
- **DPO（Direct Preference Optimization）**：在 SFT 基础上，用「好答案 vs 差答案」让模型更偏好正确格式、更少胡编槽位。

推荐顺序：**先 SFT，再 DPO**；DPO 依赖 SFT 产出的 checkpoint。

---

## 仓库结构

```text
llm_edge/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── config.py                 # 训练 / 推理超参 dataclass
├── presets.py                # 默认模型 ID、artifacts 路径
├── schemas/                  # 意图枚举、槽位名、IndustrialCommand
├── data/                     # JSONL 数据集加载与校验（接口已就绪）
├── nlp/                      # BERT 解析 + 规则 fallback + 训练 + ONNX 导出
├── llm/                      # Tinytron SFT / DPO
├── scripts/                  # 命令行入口
└── examples/                 # *.jsonl.example 格式说明（无真实样本）
```

---

## 环境安装

### 要求

- Python **3.10+**（推荐 3.10 / 3.11）
- **指令解析 demo**：CPU 即可
- **BERT / Tinytron 训练**：建议 NVIDIA GPU + CUDA

### 1. 创建虚拟环境（推荐）

```bash
# Windows (PowerShell)
cd D:\dl\xianyu
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / WSL
cd /mnt/d/dl/xianyu
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r xianyu/llm_edge/requirements.txt
```

首次训练 Tinytron 会从 Hugging Face 拉取约 **0.5B** 参数模型，请保证网络可达或已配置镜像。

### 3. 让 Python 能找到本包

本包目录名是 `llm_edge`，需把**其父目录**加入 `PYTHONPATH`（即包含 `llm_edge` 文件夹的那一层）：

```bash
# 仓库布局为: xianyu/llm_edge/  xianyu/yolo_master_edge/
# 则应设置:
export PYTHONPATH=/path/to/xianyu:$PYTHONPATH   # Linux / WSL
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "D:\dl\xianyu\xianyu"
```

验证：

```bash
python -c "from llm_edge import InstructionParser; print('ok')"
```

### 4. Windows 上 OpenMP 报错（可选）

若出现 `libiomp5md.dll already initialized`，可临时设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

---

## 5 分钟上手：解析一条指令

无需下载模型、无需 GPU，使用**规则解析**即可体验完整 JSON 输出：

```bash
python -m llm_edge.scripts.parse_demo --text "帮我把滚柱放到料箱的第三个格子中"
```

期望字段包括：`intent`、`action`、`object`、`target`、`position` 等。

指定已训练好的 BERT 权重（路径换成你的 checkpoint 目录）：

```bash
python -m llm_edge.scripts.parse_demo \
  --text "取出左侧的扳手" \
  --intent-ckpt artifacts/nlp_runs/intent \
  --slot-ckpt artifacts/nlp_runs/slot
```

强制只用规则（调试关键词表）：

```bash
python -m llm_edge.scripts.parse_demo --text "取出左侧的扳手" --rule-only
```

---

## 输出 JSON 长什么样

解析结果类型为 [`IndustrialCommand`](schemas/instruction.py)，序列化示例：

```json
{
  "raw_text": "帮我把滚柱放到料箱的第三个格子中",
  "intent": "place",
  "action": "place",
  "object": "滚柱",
  "target": "料箱",
  "position": "第3格",
  "slots": {
    "action": "place",
    "object": "滚柱",
    "target": "料箱",
    "position": "第3格"
  }
}
```

| 字段 | 含义 |
|------|------|
| `intent` | 意图类别（见上表） |
| `action` | 动作动词归纳 |
| `object` | 操作对象（零件、滚柱等） |
| `tool` | 工具（扳手、螺丝刀等） |
| `target` | 目标容器 / 工位（料箱等） |
| `position` | 格位、点位描述 |
| `direction` | 方位（左侧、右侧） |
| `slots` | 上述槽位的字典汇总，便于下游统一读取 |

---

## 数据怎么准备

本仓库**不包含真实训练数据**，只提供格式契约与校验。模板见 [`examples/`](examples/)：

| 文件 | 用途 |
|------|------|
| `sft.jsonl.example` | Tinytron SFT |
| `dpo.jsonl.example` | Tinytron DPO |
| `industrial.jsonl.example` | BERT 意图 + 槽位 |

### SFT（每行一条 JSON）

```json
{"messages": [
  {"role": "system", "content": "将工业口语指令解析为 JSON。"},
  {"role": "user", "content": "帮我把滚柱放到料箱的第三个格子中"},
  {"role": "assistant", "content": "{\"intent\":\"place\",\"action\":\"place\",\"object\":\"滚柱\",\"target\":\"料箱\",\"position\":\"第3格\"}"}
]}
```

保存为例如 `data/sft_train.jsonl`，每行一条，不要用数组包一整文件。

### DPO（偏好对）

```json
{"prompt": "帮我把滚柱放到料箱的第三个格子中", "chosen": "{\"intent\":\"place\",\"object\":\"滚柱\",\"target\":\"料箱\",\"position\":\"第3格\"}", "rejected": "{\"intent\":\"place\",\"object\":\"料箱\",\"target\":\"滚柱\"}"}
```

`chosen` 为更好答案，`rejected` 为常见错误（槽位颠倒、漏填等）。

### Industrial（BERT 标注）

```json
{"text": "帮我把滚柱放到料箱的第三个格子中", "intent": "place", "slots": {"action": "place", "object": "滚柱", "target": "料箱", "position": "第3格"}}
```

槽位名建议与 [`schemas/slots.py`](schemas/slots.py) 一致：`action`、`object`、`tool`、`target`、`position`、`direction`。

---

## 训练流程

### 路线 A：BERT 指令解析（边缘部署推荐）

```text
准备 industrial.jsonl
        ↓
train_nlp intent  →  artifacts/nlp_runs/intent/
train_nlp slot    →  artifacts/nlp_runs/slot/
        ↓
parse_demo / InstructionParser（加载上述目录）
        ↓
（可选）导出 ONNX → nlp/export.py
```

```bash
# 意图分类
python -m llm_edge.scripts.train_nlp intent --data data/industrial_train.jsonl --epochs 5

# 槽位序列标注（同一标注文件）
python -m llm_edge.scripts.train_nlp slot --data data/industrial_train.jsonl --epochs 5
```

默认 backbone：`hfl/chinese-bert-wwm-ext`（可在代码或环境变量中改为更小的 `hfl/rbt3`）。

### 路线 B：Tinytron SFT → DPO（需 GPU）

```text
准备 sft.jsonl
        ↓
train_sft  →  artifacts/tinytron_runs/sft/
        ↓
准备 dpo.jsonl
        ↓
train_dpo --sft-checkpoint ...  →  artifacts/tinytron_runs/dpo/
```

```bash
python -m llm_edge.scripts.train_sft --data data/sft_train.jsonl --epochs 3 --batch-size 4

python -m llm_edge.scripts.train_dpo \
  --sft-checkpoint artifacts/tinytron_runs/sft \
  --data data/dpo_train.jsonl \
  --beta 0.1
```

默认开启 **LoRA** 以降低显存；全参微调：

```bash
python -m llm_edge.scripts.train_sft --data data/sft_train.jsonl --no-lora
```

未传 `--data` 时，脚本会打印 JSONL 格式说明并以退出码 `1` 结束，**不会**静默失败。

### 训练产物默认路径

| 阶段 | 默认目录 |
|------|----------|
| SFT | `artifacts/tinytron_runs/sft/` |
| DPO | `artifacts/tinytron_runs/dpo/` |
| BERT intent | `artifacts/nlp_runs/intent/` |
| BERT slot | `artifacts/nlp_runs/slot/` |

可通过环境变量 `LLM_EDGE_ARTIFACTS` 修改根目录。

---

## Python API

### 解析指令

```python
from llm_edge import InstructionParser, InferenceConfig
from pathlib import Path

# 无 checkpoint：自动规则 fallback
parser = InstructionParser()
cmd = parser.parse("帮我把滚柱放到料箱的第三个格子中")
print(cmd.model_dump_json_ready())

# 加载微调后的 BERT
cfg = InferenceConfig(
    intent_ckpt=Path("artifacts/nlp_runs/intent"),
    slot_ckpt=Path("artifacts/nlp_runs/slot"),
    device="cpu",
)
parser = InstructionParser(cfg)
cmd = parser.parse("取出左侧的扳手")
```

### Tinytron 训练编排

```python
from pathlib import Path
from llm_edge import TinytronFinetunePipeline, TinytronTrainConfig, DpoTrainConfig

pipe = TinytronFinetunePipeline(
    sft_config=TinytronTrainConfig(epochs=3, batch_size=4),
    dpo_config=DpoTrainConfig(beta=0.1),
)

sft_dir = pipe.run_sft(Path("data/sft_train.jsonl"))
dpo_dir = pipe.run_dpo(sft_dir, Path("data/dpo_train.jsonl"))
```

### 加载 JSONL 数据集（校验用）

```python
from pathlib import Path
from llm_edge.data import JsonlIndustrialDataset, JsonlSFTDataset

ds = JsonlIndustrialDataset.from_path(Path("data/industrial_train.jsonl"))
print(len(ds), ds[0])
```

---

## 命令行一览

| 命令 | 作用 |
|------|------|
| `python -m llm_edge.scripts.parse_demo --text "..."` | 单条指令解析 demo |
| `python -m llm_edge.scripts.train_nlp intent --data ...` | BERT 意图训练 |
| `python -m llm_edge.scripts.train_nlp slot --data ...` | BERT 槽位训练 |
| `python -m llm_edge.scripts.train_sft --data ...` | Tinytron SFT |
| `python -m llm_edge.scripts.train_dpo --sft-checkpoint ... --data ...` | Tinytron DPO |

查看格式说明：

```bash
python -m llm_edge.scripts.train_sft --help
python -m llm_edge.scripts.train_nlp --help
```

---

## 与 yolo_master_edge 组合

典型工业流水线（概念示意）：

```text
相机图像 ──→ yolo_master_edge（检测零件/工具位置）
                    │
用户语音/文本 ──→ llm_edge（解析「做什么」）
                    │
                    ▼
              调度 / PLC / 机械臂（你的业务代码）
```

本包**只负责** `文本 → IndustrialCommand JSON`，不直接控制电机；请将 `cmd.slots` 与 YOLO 检测框在业务层做关联（例如「左侧扳手」对应画面左侧检测框）。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PYTHONPATH` | （需自行设置） | 指向包含 `llm_edge` 的父目录 |
| `LLM_EDGE_ARTIFACTS` | `artifacts` | 训练输出根目录 |
| `TINYTRON_HF_ID` | Agnuxo/Tinytron-Qwen-0.5B-... | Hugging Face 模型 ID |
| `LLM_EDGE_BERT_BACKBONE` | `hfl/chinese-bert-wwm-ext` | BERT 预训练名 |
| `KMP_DUPLICATE_LIB_OK` | — | Windows OpenMP 冲突临时 workaround |

---

## 常见问题

### `ModuleNotFoundError: No module named 'llm_edge'`

未设置 `PYTHONPATH`。应指向 **`xianyu`** 这一层（与 `yolo_master_edge` 同级），不是 `llm_edge` 内部。

### `FileNotFoundError: 未提供数据路径`

训练脚本必须加 `--data your.jsonl`。参考 [`examples/`](examples/) 新建自己的 jsonl 文件。

### 规则解析结果不准

规则仅用于 demo。产线请标注 `industrial.jsonl` 并训练 BERT，再用 `--intent-ckpt` / `--slot-ckpt` 加载。

### Tinytron 训练显存不足

- 减小 `--batch-size`
- 保持默认 LoRA（不要加 `--no-lora`）
- 在 `TinytronTrainConfig` 中增大 `gradient_accumulation_steps`

### 中文在 Windows 终端显示乱码

属控制台编码问题，可改用：

```bash
chcp 65001
python -m llm_edge.scripts.parse_demo --text "你的指令"
```

或在 Python 脚本内直接写 Unicode 字符串验证。

### Tinytron 训练后如何部署？

本包在线路径默认走 BERT。Tinytron 训练权重可用于：

- Hugging Face `transformers` 本地推理
- 官方文档中的 [Ollama](https://huggingface.co/Agnuxo/Tinytron-Qwen-0.5B-Instruct_CODE_Python_Spanish_English_16bit) 部署

与 BERT 路径相互独立，按场景选型即可。

---

## 模块索引（查代码时）

| 模块 | 说明 |
|------|------|
| [`nlp/parser.py`](nlp/parser.py) | `InstructionParser` 在线入口 |
| [`nlp/rule_parser.py`](nlp/rule_parser.py) | 规则 fallback |
| [`nlp/train.py`](nlp/train.py) | `train_intent` / `train_slots` |
| [`nlp/export.py`](nlp/export.py) | ONNX 导出与延迟 benchmark |
| [`llm/pipeline.py`](llm/pipeline.py) | `TinytronFinetunePipeline` |
| [`data/protocols.py`](data/protocols.py) | JSONL 记录类型与校验 |
| [`config.py`](config.py) | 所有 dataclass 配置 |

---

## 许可证与上游模型

- Tinytron 模型见 Hugging Face 模型卡（Apache 2.0，以官方为准）。
- `hfl/chinese-bert-wwm-ext` 等预训练模型遵循其各自许可证。

商用与闭源分发请自行做法务评估。

---

更上层的仓库说明（含 `yolo_master_edge`）见同级目录文档；视觉检测细节请参阅 [`yolo_master_edge/README.md`](../yolo_master_edge/README.md)。
