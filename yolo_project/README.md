# YOLOv8 目标检测项目

一个完整的 YOLOv8 目标检测项目，包含训练、检测、验证和模型导出功能。

## 项目结构

```
yolo_project/
├── train.py              # 训练脚本
├── detect.py             # 检测/推理脚本
├── val.py                # 验证/评估脚本
├── export.py             # 模型导出脚本
├── prepare_data.py       # 数据准备脚本
├── utils.py              # 工具函数模块
├── quickstart.py         # 快速开始向导
├── requirements.txt      # 依赖列表
├── data/                 # 数据目录
│   └── dataset_template.yaml  # 数据集配置模板
└── runs/                 # 训练结果输出目录
    ├── train/            # 训练结果
    ├── detect/           # 检测结果
    └── val/              # 验证结果
```

## 快速开始

### 1. 环境准备

```bash
# 克隆或下载项目后，进入项目目录
cd yolo_project

# 安装依赖
pip install -r requirements.txt

# 或使用快速开始脚本
python quickstart.py
```

### 2. 准备数据

```bash
# 下载 COCO128 示例数据集 (128张图片，80个类别)
python prepare_data.py --dataset coco128

# 或查看推荐的数据集列表
python prepare_data.py --dataset list

# 或创建自定义数据集结构
python prepare_data.py --dataset custom
```

### 3. 训练模型

```bash
# 基本训练
python train.py

# 使用自定义数据集训练
# 1. 修改 train.py 中的 data_yaml 路径
# 2. 运行训练
python train.py

# 或使用命令行参数 (需要修改脚本支持)
```

训练参数说明：
- `model_size`: 模型大小 (n/s/m/l/x)
- `epochs`: 训练轮数
- `batch`: 批次大小
- `imgsz`: 输入图像大小
- `device`: 训练设备

### 4. 运行检测

```bash
# 检测图片
python detect.py --source path/to/image.jpg

# 检测视频
python detect.py --source path/to/video.mp4

# 使用摄像头检测
python detect.py --source 0

# 批量检测目录中的图片
python detect.py --source path/to/images/

# 使用自定义模型
python detect.py --source image.jpg --model runs/train/exp/weights/best.pt

# 调整置信度阈值
python detect.py --source image.jpg --conf 0.5
```

### 5. 模型验证

```bash
# 验证模型性能
python val.py --model runs/train/exp/weights/best.pt --data data/coco128.yaml

# 基准测试 (测试不同配置)
python val.py --model yolov8n.pt --data data/coco128.yaml --benchmark

# 对比多个模型
python val.py --model model1.pt model2.pt model3.pt --data data/coco128.yaml --compare
```

### 6. 导出模型

```bash
# 导出为 ONNX 格式
python export.py --model runs/train/exp/weights/best.pt --format onnx

# 导出为 TensorRT 格式 (需要GPU)
python export.py --model best.pt --format engine

# 导出为 TFLite 格式 (移动端)
python export.py --model best.pt --format tflite

# 查看所有支持的格式
python export.py --model best.pt --list

# 查看部署场景推荐
python export.py --recommend
```

## 数据集推荐

### 入门级

| 数据集 | 图片数 | 类别数 | 大小 | 适用场景 |
|--------|--------|--------|------|----------|
| **COCO128** | 128 | 80 | ~20MB | 快速测试、学习入门 |
| **VOC2012** | ~17K | 20 | ~2GB | 经典数据集、算法对比 |

### 进阶级

| 数据集 | 图片数 | 类别数 | 大小 | 适用场景 |
|--------|--------|--------|------|----------|
| **COCO2017** | 118K | 80 | ~25GB | 标准benchmark、学术研究 |
| **VisDrone** | ~10K | 10 | ~5GB | 小目标检测、无人机视角 |
| **GlobalWheat** | ~3K | 1 | ~1GB | 农业检测、实例分割 |

### 高级/特殊场景

| 数据集 | 图片数 | 类别数 | 大小 | 适用场景 |
|--------|--------|--------|------|----------|
| **SKU-110K** | ~118K | 1 | ~15GB | 密集目标检测、零售场景 |
| **OpenImages** | ~9M | 600 | ~500GB | 大规模检测、通用场景 |

## 数据集格式

YOLO格式要求：
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt      # YOLO格式标注
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

YOLO标注格式 (每行一个目标)：
```
<class_id> <x_center> <y_center> <width> <height>
```

示例：
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

## 模型大小选择

| 模型 | 参数量 | FLOPs | 速度 | 适用场景 |
|------|--------|-------|------|----------|
| YOLOv8n | 3.2M | 8.7B | 最快 | 边缘设备、实时应用 |
| YOLOv8s | 11.2M | 28.6B | 快 | 轻量级应用 |
| YOLOv8m | 25.9M | 78.9B | 中等 | 平衡速度和精度 |
| YOLOv8l | 43.7M | 165.2B | 慢 | 高精度需求 |
| YOLOv8x | 68.2M | 257.8B | 最慢 | 最高精度 |

## 常见问题

### Q: 训练时出现 CUDA out of memory
A: 减小 batch size 或图像尺寸：
```python
# train.py 中修改
config = {
    'batch': 8,      # 原来是16
    'imgsz': 416,    # 原来是640
}
```

### Q: 如何训练自定义数据集？
A: 
1. 准备数据为YOLO格式
2. 创建数据集配置文件 (参考 data/dataset_template.yaml)
3. 修改 train.py 中的 data_yaml 路径
4. 运行训练

### Q: 检测速度太慢？
A:
- 使用更小的模型 (yolov8n)
- 减小输入尺寸 (--imgsz 320)
- 使用半精度 (--half)
- 导出为 TensorRT/OpenVINO 格式加速

### Q: 如何提高检测精度？
A:
- 使用更大的模型 (yolov8l/x)
- 增加训练轮数
- 使用数据增强
- 调整置信度阈值

## 工具函数使用

```python
from utils import YOLOUtils, AugmentationUtils

# COCO格式转YOLO格式
YOLOUtils.convert_coco_to_yolo('annotations.json', 'labels', 'images')

# VOC格式转YOLO格式
YOLOUtils.convert_voc_to_yolo('xml_labels', 'yolo_labels', 'images', ['cat', 'dog'])

# 可视化标注
YOLOUtils.visualize_annotations('image.jpg', 'label.txt', ['person', 'car'])

# 划分数据集
YOLOUtils.split_dataset('images', 'labels', 'output', train_ratio=0.8)

# 统计数据集信息
YOLOUtils.calculate_dataset_statistics('images/train', 'labels/train', ['class1', 'class2'])

# 检查数据集格式
from utils import check_dataset_format
check_dataset_format('path/to/dataset')
```

## 部署建议

| 平台 | 推荐格式 | 说明 |
|------|----------|------|
| NVIDIA GPU | TensorRT (.engine) | 最佳性能，需CUDA |
| Intel CPU | OpenVINO | 针对Intel优化 |
| 移动端 iOS | CoreML | Apple原生支持 |
| 移动端 Android | TFLite | 轻量高效 |
| Web浏览器 | ONNX/TFJS | 跨平台兼容 |
| 树莓派/Jetson | NCNN/TFLite | ARM优化 |

## 参考资源

- [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO数据集](https://cocodataset.org/)
- [YOLO格式说明](https://docs.ultralytics.com/datasets/detect/)

## License

本项目仅供学习和研究使用。