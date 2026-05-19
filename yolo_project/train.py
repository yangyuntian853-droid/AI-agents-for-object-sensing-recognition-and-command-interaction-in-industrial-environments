"""
YOLOv8 训练脚本
用于目标检测模型的训练
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def check_environment():
    """检查环境配置"""
    print("=" * 50)
    print("YOLOv8 训练环境检查")
    print("=" * 50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    
    # 检查 ultralytics
    try:
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        print("警告: 未安装 ultralytics，正在安装...")
        os.system("pip install ultralytics")
    
    print("=" * 50)


def train_model(data_yaml, model_size='n', epochs=100, batch=16, imgsz=640, device=''):
    """
    训练YOLOv8模型
    
    参数:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像大小
        device: 训练设备 (''=auto, '0'=GPU 0, 'cpu'=CPU)
    """
    
    # 选择预训练模型
    model_name = f'yolov8{model_size}.pt'
    print(f"\n加载预训练模型: {model_name}")
    
    # 加载模型
    model = YOLO(model_name)
    
    # 训练参数
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'device': device if device else ('0' if torch.cuda.is_available() else 'cpu'),
        'workers': 8,
        'patience': 20,  # 早停耐心值
        'save': True,
        'project': 'runs/train',
        'name': f'exp_{model_size}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',  # 自动选择优化器 (SGD/AdamW)
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,  # 余弦学习率调度
        'close_mosaic': 10,  # 最后10轮关闭mosaic增强
        'resume': False,
        'amp': True,  # 自动混合精度
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.01,  # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,  # 生成训练图表
    }
    
    print("\n开始训练...")
    print(f"数据配置: {data_yaml}")
    print(f"模型大小: yolov8{model_size}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch}")
    print(f"图像大小: {imgsz}")
    print(f"训练设备: {train_args['device']}")
    
    # 开始训练
    results = model.train(**train_args)
    
    print("\n训练完成!")
    print(f"最佳模型权重: {results.best}")
    
    return results


def main():
    """主函数"""
    check_environment()
    
    # 数据集配置文件路径
    # 这里使用示例，你可以修改为你自己的数据集
    data_yaml = 'data/dataset.yaml'
    
    # 检查数据集文件是否存在
    if not os.path.exists(data_yaml):
        print(f"\n警告: 找不到数据集配置文件 {data_yaml}")
        print("请确保:")
        print("1. 已下载并准备数据集")
        print("2. 已创建 data/dataset.yaml 配置文件")
        print("\n你可以使用以下命令下载示例数据集:")
        print("  python prepare_data.py --dataset coco128")
        return
    
    # 训练配置
    config = {
        'data_yaml': data_yaml,
        'model_size': 'n',      # 可选: 'n', 's', 'm', 'l', 'x'
        'epochs': 100,          # 训练轮数
        'batch': 16,            # 批次大小 (根据GPU内存调整)
        'imgsz': 640,           # 输入图像大小
        'device': '',           # 留空自动选择, '0'使用GPU, 'cpu'使用CPU
    }
    
    # 开始训练
    results = train_model(**config)
    
    print("\n" + "=" * 50)
    print("训练总结:")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"精确率: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"召回率: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print("=" * 50)


if __name__ == '__main__':
    main()