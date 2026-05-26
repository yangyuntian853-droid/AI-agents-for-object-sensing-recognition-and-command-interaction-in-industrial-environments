"""
数据准备脚本
用于下载和准备YOLO训练所需的数据集
"""

import os
import sys
import argparse
import shutil
import zipfile
import requests
from pathlib import Path
import yaml


def download_file(url, dest_path, chunk_size=8192):
    """
    下载文件并显示进度
    
    参数:
        url: 下载链接
        dest_path: 保存路径
        chunk_size: 分块大小
    """
    print(f"下载: {url}")
    print(f"保存到: {dest_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    print("\n下载完成!")
    return str(dest_path)


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    
    参数:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录
    """
    print(f"\n解压: {zip_path}")
    print(f"到: {extract_to}")
    
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print("解压完成!")
    return str(extract_to)


def download_coco128(data_dir='data'):
    """
    下载COCO128数据集 (128张图片的COCO子集)
    适合快速测试和学习
    """
    print("\n" + "=" * 50)
    print("准备 COCO128 数据集")
    print("=" * 50)
    
    data_dir = Path(data_dir)
    dataset_dir = data_dir / 'coco128'
    
    # 检查是否已存在
    if dataset_dir.exists():
        print(f"数据集已存在: {dataset_dir}")
        print("跳过下载")
        return str(dataset_dir)
    
    # 下载链接 (Ultralytics官方)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    zip_path = data_dir / "coco128.zip"
    
    try:
        # 下载
        download_file(url, zip_path)
        
        # 解压
        extract_zip(zip_path, data_dir)
        
        # 清理
        os.remove(zip_path)
        
        print(f"\nCOCO128 准备完成!")
        print(f"位置: {dataset_dir}")
        print(f"\n数据集信息:")
        print("  - 128张图片")
        print("  - 80个COCO类别")
        print("  - 训练集: 128张")
        
        return str(dataset_dir)
        
    except Exception as e:
        print(f"错误: {e}")
        return None


def create_dataset_yaml(dataset_path, dataset_name, num_classes, class_names, output_dir='data'):
    """
    创建YOLO数据集配置文件
    
    参数:
        dataset_path: 数据集根目录
        dataset_name: 数据集名称
        num_classes: 类别数量
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    print(f"\n创建数据集配置: {dataset_name}")
    
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test' if (dataset_path / 'images/test').exists() else None,
        'nc': num_classes,
        'names': {i: name for i, name in enumerate(class_names)},
    }
    
    # 保存配置文件
    yaml_path = output_dir / f'{dataset_name}.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    print(f"配置文件保存: {yaml_path}")
    return str(yaml_path)


def setup_coco128_yaml(data_dir='data'):
    """创建COCO128的数据集配置"""
    dataset_dir = Path(data_dir) / 'coco128'
    
    # COCO80类别名称
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    yaml_path = create_dataset_yaml(
        dataset_dir,
        'coco128',
        80,
        coco_names,
        data_dir
    )
    
    return yaml_path


def create_custom_dataset_structure(base_dir='data/custom'):
    """
    创建自定义数据集目录结构
    """
    print("\n" + "=" * 50)
    print("创建自定义数据集结构")
    print("=" * 50)
    
    base_dir = Path(base_dir)
    
    # 创建目录结构
    dirs = [
        'images/train',
        'images/val',
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test',
    ]
    
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    
    print(f"目录结构创建完成: {base_dir}")
    print("\n目录结构:")
    for d in dirs:
        print(f"  {d}/")
    
    print("\n使用说明:")
    print("1. 将训练图片放入: images/train/")
    print("2. 将验证图片放入: images/val/")
    print("3. 将标注文件(YOLO格式)放入对应labels/目录")
    print("4. 运行: python prepare_data.py --create-yaml custom")
    
    # 创建示例配置文件模板
    example_yaml = base_dir / 'dataset_template.yaml'
    example_config = """# 自定义数据集配置模板
# 修改以下配置后保存为 dataset.yaml

path: ../custom  # 数据集根目录
train: images/train  # 训练集
val: images/val      # 验证集
test: images/test    # 测试集 (可选)

# 类别数量
nc: 3

# 类别名称
names:
  0: class1
  1: class2
  2: class3
"""
    
    with open(example_yaml, 'w') as f:
        f.write(example_config)
    
    print(f"\n配置模板: {example_yaml}")
    
    return str(base_dir)


def list_available_datasets():
    """列出推荐的数据集"""
    print("\n" + "=" * 60)
    print("推荐的数据集列表")
    print("=" * 60)
    
    datasets = [
        {
            'name': 'COCO128',
            'description': 'COCO数据集前128张图片，适合快速测试',
            'classes': 80,
            'images': 128,
            'size': '~20MB',
            'difficulty': '简单',
            'use_case': '入门学习、快速验证代码',
        },
        {
            'name': 'COCO2017',
            'description': '完整COCO数据集，目标检测标准 benchmark',
            'classes': 80,
            'images': '118K(train) + 5K(val)',
            'size': '~25GB',
            'difficulty': '中等',
            'use_case': '模型训练、学术研究',
        },
        {
            'name': 'VOC2012',
            'description': 'PASCAL VOC 2012，经典目标检测数据集',
            'classes': 20,
            'images': '~17K',
            'size': '~2GB',
            'difficulty': '简单',
            'use_case': '入门学习、算法对比',
        },
        {
            'name': 'VisDrone2019',
            'description': '无人机视角目标检测，包含小目标和密集场景',
            'classes': 10,
            'images': '~10K',
            'size': '~5GB',
            'difficulty': '困难',
            'use_case': '小目标检测、无人机应用',
        },
        {
            'name': 'GlobalWheat2020',
            'description': '全球小麦穗检测，农业应用',
            'classes': 1,
            'images': '~3K',
            'size': '~1GB',
            'difficulty': '中等',
            'use_case': '农业检测、实例分割',
        },
        {
            'name': 'SKU-110K',
            'description': '零售商品检测，极度密集场景',
            'classes': 1,
            'images': '~118K',
            'size': '~15GB',
            'difficulty': '困难',
            'use_case': '密集目标检测、零售应用',
        },
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   描述: {ds['description']}")
        print(f"   类别数: {ds['classes']}")
        print(f"   图片数: {ds['images']}")
        print(f"   大小: {ds['size']}")
        print(f"   难度: {ds['difficulty']}")
        print(f"   适用: {ds['use_case']}")
    
    print("\n" + "=" * 60)
    print("下载地址:")
    print("  - COCO128/COCO2017: https://cocodataset.org/")
    print("  - VOC2012: http://host.robots.ox.ac.uk/pascal/VOC/")
    print("  - VisDrone: https://github.com/VisDrone/")
    print("  - GlobalWheat: https://www.global-wheat.com/")
    print("  - SKU-110K: https://github.com/eg4000/SKU110K_CVPR19")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='准备YOLO训练数据')
    parser.add_argument('--dataset', type=str, choices=['coco128', 'custom', 'list'],
                       default='coco128', help='要准备的数据集')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='数据保存目录')
    parser.add_argument('--create-yaml', type=str,
                       help='为指定数据集创建YAML配置文件')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("YOLO 数据准备工具")
    print("=" * 50)
    
    if args.dataset == 'list':
        list_available_datasets()
    
    elif args.dataset == 'coco128':
        # 下载COCO128
        dataset_path = download_coco128(args.data_dir)
        
        if dataset_path:
            # 创建配置文件
            yaml_path = setup_coco128_yaml(args.data_dir)
            print(f"\n✅ COCO128 准备完成!")
            print(f"   数据路径: {dataset_path}")
            print(f"   配置文件: {yaml_path}")
            print(f"\n可以开始训练了:")
            print(f"  python train.py  # 确保配置指向 {yaml_path}")
    
    elif args.dataset == 'custom':
        # 创建自定义数据集结构
        custom_path = create_custom_dataset_structure(f"{args.data_dir}/custom")
        print(f"\n✅ 自定义数据集结构创建完成!")
        print(f"   路径: {custom_path}")
        print(f"\n下一步:")
        print("  1. 添加你的图片和标注")
        print("  2. 创建 dataset.yaml 配置文件")
        print("  3. 开始训练")


if __name__ == '__main__':
    # 检查依赖
    try:
        import requests
    except ImportError:
        print("安装依赖: pip install requests")
        os.system("pip install requests")
        import requests
    
    main()