#!/usr/bin/env python3
"""
创建示例数据集
生成合成数据用于测试YOLO训练
"""

import os
import sys
import random
import shutil
from pathlib import Path
import cv2
import numpy as np


def create_sample_dataset(output_dir='data/sample', num_train=50, num_val=10):
    """
    创建示例数据集
    
    参数:
        output_dir: 输出目录
        num_train: 训练图片数量
        num_val: 验证图片数量
    """
    print('='*60)
    print('创建示例数据集')
    print('='*60)
    
    output_dir = Path(output_dir)
    
    # 创建目录结构
    dirs = [
        output_dir / 'images' / 'train',
        output_dir / 'images' / 'val',
        output_dir / 'labels' / 'train',
        output_dir / 'labels' / 'val',
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f'创建目录: {d}')
    
    # 类别定义
    classes = ['circle', 'rectangle', 'triangle']
    colors = [
        (255, 100, 100),   # 红色 - circle
        (100, 255, 100),   # 绿色 - rectangle
        (100, 100, 255),   # 蓝色 - triangle
    ]
    
    print(f'\\n类别: {classes}')
    print(f'训练图片: {num_train}张')
    print(f'验证图片: {num_val}张\\n')
    
    # 生成训练集
    print('生成训练集...')
    for i in range(num_train):
        create_image_with_shapes(
            output_dir / 'images' / 'train' / f'train_{i:04d}.jpg',
            output_dir / 'labels' / 'train' / f'train_{i:04d}.txt',
            classes, colors, num_shapes=random.randint(1, 5)
        )
        if (i + 1) % 10 == 0:
            print(f'  进度: {i+1}/{num_train}')
    
    # 生成验证集
    print('\\n生成验证集...')
    for i in range(num_val):
        create_image_with_shapes(
            output_dir / 'images' / 'val' / f'val_{i:04d}.jpg',
            output_dir / 'labels' / 'val' / f'val_{i:04d}.txt',
            classes, colors, num_shapes=random.randint(1, 5)
        )
        if (i + 1) % 5 == 0:
            print(f'  进度: {i+1}/{num_val}')
    
    # 创建数据集配置文件
    create_dataset_yaml(output_dir, classes)
    
    print('\\n' + '='*60)
    print('✓ 示例数据集创建完成!')
    print(f'位置: {output_dir.absolute()}')
    print('='*60)
    print('\\n使用方法:')
    print(f'1. 修改 train.py 中的 data_yaml = \"{output_dir}/sample.yaml\"')
    print('2. 运行 python train.py 开始训练')
    print('\\n或者直接运行:')
    print(f'  python train.py --data {output_dir}/sample.yaml')
    print('='*60)


def create_image_with_shapes(img_path, label_path, classes, colors, num_shapes=3):
    """
    创建一张包含随机形状的图片
    
    参数:
        img_path: 图片保存路径
        label_path: 标注保存路径
        classes: 类别列表
        colors: 颜色列表
        num_shapes: 形状数量
    """
    # 创建空白画布 (640x480)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    labels = []
    
    for _ in range(num_shapes):
        # 随机选择类别
        class_id = random.randint(0, len(classes) - 1)
        color = colors[class_id]
        
        # 随机位置（确保形状在图片内）
        margin = 50
        x = random.randint(margin, 640 - margin)
        y = random.randint(margin, 480 - margin)
        size = random.randint(30, 80)
        
        if classes[class_id] == 'circle':
            # 绘制圆形
            cv2.circle(img, (x, y), size, color, -1)
            # 边界框
            x1, y1 = x - size, y - size
            x2, y2 = x + size, y + size
            
        elif classes[class_id] == 'rectangle':
            # 绘制矩形
            x1, y1 = x - size, y - size
            x2, y2 = x + size, y + size
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
        else:  # triangle
            # 绘制三角形
            pts = np.array([
                [x, y - size],
                [x - size, y + size],
                [x + size, y + size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
            # 边界框
            x1, y1 = x - size, y - size
            x2, y2 = x + size, y + size
        
        # 确保边界框在图片内
        x1 = max(0, min(x1, 639))
        y1 = max(0, min(y1, 479))
        x2 = max(0, min(x2, 639))
        y2 = max(0, min(y2, 479))
        
        # 计算YOLO格式 (归一化中心点坐标和宽高)
        x_center = (x1 + x2) / 2 / 640
        y_center = (y1 + y2) / 2 / 480
        width = (x2 - x1) / 640
        height = (y2 - y1) / 480
        
        labels.append(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
    
    # 添加一些噪声和背景纹理
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # 保存图片
    cv2.imwrite(str(img_path), img)
    
    # 保存标注
    with open(label_path, 'w') as f:
        f.write('\\n'.join(labels))


def create_dataset_yaml(data_dir, class_names):
    """
    创建数据集配置文件
    """
    config = f'''# Sample Dataset Configuration
# 示例数据集 - 包含圆形、矩形、三角形

path: {data_dir.absolute()}  # 数据集根目录（绝对路径）
train: images/train  # 训练集
val: images/val      # 验证集

# 类别
nc: {len(class_names)}
names:
'''
    
    for i, name in enumerate(class_names):
        config += f'  {i}: {name}\\n'
    
    yaml_path = data_dir / 'sample.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(config)
    
    print(f'\\n配置文件创建: {yaml_path}')


def visualize_sample(data_dir='data/sample', num_samples=5):
    """
    可视化数据集中的样本
    
    参数:
        data_dir: 数据集目录
        num_samples: 显示的样本数
    """
    print('\\n显示数据样本...')
    
    data_dir = Path(data_dir)
    img_dir = data_dir / 'images' / 'train'
    label_dir = data_dir / 'labels' / 'train'
    
    img_files = list(img_dir.glob('*.jpg'))[:num_samples]
    
    # 读取类别名称
    yaml_path = data_dir / 'sample.yaml'
    class_names = []
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
            in_names = False
            for line in lines:
                if 'names:' in line:
                    in_names = True
                    continue
                if in_names and ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        class_names.append(parts[1].strip())
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    for img_file in img_files:
        img = cv2.imread(str(img_file))
        label_file = label_dir / f'{img_file.stem}.txt'
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * 640
                    y_center = float(parts[2]) * 480
                    width = float(parts[3]) * 640
                    height = float(parts[4]) * 480
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                    cv2.putText(img, class_name, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示
        cv2.imshow('Sample', img)
        print(f'显示: {img_file.name}，按任意键查看下一张，按ESC退出')
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='创建示例数据集')
    parser.add_argument('--output', type=str, default='data/sample',
                       help='输出目录 (默认: data/sample)')
    parser.add_argument('--train', type=int, default=50,
                       help='训练图片数量 (默认: 50)')
    parser.add_argument('--val', type=int, default=10,
                       help='验证图片数量 (默认: 10)')
    parser.add_argument('--visualize', action='store_true',
                       help='创建后显示样本')
    
    args = parser.parse_args()
    
    # 创建数据集
    create_sample_dataset(args.output, args.train, args.val)
    
    # 可视化样本
    if args.visualize:
        visualize_sample(args.output)


if __name__ == '__main__':
    main()