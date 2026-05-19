"""
YOLOv8 工具函数模块
提供各种辅助功能
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Union
import cv2
import numpy as np
import yaml


class YOLOUtils:
    """
    YOLO工具类
    提供数据预处理、标注转换、可视化等功能
    """
    
    @staticmethod
    def convert_coco_to_yolo(coco_json_path, output_dir, image_dir):
        """
        将COCO格式转换为YOLO格式
        
        参数:
            coco_json_path: COCO标注文件路径
            output_dir: YOLO标注输出目录
            image_dir: 图片目录
        """
        import json
        
        print(f"转换COCO格式: {coco_json_path}")
        
        # 读取COCO数据
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建类别映射
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        class_names = list(categories.values())
        
        # 构建图像信息映射
        images_info = {img['id']: img for img in coco_data['images']}
        
        # 按图像ID组织标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # 转换为YOLO格式
        converted_count = 0
        
        for img_id, img_info in images_info.items():
            img_name = Path(img_info['file_name']).stem
            img_width = img_info['width']
            img_height = img_info['height']
            
            yolo_annotations = []
            
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    # COCO bbox: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    category_id = ann['category_id']
                    
                    # 转换为YOLO格式 (归一化中心点坐标和宽高)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # 获取类别索引
                    cat_ids = list(categories.keys())
                    class_idx = cat_ids.index(category_id)
                    
                    yolo_annotations.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 保存YOLO标注文件
            label_file = output_dir / f"{img_name}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        print(f"转换完成: {converted_count} 张图片")
        print(f"类别数: {len(class_names)}")
        print(f"类别: {class_names}")
        
        return class_names
    
    @staticmethod
    def convert_voc_to_yolo(xml_dir, output_dir, image_dir, class_names):
        """
        将VOC格式(XML)转换为YOLO格式
        
        参数:
            xml_dir: VOC XML文件目录
            output_dir: YOLO标注输出目录
            image_dir: 图片目录
            class_names: 类别名称列表
        """
        import xml.etree.ElementTree as ET
        
        print(f"转换VOC格式: {xml_dir}")
        
        xml_dir = Path(xml_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        
        for xml_file in xml_dir.glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 获取图像尺寸
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            yolo_annotations = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue
                
                class_idx = class_names.index(class_name)
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为YOLO格式
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_annotations.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 保存
            output_file = output_dir / f"{xml_file.stem}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        print(f"转换完成: {converted_count} 个XML文件")
        return converted_count
    
    @staticmethod
    def visualize_annotations(image_path, label_path, class_names, save_path=None, show=True):
        """
        可视化YOLO标注
        
        参数:
            image_path: 图片路径
            label_path: YOLO标注文件路径
            class_names: 类别名称列表
            save_path: 保存路径 (可选)
            show: 是否显示
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return
        
        img_height, img_width = image.shape[:2]
        
        # 颜色列表
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        # 读取标注
        if Path(label_path).exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_idx = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 计算左上角和右下角
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # 确保在图像范围内
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # 绘制
                color = colors[class_idx % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 标签
                class_name = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
                label = f"{class_name}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存或显示
        if save_path:
            cv2.imwrite(str(save_path), image)
            print(f"保存到: {save_path}")
        
        if show:
            cv2.imshow("Annotations", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image
    
    @staticmethod
    def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        划分训练集、验证集、测试集
        
        参数:
            image_dir: 图片目录
            label_dir: 标注目录
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        random.seed(seed)
        
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        output_dir = Path(output_dir)
        
        # 创建输出目录结构
        splits = ['train', 'val', 'test']
        for split in splits:
            (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片
        image_files = list(image_dir.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # 随机打乱
        random.shuffle(image_files)
        
        # 划分
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits_dict = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # 复制文件
        for split, files in splits_dict.items():
            print(f"{split}: {len(files)} 张图片")
            
            for img_file in files:
                # 复制图片
                shutil.copy(img_file, output_dir / 'images' / split / img_file.name)
                
                # 复制标注
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy(label_file, output_dir / 'labels' / split / label_file.name)
        
        print(f"\n数据集划分完成!")
        print(f"总图片数: {total}")
        
        return output_dir
    
    @staticmethod
    def calculate_dataset_statistics(image_dir, label_dir, class_names):
        """
        计算数据集统计信息
        
        参数:
            image_dir: 图片目录
            label_dir: 标注目录
            class_names: 类别名称列表
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        stats = {
            'total_images': 0,
            'total_objects': 0,
            'objects_per_class': {name: 0 for name in class_names},
            'image_sizes': [],
            'object_sizes': []
        }
        
        for img_file in image_dir.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            stats['total_images'] += 1
            
            # 读取图片获取尺寸
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                stats['image_sizes'].append((w, h))
            
            # 读取标注
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_idx = int(parts[0])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        stats['total_objects'] += 1
                        
                        if class_idx < len(class_names):
                            stats['objects_per_class'][class_names[class_idx]] += 1
                        
                        stats['object_sizes'].append((width, height))
        
        # 计算平均值
        if stats['image_sizes']:
            avg_w = sum(s[0] for s in stats['image_sizes']) / len(stats['image_sizes'])
            avg_h = sum(s[1] for s in stats['image_sizes']) / len(stats['image_sizes'])
        else:
            avg_w = avg_h = 0
        
        # 打印统计
        print("="*50)
        print("数据集统计信息")
        print("="*50)
        print(f"总图片数: {stats['total_images']}")
        print(f"总目标数: {stats['total_objects']}")
        print(f"平均每图目标数: {stats['total_objects']/stats['total_images']:.2f}" if stats['total_images'] > 0 else "N/A")
        print(f"平均图像尺寸: {avg_w:.0f}x{avg_h:.0f}")
        print(f"\n各类别目标数:")
        for name, count in stats['objects_per_class'].items():
            print(f"  {name}: {count}")
        
        return stats


class AugmentationUtils:
    """
    数据增强工具类
    """
    
    @staticmethod
    def mosaic_augmentation(images, labels, output_size=640):
        """
        Mosaic增强: 将4张图片拼成1张
        
        参数:
            images: 图片列表 (4张)
            labels: 标注列表
            output_size: 输出尺寸
        """
        assert len(images) == 4, "需要4张图片"
        
        # 创建画布
        mosaic_img = np.zeros((output_size * 2, output_size * 2, 3), dtype=np.uint8)
        mosaic_labels = []
        
        # 4个位置
        positions = [
            (0, 0),                      # 左上
            (output_size, 0),            # 右上
            (0, output_size),            # 左下
            (output_size, output_size)   # 右下
        ]
        
        for i, (img, label) in enumerate(zip(images, labels)):
            h, w = img.shape[:2]
            
            # 计算缩放
            scale = min(output_size / w, output_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 缩放图片
            resized = cv2.resize(img, (new_w, new_h))
            
            # 放置位置
            dx, dy = positions[i]
            mosaic_img[dy:dy+new_h, dx:dx+new_w] = resized
            
            # 调整标注坐标
            for l in label:
                class_idx, x_center, y_center, bw, bh = l
                
                # 转换到mosaic坐标
                new_x = x_center * new_w + dx
                new_y = y_center * new_h + dy
                new_bw = bw * new_w / (output_size * 2)
                new_bh = bh * new_h / (output_size * 2)
                new_x_center = new_x / (output_size * 2)
                new_y_center = new_y / (output_size * 2)
                
                mosaic_labels.append([class_idx, new_x_center, new_y_center, new_bw, new_bh])
        
        return mosaic_img, mosaic_labels
    
    @staticmethod
    def mixup_augmentation(img1, labels1, img2, labels2, alpha=0.5):
        """
        MixUp增强: 混合两张图片
        
        参数:
            img1, img2: 两张图片
            labels1, labels2: 对应的标注
            alpha: 混合系数
        """
        # 确保尺寸一致
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h))
        
        # 混合图片
        mixed_img = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
        
        # 合并标注
        mixed_labels = labels1 + labels2
        
        return mixed_img, mixed_labels


def check_dataset_format(data_dir):
    """
    检查数据集格式是否正确
    
    参数:
        data_dir: 数据集目录
    """
    data_dir = Path(data_dir)
    
    print(f"\n检查数据集: {data_dir}")
    print("="*50)
    
    issues = []
    
    # 检查目录结构
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in required_dirs:
        dir_path = data_dir / d
        if not dir_path.exists():
            issues.append(f"缺少目录: {d}")
        else:
            files = list(dir_path.iterdir())
            print(f"✓ {d}: {len(files)} 个文件")
    
    # 检查图片和标注是否匹配
    splits = ['train', 'val']
    for split in splits:
        img_dir = data_dir / 'images' / split
        label_dir = data_dir / 'labels' / split
        
        if not img_dir.exists() or not label_dir.exists():
            continue
        
        img_files = set(f.stem for f in img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
        label_files = set(f.stem for f in label_dir.glob('*.txt'))
        
        # 检查是否有图片没有标注
        missing_labels = img_files - label_files
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} 张图片缺少标注")
        
        # 检查是否有标注没有图片
        missing_images = label_files - img_files
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} 个标注缺少图片")
        
        # 检查空标注文件
        empty_labels = []
        for label_file in label_dir.glob('*.txt'):
            if label_file.stat().st_size == 0:
                empty_labels.append(label_file.name)
        
        if empty_labels:
            issues.append(f"{split}: {len(empty_labels)} 个空标注文件")
    
    # 报告问题
    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ 数据集格式正确!")
    
    return len(issues) == 0


if __name__ == '__main__':
    # 示例用法
    print("YOLO工具函数模块")
    print("导入方式: from utils import YOLOUtils, AugmentationUtils")