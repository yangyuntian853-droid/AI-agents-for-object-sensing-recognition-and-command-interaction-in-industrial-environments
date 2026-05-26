#!/usr/bin/env python3
"""
改进的数据集下载脚本
使用Ultralytics内置功能或手动下载
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError


def print_progress(block_num, block_size, total_size):
    """打印下载进度"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    progress = int(percent / 2)
    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
    print(f'\\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)', end='', flush=True)


def download_file(url, dest_path):
    """
    下载文件，带重试机制
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f'下载: {url}')
    print(f'保存到: {dest_path}')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f'尝试 {attempt + 1}/{max_retries}...')
            urlretrieve(url, dest_path, reporthook=print_progress)
            print('\\n下载完成!')
            return True
        except Exception as e:
            print(f'\\n下载失败: {e}')
            if attempt < max_retries - 1:
                print('重试中...')
            else:
                print('达到最大重试次数')
                return False


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f'解压: {zip_path}')
    print(f'到: {extract_to}')
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print('解压完成!')
        
        # 删除zip文件
        zip_path.unlink()
        print(f'删除: {zip_path}')
        
        return True
    except Exception as e:
        print(f'解压失败: {e}')
        return False


def download_coco128_ultralytics():
    """
    使用Ultralytics库下载COCO128
    最推荐的方式
    """
    print('='*60)
    print('方法1: 使用Ultralytics库下载COCO128')
    print('='*60)
    
    try:
        from ultralytics import YOLO
        
        # 创建一个临时模型来触发数据集下载
        print('正在使用Ultralytics下载COCO128...')
        model = YOLO('yolov8n.pt')
        
        # 使用COCO128配置文件路径触发下载
        data_yaml = 'coco128.yaml'
        
        # 尝试验证数据集（这会触发下载）
        print('下载中，请稍候...')
        results = model.val(data=data_yaml, verbose=False)
        
        print('✓ COCO128下载成功!')
        return True
        
    except Exception as e:
        print(f'方法1失败: {e}')
        return False


def download_coco128_manual():
    """
    手动下载COCO128
    备用方案
    """
    print('='*60)
    print('方法2: 手动下载COCO128')
    print('='*60)
    
    # 多个镜像源
    urls = [
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
        'https://ultralytics.com/assets/coco128.zip',
    ]
    
    data_dir = Path('data')
    zip_path = data_dir / 'coco128.zip'
    
    # 尝试每个URL
    for url in urls:
        if download_file(url, zip_path):
            if extract_zip(zip_path, data_dir):
                # 创建配置文件
                create_coco128_config(data_dir)
                print('✓ COCO128准备完成!')
                return True
    
    return False


def download_coco128_from_alternative():
    """
    使用替代下载方式（国内镜像）
    """
    print('='*60)
    print('方法3: 使用替代源下载')
    print('='*60)
    
    # 尝试使用github代理
    urls = [
        'https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
        'https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
    ]
    
    data_dir = Path('data')
    zip_path = data_dir / 'coco128.zip'
    
    for url in urls:
        print(f'尝试: {url}')
        if download_file(url, zip_path):
            if extract_zip(zip_path, data_dir):
                create_coco128_config(data_dir)
                print('✓ COCO128准备完成!')
                return True
    
    return False


def create_coco128_config(data_dir):
    """
    创建COCO128的数据集配置文件
    """
    print('创建数据集配置...')
    
    config_content = '''# COCO128 Dataset Configuration
path: ../data/coco128  # dataset root dir
train: images/train2017  # train images
val: images/train2017  # val images (same as train for demo)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
'''
    
    config_path = data_dir / 'coco128.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f'配置文件创建: {config_path}')


def create_sample_dataset():
    """
    创建一个示例数据集（当下载失败时使用）
    使用用户桌面上的图片
    """
    print('='*60)
    print('方法4: 创建示例数据集')
    print('='*60)
    
    import cv2
    import numpy as np
    
    data_dir = Path('data/sample')
    images_dir = data_dir / 'images' / 'train'
    labels_dir = data_dir / 'labels' / 'train'
    val_images_dir = data_dir / 'images' / 'val'
    val_labels_dir = data_dir / 'labels' / 'val'
    
    # 创建目录
    for d in [images_dir, labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 创建示例图片（简单的几何图形）
    print('创建示例图片...')
    
    # 生成训练图片
    for i in range(10):
        # 创建空白图片
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 随机添加形状（模拟目标）
        num_objects = np.random.randint(1, 4)
        labels = []
        
        for j in range(num_objects):
            # 随机类别 0-2 (person, car, bicycle)
            class_id = np.random.randint(0, 3)
            
            # 随机位置
            x = np.random.randint(100, 540)
            y = np.random.randint(100, 380)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            # 绘制矩形
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][class_id]
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            
            # 计算YOLO格式坐标 (归一化)
            x_center = x / 640
            y_center = y / 480
            width = w / 640
            height = h / 480
            
            labels.append(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
        
        # 保存图片
        img_path = images_dir / f'img_{i:03d}.jpg'
        cv2.imwrite(str(img_path), img)
        
        # 保存标签
        label_path = labels_dir / f'img_{i:03d}.txt'
        with open(label_path, 'w') as f:
            f.write('\\n'.join(labels))
    
    # 复制一些到验证集
    for i in range(3):
        import shutil
        shutil.copy(images_dir / f'img_{i:03d}.jpg', val_images_dir / f'val_{i:03d}.jpg')
        shutil.copy(labels_dir / f'img_{i:03d}.txt', val_labels_dir / f'val_{i:03d}.txt')
    
    # 创建配置文件
    config = '''path: ../data/sample
train: images/train
val: images/val

nc: 3
names:
  0: person
  1: car
  2: bicycle
'''
    config_path = data_dir / 'sample.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f'✓ 示例数据集创建完成!')
    print(f'  位置: {data_dir}')
    print(f'  训练图片: 10张')
    print(f'  验证图片: 3张')
    print(f'  类别: person, car, bicycle')
    print(f'\\n使用方法: 修改 train.py 中的 data_yaml = \"data/sample/sample.yaml\"')
    
    return True


def main():
    """主函数"""
    print('='*60)
    print('YOLOv8 数据集下载工具')
    print('='*60)
    
    # 方法1: 使用Ultralytics（最简单）
    if download_coco128_ultralytics():
        print('\\n✓ 数据集准备完成!')
        return
    
    print('\\n方法1失败，尝试其他方法...\\n')
    
    # 方法2: 手动下载
    if download_coco128_manual():
        print('\\n✓ 数据集准备完成!')
        return
    
    print('\\n方法2失败，尝试使用代理...\\n')
    
    # 方法3: 使用代理
    if download_coco128_from_alternative():
        print('\\n✓ 数据集准备完成!')
        return
    
    print('\\n所有下载方法都失败了。')
    print('可能的原因:')
    print('  - 网络连接问题')
    print('  - GitHub访问受限')
    print('  - 防火墙/代理设置')
    print('\\n创建示例数据集供测试使用...')
    
    # 方法4: 创建示例数据集
    create_sample_dataset()


if __name__ == '__main__':
    main()