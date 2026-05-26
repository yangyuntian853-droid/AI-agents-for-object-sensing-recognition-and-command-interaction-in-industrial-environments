#!/usr/bin/env python3
"""
数据集结构检查脚本
"""
import os
from pathlib import Path

def check_dataset():
    # 检查数据集结构
    coco_path = Path('data/coco128')
    print('=' * 50)
    print('YOLOv8 数据集结构检查')
    print('=' * 50)
    print(f'数据集路径: {coco_path.absolute()}')
    print(f'路径存在: {coco_path.exists()}')

    if coco_path.exists():
        print('\n数据集目录内容:')
        for item in coco_path.iterdir():
            print(f'  - {item.name}')
        
        # 检查图片和标注
        images_path = coco_path / 'images' / 'train2017'
        labels_path = coco_path / 'labels' / 'train2017'
        
        print(f'\n图片目录: {images_path}')
        print(f'  存在: {images_path.exists()}')
        if images_path.exists():
            imgs = list(images_path.glob('*.jpg'))
            print(f'  图片数量: {len(imgs)}')
            if imgs:
                print(f'  示例: {imgs[0].name}')
        
        print(f'\n标注目录: {labels_path}')
        print(f'  存在: {labels_path.exists()}')
        if labels_path.exists():
            labels = list(labels_path.glob('*.txt'))
            print(f'  标注文件数量: {len(labels)}')
            if labels:
                print(f'  示例: {labels[0].name}')
                # 显示第一个标注文件内容
                print(f'\n  标注示例 ({labels[0].name}):')
                with open(labels[0]) as f:
                    content = f.read().strip()
                    for line in content.split('\n')[:3]:
                        print(f'    {line}')
        
        # 检查配置文件
        config_path = Path('data/dataset.yaml')
        print(f'\n配置文件: {config_path.absolute()}')
        print(f'  存在: {config_path.exists()}')
        
        if config_path.exists():
            print('\n✅ 数据集准备完成！可以开始训练')
            print(f'\n运行命令:')
            print(f'  python train.py')
        else:
            print('\n❌ 缺少配置文件')
            
    else:
        # 列出data目录内容
        data_path = Path('data')
        print(f'\ndata目录内容:')
        for item in data_path.iterdir():
            item_type = "(目录)" if item.is_dir() else "(文件)"
            print(f'  - {item.name} {item_type}')
    
    print('=' * 50)

if __name__ == '__main__':
    check_dataset()