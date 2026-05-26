#!/usr/bin/env python3
"""
YOLOv8 快速开始脚本
一键完成环境检查、数据准备、训练和测试
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              YOLOv8 目标检测 - 快速开始向导                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """检查Python版本"""
    print("[1/5] 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ 需要Python 3.8+, 当前为 {version.major}.{version.minor}")
        return False


def install_dependencies():
    """安装依赖"""
    print("\n[2/5] 安装依赖包...")
    
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("  ✗ 未找到 requirements.txt")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file), 
                       "-q"], check=True)
        print("  ✓ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 安装失败: {e}")
        return False


def download_dataset():
    """下载示例数据集"""
    print("\n[3/5] 准备示例数据集...")
    
    try:
        # 运行数据准备脚本
        result = subprocess.run([sys.executable, "prepare_data.py", "--dataset", "coco128"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ COCO128数据集准备完成")
            return True
        else:
            print(f"  ✗ 数据集准备失败")
            return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False


def quick_train():
    """快速训练示例"""
    print("\n[4/5] 开始快速训练 (10轮, 用于测试)...")
    print("  提示: 按Ctrl+C可随时停止")
    
    try:
        # 快速训练配置
        from ultralytics import YOLO
        
        model = YOLO('yolov8n.pt')
        results = model.train(
            data='data/coco128.yaml',
            epochs=10,
            imgsz=320,
            batch=8,
            patience=5,
            save=True,
            project='runs/quickstart',
            name='exp',
            exist_ok=True,
            verbose=True
        )
        
        print("  ✓ 快速训练完成!")
        print(f"  最佳模型: {results.best}")
        return True
        
    except KeyboardInterrupt:
        print("\n  ! 训练被用户中断")
        return True
    except Exception as e:
        print(f"  ✗ 训练失败: {e}")
        return False


def quick_detect():
    """快速检测示例"""
    print("\n[5/5] 运行检测示例...")
    
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        # 加载模型
        model_path = 'yolov8n.pt'
        if not Path(model_path).exists():
            print("  正在下载预训练模型...")
        
        model = YOLO(model_path)
        
        # 创建测试图片 (如果找不到真实图片)
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 运行检测
        results = model.predict(test_img, verbose=False)
        
        print("  ✓ 检测功能正常")
        print(f"  检测到 {len(results[0].boxes) if results[0].boxes else 0} 个目标")
        return True
        
    except Exception as e:
        print(f"  ✗ 检测失败: {e}")
        return False


def show_next_steps():
    """显示后续步骤"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                        后续步骤                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. 完整训练:                                                 ║
║     python train.py                                           ║
║                                                               ║
║  2. 图片检测:                                                 ║
║     python detect.py --source path/to/image.jpg               ║
║                                                               ║
║  3. 视频检测:                                                 ║
║     python detect.py --source path/to/video.mp4               ║
║                                                               ║
║  4. 摄像头检测:                                               ║
║     python detect.py --source 0                               ║
║                                                               ║
║  5. 模型验证:                                                 ║
║     python val.py --model runs/train/exp/weights/best.pt      ║
║        --data data/coco128.yaml                               ║
║                                                               ║
║  6. 导出模型:                                                 ║
║     python export.py --model runs/train/exp/weights/best.pt   ║
║        --format onnx                                          ║
║                                                               ║
║  7. 查看数据集列表:                                           ║
║     python prepare_data.py --dataset list                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    """主函数"""
    print_banner()
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 询问是否安装依赖
    print("\n是否安装依赖? (y/n): ", end='')
    if input().lower() != 'y':
        print("跳过依赖安装")
    else:
        if not install_dependencies():
            return
    
    # 询问是否下载数据集
    print("\n是否下载COCO128数据集? (y/n): ", end='')
    if input().lower() == 'y':
        download_dataset()
    
    # 询问是否快速训练
    print("\n是否运行快速训练测试? (y/n): ", end='')
    if input().lower() == 'y':
        quick_train()
    
    # 测试检测功能
    print("\n是否测试检测功能? (y/n): ", end='')
    if input().lower() == 'y':
        quick_detect()
    
    # 显示后续步骤
    show_next_steps()
    
    print("\n✓ 快速开始完成! 祝你使用愉快!\n")


if __name__ == '__main__':
    main()