"""
YOLOv8 训练脚本
用于目标检测模型的训练（支持GPU/CPU自动切换）
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def check_environment():
    """检查环境配置，包括GPU状态"""
    print("=" * 60)
    print("YOLOv8 训练环境检查")
    print("=" * 60)
    
    # 检查Python版本
    print(f"Python版本: {sys.version.split()[0]}")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"  GPU {i}: {gpu_name}")
            print(f"    显存: {gpu_memory:.2f} GB")
        
        # 检查当前GPU是否可用
        try:
            test_tensor = torch.zeros(1).cuda()
            print("  GPU测试: 通过 ✓")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  GPU测试: 失败 ✗ ({e})")
    else:
        print("警告: CUDA不可用，将使用CPU训练（速度较慢）")
        print("如需使用GPU，请检查:")
        print("  1. 是否安装CUDA Toolkit")
        print("  2. PyTorch是否为CUDA版本")
        print("  3. 显卡驱动是否最新")
    
    # 检查 ultralytics
    try:
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        print("警告: 未安装 ultralytics，正在安装...")
        os.system("pip install ultralytics")
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
    
    print("=" * 60)
    return cuda_available


def get_device(preferred_device=''):
    """
    获取训练设备
    
    参数:
        preferred_device: 首选设备 ('0', '0,1', 'cpu', ''=自动)
    返回:
        设备字符串
    """
    if preferred_device:
        return preferred_device
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            # 多GPU，使用所有可用GPU
            devices = ','.join(str(i) for i in range(gpu_count))
            print(f"检测到多GPU，使用: {devices}")
            return devices
        else:
            print("使用GPU: 0")
            return '0'
    else:
        print("使用CPU训练")
        return 'cpu'


def get_model_path(script_dir, model_size='n', custom_path=None):
    """
    获取模型路径
    优先顺序: 自定义路径 > 项目目录 > 自动下载
    """
    model_name = f'yolov8{model_size}.pt'
    
    # 1. 优先使用自定义路径
    if custom_path:
        custom_path = Path(custom_path)
        if custom_path.exists():
            print(f"使用自定义模型: {custom_path}")
            return str(custom_path)
        else:
            print(f"警告: 自定义模型路径不存在: {custom_path}")
    
    # 2. 检查项目根目录
    local_model = script_dir / model_name
    if local_model.exists():
        print(f"使用本地模型: {local_model}")
        return str(local_model)
    
    # 3. 检查当前工作目录
    cwd_model = Path.cwd() / model_name
    if cwd_model.exists():
        print(f"使用当前目录模型: {cwd_model}")
        return str(cwd_model)
    
    # 4. 使用模型名称（自动下载）
    print(f"使用自动下载模型: {model_name}")
    return model_name


def train_model(data_yaml, model_size='n', epochs=100, batch=16, imgsz=640, 
                device='', model_path=None, workers=8, resume=False):
    """
    训练YOLOv8模型
    
    参数:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        epochs: 训练轮数
        batch: 批次大小（根据GPU显存调整）
        imgsz: 输入图像大小
        device: 训练设备 ('0'=GPU0, '0,1'=多GPU, 'cpu'=CPU, ''=自动)
        model_path: 自定义模型路径
        workers: 数据加载线程数
        resume: 是否从上次中断处继续训练
    """
    
    # 获取项目目录（脚本所在目录）
    script_dir = Path(__file__).parent.resolve()
    print(f"\n项目目录: {script_dir}")
    
    # 获取模型路径
    model_file = get_model_path(script_dir, model_size, model_path)
    
    # 转换数据配置路径为绝对路径
    data_yaml_path = Path(data_yaml).resolve()
    print(f"数据配置: {data_yaml_path}")
    
    # 检查数据配置是否存在
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {data_yaml_path}")
    
    # 确保输出目录存在（项目目录下）
    output_dir = script_dir / 'runs' / 'train'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载模型
    print(f"\n加载模型...")
    try:
        model = YOLO(model_file)
        print(f"模型加载成功: {model_file}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("\n请检查:")
        print(f"1. 模型文件是否存在: {script_dir / f'yolov8{model_size}.pt'}")
        print("2. 网络连接（如需自动下载）")
        print("3. 从 https://github.com/ultralytics/assets/releases 手动下载")
        raise
    
    # 确定训练设备
    train_device = get_device(device)
    
    # 打印训练配置
    print(f"\n训练配置:")
    print(f"  模型: yolov8{model_size}")
    print(f"  数据: {data_yaml_path}")
    print(f"  轮数: {epochs}")
    print(f"  批次: {batch}")
    print(f"  图像大小: {imgsz}")
    print(f"  设备: {train_device}")
    print(f"  线程数: {workers}")
    
    # GPU优化提示
    if train_device != 'cpu' and torch.cuda.is_available():
        print(f"\nGPU优化提示:")
        print(f"  - 使用混合精度训练(AMP)加速")
        print(f"  - 批次大小可根据显存调整")
        print(f"  - 多GPU训练: device='0,1,2,3'")
    
    # 开始训练
    print(f"\n开始训练...")
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=train_device,
            workers=workers,
            project=str(output_dir),
            name=f'exp_{model_size}',
            exist_ok=True,
            resume=resume,
            pretrained=True,
            patience=20,
            save=True,
            plots=True,
            amp=True,  # 自动混合精度（GPU加速）
        )
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        
        # 获取最佳模型路径
        best_model_path = Path(results.best)
        print(f"最佳模型: {best_model_path}")
        
        # 返回训练结果和模型路径
        return {
            'results': results,
            'best_model_path': str(best_model_path),
            'model_size': model_size,
            'device': train_device,
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n错误: GPU显存不足！")
            print(f"建议:")
            print(f"  1. 减小批次大小: batch=8 或 batch=4")
            print(f"  2. 减小图像大小: imgsz=416 或 imgsz=320")
            print(f"  3. 使用更小的模型: model_size='n'")
            print(f"  4. 关闭其他占用显存的程序")
        raise
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise


def main():
    """主函数"""
    # 检查环境
    cuda_available = check_environment()
    
    # 获取项目目录
    script_dir = Path(__file__).parent.resolve()
    
    # 数据配置文件路径（绝对路径）
    data_yaml = script_dir / 'data' / 'dataset.yaml'
    
    # 检查数据配置
    if not data_yaml.exists():
        print(f"\n错误: 找不到数据配置文件: {data_yaml}")
        print("\n请确保:")
        print("1. 已下载数据集: python download_dataset.py")
        print("2. 已创建数据配置: data/dataset.yaml")
        return 1
    
    # ==================== 训练配置 ====================
    config = {
        'data_yaml': str(data_yaml),
        'model_size': 'n',      # n, s, m, l, x （n最小最快，x最大最准）
        'epochs': 100,          # 训练轮数
        'batch': 16,            # 批次大小（根据显存调整: 16, 32, 64）
        'imgsz': 640,           # 图像大小（640标准，416快速）
        'device': '',           # 设备: ''=自动, '0'=GPU0, '0,1'=多GPU, 'cpu'=CPU
        'model_path': None,     # 自定义模型路径，None则自动查找
        'workers': 8,           # 数据加载线程（根据CPU核心数调整）
    }
    
    # 根据GPU显存推荐批次大小
    if cuda_available:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 4:
                config['batch'] = 8
                print(f"\n检测到显存较小 ({gpu_memory:.1f}GB)，建议批次大小: 8")
            elif gpu_memory < 8:
                config['batch'] = 16
                print(f"\n检测到中等显存 ({gpu_memory:.1f}GB)，建议批次大小: 16")
            else:
                config['batch'] = 32
                print(f"\n检测到较大显存 ({gpu_memory:.1f}GB)，建议批次大小: 32")
        except:
            pass
    
    try:
        # 开始训练
        train_result = train_model(**config)
        
        # 打印训练结果
        results = train_result['results']
        print(f"\n训练结果:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
        
        print(f"\n最佳模型已保存到: {train_result['best_model_path']}")
        print(f"\n后续使用:")
        print(f"  检测: python detect.py --model {train_result['best_model_path']} --source image.jpg")
        print(f"  验证: python val.py --model {train_result['best_model_path']} --data {data_yaml}")
        print(f"  导出: python export.py --model {train_result['best_model_path']} --format onnx")
        
        return 0
        
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())