"""
YOLOv8 验证/评估脚本
用于评估训练好的模型性能
"""

import os
import sys
import argparse
from pathlib import Path
import json
import torch
from ultralytics import YOLO


class ModelValidator:
    """
    YOLO模型验证器
    提供全面的模型性能评估
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化验证器
        
        参数:
            model_path: 模型路径
            device: 运行设备
        """
        # 获取项目目录
        self.script_dir = Path(__file__).parent.resolve()
        
        # 解析模型路径
        self.model_path = self._resolve_model_path(model_path)
        self.device = device if device else ('0' if torch.cuda.is_available() else 'cpu')
        
        print(f"加载模型: {self.model_path}")
        print(f"设备: {self.device}")
        
        # 加载模型
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"模型加载完成!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"\n请检查模型路径: {model_path}")
            raise
    
    def _resolve_model_path(self, model_path):
        """解析模型路径"""
        path = Path(model_path)
        
        # 1. 如果是绝对路径且存在，直接使用
        if path.is_absolute() and path.exists():
            return str(path)
        
        # 2. 如果是相对路径且存在，直接使用
        if path.exists():
            return str(path.resolve())
        
        # 3. 检查是否在项目目录下
        project_path = self.script_dir / path
        if project_path.exists():
            return str(project_path)
        
        # 4. 尝试添加weights/best.pt
        if not path.is_absolute():
            weights_path = self.script_dir / path / 'weights' / 'best.pt'
            if weights_path.exists():
                return str(weights_path)
        
        # 5. 返回原始路径（可能是预训练模型名称）
        return model_path
    
    def validate(self, data_yaml, imgsz=640, batch=16, conf=0.001, iou=0.6, 
                 save_json=False, save_hybrid=False, half=False):
        """
        验证模型
        
        参数:
            data_yaml: 数据集配置文件（支持绝对路径）
            imgsz: 输入图像大小
            batch: 批次大小
            conf: 置信度阈值
            iou: NMS IoU阈值
            save_json: 保存结果为JSON
            save_hybrid: 保存混合标签
            half: 使用半精度
        """
        # 确保数据配置使用绝对路径
        data_yaml_path = Path(data_yaml).resolve()
        
        print(f"\n{'='*50}")
        print("开始验证")
        print(f"{'='*50}")
        print(f"数据集: {data_yaml_path}")
        print(f"图像大小: {imgsz}")
        print(f"批次大小: {batch}")
        print(f"置信度阈值: {conf}")
        print(f"IoU阈值: {iou}")
        
        # 设置输出目录（项目目录下的runs/val）
        output_dir = self.script_dir / 'runs' / 'val'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行验证
        try:
            results = self.model.val(
                data=str(data_yaml_path),
                imgsz=imgsz,
                batch=batch,
                conf=conf,
                iou=iou,
                device=self.device,
                save_json=save_json,
                save_hybrid=save_hybrid,
                half=half,
                plots=True,
                project=str(output_dir),
                name='exp',
                exist_ok=True
            )
        except Exception as e:
            print(f"\n验证失败: {e}")
            print(f"\n请检查:")
            print(f"1. 数据配置文件是否存在: {data_yaml_path}")
            print(f"2. 数据集中的图片路径是否正确")
            raise
        
        # 打印结果
        print(f"\n{'='*50}")
        print("验证结果")
        print(f"{'='*50}")
        
        # 主要指标
        metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP75': results.results_dict.get('metrics/mAP75(B)', 0),
            'Precision': results.results_dict.get('metrics/precision(B)', 0),
            'Recall': results.results_dict.get('metrics/recall(B)', 0),
            'F1-Score': results.results_dict.get('metrics/F1(B)', 0),
        }
        
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 速度统计
        print(f"\n推理速度:")
        if hasattr(results, 'speed'):
            for key, value in results.speed.items():
                print(f"  {key}: {value:.2f}ms")
        
        # 类别详细结果
        if hasattr(results, 'box') and results.box is not None:
            print(f"\n各类别AP50:")
            ap50 = results.box.ap50
            if ap50 is not None:
                for i, ap in enumerate(ap50):
                    if ap > 0:  # 只显示有结果的类别
                        class_name = self.model.names.get(i, f'class_{i}')
                        print(f"  {class_name}: {ap:.4f}")
        
        return results
    
    def benchmark(self, data_yaml, imgsz_list=[320, 416, 512, 640, 768, 896, 1024], 
                  batch_list=[1, 8, 16]):
        """
        基准测试 - 测试不同配置下的性能
        
        参数:
            data_yaml: 数据集配置（支持绝对路径）
            imgsz_list: 图像大小列表
            batch_list: 批次大小列表
        """
        # 确保数据配置使用绝对路径
        data_yaml_path = Path(data_yaml).resolve()
        
        # 设置输出目录（项目目录下的runs/benchmark）
        output_dir = self.script_dir / 'runs' / 'benchmark'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*50}")
        print("开始基准测试")
        print(f"{'='*50}")
        
        results = []
        
        for imgsz in imgsz_list:
            for batch in batch_list:
                print(f"\n测试配置: imgsz={imgsz}, batch={batch}")
                
                try:
                    result = self.model.val(
                        data=str(data_yaml_path),
                        imgsz=imgsz,
                        batch=batch,
                        conf=0.001,
                        iou=0.6,
                        device=self.device,
                        plots=False,
                        verbose=False,
                        project=str(output_dir),
                        name=f'benchmark_{imgsz}_{batch}',
                        exist_ok=True
                    )
                    
                    res = {
                        'imgsz': imgsz,
                        'batch': batch,
                        'mAP50': result.results_dict.get('metrics/mAP50(B)', 0),
                        'mAP50-95': result.results_dict.get('metrics/mAP50-95(B)', 0),
                        'speed': result.speed if hasattr(result, 'speed') else {}
                    }
                    results.append(res)
                    
                    print(f"  mAP50: {res['mAP50']:.4f}")
                    print(f"  速度: {res['speed']}")
                    
                except Exception as e:
                    print(f"  错误: {e}")
        
        # 保存结果
        print(f"\n{'='*50}")
        print("基准测试完成")
        print(f"{'='*50}")
        
        # 找到最佳配置
        best_map = max(results, key=lambda x: x['mAP50'])
        print(f"\n最佳mAP50配置:")
        print(f"  图像大小: {best_map['imgsz']}")
        print(f"  批次大小: {best_map['batch']}")
        print(f"  mAP50: {best_map['mAP50']:.4f}")
        
        # 保存到文件
        output_file = output_dir / 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存: {output_file}")
        
        return results


def find_latest_model(script_dir):
    """查找最新的训练模型"""
    runs_dir = script_dir / 'runs' / 'train'
    
    if not runs_dir.exists():
        return None
    
    # 查找所有exp_*目录
    exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    
    if not exp_dirs:
        return None
    
    # 按修改时间排序
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_exp = exp_dirs[0]
    
    # 查找best.pt
    best_model = latest_exp / 'weights' / 'best.pt'
    if best_model.exists():
        return str(best_model)
    
    last_model = latest_exp / 'weights' / 'last.pt'
    if last_model.exists():
        return str(last_model)
    
    return None


def compare_models(model_paths, data_yaml, imgsz=640, batch=16, script_dir=None):
    """
    比较多个模型的性能
    
    参数:
        model_paths: 模型路径列表
        data_yaml: 数据集配置（支持绝对路径）
        imgsz: 图像大小
        batch: 批次大小
        script_dir: 项目目录
    """
    if script_dir is None:
        script_dir = Path(__file__).parent.resolve()
    
    # 设置输出目录
    output_dir = script_dir / 'runs' / 'compare'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("模型对比")
    print(f"{'='*60}")
    
    results = []
    
    for model_path in model_paths:
        print(f"\n评估模型: {model_path}")
        
        validator = ModelValidator(model_path)
        result = validator.validate(
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=0.001,
            iou=0.6,
            plots=False
        )
        
        res = {
            'model': Path(model_path).name,
            'mAP50': result.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': result.results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': result.results_dict.get('metrics/precision(B)', 0),
            'recall': result.results_dict.get('metrics/recall(B)', 0),
        }
        results.append(res)
    
    # 打印对比表
    print(f"\n{'='*60}")
    print("对比结果")
    print(f"{'='*60}")
    print(f"{'模型':<20} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print('-'*60)
    
    for r in results:
        print(f"{r['model']:<20} {r['mAP50']:<10.4f} {r['mAP50-95']:<10.4f} "
              f"{r['precision']:<10.4f} {r['recall']:<10.4f}")
    
    # 保存结果到项目目录
    output_file = output_dir / 'model_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_file}")
    
    # 找出最佳模型
    best = max(results, key=lambda x: x['mAP50'])
    print(f"\n最佳模型: {best['model']}")
    print(f"  mAP50: {best['mAP50']:.4f}")


def main():
    """主函数"""
    # 获取项目目录
    script_dir = Path(__file__).parent.resolve()
    
    # 查找最新的训练模型
    latest_model = find_latest_model(script_dir)
    default_model = latest_model if latest_model else 'yolov8n.pt'
    
    parser = argparse.ArgumentParser(description='YOLO模型验证')
    parser.add_argument('--model', type=str, default=default_model,
                       help=f'模型路径 (默认: {default_model})')
    parser.add_argument('--data', type=str, 
                       default=str(script_dir / 'data' / 'dataset.yaml'),
                       help='数据集配置文件 (YAML)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像大小 (默认: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小 (默认: 16)')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='置信度阈值 (默认: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='IoU阈值 (默认: 0.6)')
    parser.add_argument('--device', type=str, default=None,
                       help='运行设备 (默认: 自动)')
    parser.add_argument('--save-json', action='store_true',
                       help='保存结果为JSON')
    parser.add_argument('--benchmark', action='store_true',
                       help='执行基准测试')
    parser.add_argument('--compare', nargs='+',
                       help='对比多个模型 (提供多个模型路径)')
    
    args = parser.parse_args()
    
    # 检查数据集
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    if not data_path.exists():
        print(f"错误: 找不到数据集配置文件 {data_path}")
        print("请先准备数据集:")
        print("  python download_dataset.py")
        return
    
    # 模型对比模式
    if args.compare:
        compare_models(args.compare, str(data_path), args.imgsz, args.batch, script_dir)
        return
    
    # 创建验证器
    validator = ModelValidator(args.model, args.device)
    
    # 基准测试模式
    if args.benchmark:
        validator.benchmark(str(data_path), imgsz_list=[416, 512, 640, 768])
        return
    
    # 标准验证
    results = validator.validate(
        data_yaml=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json
    )
    
    print(f"\n{'='*50}")
    print("验证完成!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()