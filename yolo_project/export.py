"""
YOLOv8 模型导出脚本
用于将训练好的模型导出为不同格式，便于部署
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


class ModelExporter:
    """
    YOLO模型导出器
    支持多种导出格式
    """
    
    # 支持的导出格式
    FORMATS = {
        'torchscript': {'ext': '.torchscript', 'desc': 'PyTorch TorchScript'},
        'onnx': {'ext': '.onnx', 'desc': 'Open Neural Network Exchange'},
        'openvino': {'ext': '_openvino_model', 'desc': 'Intel OpenVINO'},
        'engine': {'ext': '.engine', 'desc': 'TensorRT (GPU加速)'},
        'coreml': {'ext': '.mlpackage', 'desc': 'Apple CoreML'},
        'saved_model': {'ext': '_saved_model', 'desc': 'TensorFlow SavedModel'},
        'pb': {'ext': '.pb', 'desc': 'TensorFlow GraphDef'},
        'tflite': {'ext': '.tflite', 'desc': 'TensorFlow Lite (移动端)'},
        'edgetpu': {'ext': '_edgetpu.tflite', 'desc': 'Google Edge TPU'},
        'tfjs': {'ext': '_web_model', 'desc': 'TensorFlow.js (浏览器)'},
        'paddle': {'ext': '_paddle_model', 'desc': 'PaddlePaddle'},
        'ncnn': {'ext': '_ncnn_model', 'desc': 'Tencent NCNN (移动端)'},
    }
    
    def __init__(self, model_path):
        """
        初始化导出器
        
        参数:
            model_path: 模型路径 (.pt文件)
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        print(f"模型加载完成!")
    
    def list_formats(self):
        """列出支持的导出格式"""
        print("\n" + "="*60)
        print("支持的导出格式:")
        print("="*60)
        
        for i, (fmt, info) in enumerate(self.FORMATS.items(), 1):
            print(f"{i:2d}. {fmt:<15} - {info['desc']}")
            print(f"    扩展名: {info['ext']}")
        print("="*60)
    
    def export(self, format='onnx', imgsz=640, half=False, int8=False, 
               dynamic=False, simplify=False, opset=12, workspace=4, 
               nms=False, batch=1):
        """
        导出模型
        
        参数:
            format: 导出格式
            imgsz: 输入图像大小
            half: 使用FP16半精度
            int8: 使用INT8量化
            dynamic: 动态输入尺寸
            simplify: 简化ONNX模型
            opset: ONNX opset版本
            workspace: TensorRT工作空间大小(GB)
            nms: 在模型中包含NMS
            batch: 批次大小
        """
        if format not in self.FORMATS:
            print(f"错误: 不支持的格式 '{format}'")
            print(f"支持的格式: {list(self.FORMATS.keys())}")
            return None
        
        print(f"\n{'='*60}")
        print(f"导出模型为 {format.upper()} 格式")
        print(f"{'='*60}")
        print(f"输入尺寸: {imgsz}")
        print(f"半精度(FP16): {half}")
        print(f"INT8量化: {int8}")
        print(f"动态尺寸: {dynamic}")
        print(f"简化模型: {simplify}")
        print(f"批次大小: {batch}")
        
        # 导出参数
        export_args = {
            'format': format,
            'imgsz': imgsz,
            'half': half,
            'int8': int8,
            'dynamic': dynamic,
            'simplify': simplify,
            'opset': opset,
            'workspace': workspace,
            'nms': nms,
            'batch': batch,
        }
        
        try:
            # 执行导出
            print(f"\n开始导出...")
            export_path = self.model.export(**export_args)
            
            print(f"\n{'='*60}")
            print("导出成功!")
            print(f"{'='*60}")
            print(f"导出路径: {export_path}")
            
            # 获取文件信息
            export_file = Path(export_path)
            if export_file.exists():
                size_mb = export_file.stat().st_size / (1024 * 1024)
                print(f"文件大小: {size_mb:.2f} MB")
            
            return export_path
            
        except Exception as e:
            print(f"\n导出失败!")
            print(f"错误: {e}")
            
            # 提供解决方案
            if format == 'engine':
                print("\nTensorRT导出提示:")
                print("  1. 确保安装了TensorRT: pip install tensorrt")
                print("  2. 确保CUDA和cuDNN正确安装")
                print("  3. 尝试减小workspace大小")
            
            elif format == 'openvino':
                print("\nOpenVINO导出提示:")
                print("  1. 安装OpenVINO: pip install openvino-dev")
            
            elif format == 'coreml':
                print("\nCoreML导出提示:")
                print("  1. 仅在macOS上支持")
                print("  2. 安装coremltools: pip install coremltools")
            
            elif format == 'edgetpu':
                print("\nEdge TPU导出提示:")
                print("  1. 先导出为tflite格式")
                print("  2. 安装Edge TPU编译器")
            
            return None
    
    def export_all(self, imgsz=640, output_dir='exports'):
        """
        导出为所有可用格式
        
        参数:
            imgsz: 输入图像大小
            output_dir: 输出目录
        """
        print(f"\n{'='*60}")
        print("批量导出所有格式")
        print(f"{'='*60}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 按平台推荐排序
        priority_formats = ['onnx', 'torchscript', 'engine', 'openvino', 
                           'tflite', 'ncnn', 'coreml']
        
        for fmt in priority_formats:
            if fmt in self.FORMATS:
                print(f"\n{'-'*40}")
                print(f"导出: {fmt}")
                print(f"{'-'*40}")
                
                path = self.export(format=fmt, imgsz=imgsz)
                results[fmt] = path
        
        # 保存汇总
        print(f"\n{'='*60}")
        print("批量导出完成!")
        print(f"{'='*60}")
        
        successful = {k: v for k, v in results.items() if v is not None}
        failed = {k: v for k, v in results.items() if v is None}
        
        print(f"\n成功: {len(successful)}/{len(results)}")
        for fmt, path in successful.items():
            print(f"  ✓ {fmt}: {path}")
        
        if failed:
            print(f"\n失败: {len(failed)}")
            for fmt in failed.keys():
                print(f"  ✗ {fmt}")
        
        return results


def get_recommendations():
    """获取部署场景推荐"""
    print("\n" + "="*70)
    print("部署场景推荐")
    print("="*70)
    
    scenarios = [
        {
            'name': '服务器/云端 (GPU)',
            'formats': ['engine', 'onnx'],
            'reason': 'TensorRT提供最佳GPU性能，ONNX通用性强',
            'notes': '需要NVIDIA GPU'
        },
        {
            'name': '服务器/云端 (CPU)',
            'formats': ['openvino', 'onnx'],
            'reason': 'OpenVINO针对Intel CPU优化，ONNX Runtime也很好',
            'notes': '推荐Intel处理器'
        },
        {
            'name': '边缘设备 (ARM)',
            'formats': ['tflite', 'ncnn'],
            'reason': '轻量级，适合树莓派、Jetson等',
            'notes': 'TFLite支持量化，NCNN在移动端表现好'
        },
        {
            'name': '移动端 (iOS)',
            'formats': ['coreml'],
            'reason': 'Apple原生支持，性能最佳',
            'notes': '仅支持Apple设备'
        },
        {
            'name': '移动端 (Android)',
            'formats': ['tflite', 'ncnn'],
            'reason': 'TFLite官方支持，NCNN速度快',
            'notes': '可考虑量化加速'
        },
        {
            'name': 'Web浏览器',
            'formats': ['tfjs', 'onnx'],
            'reason': 'TensorFlow.js直接支持，ONNX通过WebAssembly',
            'notes': '注意模型大小限制'
        },
        {
            'name': '嵌入式/FPGA',
            'formats': ['tflite', 'onnx'],
            'reason': '标准格式，工具链支持好',
            'notes': '可能需要特定转换'
        },
    ]
    
    for i, s in enumerate(scenarios, 1):
        print(f"\n{i}. {s['name']}")
        print(f"   推荐格式: {', '.join(s['formats'])}")
        print(f"   理由: {s['reason']}")
        if s['notes']:
            print(f"   注意: {s['notes']}")
    
    print("\n" + "="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO模型导出')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pt文件)')
    parser.add_argument('--format', type=str, default='onnx',
                       help='导出格式 (默认: onnx)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像大小 (默认: 640)')
    parser.add_argument('--half', action='store_true',
                       help='使用FP16半精度')
    parser.add_argument('--int8', action='store_true',
                       help='使用INT8量化')
    parser.add_argument('--dynamic', action='store_true',
                       help='动态输入尺寸')
    parser.add_argument('--simplify', action='store_true',
                       help='简化ONNX模型')
    parser.add_argument('--batch', type=int, default=1,
                       help='批次大小 (默认: 1)')
    parser.add_argument('--list', action='store_true',
                       help='列出支持的格式')
    parser.add_argument('--all', action='store_true',
                       help='导出所有格式')
    parser.add_argument('--recommend', action='store_true',
                       help='显示部署推荐')
    
    args = parser.parse_args()
    
    # 显示推荐
    if args.recommend:
        get_recommendations()
        return
    
    # 创建导出器
    exporter = ModelExporter(args.model)
    
    # 列出格式
    if args.list:
        exporter.list_formats()
        return
    
    # 批量导出
    if args.all:
        exporter.export_all(imgsz=args.imgsz)
        return
    
    # 单格式导出
    exporter.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        batch=args.batch
    )


if __name__ == '__main__':
    main()