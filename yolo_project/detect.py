"""
YOLOv8 推理/检测脚本
用于对图片、视频或摄像头进行目标检测
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    YOLO目标检测器类
    支持图片、视频、摄像头检测
    """
    
    # COCO类别颜色映射 (BGR格式)
    COLORS = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
        (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
        (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
    ]
    
    def __init__(self, model_path='yolov8n.pt', conf_thres=0.25, iou_thres=0.45, 
                 device=None, imgsz=640, half=False):
        """
        初始化检测器
        
        参数:
            model_path: 模型路径 (预训练模型或训练好的模型)
            conf_thres: 置信度阈值
            iou_thres: NMS IoU阈值
            device: 运行设备 (None=自动, '0'=GPU, 'cpu'=CPU)
            imgsz: 输入图像大小
            half: 是否使用半精度(FP16)
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.half = half
        
        # 自动选择设备
        if device is None:
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"加载模型: {model_path}")
        print(f"设备: {self.device}")
        
        # 加载模型
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # 获取类别名称
        self.class_names = self.model.names
        print(f"类别数: {len(self.class_names)}")
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """模型预热"""
        print("模型预热...")
        dummy_input = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
        for _ in range(3):
            self.model.predict(dummy_input, verbose=False)
        print("预热完成")
    
    def detect_image(self, image_path, save=True, show=True, save_dir='runs/detect'):
        """
        检测单张图片
        
        参数:
            image_path: 图片路径
            save: 是否保存结果
            show: 是否显示结果
            save_dir: 保存目录
        """
        print(f"\n检测图片: {image_path}")
        
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return None
        
        # 执行检测
        start_time = time.time()
        results = self.model.predict(
            image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        # 处理结果
        result = results[0]
        detections = []
        
        # 绘制检测结果
        annotated_image = image.copy()
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[cls]
                color = self.COLORS[cls % len(self.COLORS)]
                
                # 绘制边界框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_name} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                detections.append({
                    'class': class_name,
                    'class_id': cls,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # 添加信息
        info_text = f"检测时间: {inference_time*1000:.1f}ms | 目标数: {len(detections)}"
        cv2.putText(annotated_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"  检测到 {len(detections)} 个目标")
        print(f"  推理时间: {inference_time*1000:.1f}ms")
        
        # 保存结果
        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"result_{Path(image_path).name}"
            cv2.imwrite(str(save_path), annotated_image)
            print(f"  结果保存: {save_path}")
        
        # 显示结果
        if show:
            window_name = "YOLO Detection"
            cv2.imshow(window_name, annotated_image)
            print("  按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'image': annotated_image,
            'detections': detections,
            'inference_time': inference_time
        }
    
    def detect_video(self, source, save=True, show=True, save_dir='runs/detect'):
        """
        检测视频或摄像头
        
        参数:
            source: 视频路径或摄像头ID (0=默认摄像头)
            save: 是否保存结果
            show: 是否显示结果
            save_dir: 保存目录
        """
        # 打开视频源
        if isinstance(source, int) or source.isdigit():
            source = int(source)
            source_name = f"camera_{source}"
            print(f"\n打开摄像头: {source}")
        else:
            source_name = Path(source).stem
            print(f"\n检测视频: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"错误: 无法打开视频源 {source}")
            return None
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        if total_frames > 0:
            print(f"  总帧数: {total_frames}")
        
        # 准备视频写入
        video_writer = None
        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"result_{source_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
            print(f"  将保存到: {save_path}")
        
        # 处理帧
        frame_count = 0
        total_inference_time = 0
        
        print("\n开始检测 (按 'q' 退出)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测
            start_time = time.time()
            results = self.model.predict(
                frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # 绘制结果
            result = results[0]
            annotated_frame = frame.copy()
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.class_names[cls]
                    color = self.COLORS[cls % len(self.COLORS)]
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 添加信息
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            info_text = f"FPS: {avg_fps:.1f} | Objects: {len(result.boxes) if result.boxes else 0}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存
            if video_writer:
                video_writer.write(annotated_frame)
            
            # 显示
            if show:
                cv2.imshow("YOLO Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断")
                    break
            
            # 进度显示
            if total_frames > 0 and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # 清理
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 统计
        avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        print(f"\n检测完成!")
        print(f"  处理帧数: {frame_count}")
        print(f"  平均推理时间: {avg_inference_time*1000:.1f}ms")
        print(f"  平均FPS: {1/avg_inference_time:.1f}" if avg_inference_time > 0 else "  平均FPS: N/A")
        
        return {
            'frames_processed': frame_count,
            'avg_inference_time': avg_inference_time
        }
    
    def detect_batch(self, image_paths, save=True, save_dir='runs/detect'):
        """
        批量检测图片
        
        参数:
            image_paths: 图片路径列表
            save: 是否保存结果
            save_dir: 保存目录
        """
        print(f"\n批量检测: {len(image_paths)} 张图片")
        
        results_list = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}]")
            result = self.detect_image(img_path, save=save, show=False, save_dir=save_dir)
            if result:
                results_list.append({
                    'path': str(img_path),
                    'detections': result['detections'],
                    'inference_time': result['inference_time']
                })
        
        # 统计
        total_time = sum(r['inference_time'] for r in results_list)
        total_objects = sum(len(r['detections']) for r in results_list)
        
        print(f"\n{'='*50}")
        print("批量检测完成!")
        print(f"  成功: {len(results_list)}/{len(image_paths)}")
        print(f"  总目标数: {total_objects}")
        print(f"  总时间: {total_time:.2f}s")
        print(f"  平均每张: {total_time/len(results_list)*1000:.1f}ms" if results_list else "  平均每张: N/A")
        
        return results_list


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO目标检测')
    parser.add_argument('--source', type=str, required=True,
                       help='检测源: 图片路径、视频路径、目录、或摄像头ID (0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型路径 (默认: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU阈值 (默认: 0.45)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像大小 (默认: 640)')
    parser.add_argument('--device', type=str, default=None,
                       help='运行设备 (默认: 自动)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='保存结果')
    parser.add_argument('--show', action='store_true', default=True,
                       help='显示结果')
    parser.add_argument('--nosave', action='store_true',
                       help='不保存结果')
    parser.add_argument('--noshow', action='store_true',
                       help='不显示结果')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = YOLODetector(
        model_path=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
        imgsz=args.imgsz
    )
    
    save = not args.nosave
    show = not args.noshow
    
    # 判断检测源类型
    source = args.source
    
    if source.isdigit():
        # 摄像头
        detector.detect_video(int(source), save=save, show=show)
    
    elif Path(source).is_file():
        # 单个文件
        ext = Path(source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            detector.detect_image(source, save=save, show=show)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            detector.detect_video(source, save=save, show=show)
        else:
            print(f"不支持的文件类型: {ext}")
    
    elif Path(source).is_dir():
        # 目录 - 批量检测
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = [p for p in Path(source).iterdir() 
                      if p.suffix.lower() in image_exts]
        
        if not image_paths:
            print(f"目录中没有图片: {source}")
            return
        
        detector.detect_batch(image_paths, save=save)
    
    else:
        print(f"无效的检测源: {source}")


if __name__ == '__main__':
    main()