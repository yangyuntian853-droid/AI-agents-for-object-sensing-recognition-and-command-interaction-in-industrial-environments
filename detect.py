"""
YOLOv8 推理/检测脚本
用于对图片、视频或摄像头进行目标检测
支持结果保存为JSON、CSV格式
"""

import os
import sys
import argparse
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class ResultStorage:
    """
    检测结果存储类
    支持JSON、CSV格式保存
    """
    
    def __init__(self, save_dir=None):
        """初始化存储"""
        if save_dir is None:
            self.save_dir = Path(__file__).parent.resolve() / 'runs' / 'detect'
        else:
            self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建结果目录
        self.result_dir = self.save_dir / f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据存储
        self.all_results = []
        self.session_start = datetime.now().isoformat()
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'class_counts': defaultdict(int),
            'total_inference_time': 0
        }
    
    def save_detection(self, source, detections, inference_time, image_shape=None):
        """
        保存单次检测结果
        
        参数:
            source: 检测源（图片路径或摄像头ID）
            detections: 检测到的目标列表
            inference_time: 推理时间
            image_shape: 图像尺寸 (h, w)
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'source': str(source),
            'image_shape': image_shape,
            'inference_time_ms': round(inference_time * 1000, 2),
            'num_detections': len(detections),
            'detections': detections
        }
        
        self.all_results.append(result)
        
        # 更新统计
        self.stats['total_images'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['total_inference_time'] += inference_time
        
        for det in detections:
            self.stats['class_counts'][det['class']] += 1
        
        return result
    
    def save_json(self, filename='detection_results.json'):
        """保存结果为JSON文件"""
        output = {
            'session_info': {
                'start_time': self.session_start,
                'end_time': datetime.now().isoformat(),
                'statistics': dict(self.stats)
            },
            'results': self.all_results
        }
        
        json_path = self.result_dir / filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"  JSON结果保存: {json_path}")
        return json_path
    
    def save_csv(self, filename='detection_results.csv'):
        """保存结果为CSV文件（平铺格式）"""
        csv_path = self.result_dir / filename
        
        rows = []
        for result in self.all_results:
            if result['detections']:
                for det in result['detections']:
                    rows.append({
                        'timestamp': result['timestamp'],
                        'source': result['source'],
                        'class': det['class'],
                        'class_id': det['class_id'],
                        'confidence': round(det['confidence'], 4),
                        'bbox_x1': det['bbox'][0],
                        'bbox_y1': det['bbox'][1],
                        'bbox_x2': det['bbox'][2],
                        'bbox_y2': det['bbox'][3],
                        'inference_time_ms': result['inference_time_ms']
                    })
            else:
                rows.append({
                    'timestamp': result['timestamp'],
                    'source': result['source'],
                    'class': 'None',
                    'class_id': -1,
                    'confidence': 0,
                    'bbox_x1': '',
                    'bbox_y1': '',
                    'bbox_x2': '',
                    'bbox_y2': '',
                    'inference_time_ms': result['inference_time_ms']
                })
        
        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"  CSV结果保存: {csv_path}")
        
        return csv_path
    
    def save_summary(self, filename='summary.txt'):
        """保存文本摘要"""
        summary_path = self.result_dir / filename
        
        avg_time = (self.stats['total_inference_time'] / self.stats['total_images'] * 1000 
                   if self.stats['total_images'] > 0 else 0)
        
        lines = [
            "=" * 60,
            "YOLO 检测结果摘要",
            "=" * 60,
            f"会话开始: {self.session_start}",
            f"会话结束: {datetime.now().isoformat()}",
            "",
            "[统计信息]",
            f"  处理图片数: {self.stats['total_images']}",
            f"  总检测目标: {self.stats['total_detections']}",
            f"  平均推理时间: {avg_time:.2f}ms",
            f"  平均FPS: {1000/avg_time:.1f}" if avg_time > 0 else "  平均FPS: N/A",
            "",
            "[类别统计]"
        ]
        
        # 按数量排序
        sorted_classes = sorted(self.stats['class_counts'].items(), 
                               key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes:
            lines.append(f"  {class_name}: {count}")
        
        lines.extend([
            "",
            "=" * 60
        ])
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"  摘要保存: {summary_path}")
        return summary_path
    
    def finalize(self):
        """完成会话，保存所有结果"""
        print(f"\n正在保存结果到: {self.result_dir}")
        self.save_json()
        self.save_csv()
        self.save_summary()
        return self.result_dir


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
            model_path: 模型路径 (支持预训练模型、训练好的模型路径)
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
        
        # 获取项目目录
        self.script_dir = Path(__file__).parent.resolve()
        
        # 自动选择设备
        if device is None:
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 解析模型路径
        model_file = self._resolve_model_path(model_path)
        
        print(f"加载模型: {model_file}")
        print(f"设备: {self.device}")
        
        # 加载模型
        try:
            self.model = YOLO(model_file)
            self.model.to(self.device)
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"\n请检查模型路径: {model_path}")
            print("支持的格式:")
            print("  - 预训练模型: yolov8n.pt, yolov8s.pt, ...")
            print("  - 训练后的模型: runs/train/exp_n/weights/best.pt")
            raise
        
        # 获取类别名称
        self.class_names = self.model.names
        print(f"类别数: {len(self.class_names)}")
        print(f"类别: {list(self.class_names.values())[:5]}..." if len(self.class_names) > 5 else f"类别: {list(self.class_names.values())}")
        
        # 初始化结果存储
        self.storage = ResultStorage()
        
        # 预热
        self._warmup()
    
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
        
        # 4. 检查是否在runs/train目录下（训练后的模型）
        if not path.is_absolute():
            # 尝试解析 runs/train 路径
            runs_path = self.script_dir / path
            if runs_path.exists():
                return str(runs_path)
            
            # 尝试添加weights/best.pt
            weights_path = self.script_dir / path / 'weights' / 'best.pt'
            if weights_path.exists():
                return str(weights_path)
        
        # 5. 返回原始路径（可能是预训练模型名称，如yolov8n.pt）
        return model_path
    
    def _warmup(self):
        """模型预热"""
        print("模型预热...")
        try:
            dummy_input = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
            for _ in range(3):
                self.model.predict(dummy_input, verbose=False)
            print("预热完成")
        except Exception as e:
            print(f"预热警告: {e}")
    
    def process_results(self, result, original_image):
        """
        处理检测结果，绘制边界框
        
        参数:
            result: YOLO检测结果
            original_image: 原始图像
        
        返回:
            annotated_image: 标注后的图像
            detections: 检测信息列表
        """
        detections = []
        annotated_image = original_image.copy()
        h, w = original_image.shape[:2]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
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
                    'class_id': int(cls),
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return annotated_image, detections
    
    def detect_image(self, image_path, save=True, show=True, save_dir=None):
        """
        检测单张图片
        
        参数:
            image_path: 图片路径
            save: 是否保存结果
            show: 是否显示结果
            save_dir: 保存目录，默认为项目目录下的runs/detect
        """
        # 设置默认保存目录
        if save_dir is None:
            save_dir = self.storage.result_dir
        else:
            save_dir = Path(save_dir)
        
        print(f"\n检测图片: {image_path}")
        
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return None
        
        h, w = image.shape[:2]
        
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
        annotated_image, detections = self.process_results(result, image)
        
        # 保存到存储
        self.storage.save_detection(image_path, detections, inference_time, (h, w))
        
        # 添加信息
        info_text = f"检测时间: {inference_time*1000:.1f}ms | 目标数: {len(detections)}"
        cv2.putText(annotated_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"  检测到 {len(detections)} 个目标")
        print(f"  推理时间: {inference_time*1000:.1f}ms")
        
        # 保存结果图片
        if save:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"result_{Path(image_path).name}"
            cv2.imwrite(str(save_path), annotated_image)
            print(f"  结果图片保存: {save_path}")
        
        # 显示结果
        if show:
            window_name = "YOLO Detection - 按任意键关闭"
            cv2.imshow(window_name, annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'image': annotated_image,
            'detections': detections,
            'inference_time': inference_time
        }
    
    def detect_camera(self, camera_id=0, save=True, show=True, save_dir=None, 
                      max_frames=None, capture_key='c', quit_key='q'):
        """
        摄像头实时检测
        
        参数:
            camera_id: 摄像头ID (0=默认摄像头)
            save: 是否保存结果视频
            show: 是否显示结果
            save_dir: 保存目录
            max_frames: 最大处理帧数 (None=无限制)
            capture_key: 截图按键
            quit_key: 退出按键
        """
        if save_dir is None:
            save_dir = self.storage.result_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n打开摄像头: {camera_id}")
        print(f"控制键: '{capture_key}'=截图保存, '{quit_key}'=退出")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_id}")
            return None
        
        # 获取摄像头信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # 准备视频写入
        video_writer = None
        video_path = None
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = save_dir / f"camera_{camera_id}_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            print(f"  视频将保存到: {video_path}")
        
        # 截图保存目录
        capture_dir = save_dir / 'captures'
        capture_dir.mkdir(exist_ok=True)
        capture_count = 0
        
        # 处理帧
        frame_count = 0
        total_inference_time = 0
        is_running = True
        
        print("\n开始检测...")
        
        while is_running:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取帧")
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
            
            # 处理结果
            result = results[0]
            annotated_frame, detections = self.process_results(result, frame)
            
            # 实时保存检测结果
            self.storage.save_detection(f"camera_{camera_id}_frame_{frame_count}", 
                                        detections, inference_time, (height, width))
            
            # 添加信息
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            obj_count = len(detections)
            info_text = f"FPS: {avg_fps:.1f} | Objects: {obj_count} | Frame: {frame_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 添加控制提示
            hint_text = f"Press '{capture_key}'=Capture, '{quit_key}'=Quit"
            cv2.putText(annotated_frame, hint_text, (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 保存视频
            if video_writer:
                video_writer.write(annotated_frame)
            
            # 显示
            if show:
                cv2.imshow("YOLO Camera Detection", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(quit_key):
                    print(f"\n用户退出 (按 '{quit_key}')")
                    is_running = False
                elif key == ord(capture_key):
                    # 截图
                    capture_count += 1
                    capture_path = capture_dir / f"capture_{timestamp}_{capture_count:04d}.jpg"
                    cv2.imwrite(str(capture_path), annotated_frame)
                    print(f"  截图保存: {capture_path}")
            
            # 检查最大帧数
            if max_frames and frame_count >= max_frames:
                print(f"\n达到最大帧数: {max_frames}")
                break
        
        # 清理
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 统计
        avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        print(f"\n摄像头检测完成!")
        print(f"  处理帧数: {frame_count}")
        print(f"  截图数量: {capture_count}")
        print(f"  平均推理时间: {avg_inference_time*1000:.1f}ms")
        if avg_inference_time > 0:
            print(f"  平均FPS: {1/avg_inference_time:.1f}")
        
        return {
            'frames_processed': frame_count,
            'captures': capture_count,
            'avg_inference_time': avg_inference_time,
            'video_path': str(video_path) if video_path else None
        }
    
    def detect_video(self, source, save=True, show=True, save_dir=None):
        """
        检测视频文件
        
        参数:
            source: 视频路径
            save: 是否保存结果
            show: 是否显示结果
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = self.storage.result_dir
        else:
            save_dir = Path(save_dir)
        
        source_name = Path(source).stem
        print(f"\n检测视频: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {source}")
            return None
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        if total_frames > 0:
            print(f"  总帧数: {total_frames}")
        
        # 准备视频写入
        video_writer = None
        video_path = None
        if save:
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = save_dir / f"result_{source_name}_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            print(f"  将保存到: {video_path}")
        
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
            
            # 处理结果
            result = results[0]
            annotated_frame, detections = self.process_results(result, frame)
            
            # 保存检测结果
            self.storage.save_detection(f"{source_name}_frame_{frame_count}", 
                                        detections, inference_time, (height, width))
            
            # 添加信息
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            obj_count = len(detections)
            info_text = f"FPS: {avg_fps:.1f} | Objects: {obj_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存
            if video_writer:
                video_writer.write(annotated_frame)
            
            # 显示
            if show:
                cv2.imshow("YOLO Video Detection", annotated_frame)
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
        print(f"\n视频检测完成!")
        print(f"  处理帧数: {frame_count}")
        print(f"  平均推理时间: {avg_inference_time*1000:.1f}ms")
        if avg_inference_time > 0:
            print(f"  平均FPS: {1/avg_inference_time:.1f}")
        
        return {
            'frames_processed': frame_count,
            'avg_inference_time': avg_inference_time,
            'video_path': str(video_path) if video_path else None
        }
    
    def detect_batch(self, image_paths, save=True, show=False, save_dir=None):
        """
        批量检测图片
        
        参数:
            image_paths: 图片路径列表
            save: 是否保存结果
            show: 是否显示结果（批量时建议False）
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = self.storage.result_dir
        else:
            save_dir = Path(save_dir)
        
        print(f"\n批量检测: {len(image_paths)} 张图片")
        
        results_list = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}]")
            result = self.detect_image(img_path, save=save, show=show, save_dir=save_dir)
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
        if results_list:
            print(f"  平均每张: {total_time/len(results_list)*1000:.1f}ms")
        
        return results_list
    
    def finalize(self):
        """完成检测，保存所有结果"""
        return self.storage.finalize()


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


def interactive_mode(detector):
    """
    交互式模式
    提供菜单选择检测方式
    """
    print("\n" + "="*60)
    print("YOLO 目标检测系统")
    print("="*60)
    
    while True:
        print("\n请选择检测方式:")
        print("  1. 摄像头实时检测")
        print("  2. 上传图片检测")
        print("  3. 检测视频文件")
        print("  4. 批量检测文件夹")
        print("  5. 退出")
        
        choice = input("\n输入选项 (1-5): ").strip()
        
        if choice == '1':
            # 摄像头检测
            camera_id = input("请输入摄像头ID (默认0): ").strip() or "0"
            try:
                camera_id = int(camera_id)
                detector.detect_camera(camera_id=camera_id)
            except ValueError:
                print("无效的摄像头ID")
        
        elif choice == '2':
            # 图片检测
            image_path = input("请输入图片路径: ").strip()
            if Path(image_path).exists():
                show = input("是否显示结果? (y/n, 默认y): ").strip().lower() != 'n'
                detector.detect_image(image_path, show=show)
            else:
                print(f"文件不存在: {image_path}")
        
        elif choice == '3':
            # 视频检测
            video_path = input("请输入视频路径: ").strip()
            if Path(video_path).exists():
                detector.detect_video(video_path)
            else:
                print(f"文件不存在: {video_path}")
        
        elif choice == '4':
            # 批量检测
            folder_path = input("请输入文件夹路径: ").strip()
            if Path(folder_path).is_dir():
                image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                image_paths = [p for p in Path(folder_path).iterdir() 
                              if p.suffix.lower() in image_exts]
                if image_paths:
                    detector.detect_batch(image_paths)
                else:
                    print(f"目录中没有图片文件")
            else:
                print(f"目录不存在: {folder_path}")
        
        elif choice == '5':
            print("\n退出程序")
            break
        
        else:
            print("无效选项，请重新选择")
        
        # 询问是否继续
        if choice in ['1', '2', '3', '4']:
            cont = input("\n是否继续检测? (y/n, 默认y): ").strip().lower()
            if cont == 'n':
                break
    
    # 保存所有结果
    result_dir = detector.finalize()
    print(f"\n所有结果已保存到: {result_dir}")


def main():
    """主函数"""
    # 获取项目目录
    script_dir = Path(__file__).parent.resolve()
    
    # 查找最新的训练模型
    latest_model = find_latest_model(script_dir)
    default_model = latest_model if latest_model else 'yolov8n.pt'
    
    parser = argparse.ArgumentParser(description='YOLO目标检测')
    parser.add_argument('--source', type=str, default=None,
                       help='检测源: 图片路径、视频路径、目录、或摄像头ID (0)')
    parser.add_argument('--model', type=str, default=default_model,
                       help=f'模型路径 (默认: {default_model})')
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
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='交互式模式')
    
    args = parser.parse_args()
    
    print(f"使用模型: {args.model}")
    
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
    
    # 交互式模式
    if args.interactive or args.source is None:
        interactive_mode(detector)
        detector.finalize()
        return
    
    # 命令行模式
    source = args.source
    
    if source.isdigit():
        # 摄像头
        detector.detect_camera(int(source), save=save, show=show)
    
    elif Path(source).is_file():
        # 单个文件
        ext = Path(source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            detector.detect_image(source, save=save, show=show)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            detector.detect_video(source, save=save, show=show)
        else:
            print(f"不支持的文件类型: {ext}")
            return
    
    elif Path(source).is_dir():
        # 目录 - 批量检测
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = [p for p in Path(source).iterdir() 
                      if p.suffix.lower() in image_exts]
        
        if not image_paths:
            print(f"目录中没有图片: {source}")
            return
        
        detector.detect_batch(image_paths, save=save, show=False)
    
    else:
        print(f"无效的检测源: {source}")
        return
    
    # 保存所有结果
    result_dir = detector.finalize()
    print(f"\n所有结果已保存到: {result_dir}")


if __name__ == '__main__':
    main()