# 🚨 RTSP Multi-Camera Fall Detection System 
**基于RT-DETR+YOLO-Pose的多摄像头跌倒检测系统**  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 项目概述
通过**RT-DETR目标检测**与**YOLOv8-Pose姿态估计**实现多路RTSP视频流的实时跌倒检测，具备：
• **多摄像头并发处理**（支持8路RTSP流）
• **双重检测机制**（目标检测+姿态验证）
• **动态阈值调整**（根据检测置信度自动调整姿态分析阈值）
• **智能重复过滤**（基于IoU的去重算法）

## 🛠️ 快速部署
### 环境要求
```bash
pip install av opencv-python-headless ultralytics==8.1.0 
# 需提前安装CUDA 11.7+和cuDNN
```

### 配置说明
1. **RTSP配置**  
修改`rtsp_urls`列表（[代码L55-63](fall_detection.py#L55-L63)）：
```python
rtsp_urls = [
    'rtsp://user:password@ip:port/path',
    # 添加更多摄像头...
]
```

2. **模型路径**  
• RT-DETR引擎文件路径：[代码L42](fall_detection.py#L42)
• YOLO-Pose模型路径：[代码L44](fall_detection.py#L44)

3. **目录设置**  
修改输出路径（[代码L66-68](fall_detection.py#L66-L68)）：
```python
original_frames_dir = "path/to/original" 
annotated_frames_dir = "path/to/annotated"
cropped_frames_dir = "path/to/cropped"
```

## 🚀 运行方法
```bash
python fall_detection.py
```
**输出文件结构**：
```
├── original_frames/    # 原始检测帧
├── annotated_frames/   # 带标注结果帧 
├── cropped_frames/     # 局部裁剪帧
└── fall_detection.log  # 运行日志
```

## 🔍 高级功能
### 检测逻辑优化
• **双重验证机制**（[代码L168-254](fall_detection.py#L168-L254)）
  1. RT-DETR初检（置信度>0.7）
  2. 5秒后二次验证（IoU>0.8）
  3. YOLO-Pose膝关节置信度分析

• **动态阈值调整**（[代码L75-76](fall_detection.py#L75-L76)）
  ```python
  knee_confidence_threshold_pose_high = 0.7  # RTDETR高置信度时使用
  knee_confidence_threshold_pose_low = 0.85 # RTDETR低置信度时使用
  ```

### 性能优化
• **关键帧提取**（[KeyFrameThread类](fall_detection.py#L111-L135)）
• **多线程架构**（[main()函数](fall_detection.py#L358-L377)）

## 📂 项目结构
```
fall_detection/
├── configs/                  # 配置文件
├── models/                   # 模型文件
│   ├── detr_fallv2/          # RT-DETR模型
│   └── yolov8n-pose.pt      # YOLO-Pose模型
├── utils/                    # 工具类
│   ├── visualization.py      # 可视化工具
│   └── logger.py             # 日志模块
└── fall_detection.py         # 主程序
```

## 📝 日志系统
• **日志格式**（[代码L27-34](fall_detection.py#L27-L34)）
  ```python
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      handlers=[logging.FileHandler(), logging.StreamHandler()]
  )
  ```
• **典型日志输出**：
  ```
  2025-03-24 14:30:45 - INFO - First detection for camera 0..., RTDETR confidence: 0.82
  2025-03-24 14:30:50 - INFO - High IOU (0.85 > 0.8)..., proceeding with pose estimation
  ```

## 🤝 贡献指南
1. 提交Issue说明问题
2. Fork仓库进行修改
3. 提交Pull Request附修改说明

## 📜 许可证
MIT License - 详见 [LICENSE](LICENSE)

---

> 提示：建议在README中添加**系统架构图**和**检测效果截图**（可参考[网页3]的视觉效果建议）。完整配置模板请参考[网页2]的经典结构和[网页4]的项目状态说明。
 
