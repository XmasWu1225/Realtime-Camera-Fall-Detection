```markdown
# 基于 RT-DETR 和 Yolopose 的跌倒检测系统

## 概述

本项目实现了一个使用计算机视觉技术的实时跌倒检测系统。它利用 RT-DETR 模型进行目标检测，并使用 Yolopose 进行人体姿态估计，从而从多个 RTSP 摄像头流中识别潜在的跌倒事件。该系统旨在处理视频流，检测人物，并分析其姿势以判断是否发生跌倒。一旦检测到潜在的跌倒，系统会保存原始帧、带注释的帧和裁剪帧，并将事件记录下来以供后续审查。

## 功能

- **实时跌倒检测:** 并发处理多个 RTSP 摄像头流，以实时检测跌倒事件。
- **RT-DETR 用于人物检测:** 利用高精度的 RT-DETR 模型可靠地检测视频帧中的人物。
- **Yolopose 用于姿态估计:** 使用 Yolopose 估计人体姿态并识别关键点，特别关注膝关节置信度以进行跌倒检测。
- **基于 IOU 的重复检测预防:** 通过计算连续检测之间的交并比 (IOU)，防止保存冗余帧。
- **可配置的置信度阈值:** 允许调整 RT-DETR 和 Yolopose 模型的置信度阈值，以优化性能并减少误报/漏报。
- **可调节的 IOU 阈值:** 提供对重复检测的 IOU 阈值的控制。
- **帧裁剪:** 裁剪检测到的人物区域并添加填充，以便进行详细的姿态分析并保存裁剪帧。
- **日志记录:** 将检测事件、警告和错误全面记录到日志文件和控制台。
- **帧保存:** 当检测到潜在的跌倒时，保存检测到的人物的原始帧、带注释的帧（带有 RT-DETR 边界框）和裁剪帧。
- **多摄像头支持:** 设计用于同时处理多个 RTSP 摄像头流。
- **引擎支持:** 针对 TensorRT 引擎进行了优化，以实现更快的推理速度（RT-DETR 模型以 `.engine` 格式加载）。

## 依赖

在运行脚本之前，请确保已安装以下 Python 库。您可以使用 pip 安装它们：

```bash
pip install torch torchvision ultralytics opencv-python-headless av tensorrt
```

- **torch**: PyTorch 深度学习框架。
- **torchvision**: PyTorch 用于计算机视觉任务的库。
- **ultralytics**: 用于目标检测和姿态估计的 YOLOv8 和 RT-DETR 框架。
- **opencv-python-headless**: OpenCV 库，用于图像和视频处理（无头版本，适用于没有显示器的环境）。
- **av**: PyAV 库，用于访问多媒体容器（用于 RTSP 流处理）。
- **tensorrt**: NVIDIA TensorRT，用于高性能推理（如果使用 `.engine` RT-DETR 模型，则需要）。

**注意:** 如果您未使用 TensorRT 优化的模型，可能需要调整脚本中的模型加载部分。

## 安装

1. **克隆仓库（如果适用）或创建一个新的项目目录。**
2. **将 Python 脚本 (`your_script_name.py`) 放置在您的项目目录中。** (假设您将提供的脚本保存为 `fall_detection.py`)
3. **下载预训练模型:**
    - **RT-DETR 模型:** 您需要拥有一个预训练的 RT-DETR 模型，格式为 `.engine`。脚本配置为从 `/home/aa/workspace/runs/detect/Fall/059/0318_3w_detr_fallv2/weights/best.engine` 加载模型。 您需要将您的 `.engine` 模型放置在此路径，或修改脚本中的 `detr_model` 加载行以指向您的模型位置。如果您没有 `.engine` 模型，可以使用 Ultralytics 训练您自己的 RT-DETR 模型，或者将 YOLOv8 模型转换为 RT-DETR，然后再转换为 TensorRT 引擎。
    - **Yolov8-Pose 模型:** 脚本在首次使用时会自动从 Ultralytics 下载 `yolov8n-pose.pt` 模型。 Yolopose 不需要手动下载。

4. **设置目录结构:** 确保存在以下目录结构，或者脚本将会创建它们：
    ```
    Result/
    └── FallDetection/
        └── 20250325test/  (或者您期望的测试目录 - 可在脚本中配置)
            ├── fall_detection.log  (日志文件将在此处创建)
            ├── original_frames/    (原始帧将保存到此处)
            ├── output_frames/      (带注释的帧将保存到此处)
            └── cropped_frames/     (裁剪帧将保存到此处)
    ```
    脚本默认使用这些目录，由以下代码定义：
    ```python
    log_filename = "/home/aa/workspace/test/Result/FallDetection/20250325test/fall_detection.log"
    original_frames_dir = "/home/aa/workspace/test/Result/FallDetection/20250325test/original_frames"
    annotated_frames_dir = "/home/aa/workspace/test/Result/FallDetection/20250325test/output_frames"
    cropped_frames_dir = "/home/aa/workspace/test/Result/FallDetection/20250325test/cropped_frames"
    ```
    如果需要，您可以在脚本中修改这些路径。

## 配置

您可以直接在 Python 脚本中配置以下参数：

- **RTSP URL (`rtsp_urls`):**
    - 修改 `rtsp_urls` 列表以包含您的 IP 摄像头的 RTSP URL。
    ```python
    rtsp_urls = [
        'rtsp://admin:weilexinxi0608@192.168.1.201:554/Streaming/Channels/101',
        'rtsp://admin:weilexinxi0608@192.168.1.202:554/Streaming/Channels/101',
        # ... 更多 URL
    ]
    ```
    - **重要:** 根据您的摄像头设置更新用户名、密码、IP 地址和端口。

- **模型路径:**
    - **RT-DETR 模型 (`detr_model`):**
        ```python
        detr_model = RTDETR("/home/aa/workspace/runs/detect/Fall/059/0318_3w_detr_fallv2/weights/best.engine")
        ```
        将路径更改为您 RT-DETR `.engine` 模型文件的位置。如果您使用不同格式的模型（例如 `.pt`），您可能需要相应地调整模型加载部分。
    - **Yolov8-Pose 模型 (`pose_model`):**
        ```python
        pose_model = YOLO("yolov8n-pose.pt")
        ```
        脚本使用预训练的 `yolov8n-pose.pt` 模型。 如果需要，您可以更改为其他 Yolopose 模型，但建议使用 `yolov8n-pose.pt` 以平衡速度和准确性。

- **阈值:**
    - **RT-DETR 置信度阈值 (`confidence_threshold`):**
        ```python
        confidence_threshold = 0.7
        ```
        调整此值（介于 0 和 1 之间）以控制 RT-DETR 的检测灵敏度。 较高的值意味着检测需要更高的置信度。
    - **重复检测的 IOU 阈值 (`iou_threshold`):**
        ```python
        iou_threshold = 0.8
        ```
        修改此值（介于 0 和 1 之间）以调整将检测视为重复项的 IOU 阈值。 较高的值意味着检测需要更多重叠才能被视为重复项。
    - **姿态估计的膝关节置信度阈值 (`knee_confidence_threshold_pose_high`, `knee_confidence_threshold_pose_low`):**
        ```python
        knee_confidence_threshold_pose_high = 0.7
        knee_confidence_threshold_pose_low = 0.85
        ```
        这些阈值确定了 Yolopose 检测到的膝关节关键点的最低置信度，跌倒检测需要达到此置信度。 当 RT-DETR 置信度较高（>= 0.8）时，使用 `knee_confidence_threshold_pose_high`，当 RT-DETR 置信度较低时，使用 `knee_confidence_threshold_pose_low`。 调整这些值以微调基于姿态的跌倒检测。

- **关键点索引:**
    - **左膝关节关键点索引 (`left_knee_keypoint_index`):**
    - **右膝关节关键点索引 (`right_knee_keypoint_index`):**
        ```python
        left_knee_keypoint_index = 13
        right_knee_keypoint_index = 14
        ```
        这些是 Yolopose 模型输出中左膝和右膝的关键点索引。 通常，除非您使用具有不同关键点排序的不同姿态估计模型，否则无需更改这些值。

- **裁剪填充 (`crop_padding`):**
    ```python
        crop_padding = 40
    ```
    此值（以像素为单位）确定裁剪帧以进行姿态估计时，检测到的人物边界框周围的填充量。 调整此值以确保裁剪帧中包含整个人物。

- **输出目录:**
    - **日志文件路径 (`log_filename`):**
    - **原始帧目录 (`original_frames_dir`):**
    - **带注释的帧目录 (`annotated_frames_dir`):**
    - **裁剪帧目录 (`cropped_frames_dir`):**
        ```python
        log_filename = "/path/to/your/log/fall_detection.log"
        original_frames_dir = "/path/to/your/output/original_frames"
        annotated_frames_dir = "/path/to/your/output/output_frames"
        cropped_frames_dir = "/path/to/your/output/cropped_frames"
        ```
        修改这些路径以指定您要保存日志文件和输出帧的位置。

## 使用方法

1. **确保所有依赖项都已安装。**
2. **配置脚本**，方法是修改 [配置](#配置) 部分中描述的参数，特别是更新 `rtsp_urls` 和模型路径（如果需要）。
3. **运行脚本:**
    ```bash
    python fall_detection.py
    ```
    将 `fall_detection.py` 替换为您的 Python 脚本文件的实际名称。
4. **监控输出:**
    - 检查控制台输出以获取实时日志和信息。
    - 查看日志文件 (`fall_detection.log`) 以获取详细日志。
    - 检查 `original_frames`、`output_frames` 和 `cropped_frames` 目录，以查看在检测到跌倒时保存的帧。

## 输出

脚本生成以下输出：

- **日志文件 (`fall_detection.log`):** 一个文本文件，其中包含系统运行的详细日志，包括：
    - 事件的时间戳。
    - 来自 RT-DETR 的检测信息。
    - 来自 Yolopose 的姿态估计详细信息。
    - 跌倒检测决策和原因。
    - 遇到的任何警告或错误。

- **原始帧 (`original_frames_dir`):** JPG 图像，是在检测到跌倒时从 RTSP 流中获取的原始帧。文件名包括时间戳和摄像头 ID（例如，`frame_20250325_103000_cam0.jpg`）。

- **带注释的帧 (`output_frames_dir`):** JPG 图像，与 `original_frames_dir` 中的帧相同，但使用 RT-DETR 绘制的边界框进行了注释。文件名包括时间戳、摄像头 ID 和 `_annotated` 后缀（例如，`frame_20250325_103000_cam0_annotated.jpg`）。

- **裁剪帧 (`cropped_frames_dir`):** JPG 图像，是围绕检测到的人物进行裁剪的区域，用于姿态分析。 这些帧在检测到潜在的跌倒时保存，提供人物姿势的更近视图。文件名包括时间戳、摄像头 ID 和 `_cropped` 后缀（例如，`frame_20250325_103000_cam0_cropped.jpg`）。