import av
import cv2
from ultralytics import RTDETR, YOLO  # 修改：导入 YOLO for yolopose
import os
import time
from datetime import datetime
import threading
import logging

# torch torchvision torchaudio
# ultralytics
# opencv-python-headless
# av
# tensorrt

# 设置日志记录并将其保存到文件中
log_filename = "/home/aa/workspace/test/Result/FallDetection/059/FallDetection/20250318detr/RTSP_test_v2/fall_detection.log" # 修改日志路径
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
# 设置日志记录格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# 加载预训练的 RT-DETR 模型（替换原来的 YOLO 模型）
detr_model = RTDETR("/home/aa/workspace/runs/detect/Fall/059/0318_3w_detr_fallv2/weights/best.engine")  
# 加载 YOLOv8-Pose 模型
pose_model = YOLO("yolov8n-pose.pt")  # 使用 YOLOv8-Pose 模型

# 定义RTSP流URL列表
rtsp_urls = [
    'rtsp://admin:weilexinxi0608@192.168.1.201:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.202:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.203:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.204:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.205:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.206:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.207:554/Streaming/Channels/101',
    'rtsp://admin:weilexinxi0608@192.168.1.208:554/Streaming/Channels/101'
    # 'rtsp://192.168.0.213:8554/test1'
]

original_frames_dir = "/home/aa/workspace/test/Result/FallDetection/059/FallDetection/20250318detr/RTSP_test_v2/original_frames" # 修改输出路径
annotated_frames_dir = "/home/aa/workspace/test/Result/FallDetection/059/FallDetection/20250318detr/RTSP_test_v2/output_frames" # 修改输出路径
cropped_frames_dir = "/home/aa/workspace/test/Result/FallDetection/059/FallDetection/20250318detr/RTSP_test_v2/cropped_frames" # 新增裁剪帧保存路径

# 确保目录存在
os.makedirs(original_frames_dir, exist_ok=True)
os.makedirs(annotated_frames_dir, exist_ok=True)
os.makedirs(cropped_frames_dir, exist_ok=True) # 确保裁剪帧目录存在

confidence_threshold = 0.7  # 设置RTDETR置信度阈值
iou_threshold = 0.8  # 设置IOU阈值用于去重
knee_confidence_threshold_pose_high = 0.7 # 膝关节置信度阈值(RTDETR high confidence)，用于姿态检测
knee_confidence_threshold_pose_low = 0.85 # 膝关节置信度阈值(RTDETR low confidence)，用于姿态检测

left_knee_keypoint_index = 13  # 左膝关节关键点索引
right_knee_keypoint_index = 14  # 右膝关节关键点索引
crop_padding = 40 # 裁剪区域外扩像素点

def calculate_iou(box1, box2):  # 计算IOU,交并比
    """
    计算两个边界框的交并比（IoU）。

    参数:
    box1 (array): 第一个边界框坐标 [x1, y1, x2, y2]
    box2 (array): 第二个边界框坐标 [x1, y1, y2, y2]

    返回:
    float: IoU值
    """
    x1_min = max(box1[0], box2[0])
    y1_min = max(box1[1], box2[1])
    x2_max = min(box1[2], box2[2])
    y2_max = min(box1[3], box2[3])

    intersection_area = max(0, x2_max - x1_min) * max(0, y2_max - y1_min)  # 交集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area  # 并集面积
    return intersection_area / union_area if union_area > 0 else 0  # 返回IoU值

class KeyFrameThread(threading.Thread):
    def __init__(self, rtsp_url):
        super(KeyFrameThread, self).__init__()
        self.rtsp_url = rtsp_url  # 储存RTSP 流地址
        self.container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})  # 用av库打开RTSP流，指定传输协议为TCP
        self.stream = self.container.streams.video[0]  # 获取视频流的第一个视频流对象
        self.running = True
        self.latest_frame = None  # 存储最新的关键帧数据，默认为 None

    def run(self):  # 运行方法
        try:
            for packet in self.container.demux(self.stream):
                if not self.running:
                    break
                if packet.is_keyframe:
                    for frame in packet.decode():
                        ndarray_frame = frame.to_ndarray(format='bgr24')
                        self.latest_frame = ndarray_frame  # 更新最新的关键帧数据

        except Exception as e:
            logging.exception(f"Exception in KeyFrameThread for URL {self.rtsp_url}: {e}")

        finally:
            self.container.close()

    def get_latest_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False

def process_camera(rtsp_url, cam_id, previous_boxes, keyframe_thread):  # 处理摄像头
    """
    处理单个摄像头的视频流，进行目标检测并保存非重复帧。
    ... (文档字符串) ...
    """
    detection_result_1 = None # 存储第一次检测结果
    frame_result_1 = None # 存储第一次检测的帧
    timestamp_result_1 = None # 存储第一次检测的时间戳
    rtdetr_confidence = None # 存储第一次检测的RTDETR confidence
    box_result_1_xyxy = None # 存储第一次检测的边界框坐标 (xyxy 格式)

    while True:
        try:
            frame = keyframe_thread.get_latest_frame()
            if frame is None:
                logging.warning(f"No frame received from camera {cam_id}. Retrying...")
                time.sleep(5)
                continue

            start_time = time.time()

            # 修改：直接对每一帧进行检测
            result = detr_model.predict(source=frame, classes=[1], conf=confidence_threshold)
            boxes = result[0].boxes.cpu().numpy()

            if len(boxes) > 0:  # 如果检测到目标
                current_boxes = [(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]) for box in boxes]

                if detection_result_1 is None:  # 第一次检测到目标
                    detection_result_1 = result
                    frame_result_1 = frame.copy()
                    timestamp_result_1 = datetime.now()
                    rtdetr_confidence = boxes[0].conf[0] # 保存第一次检测的 RTDETR 的置信度
                    box_result_1_xyxy = boxes[0].xyxy[0].copy() # 保存第一次检测的边界框坐标
                    logging.info(f"First detection for camera {cam_id} at {timestamp_result_1.strftime('%Y%m%d_%H%M%S')}, RTDETR confidence: {rtdetr_confidence:.2f}, waiting for confirmation...")
                    time.sleep(5)

                    # 等待 5 秒后，再次检测目标
                    frame = keyframe_thread.get_latest_frame()
                    confirmation_result = detr_model.predict(source=frame, classes=[1], conf=confidence_threshold)
                    confirmation_boxes = confirmation_result[0].boxes.cpu().numpy()

                    if len(confirmation_boxes) == 0:  # 5 秒后未检测到目标
                        logging.info(f"Confirmation failed for camera {cam_id} at {datetime.now().strftime('%Y%m%d_%H%M%S')}, resetting detection cycle.")
                        detection_result_1 = None
                        frame_result_1 = None
                        timestamp_result_1 = None
                        rtdetr_confidence = None
                        box_result_1_xyxy = None # 重置边界框
                        continue # continue跳到下一次循环

                    # 现在已经确认了目标仍然存在
                    detection_result_2 = confirmation_result # 将第二次检测的结果赋值给 detection_result_2
                    boxes_result_2 = detection_result_2[0].boxes.cpu().numpy()
                    current_boxes_result_2 = [(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]) for box in boxes_result_2]
                    box_result_2_xyxy = boxes_result_2[0].xyxy[0] # 获取第二次检测的边界框坐标

                    # 计算第一次和第二次检测框的IOU
                    iou_confirmation = calculate_iou(box_result_1_xyxy, box_result_2_xyxy)

                    if iou_confirmation > 0.8: # IOU 大于 0.8，认为人静止不动，可能是摔倒
                        logging.info(f"High IOU ({iou_confirmation:.2f} > 0.8) between first and confirmation detections for camera {cam_id}, proceeding with pose estimation.")

                        # 裁剪区域
                        box_result_2 = boxes_result_2[0].xyxy[0] # 取第二个检测框 (实际上用哪个框都可以，因为IOU很高)
                        x1, y1, x2, y2 = map(int, box_result_2)

                        # 扩展裁剪区域
                        x1_crop = max(0, int(x1 - crop_padding))
                        y1_crop = max(0, int(y1 - crop_padding))
                        x2_crop = min(frame.shape[1], int(x2 + crop_padding))
                        y2_crop = min(frame.shape[0], int(y2 + crop_padding))
                        cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                        # Determine knee confidence threshold for pose estimation
                        if rtdetr_confidence >= 0.8:
                            knee_confidence_threshold_pose = knee_confidence_threshold_pose_high  # Use lower threshold
                            logging.info(f"Camera {cam_id}: Using low pose confidence threshold: {knee_confidence_threshold_pose:.2f}")
                        else:
                            knee_confidence_threshold_pose = knee_confidence_threshold_pose_low  # Use higher threshold
                            logging.info(f"Camera {cam_id}: Using high pose confidence threshold: {knee_confidence_threshold_pose:.2f}")

                        # 使用 Yolopose 进行姿态检测
                        pose_results = pose_model.predict(cropped_frame,conf=0.5)
                        if pose_results and len(pose_results) > 0:
                            keypoints = pose_results[0].keypoints.cpu().numpy()

                            # 检查 keypoints.conf 是否为 None
                            if keypoints.conf is None:
                                logging.info(f"YoloPose returned None for keypoints.conf for camera {cam_id} at {datetime.now().strftime('%Y%m%d_%H%M%S')}, resetting detection cycle.")
                                detection_result_1 = None
                                frame_result_1 = None
                                timestamp_result_1 = None
                                rtdetr_confidence = None
                                box_result_1_xyxy = None # 重置边界框
                                continue

                            # 获取左膝和右膝的置信度
                            left_knee_confidence = keypoints.conf[0][left_knee_keypoint_index]
                            right_knee_confidence = keypoints.conf[0][right_knee_keypoint_index]

                            # 选择置信度较高的膝关节
                            knee_confidence = max(left_knee_confidence, right_knee_confidence)

                            if knee_confidence < knee_confidence_threshold_pose:
                                logging.info(f"Low knee confidence ({knee_confidence:.2f} < {knee_confidence_threshold_pose}) for camera {cam_id} at {datetime.now().strftime('%Y%m%d_%H%M%S')}, resetting detection cycle.")
                                detection_result_1 = None
                                frame_result_1 = None
                                timestamp_result_1 = None
                                rtdetr_confidence = None
                                box_result_1_xyxy = None # 重置边界框
                                continue
                            else:
                                # 膝关节置信度足够高，继续进行后续处理
                                logging.info(f"Compliant knee confidence ({knee_confidence:.2f} > {knee_confidence_threshold_pose}) for camera {cam_id} at {datetime.now().strftime('%Y%m%d_%H%M%S')}, going to save frame.")
                                save_frame = True
                                previous_box = previous_boxes.get(cam_id)
                                if previous_box is not None:
                                    iou = calculate_iou(previous_box, current_boxes_result_2[0])
                                    if iou > iou_threshold:
                                        save_frame = False
                                        now = datetime.now()
                                        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                                        logging.info(f"Duplicate detection {cam_id} at {timestamp_str}, IOU: {iou:.2f} > {iou_threshold}")
                                    else:
                                        logging.info(f"IOU: {iou:.2f} <= {iou_threshold}, proceeding to save frame.")

                                if save_frame:
                                    now = datetime.now()
                                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

                                    original_frame_path = os.path.join(original_frames_dir, f"frame_{timestamp_str}_cam{cam_id}.jpg")
                                    annotated_frame_path = os.path.join(annotated_frames_dir, f"frame_{timestamp_str}_cam{cam_id}_annotated.jpg")
                                    cropped_frame_path = os.path.join(cropped_frames_dir, f"frame_{timestamp_str}_cam{cam_id}_cropped.jpg") # cropped frame path

                                    cv2.imwrite(original_frame_path, frame) # 保存第二次检测到的完整帧
                                    annotated_frame = detection_result_2[0].plot() # 使用第二次检测结果进行标注
                                    cv2.imwrite(annotated_frame_path, annotated_frame)
                                    cv2.imwrite(cropped_frame_path, cropped_frame) # 保存裁剪的帧
                                    logging.info(f"Fall detected and Saved frames for camera {cam_id} at {timestamp_str}")

                                if len(current_boxes_result_2) > 0:
                                    previous_boxes[cam_id] = current_boxes_result_2[0] # 更新为第二次检测的框

                        else:
                            logging.info(f"No pose results from Yolopose for camera {cam_id} at {datetime.now().strftime('%Y%m%d_%H%M%S')}, resetting detection cycle.")
                            detection_result_1 = None
                            frame_result_1 = None
                            timestamp_result_1 = None
                            rtdetr_confidence = None
                            box_result_1_xyxy = None # 重置边界框
                            continue


                    elif iou_confirmation <= 0.8: # IOU 小于等于 0.8，认为人移动了，不是静止摔倒
                        logging.info(f"Low IOU ({iou_confirmation:.2f} <= 0.8) between first and confirmation detections for camera {cam_id}, assuming person is moving, resetting detection cycle.")
                        detection_result_1 = None
                        frame_result_1 = None
                        timestamp_result_1 = None
                        rtdetr_confidence = None
                        box_result_1_xyxy = None # 重置边界框
                        continue # 重置检测周期


                    else: # 理论上不应该到这里，以防万一
                        logging.warning(f"Unexpected IOU confirmation result for camera {cam_id}, resetting detection cycle.")
                        detection_result_1 = None
                        frame_result_1 = None
                        timestamp_result_1 = None
                        rtdetr_confidence = None
                        box_result_1_xyxy = None # 重置边界框
                        continue


                else:  # 不是第一次检测
                    # 忽略后续的检测，直到重置
                    pass

            else:  # RT-DETR 未检测到目标
                # logging.info(f"Camera {cam_id}: RT-DETR did not detect a person at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
                detection_result_1 = None
                frame_result_1 = None
                timestamp_result_1 = None
                rtdetr_confidence = None
                box_result_1_xyxy = None # 重置边界框
                pass

            elapsed_time = time.time() - start_time  # 计算处理时间
            wait_time = max(0, 1 - elapsed_time)  # 等待时间
            if wait_time > 0:
                time.sleep(wait_time)

        except Exception as e:
            logging.exception(f"Exception in process_camera for camera {cam_id}: {e}")
            time.sleep(5)
            detection_result_1 = None # 异常发生时也重置检测状态，避免卡住
            frame_result_1 = None
            timestamp_result_1 = None
            rtdetr_confidence = None
            box_result_1_xyxy = None # 重置边界框

def main():
    """
    主函数，启动所有摄像头的处理线程。
    """
    threads = []
    keyframe_threads = {}  # 存储关键帧线程的字典
    previous_boxes = {}  # 存储每个摄像头的前一个边界框的字典

    for i, rtsp_url in enumerate(rtsp_urls):
        keyframe_thread = KeyFrameThread(rtsp_url)
        keyframe_thread.start()  # 调用后自动执行run方法
        keyframe_threads[i] = keyframe_thread

        thread = threading.Thread(target=process_camera, args=(rtsp_url, i, previous_boxes, keyframe_thread))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()